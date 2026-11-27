"""Bounded dialogue grouping over stable lines, independent of Qt and model runtimes."""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import groupby
from statistics import median

from dantetl.domain import Rect, union_bounds
from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.lettering import Lettering
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.tracking import TextActivity
from dantetl.translation.dialogue_paths import prose_segments, reading_paths
from dantetl.translation.limits import MAX_SOURCE_CHARACTERS

_MAX_ROW_OVERLAP = 0.1
_MAX_ROW_GAP = 0.95
_PERCENT_NUMBER = r"[0-9０-９]+(?:[.．][0-9０-９]+)?[%％]"
_PERCENT = re.compile(rf"[tTｔＴ*＊]?[+＋\-−±:：]?{_PERCENT_NUMBER}")
_CLAUSE_VALUE = re.compile(rf"[+＋\-−±:：]?\s*{_PERCENT_NUMBER}")
_SIGNED_PERCENT = re.compile(rf"[+＋\-−±]?{_PERCENT_NUMBER}")


def _percentage_row(language: SourceLanguage, text: str, *, clause: bool = False) -> bool:
    if text.endswith((language.period, ".", "!", "！")):
        text = text[:-1]
    if clause:
        lead = language.percentage_lead and language.percentage_lead.match(text)
        return bool(lead and _CLAUSE_VALUE.fullmatch(text, lead.end()))
    return bool(_PERCENT.fullmatch(text))


def _continues_percentage(language: SourceLanguage, text: str, *, in_paragraph: bool) -> bool:
    if _percentage_row(language, text, clause=True):
        return True
    if (
        not in_paragraph
        or not language.percentage_label_terms
        or language.lists.line_marker.match(text)
    ):
        return False
    label = language.lists.inline_label.match(text)
    if not label or not any(term in label[1] for term in language.percentage_label_terms):
        return False
    # A stat label and value alone cannot continue prose. Require more text after the operand.
    operand = _SIGNED_PERCENT.match(text, label.end())
    if not operand:
        return False
    tail = text[operand.end() :]
    return tail.startswith((language.period, ".", "!", "！")) and language.contains_text(tail[1:])


def join_lines(lines: tuple[TextTrack, ...], language: SourceLanguage = DEFAULT_SOURCE) -> str:
    text = ""
    for line in lines:
        part = line.text.strip()
        if language.needs_space(text[-1:], part[:1]):
            text += " "
        text += part
    return text


@dataclass(frozen=True, slots=True)
class TextBlock:
    lines: tuple[TextTrack, ...]
    dialogue: bool = field(default=False, compare=False)
    language: SourceLanguage = field(default=DEFAULT_SOURCE, kw_only=True)

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError(tr("A text block needs at least one line."))

    @property
    def text(self) -> str:
        return join_lines(self.lines, self.language)

    @property
    def bounds(self) -> Rect:
        return union_bounds(tuple(line.bounds for line in self.lines))

    @property
    def line_height(self) -> int:
        return round(median(line.bounds.height for line in self.lines))

    @property
    def key(self) -> tuple[tuple[int, int, str], ...]:
        return tuple((line.track_id, line.text_revision, line.text) for line in self.lines)


@dataclass(frozen=True, slots=True)
class GroupedText:
    blocks: tuple[TextBlock, ...]
    ready: tuple[TextBlock, ...]


def _on_column_pitch(previous: Rect, current: Rect, following: Rect) -> bool:
    """Predict the next row center from the last two, allowing for shorter glyphs."""
    pitch = current.y - previous.y + (current.height - previous.height) / 2
    step = following.y - current.y + (following.height - current.height) / 2
    unit = current.height
    return (
        unit * 0.9 <= pitch <= unit * 1.9
        and abs(step - pitch) <= unit * 0.25
        and abs(current.x - following.x) <= unit * 0.5
    )


@dataclass(frozen=True, slots=True)
class _WrapLimits:
    height_ratio: float
    max_width: float
    quoted: bool
    item: bool
    ends_sentence: bool


def _wrap_limits(
    language: SourceLanguage,
    upper: TextTrack,
    lower: TextTrack,
    *,
    revealing: bool,
    paragraph: bool,
    opening_quote: bool,
    on_pitch: bool,
) -> _WrapLimits | None:
    """Reject independent items, or derive the geometry allowed by their text."""
    a, b = upper.bounds, lower.bounds
    first, second = upper.text.strip(), lower.text.strip()
    introduces_value = first.endswith(language.value_introductions)
    ends_sentence = first.endswith(language.endings)
    percentage = (
        (paragraph or introduces_value) and len(second) <= 16 and _percentage_row(language, second)
    )
    numeric_tail = (
        percentage
        and not ends_sentence
        and (not second.startswith(("*", "＊")) or on_pitch)
        and b.width <= min(a.width * 0.35, a.height * 4)
    )
    if (
        upper.vertical
        or lower.vertical
        or not language.contains_text(first)
        or not (language.contains_text(second) or numeric_tail)
    ):
        return None
    labelled = language.lists.has_inline_label(first)
    if (
        (len(first) < 10 and not (introduces_value and numeric_tail))
        or not second
        or (first.endswith((":", "：")) and not labelled)
    ):
        return None
    quoted = opening_quote or bool(language.quote_start.match(first))
    if language.is_choice(first) and not (labelled or opening_quote):
        return None  # Bare menu choices stay independent; labelled descriptions can wrap.
    inline_quote = False
    if not numeric_tail and language.quote_start.match(second):
        inline_quote = on_pitch and not ends_sentence and language.quoted_term_with_suffix(second)
        if not inline_quote:
            return None
    if (
        not numeric_tail
        and language.starts_item(second)
        and (
            ends_sentence
            or not _continues_percentage(language, second, in_paragraph=paragraph or labelled)
        )
    ):
        return None
    item = not opening_quote and language.starts_item(first)
    # A few final glyphs can have shorter ink bounds than a full line.
    short_ending = numeric_tail or (
        1 < len(second) <= 4
        and second.endswith(language.endings)
        and b.width <= min(a.width * 0.35, a.height * 4)
        and b.height < a.height
    )
    width_growth = (
        paragraph
        or revealing
        or language.internal_prose(first)
        or first.endswith((*language.endings, *language.commas))
        or quoted
    )
    height_ratio = 1.45 if short_ending or inline_quote else 1.3
    if numeric_tail and on_pitch:
        height_ratio = 1.65  # Digits may occupy much less of the same physical row.
    height = max(1, min(a.height, b.height))
    max_width = max(a.width * 1.5, a.width + height * 4) if width_growth else a.width * 1.25
    return _WrapLimits(height_ratio, max_width, quoted, item, ends_sentence)


def _can_follow(
    language: SourceLanguage,
    upper: TextTrack,
    lower: TextTrack,
    *,
    revealing: bool = False,
    previous_row: Rect | None = None,
    opening_quote: bool = False,
    lettering: dict[TextTrack, Lettering] | None = None,
) -> bool:
    a, b = upper.bounds, lower.bounds
    height = max(1, min(a.height, b.height))
    gap = b.y - a.y - a.height
    if not -height * _MAX_ROW_OVERLAP <= gap <= height * _MAX_ROW_GAP:
        return False  # Distant rows need no text/quote analysis in dense layouts.
    on_pitch = previous_row is not None and _on_column_pitch(previous_row, a, b)
    limits = _wrap_limits(
        language,
        upper,
        lower,
        revealing=revealing,
        paragraph=previous_row is not None,
        opening_quote=opening_quote,
        on_pitch=on_pitch,
    )
    if limits is None:
        return False
    height_ratio = limits.height_ratio
    if (
        height_ratio < 1.45
        and previous_row
        and on_pitch
        and not limits.ends_sentence
        and abs(previous_row.x - a.x) <= height * 0.5
        and min(a.width, b.width) >= 8 * height
        and max(a.height, b.height) / 1.3 <= previous_row.height <= height * 1.3
    ):
        # Two successive ink heights may differ even when each matches the preceding
        # body row. Require the same margin and row pitch to bridge that variation.
        height_ratio = 1.45
    if (
        lettering
        and height * height_ratio < max(a.height, b.height) <= height * 1.45
        and len(_panel_paragraph(language, (upper, lower), lettering)) == 2
    ):
        # Shorter words may have smaller OCR ink boxes even in the same font.
        height_ratio = 1.45
    return (
        max(a.height, b.height) <= height * height_ratio
        # Wrapped rows can widen. A heading/name is substantially narrower than body
        # prose; measure that distinction in font units as well as relative width.
        and b.width <= limits.max_width
        and (
            abs(a.x - b.x) <= max(6, height * 0.75)
            or (not limits.item and _centered_wrap(language, upper, lower))
            # Ink extents include or omit the leading dash/bracket depending on the
            # scan. Allow a hanging mark or indent; overlap/gap/font checks still apply.
            or (limits.quoted and abs(a.x - b.x) <= height * 2)
            # A list marker/inline label may hang to the left of wrapped body text.
            or (limits.item and 0 <= b.x - a.x <= height * 2)
        )
        and min(a.x + a.width, b.x + b.width) - max(a.x, b.x) >= min(a.width, b.width) * 0.8
        # A sentence boundary is weaker evidence of continuity, not a hard stop.
        # Require tightly spaced full prose rows; marked choices remain excluded.
        and (
            revealing
            or not limits.ends_sentence
            or (gap <= height * 0.55 and min(a.width, b.width) >= height * 8)
        )
    )


_SUBTITLE_TAIL_MAX_WIDTH_RATIO = 0.7


def _centered_subtitle_tail(language: SourceLanguage, upper: TextTrack, lower: TextTrack) -> bool:
    """Recognize a short final row from its inset on both sides and a textual continuation."""
    a, b = upper.bounds, lower.bounds
    unit = min(a.height, b.height)
    gap = b.y - a.y - a.height
    if not (
        unit > 0
        and not lower.vertical
        and max(a.height, b.height) <= unit * 1.4
        and -0.25 * unit <= gap <= 0.55 * unit
        and a.width >= 14 * unit
        and 0 < b.width <= _SUBTITLE_TAIL_MAX_WIDTH_RATIO * a.width
        and abs((a.x + a.width / 2) - (b.x + b.width / 2)) <= 0.75 * unit
    ):
        return False
    first, second = upper.text.rstrip(), lower.text.strip()
    if not second or language.starts_item(second):
        return False
    body = second[:-1]
    if not body or any(mark in body for mark in language.terminal):
        return False
    return language.closes_wrapped_quote(first, second) or (
        max(a.height, b.height) <= unit * 1.3
        and gap >= -0.1 * unit
        and second[-1] in language.terminal
        and language.subtitle_continuation(first, body)
    )


def _can_start_subtitle(language: SourceLanguage, first: TextTrack) -> bool:
    text = first.text.strip()
    return (
        not first.vertical
        and len(text) >= 14
        and not text.endswith((*language.endings, "：", ":"))
        and language.contains_text(text)
        and not language.starts_item(text)
    )


def _centered_subtitle(
    language: SourceLanguage, first: TextTrack, available: Iterable[TextTrack]
) -> tuple[TextTrack, ...]:
    if not _can_start_subtitle(language, first):
        return (first,)
    tails = [row for row in available if _centered_subtitle_tail(language, first, row)]
    if len(tails) == 1:
        pair = (first, tails[0])
        if len(join_lines(pair, language)) <= MAX_SOURCE_CHARACTERS:
            return pair
    return (first,)


def _centered_wrap(language: SourceLanguage, upper: TextTrack, lower: TextTrack) -> bool:
    """A centered continuation under prose, not generally centered menu rows.

    Similar-width rows use the ordinary alignment tolerance. A narrow final row gets
    extra leeway: missing edge glyphs on its much longer predecessor can shift the center.
    Font, gap, choice and unambiguous-successor checks still belong to _can_follow.
    """
    a, b = upper.bounds, lower.bounds
    unit = max(1, min(a.height, b.height))
    tolerance = 1.25 if b.width <= a.width * 0.7 else 0.75
    text = upper.text.strip()
    return abs((a.x + a.width / 2) - (b.x + b.width / 2)) <= unit * tolerance and (
        language.internal_prose(text)
        or text.endswith(language.commas)
        or bool(language.quote_start.match(text))
    )


def _wrapped_prose(language: SourceLanguage, lines: tuple[TextTrack, ...]) -> bool:
    if len(lines) < 2:
        return False
    first, second = lines[:2]
    if first.text.rstrip().endswith(language.endings):
        return False
    return (
        _centered_wrap(language, first, second)
        or (
            len(second.text.strip()) >= 10
            and min(first.bounds.width, second.bounds.width)
            >= max(first.bounds.width, second.bounds.width) * 0.8
        )
        # A hanging marker can make the opening much wider than the next row.
        # Two body rows at the predicted spacing independently support that opening.
        or (
            len(lines) >= 3
            and language.lists.has_inline_label(first.text.lstrip())
            and first.bounds.x + second.bounds.height * 0.5 < second.bounds.x
            and _on_column_pitch(first.bounds, second.bounds, lines[2].bounds)
        )
    )


def _percentage_ending(language: SourceLanguage, text: str) -> bool:
    return _percentage_row(language, text) or _percentage_row(language, text, clause=True)


def _dialogue_evidence(
    language: SourceLanguage,
    lines: tuple[TextTrack, ...],
    *,
    prior_comma: bool = False,
    prior_prose: bool = False,
) -> bool:
    first, last = lines[0].text.lstrip(), lines[-1].text.strip()
    return (
        last.endswith(language.endings)
        or bool(language.quote_start.match(first))
        or (len(lines) > 1 and _percentage_ending(language, last))
        or prior_comma
        or (prior_prose and _wrapped_prose(language, lines))
    )


def _paragraph_boundary(
    language: SourceLanguage, lines: tuple[TextTrack, ...], upper: TextTrack, lower: TextTrack
) -> bool:
    """A ragged terminal row followed by a full row starts a new paragraph.

    Establish the column from earlier rows, so a sentence ending on a full-width
    row stays joined. A trailing dash is continuation, not terminal punctuation.
    """
    if len(lines) < 2 or not upper.text.rstrip().endswith(language.endings):
        return False
    width = max(line.bounds.width for line in lines)
    unit = max(1, min(upper.bounds.height, lower.bounds.height))
    return (
        upper.bounds.width <= width * 0.85
        and width - upper.bounds.width >= unit * 2
        and lower.bounds.width >= width * 0.9
    )


def _row_fragments(
    language: SourceLanguage,
    first: TextTrack,
    available: tuple[TextTrack, ...],
    column: Rect,
    limit: int,
) -> tuple[TextTrack, ...]:
    """Join split detections only inside a column established by earlier prose rows."""
    row = (first,)
    while len(row) < limit:
        a = row[-1].bounds
        candidates = []
        for next_line in available:
            b = next_line.bounds
            height = max(1, min(a.height, b.height))
            if (
                next_line not in row
                and not next_line.vertical
                and language.contains_text(next_line.text)
                and not language.is_choice(next_line.text.lstrip())
                and 0 <= b.x - a.x - a.width <= height * 2
                and max(a.height, b.height) <= height * 1.3
                and abs((a.y + a.height / 2) - (b.y + b.height / 2)) <= height * 0.25
                and b.x + b.width <= column.x + column.width + height * 0.75
            ):
                candidates.append(next_line)
        if len(candidates) != 1:
            break
        row = (*row, candidates[0])
    return row


def _same_row(previous: TextTrack, current: TextTrack) -> bool:
    a, b = previous.bounds, current.bounds
    unit = min(a.height, b.height)
    return (
        previous.text == current.text
        and abs(a.y - b.y) <= unit * 0.25
        and abs(a.height - b.height) <= unit * 0.25
        and abs(a.x - b.x) <= unit * 0.5
        and abs(a.x + a.width - b.x - b.width) <= unit * 0.5
    )


def _supported_row(
    language: SourceLanguage,
    row: TextTrack,
    parts: tuple[TextTrack, ...],
    nearby: tuple[TextTrack, ...],
) -> bool:
    """Infer a new horizontal join only at character-like spacing inside wrapped prose."""
    if len(row.text) < 10 or (
        all(len(t.text.strip()) < 10 for t in parts)
        and row.bounds.width <= row.bounds.height * 6.5 * len(parts)
        and not language.internal_prose(row.text)
        and not language.quote_start.match(row.text)
    ):
        # A strip of compact labels above body text can fill precisely the same width.
        # Only an intact observation can disambiguate such an unmarked pair.
        return False
    for left, right in zip(parts, parts[1:], strict=False):
        a, b = left.bounds, right.bounds
        # Ink-box width per character estimates pitch on the first observation.
        pitch = min(a.height, b.height, a.width / len(left.text), b.width / len(right.text))
        if b.x - (a.x + a.width) > pitch * 0.35:
            return False

    a = row.bounds
    supporting: list[TextTrack] = []
    for t in nearby:
        b = t.bounds
        if language.starts_item(t.text.lstrip()) or abs(a.x - b.x) > a.height * 0.75:
            continue
        if not (a.height * 0.75 <= abs(a.y - b.y) <= a.height * 3.8):
            continue
        if max(a.height, b.height) > min(a.height, b.height) * 1.3:
            continue
        # An intact body row should extend across the seam and occupy this column.
        right_edge = b.x + b.width
        ragged_ending = (
            b.y > a.y
            and t.text.rstrip().endswith(language.endings)
            and parts[-1].bounds.x + a.height <= right_edge <= a.x + a.width
        )
        if ragged_ending or abs((a.x + a.width) - right_edge) <= a.height * 1.5:
            supporting.append(t)
    adjacent = [
        t for t in supporting if _can_follow(language, row, t) or _can_follow(language, t, row)
    ]
    below = sum(t.bounds.y > a.y for t in adjacent)
    if below > 1 or len(adjacent) - below > 1:
        return False
    if len(adjacent) == 2:
        return True  # An intact row on either side supports the same column.
    for t in adjacent:
        pair = (row, t) if a.y < t.bounds.y else (t, row)
        head = pair[0].text.strip()
        if _dialogue_evidence(
            language,
            pair,
            prior_comma=head.endswith(language.commas),
            prior_prose=language.internal_prose(head),
        ) or any(
            _can_follow(language, t, other) or _can_follow(language, other, t)
            for other in supporting
        ):
            return True
    return False


def _physical_rows(
    language: SourceLanguage, tracks: tuple[TextTrack, ...], previous: dict[int, TextTrack]
) -> dict[TextTrack, tuple[TextTrack, ...]]:
    """Find unambiguous left-to-right chains before walking the paragraph top-to-bottom."""
    ordered = sorted(
        (
            t
            for t in tracks
            if not t.vertical and min(t.bounds.size) > 0 and language.contains_text(t.text)
        ),
        key=lambda t: t.bounds.y + t.bounds.height / 2,
    )
    centers = [t.bounds.y + t.bounds.height / 2 for t in ordered]

    def near(center: float, radius: float) -> list[TextTrack]:
        return ordered[
            bisect_left(centers, center - radius) : bisect_right(centers, center + radius)
        ]

    successors: dict[TextTrack, TextTrack | None] = {}
    predecessors: dict[TextTrack, int] = {}
    for center, right in zip(centers, ordered, strict=True):
        b = right.bounds
        text = right.text.lstrip()
        if language.starts_item(text) or language.quote_start.match(text):
            continue
        lefts = []
        for left in near(center, b.height * 0.25):
            a = left.bounds
            unit = min(a.height, b.height)
            if (
                0 <= b.x - a.x - a.width <= unit * 0.8
                and max(a.height, b.height) <= unit * 1.3
                and abs(center - a.y - a.height / 2) <= unit * 0.25
                and not language.starts_item(left.text.lstrip())
            ):
                lefts.append(left)
        # Cut the edge at a recurring column margin, preserving joins inside either column.
        if not lefts or any(
            abs(t.bounds.x - b.x) <= b.height * 0.5 and abs(t.bounds.y - b.y) >= b.height * 0.75
            for t in near(center, b.height * 2)
        ):
            continue
        predecessors[right] = len(lefts)
        for left in lefts:
            successors[left] = None if left in successors else right

    rows = {}
    for first in successors:
        if first in predecessors:
            continue
        chain = [first]
        while (following := successors.get(chain[-1])) and predecessors[following] == 1:
            chain.append(following)
        if len(chain) < 2 or chain[-1] in successors:
            continue
        parts = tuple(chain)
        row = replace(
            first,
            text=join_lines(parts, language),
            bounds=union_bounds(tuple(t.bounds for t in parts)),
        )
        if len(row.text) > MAX_SOURCE_CHARACTERS:
            continue
        center = row.bounds.y + row.bounds.height / 2
        if any(
            (old := previous.get(t.track_id)) is not None and _same_row(old, row) for t in parts
        ) or _supported_row(language, row, parts, tuple(near(center, row.bounds.height * 4))):
            rows[row] = parts
    return rows


def _quote_openings(
    language: SourceLanguage, rows: list[TextTrack], dash_edges: dict[TextTrack, int]
) -> dict[int, tuple[TextTrack, int]]:
    """Geometry for a supported opening and the only row it may newly join."""
    quotes = language.dash_quotes
    if quotes is None:
        return {}
    openings = {}
    for row in rows:
        text = row.text.strip()
        edge = dash_edges.get(row)
        if row.vertical or not text:
            continue
        body = quotes.opening(text, recovered=edge is not None)
        if (
            not body
            or not language.contains_text(body)
            or language.lists.has_inline_label(body)
            or "%" in body
            or "％" in body
        ):
            continue
        a = row.bounds
        if edge is not None:
            a = replace(a, x=edge, width=a.width + a.x - edge)
        above, below = [], []
        for other in rows:
            b = other.bounds
            h = min(a.height, b.height)
            if other.vertical or abs(a.x - b.x) > h * 1.2 or max(a.height, b.height) > h * 1.32:
                continue
            step = b.y - a.y + (b.height - a.height) / 2
            if h * 0.9 <= abs(step) <= h * 1.95:
                (above if step < 0 else below).append(other)
        if len(above) > 1 or len(below) != 1:
            continue
        if above and (
            not above[0].text.rstrip().endswith(language.endings)
            or language.starts_item(above[0].text)
        ):
            continue
        next_text = below[0].text.strip()
        if language.starts_item(next_text) or not (
            next_text.endswith((*quotes.closers, *language.commas, "/"))
            or language.internal_prose(next_text)
        ):
            continue
        opening = replace(row, bounds=a)
        if _can_follow(language, opening, below[0], opening_quote=True):
            openings[row.track_id] = (opening, below[0].track_id)
    return openings


def _panel_paragraph(
    language: SourceLanguage, lines: tuple[TextTrack, ...], lettering: dict[TextTrack, Lettering]
) -> tuple[TextTrack, ...]:
    """Recover a ragged, unpunctuated wrap inside one closely fitting text panel."""
    first = lines[0]
    a = first.bounds
    style = lettering.get(first)
    opening = first.text.strip()
    if (
        style is None
        or style.panel is None
        or a.width < 12 * a.height
        or language.starts_item(opening)
        or opening.endswith((*language.endings, "...", "…", "~", "～"))
        or not 0 <= a.y - style.panel.top <= 1.5 * a.height
    ):
        return (first,)
    words = opening.split()
    if not (
        language.internal_prose(opening)
        or (language.prose_script and language.prose_script.search(opening))
        or (len(words) >= 4 and any(word[0].islower() for word in words[1:]))
    ):
        return (first,)  # A wide heading and its byline can fit the same card.
    panel = style.panel
    for index, line in enumerate(lines[1:], start=1):
        b = line.bounds
        unit = min(a.height, b.height)
        other = lettering.get(line)
        if (
            other is None
            or other.panel is None
            or not panel.agrees(other.panel, 0.35 * unit)
            or abs(b.x - a.x) > 0.55 * unit
            or line.text.strip() in language.standalone_menu_labels
            or any(abs(x - y) >= 85 for x, y in zip(style.ink, other.ink, strict=True))
        ):
            break
        if b.width <= 0.8 * a.width and 0 <= panel.bottom - b.y - b.height <= 1.5 * unit:
            return lines[: index + 1]
    return (first,)


def _adjacent_blocks(
    language: SourceLanguage,
    tracks: tuple[TextTrack, ...],
    growing: frozenset[int],
    dash_edges: dict[TextTrack, int],
    lettering: dict[TextTrack, Lettering],
) -> tuple[TextBlock, ...]:
    ordered = sorted(tracks[:128], key=lambda t: (t.bounds.y, t.bounds.x, t.track_id))
    row_tops = [t.bounds.y for t in ordered]
    openings = _quote_openings(language, ordered, dash_edges)
    # A rejected chain can be retried starting one row later. Classify each
    # row once and update evidence only for rows newly preceding the tail.
    comma_rows = {t.track_id for t in ordered if t.text.rstrip().endswith(language.commas)}
    prose_rows = {t.track_id for t in ordered if language.internal_prose(t.text.strip())}
    narrowest = min((t.bounds.width for t in ordered), default=0)
    # With no textual, visual or subtitle evidence, every attempted chain
    # would be rejected. A menu can then skip the adjacency search altogether.
    if not (
        growing
        or openings
        or comma_rows
        or prose_rows
        or any(style.panel is not None for style in lettering.values())
        or any(
            _dialogue_evidence(language, (t,))
            or _percentage_ending(language, t.text.strip())
            or (
                narrowest <= _SUBTITLE_TAIL_MAX_WIDTH_RATIO * t.bounds.width
                and _can_start_subtitle(language, t)
            )
            for t in ordered
        )
    ):
        return tuple(TextBlock((t,), language=language) for t in ordered)

    @lru_cache(maxsize=256)
    def successors(
        upper: TextTrack, previous: Rect | None, revealing: bool, continuation: int | None
    ) -> tuple[TextTrack, ...]:
        # A rejected paragraph can be retraced from its second row. Keep its
        # scored edges for this layout, including pitch, growth and quote context.
        bounds = upper.bounds
        bottom = bounds.y + bounds.height
        height = max(1, bounds.height)
        start = bisect_left(row_tops, bottom - height * _MAX_ROW_OVERLAP)
        stop = bisect_right(row_tops, bottom + height * _MAX_ROW_GAP)
        return tuple(
            line
            for line in ordered[start:stop]
            if line.track_id not in openings
            and _can_follow(
                language,
                upper,
                line,
                revealing=revealing or line.track_id in growing,
                previous_row=previous,
                opening_quote=line.track_id == continuation,
                lettering=lettering,
            )
        )

    used: set[int] = set()
    blocks: list[TextBlock] = []
    for first in ordered:
        if first.track_id in used:
            continue
        lines = (first,)
        source = first.text.strip()
        members = {first.track_id}
        revealing = first.track_id in growing
        last_row, continuation = openings.get(first.track_id, (first, None))
        previous_row = None
        centered_pair = False
        prior_comma = prior_prose = False
        while len(lines) < len(ordered):
            candidates = [
                line
                for line in successors(last_row, previous_row, revealing, continuation)
                if line.track_id not in used and line.track_id not in members
            ]
            if len(candidates) != 1:
                # Recover only an isolated opening pair. A new terminal join cannot
                # extend an existing paragraph or recruit a third row below the pair.
                if not candidates and len(lines) == 1 and continuation is None:
                    lines = _centered_subtitle(
                        language,
                        first,
                        (
                            t
                            for t in ordered
                            if t.track_id not in used and t.track_id not in openings
                        ),
                    )
                    centered_pair = len(lines) == 2
                break  # Ambiguous columns/layouts stay separate.
            if (
                len(lines) == 1
                and continuation is not None
                and candidates[0].track_id != continuation
            ):
                break
            if _paragraph_boundary(language, lines, last_row, candidates[0]):
                break
            row = (candidates[0],)
            if len(lines) >= 2 and _dialogue_evidence(
                language, lines, prior_comma=prior_comma, prior_prose=prior_prose
            ):
                row = _row_fragments(
                    language,
                    candidates[0],
                    tuple(
                        t for t in ordered if t.track_id not in used and t.track_id not in members
                    ),
                    union_bounds(tuple(t.bounds for t in lines)),
                    len(ordered) - len(lines),
                )
            row_text = join_lines(row, language)
            separator = " " if language.needs_space(source[-1:], row_text[:1]) else ""
            if len(source) + len(separator) + len(row_text) > MAX_SOURCE_CHARACTERS:
                break
            source += separator + row_text
            for prior in (*lines[-1:], *row[:-1]):
                prior_comma |= prior.track_id in comma_rows
                prior_prose |= prior.track_id in prose_rows
            lines += row
            members.update(t.track_id for t in row)
            revealing = revealing or any(t.track_id in growing for t in row)
            # Aggregate geometry is private to adjacency; requests retain every raw member.
            previous_row = last_row.bounds
            last_row = (
                row[0]
                if len(row) == 1 and row[0].text == row_text
                else replace(
                    row[0], text=row_text, bounds=union_bounds(tuple(t.bounds for t in row))
                )
            )
            # An unmarked button can share the column width and pitch immediately
            # below the closing quote. Its proximity adds no continuation evidence.
            quotes = language.dash_quotes
            if (
                continuation is not None
                and quotes
                and last_row.text.rstrip().endswith((*quotes.closers, "/"))
                and quotes.complete(first.text.strip(), source)
            ):
                break
        if len(lines) > 1 and not (
            centered_pair
            or first.track_id in openings
            or _dialogue_evidence(language, lines, prior_comma=prior_comma, prior_prose=prior_prose)
            or any(line.track_id in growing for line in lines)
        ):
            lines = _panel_paragraph(language, lines, lettering)
        used.update(line.track_id for line in lines)
        dialogue = len(lines) > 1 or (
            len(join_lines(lines, language)) >= 10
            and _dialogue_evidence(language, lines)
            and (first.track_id in growing or bool(language.quote_start.match(first.text.lstrip())))
        )
        blocks.append(TextBlock(lines, dialogue=dialogue, language=language))
    return tuple(blocks)


def _group_rows(
    language: SourceLanguage,
    rows: tuple[TextTrack, ...],
    enabled: bool,
    growing: frozenset[int],
    dash_edges: dict[TextTrack, int],
    lettering: dict[TextTrack, Lettering],
) -> tuple[TextBlock, ...]:
    """Assign each physical row once, in reading order.

    Adjacency is the default. A prose path can supersede those groups only when
    it covers every group it touches. Unclaimed runs retain their adjacency.
    """
    if not enabled:
        ordered = sorted(rows, key=lambda t: (t.bounds.y, t.bounds.x, t.track_id))
        return tuple(TextBlock((row,), language=language) for row in ordered)

    adjacent = _adjacent_blocks(language, rows, growing, dash_edges, lettering)
    owners = {row: block.lines for block in adjacent for row in block.lines}
    paragraphs = []
    for path in reading_paths(language, rows):
        members = set(path)
        if all(members.issuperset(group) for group in {owners[row] for row in path}):
            paragraphs.extend(prose_segments(rows, path, lettering, language))
    if not paragraphs:
        return adjacent

    claimed = {row for paragraph in paragraphs for row in paragraph}
    result = [TextBlock(paragraph, dialogue=True, language=language) for paragraph in paragraphs]
    for block in adjacent:
        if claimed.isdisjoint(block.lines):
            result.append(block)
            continue
        for taken, run in groupby(block.lines, claimed.__contains__):
            if not taken:
                lines = tuple(run)
                result.append(TextBlock(lines, dialogue=len(lines) > 1, language=language))
    return tuple(
        sorted(
            result, key=lambda b: (b.lines[0].bounds.y, b.lines[0].bounds.x, b.lines[0].track_id)
        )
    )


class DialogueGrouper:
    """Wait briefly for a whole group to settle; never queue partial old paragraphs."""

    def __init__(
        self, settle_seconds: float = 0.35, *, language: SourceLanguage = DEFAULT_SOURCE
    ) -> None:
        self._language = language
        self._settle_seconds = settle_seconds
        self._since: dict[tuple[tuple[int, int, str], ...], float] = {}
        self._layout_key: (
            tuple[
                tuple[TextTrack, ...],
                bool,
                frozenset[int],
                tuple[tuple[TextTrack, int], ...],
                tuple[tuple[TextTrack, Lettering], ...],
            ]
            | None
        ) = None
        self._layout: tuple[TextBlock, ...] = ()
        self._prior_rows: dict[int, TextTrack] = {}
        self._fragment_ids: set[int] = set()

    def reset(self) -> None:
        self._since.clear()
        self._layout_key, self._layout = None, ()
        self._prior_rows.clear()
        self._fragment_ids.clear()

    def _regroup(
        self,
        tracks: tuple[TextTrack, ...],
        enabled: bool,
        growing: frozenset[int],
        dash_edges: dict[TextTrack, int],
        lettering: dict[TextTrack, Lettering],
    ) -> tuple[TextBlock, ...]:
        rows = _physical_rows(self._language, tracks, self._prior_rows) if enabled else {}
        self._fragment_ids = {t.track_id for parts in rows.values() for t in parts}
        # Carry intact-row evidence only through an exact reconstruction in this scene.
        self._prior_rows = {
            t.track_id: old
            for row, parts in rows.items()
            for t in parts
            if (old := self._prior_rows.get(t.track_id)) and _same_row(old, row)
        }
        physical = tuple(t for t in tracks if t.track_id not in self._fragment_ids) + tuple(rows)
        row_growth = growing | {
            row.track_id for row, parts in rows.items() if any(t.track_id in growing for t in parts)
        }
        blocks = _group_rows(self._language, physical, enabled, row_growth, dash_edges, lettering)
        if not rows:
            return blocks
        return tuple(
            replace(b, lines=tuple(t for row in b.lines for t in rows.get(row, (row,))))
            for b in blocks
        )

    def update(
        self,
        visible: tuple[TextTrack, ...],
        confirmed: tuple[TextTrack, ...],
        now: float,
        *,
        enabled: bool = True,
        activity: tuple[TextActivity, ...] | None = None,
        complete_lines: frozenset[TextTrack] = frozenset(),
        dash_edges: dict[TextTrack, int] | None = None,
        lettering: dict[TextTrack, Lettering] | None = None,
    ) -> GroupedText:
        observed = {value.observation.track_id: value for value in activity or ()}
        growing = frozenset(key for key, value in observed.items() if value.growing)
        # Only reveal growth can reserve a different layout before commitment. Ordinary
        # candidate readings must not split/merge accepted groups and revoke their English.
        layout = tuple(
            value.observation
            if (value := observed.get(t.track_id)) and value.pending and value.growing
            else t
            for t in visible
        )
        displayed = {t.track_id: t for t in visible}
        dash_edges = dash_edges or {}
        lettering = lettering or {}
        layout_key = (layout, enabled, growing, tuple(dash_edges.items()), tuple(lettering.items()))
        if layout_key != self._layout_key:
            self._layout = self._regroup(layout[:128], enabled, growing, dash_edges, lettering)
            self._layout_key = layout_key
        if enabled:
            self._prior_rows.update(
                (t.track_id, t)
                for t in confirmed[:128]
                if not t.vertical
                and t.track_id not in self._fragment_ids
                and displayed.get(t.track_id) == t
                and not ((value := observed.get(t.track_id)) and value.pending)
            )
        blocks = tuple(
            replace(b, lines=tuple(displayed[t.track_id] for t in b.lines)) for b in self._layout
        )
        ready_keys = {(line.track_id, line.text_revision, line.text) for line in confirmed}
        since = {block.key: self._since.get(block.key, now) for block in blocks}
        ready = tuple(
            block
            for block in blocks
            if (
                enabled
                and len(block.lines) == 1
                and block.lines[0] in complete_lines
                and (member := observed.get(block.lines[0].track_id)) is not None
                and member.observation == block.lines[0]
                and not member.growing
                and not member.pending
                and self._language.explicit_sentence_end(block.text)
            )
            or (
                all(key in ready_keys for key in block.key)
                and (
                    self._activity_ready(block, observed, now, enabled)
                    if activity is not None
                    else len(block.lines) == 1 or now - since[block.key] >= self._settle_seconds
                )
            )
        )
        self._since = since
        return GroupedText(blocks, ready)

    def _activity_ready(
        self, block: TextBlock, activity: dict[int, TextActivity], now: float, enabled: bool
    ) -> bool:
        members = [activity[t.track_id] for t in block.lines if t.track_id in activity]
        if len(members) != len(block.lines) or any(t.pending for t in members):
            return False
        revealing = any(t.growing for t in members)
        text = block.text
        dialogue = enabled and (
            len(block.lines) > 1
            or (
                len(text) >= 10
                and not self._language.is_choice(text)
                and not self._language.is_status_counter(text)
            )
        )
        if not revealing and not dialogue:
            return True  # Ordinary short menu labels need no extra debounce.
        complete = text.rstrip().endswith(self._language.endings)
        unfinished_prose = revealing or len(block.lines) > 1 or self._language.internal_prose(text)
        quiet = 0.18 if complete else 0.55 if unfinished_prose else 0.35
        if (
            enabled
            and (revealing or len(block.lines) == 1)
            and text.rstrip().endswith(self._language.commas)
        ):
            # Allow a clause pause before the continuation. OCR sometimes mistakes
            # a final period for a comma, so even this wait must be bounded.
            quiet = 1.5
        return now - max(t.changed_at for t in members) >= quiet
