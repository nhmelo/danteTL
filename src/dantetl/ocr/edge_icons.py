"""Separate a pictorial unit at either end of a horizontal line from the lettering beside it.

OCR reads an icon beside a label as extra characters (み軌跡, 8通常攻撃, 〇戻る) and can lose the
neighbouring glyph. Measured against the line's own glyphs, an icon is taller and heavier, or at
glyph height much heavier. A unit OCR read only weakly needs less: at glyph height and heavier,
set apart from the lettering by at least twice the spacing between its glyphs. Its characters
are dropped and the line is re-read without it. The re-read is accepted only if it keeps every
character on the text side.
"""

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from statistics import median

import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.ink import ink
from dantetl.ocr.values import TextCharacter, TextRegion

_GLYPH_PIXELS = 32  # Measure units at about this glyph height, whatever the capture scale.
_MAX_READS = 16
_CONFIDENT_LETTER = 0.9
_WORD_LETTER = 0.5  # Weaker letters are no evidence of a word; see text_content.
# Read as list bullets or brackets, these glyphs carry structure that grouping relies on.
_MARKERS = frozenset("•●○◆◇▶▷→・※")
_BRACKETS = frozenset(("Ps", "Pe", "Pi", "Pf"))


@dataclass(frozen=True, slots=True)
class _Unit:
    left: float
    right: float
    height: float
    stroke: float
    partial: bool


@dataclass(frozen=True, slots=True)
class _Cut:
    """An icon's span, and a re-read box when the icon displaced a glyph beside it."""

    index: int
    icon: tuple[float, float]
    leading: bool
    reread: Rect | None


def _units(bgr: np.ndarray, bounds: Rect) -> tuple[_Unit, ...]:
    """Ink units across the line in source pixels.

    Components stacked over each other form one unit; side-by-side radicals stay separate.
    """
    pad_x, pad_y = round(bounds.height * 0.5), max(2, round(bounds.height * 0.15))
    area = Rect(
        bounds.x - pad_x, bounds.y - pad_y, bounds.width + 2 * pad_x, bounds.height + 2 * pad_y
    )
    found = ink(bgr, area, min(1.0, _GLYPH_PIXELS / bounds.height))
    if found is None:
        return ()
    # [left, right, top, bottom, ink pixels, summed depth] for each unit, built left to right.
    groups: list[list[float]] = []
    for c in found.components:
        if groups and groups[-1][1] - c.left >= 0.5 * min(
            c.right - c.left, groups[-1][1] - groups[-1][0]
        ):
            g = groups[-1]
            g[1], g[2], g[3] = max(g[1], c.right), min(g[2], c.top), max(g[3], c.bottom)
            g[4] += c.pixels
            g[5] += c.depth
        else:
            groups.append([c.left, c.right, c.top, c.bottom, c.pixels, c.depth])
    return tuple(
        _Unit(
            found.source_x(left),
            found.source_x(right),
            found.source_length(bottom - top),
            found.source_length(2 * depth / pixels),
            left == 0 or right == found.width,
        )
        for left, right, top, bottom, pixels, depth in groups
    )


def _center(char: TextCharacter) -> float:
    return char.bounds.x + char.bounds.width / 2


def _over(chars: tuple[TextCharacter, ...], span: tuple[float, float]) -> tuple[bool, ...]:
    return tuple(span[0] <= _center(c) <= span[1] for c in chars)


def _pictorial(unit: _Unit, height: float, stroke: float) -> bool:
    return (unit.height >= 1.12 * height and unit.stroke >= 1.2 * stroke) or (
        unit.height >= height and unit.stroke >= 1.8 * stroke
    )


def _gap(a: _Unit, b: _Unit) -> float:
    return max(b.left - a.right, a.left - b.right)


def _set_apart(units: Sequence[_Unit], height: float, stroke: float) -> bool:
    """A heavier glyph-height unit separated from the lettering by twice its own spacing."""
    outer = units[0]
    spacing = max((_gap(a, b) for a, b in zip(units[1:], units[2:], strict=False)), default=0.0)
    return (
        outer.height >= 0.95 * height
        and outer.stroke >= 1.4 * stroke
        and _gap(outer, units[1]) >= max(0.4 * height, 2 * spacing)
    )


def _icon(units: Sequence[_Unit], line_height: int, weak: bool) -> int:
    """How many outer units (at most two) form an icon against the glyphs inside them.

    ``units`` start at the line's edge. A second unit belongs to the icon when it touches the
    first and shares its extra height; glyphs are never that much taller than their line. When
    OCR read the outer unit only weakly (``weak``), a set-apart heavier unit is an icon too.
    """
    for size in (2, 1):
        reference = [u for u in units[size:] if u.height >= 0.5 * line_height]
        if not reference:
            continue
        height = median(u.height for u in reference)
        stroke = median(u.stroke for u in reference)
        if sum(u.right - u.left for u in reference) < 1.5 * height:
            continue
        if size == 1:
            pictorial = _pictorial(units[0], height, stroke)
            return int(pictorial or (weak and _set_apart(units, height, stroke)))
        if not _pictorial(units[0], height, stroke):
            continue
        if units[1].height >= 1.12 * height and _gap(units[0], units[1]) <= 0.25 * height:
            return 2
    return 0


def _paired(char: TextCharacter, chars: tuple[TextCharacter, ...]) -> bool:
    """An opening bracket closed later in the line, or a closing one opened earlier."""
    i = next(i for i, c in enumerate(chars) if c is char)
    opening = unicodedata.category(char.text[:1]) in ("Ps", "Pi")
    others = chars[i + 1 :] if opening else chars[:i]
    closing = ("Pe", "Pf") if opening else ("Ps", "Pi")
    return any(unicodedata.category(c.text[:1]) in closing for c in others)


def _removable(char: TextCharacter, chars: tuple[TextCharacter, ...]) -> bool:
    """Pixels alone never remove a confidently read letter, such as a bold word, nor bullets
    and brackets that carry structure. A weak bracket with no partner in its line carries none:
    OCR reads round icons as ( or ）."""
    if unicodedata.category(char.text[:1]) in _BRACKETS:
        return char.confidence < _WORD_LETTER and not _paired(char, chars)
    return not (
        char.text in _MARKERS or (char.text.isalpha() and char.confidence >= _CONFIDENT_LETTER)
    )


def _window(bounds: Rect, leading: bool) -> Rect:
    """The edge of the line holding an icon and enough glyphs to compare it with."""
    width = min(bounds.width, 6 * bounds.height)
    x = bounds.x if leading else bounds.x + bounds.width - width
    return Rect(x, bounds.y, width, bounds.height)


def _cut(index: int, region: TextRegion, units: tuple[_Unit, ...], leading: bool) -> _Cut | None:
    whole = [u for u in units if not u.partial]
    ordered = whole if leading else whole[::-1]
    b = region.bounds
    if len(ordered) < 2:
        return None
    outer = _over(region.characters, (ordered[0].left, ordered[0].right))
    read = [c for c, inside in zip(region.characters, outer, strict=True) if inside]
    weak = bool(read) and all(c.confidence < _WORD_LETTER for c in read)
    size = _icon(ordered, b.height, weak)
    if not size:
        return None
    icon = ordered[:size]
    # The first glyph-sized unit, past specks of background between it and the icon.
    neighbour = next(u for u in ordered[size:] if u.height >= 0.5 * b.height)
    span = (min(u.left for u in icon), max(u.right for u in icon))
    over = _over(region.characters, span)
    marked = [c for c, inside in zip(region.characters, over, strict=True) if inside]
    kept = [c for c, inside in zip(region.characters, over, strict=True) if not inside]
    # Icons beside words only: a number keeps its digits.
    if not 1 <= len(marked) <= 2 or not all(_removable(c, region.characters) for c in marked):
        return None
    if not any(c.text.isalpha() and c.confidence >= _WORD_LETTER for c in kept):
        return None
    # Ink beside the icon that no kept character covers was displaced by it: re-read the line.
    first = kept[0] if leading else kept[-1]
    middle = (neighbour.left + neighbour.right) / 2
    displaced = middle < first.bounds.x if leading else middle > first.bounds.x + first.bounds.width
    if not displaced:
        return _Cut(index, span, leading, None)
    if leading:
        left, right = round((span[1] + neighbour.left) / 2), b.x + b.width
    else:
        left, right = b.x, round((neighbour.right + span[0]) / 2)
    if right - left < b.height:
        return None
    return _Cut(index, span, leading, Rect(left, b.y, right - left, b.height))


def _accepts(kept: str, reread: TextRegion, leading: bool) -> bool:
    """The text side survives unchanged; at most two confident glyphs return at the cut."""
    text, chars = reread.text, reread.characters
    if (
        not kept
        or "".join(c.text for c in chars) != text
        or any(len(c.text) != 1 for c in chars)
        or not (text.endswith(kept) if leading else text.startswith(kept))
    ):
        return False
    extra = chars[: len(chars) - len(kept)] if leading else chars[len(kept) :]
    return len(extra) <= 2 and all(c.text.isalnum() and c.confidence >= 0.6 for c in extra)


def trim_edge_icons(
    bgr: np.ndarray,
    regions: tuple[TextRegion, ...],
    read: Callable[[list[Rect]], tuple[TextRegion, ...]],
) -> tuple[TextRegion, ...]:
    """Use at most one bounded batch of re-reads; preserve each line's detector bounds."""
    cuts: list[_Cut] = []
    reads = 0
    for index, region in enumerate(regions):
        chars = region.characters
        b = region.bounds
        if (
            region.vertical
            or len(chars) < 2
            or "".join(c.text for c in chars) != region.text
            or b.height < 8
            or b.width < 2 * b.height
        ):
            continue
        # An icon's glyphs lead or end the line; inspect pixels only where one could go.
        for leading, edge in ((True, chars[0]), (False, chars[-1])):
            rest = chars[1:] if leading else chars[:-1]
            if not _removable(edge, chars) or not any(
                c.text.isalpha() and c.confidence >= _WORD_LETTER for c in rest
            ):
                continue
            cut = _cut(index, region, _units(bgr, _window(b, leading)), leading)
            if cut is not None and (cut.reread is None or reads < _MAX_READS):
                cuts.append(cut)
                reads += cut.reread is not None
                break
    if not cuts:
        return regions
    boxes = [cut.reread for cut in cuts if cut.reread is not None]
    results = read(boxes) if boxes else ()
    # Pair answers with their lines only when every box was read; otherwise drop glyphs alone.
    answers = iter(results if len(results) == len(boxes) else ())
    trimmed = list(regions)
    for cut in cuts:
        original = regions[cut.index]
        inside = _over(original.characters, cut.icon)
        kept = tuple(c for c, over in zip(original.characters, inside, strict=True) if not over)
        reread = next(answers, None) if cut.reread is not None else None
        text = "".join(c.text for c in kept)
        chars = reread.characters if reread and _accepts(text, reread, cut.leading) else kept
        trimmed[cut.index] = replace(
            original, text="".join(c.text for c in chars), characters=chars
        )
    return tuple(trimmed)
