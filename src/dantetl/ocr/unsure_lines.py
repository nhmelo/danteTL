"""Keep lines the detector doubted when the recognizer reads them with confidence.

A detector with a fixed input shrinks a 4K capture fourfold, and small labels then score just
below its acceptance threshold, often in pieces or cut short. Doubted boxes along one line are
merged and grown along their own ink to the line's ends, then read once: labels read with high,
even confidence, while icons and backgrounds read as weak fragments. A line that cannot be
completed is not read, and single characters are not trusted this way.

Reads are bounded per scan and remembered by the exact pixels of their box, accepted or not,
so an unchanged screen is read once.
"""

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dantetl.domain import Rect, union_bounds
from dantetl.ocr.ink import Ink, ink
from dantetl.ocr.regions import intersects
from dantetl.ocr.values import TextRegion

UNSURE = 0.3  # Detector score floor; acceptance is 0.5. Background clutter scores lower.
_READS = 6  # New reads per scan, most confident lines first.
_BOXES = 64  # Most confident doubted boxes considered per scan.
_MEMORY = 256
_MEAN = 0.8
_LEAST = 0.5
_REACH = 4.0  # Line heights searched beyond a doubted box for the rest of its line.
_GAP = 0.6  # Glyphs of one line sit closer than this many line heights apart.
_GLYPH_PIXELS = 32
_GROWTH_STEPS = 4
_LONGEST = 40  # Line heights; longer doubted lines are left to the coarse read.


@dataclass(frozen=True, slots=True)
class Unsure:
    bounds: Rect
    score: float


def _line_shaped(box: Rect) -> bool:
    return box.width >= 1.5 * box.height or box.height >= 1.5 * box.width


def confident(line: TextRegion) -> bool:
    """Two or more characters, every one plausible and most read firmly."""
    chars = line.characters
    return (
        len(chars) >= 2
        and "".join(c.text for c in chars) == line.text
        and min(c.confidence for c in chars) >= _LEAST
        and sum(c.confidence for c in chars) / len(chars) >= _MEAN
    )


def _digest(image: np.ndarray, box: Rect) -> bytes:
    pixels = np.ascontiguousarray(image[box.y : box.y + box.height, box.x : box.x + box.width])
    return hashlib.blake2b(pixels.tobytes(), digest_size=16).digest()


def _same_line(a: Rect, b: Rect) -> bool:
    overlap = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    gap = max(a.x - (b.x + b.width), b.x - (a.x + a.width))
    return overlap >= 0.5 * min(a.height, b.height) and gap <= max(a.height, b.height)


def _rows(boxes: Sequence[Rect]) -> list[Rect]:
    """Boxes chained along one text line, as one box per line (the detector splits and
    truncates small lines unpredictably when shrunk). Chaining is transitive."""
    rows = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if _same_line(rows[i], rows[j]):
                    rows[i] = union_bounds((rows[i], rows.pop(j)))
                    merged = True
                    break
            if merged:
                break
    return sorted(rows, key=lambda r: (r.y, r.x))


def _pieces(found: Ink, middle: float, h: int) -> list[tuple[float, float]]:
    """Horizontal extents of ink that can continue a line of height ``h`` centered at ``middle``.

    Glyph-sized ink continues a line, and so do thin glyph-width strokes such as ー or 一; a long
    rule (a UI separator) does not.
    """
    return sorted(
        (found.source_x(c.left), found.source_x(c.right))
        for c in found.components
        if abs(found.y + (c.top + c.bottom) / 2 / found.scale - middle) <= 0.4 * h
        and found.source_length(c.bottom - c.top) <= 1.6 * h
        and (
            found.source_length(c.bottom - c.top) >= 0.3 * h
            or found.source_length(c.right - c.left) <= 1.5 * h
        )
    )


def _complete(image: np.ndarray, box: Rect, accepted: Sequence[Rect]) -> Rect | None:
    """The box grown along its own ink to the line's ends, or None if it cannot be completed.

    A detector box from a shrunk image can stop inside a word, and the recognizer then reads
    the part confidently (クエス for クエスト). Ink within a glyph gap of the line continues it;
    accepted lines are barriers. The search re-centres on the grown line until a clear gap is
    seen inside the searched area at both ends, or the image edge is reached.
    """
    h = box.height
    if box.width < box.height:  # Vertical lines keep their detected extent.
        return box
    reach, gap = round(_REACH * h), _GAP * h
    left, right = float(box.x), float(box.x + box.width)
    for _ in range(_GROWTH_STEPS):
        area = Rect(
            round(left) - reach,
            box.y - round(0.3 * h),
            round(right - left) + 2 * reach,
            round(1.6 * h),
        )
        found = ink(image, area, min(1.0, _GLYPH_PIXELS / h))
        if found is None:
            return box
        barriers = [a for a in accepted if intersects(a, area)]
        pieces = _pieces(found, box.y + h / 2, h)
        grown = True
        while grown:
            grown = False
            for start, end in pieces:
                if end < left - gap or start > right + gap or (left <= start and end <= right):
                    continue
                span = Rect(round(start), box.y, max(1, round(end - start)), h)
                if any(intersects(span, a) for a in barriers):
                    continue
                left, right, grown = min(left, start), max(right, end), True
        # A clear gap must lie inside what was searched, unless the image itself ends there.
        seen_left = found.x == 0 or left - gap >= found.x
        seen_right = found.x + found.source_length(found.width) >= image.shape[
            1
        ] - 1 or right + gap <= found.x + found.source_length(found.width)
        if seen_left and seen_right:
            x = max(0, round(left))
            return Rect(x, box.y, min(image.shape[1], round(right)) - x, h)
        if right - left > _LONGEST * h:
            break
    return None  # Still continuing: too long to be a doubted label, or not a line.


class UnsureReads:
    """Owned by one recognizer on its worker thread."""

    def __init__(self) -> None:
        self._memory: OrderedDict[tuple[Rect, bytes], TextRegion | None] = OrderedDict()

    def lines(
        self,
        image: np.ndarray,
        unsure: Sequence[Unsure],
        accepted: Sequence[Rect],
        read: Callable[[list[Rect]], tuple[TextRegion, ...]],
    ) -> tuple[TextRegion, ...]:
        """Confident complete lines where the detector doubted unexplained boxes.

        Doubted boxes along one line are read once, grown to the line's own ends; a line that
        cannot be completed is not read, so a fragment never passes for a word.
        """
        boxes = [
            u.bounds
            for u in sorted(unsure, key=lambda u: -u.score)
            if u.score >= UNSURE
            and min(u.bounds.size) > 0
            and _line_shaped(u.bounds)
            and not any(intersects(u.bounds, a) for a in accepted)
        ][:_BOXES]
        rows = [r for row in _rows(boxes) if (r := _complete(image, row, accepted)) is not None]
        keys = [(row, _digest(image, row)) for row in rows]
        missing = [key for key in dict.fromkeys(keys) if key not in self._memory][:_READS]
        if missing:
            answers = read([box for box, _ in missing])
            if len(answers) == len(missing):
                for key, answer in zip(missing, answers, strict=True):
                    self._remember(key, answer if confident(answer) else None)
        known = [self._memory[key] for key in keys if key in self._memory]
        for key in keys:
            if key in self._memory:
                self._memory.move_to_end(key)
        found: list[TextRegion] = []
        lines = [line for line in known if line is not None]
        for line in sorted(lines, key=lambda r: -r.bounds.width * r.bounds.height):
            if not any(intersects(line.bounds, f.bounds) for f in found):
                found.append(line)
        return tuple(found)

    def _remember(self, key: tuple[Rect, bytes], line: TextRegion | None) -> None:
        self._memory[key] = line
        while len(self._memory) > _MEMORY:
            self._memory.popitem(last=False)
