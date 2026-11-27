"""Split a line of labels that sit under (or over) a row of evenly spaced icons.

Menus repeat a cell of icon and centered label at a constant pitch. Their labels can sit as
close together as the glyphs inside a word, so detection merges the row. The icons fix the
cells: fit their pitch, cut the line midway between cells where the line has no ink, and keep
each label's own characters. OCR rows that are only readings of those icons are removed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from dantetl.domain import Rect, union_bounds
from dantetl.ocr.ink import Ink, ink
from dantetl.ocr.values import TextCharacter, TextRegion

_BAND = 3.0  # Search this many line heights beside the line for its icons.
_PIXELS = 16  # Line height in analyzed pixels; icons only need their extents.
_MAX_LINES = 16  # Plausible merged-label lines analyzed per scan.
_MAX_LABEL = 12
_LABEL_STEP = 1.13
_ICON_READING = 0.5
_SENTENCE = frozenset("、。，．,.!?！？「」『』")


@dataclass(frozen=True, slots=True)
class _Grid:
    pitch: float
    phase: float  # Center of cell zero, in source pixels.
    icons: tuple[tuple[float, float], ...]  # Horizontal spans of the found icons.
    top: float
    bottom: float

    def center(self, cell: int) -> float:
        return self.phase + cell * self.pitch

    def cell(self, x: float) -> int:
        return round((x - self.phase) / self.pitch)


def _blobs(found: Ink) -> list[tuple[float, float, float, float]]:
    """Horizontally overlapping components merged, as source (left, right, top, bottom)."""
    merged: list[list[int]] = []
    for c in found.components:
        if merged and c.left < merged[-1][1]:
            m = merged[-1]
            m[1], m[2], m[3] = max(m[1], c.right), min(m[2], c.top), max(m[3], c.bottom)
        else:
            merged.append([c.left, c.right, c.top, c.bottom])
    return [
        (
            found.source_x(left),
            found.source_x(right),
            found.y + top / found.scale,
            found.y + bottom / found.scale,
        )
        for left, right, top, bottom in merged
        if left > 0 and right < found.width
    ]


def _grid(bgr: np.ndarray, line: Rect, above: bool) -> _Grid | None:
    h = line.height
    reach = round(_BAND * h)  # Wide enough for whole icons beside the line's ends.
    band = Rect(
        line.x - reach, line.y - reach if above else line.y + h, line.width + 2 * reach, reach
    )
    found = ink(bgr, band, min(1.0, _PIXELS / h))
    if found is None:
        return None
    blobs = [
        b for b in _blobs(found) if b[3] - b[2] >= 0.9 * h and 0.5 * h <= b[1] - b[0] <= 3.5 * h
    ]
    if len(blobs) < 2:
        return None
    height = median(b[3] - b[2] for b in blobs)
    middle = median((b[2] + b[3]) / 2 for b in blobs)
    icons = [
        b
        for b in blobs
        if abs((b[2] + b[3]) / 2 - middle) <= 0.35 * height
        and 0.7 * height <= b[3] - b[2] <= 1.4 * height
    ]
    if len(icons) < 2:
        return None
    # Icons belong to this line: they end within one line height of it.
    edge = median(b[3] for b in icons) if above else median(b[2] for b in icons)
    if (line.y - edge if above else edge - (line.y + h)) > h:
        return None
    centers = [(b[0] + b[1]) / 2 for b in icons]
    steps = np.diff(centers)
    pitch = float(np.median(steps))
    if pitch < 1.5 * h:
        return None
    cells = np.rint((np.array(centers) - centers[0]) / pitch)
    # One missing icon at most between neighbours, and no irregular spacing.
    if np.any(np.diff(cells) < 1) or np.any(np.diff(cells) > 2):
        return None
    if np.any(np.abs(steps - np.diff(cells) * pitch) > 0.15 * pitch):
        return None
    slope, intercept = np.polyfit(cells, centers, 1)
    return _Grid(
        float(slope),
        float(intercept),
        tuple((b[0], b[1]) for b in icons),
        min(b[2] for b in icons),
        max(b[3] for b in icons),
    )


def _center(char: TextCharacter) -> float:
    return char.bounds.x + char.bounds.width / 2


def _cuts(bgr: np.ndarray, line: TextRegion, grid: _Grid) -> list[float] | None:
    """Cell boundaries across the line, each moved into a gap of the line's own ink."""
    chars = line.characters
    first, last = grid.cell(_center(chars[0])), grid.cell(_center(chars[-1]))
    icon_cells = [grid.cell((a + b) / 2) for a, b in grid.icons]
    # Three or more icons fix the pitch well enough to reach one cell past them; two do not.
    spare = 1 if len(icon_cells) >= 3 else 0
    # The labels lie under the icon row, at most one cell past its ends; prose runs further.
    if last == first or first < min(icon_cells) - spare or last > max(icon_cells) + spare:
        return None
    b = line.bounds
    found = ink(bgr, b, min(1.0, _PIXELS * 2 / b.height))
    if found is None:
        return None
    strokes = [
        (found.source_x(c.left), found.source_x(c.right))
        for c in found.components
        if found.source_length(c.bottom - c.top) >= 0.3 * b.height
    ]
    cuts = []
    for cell in range(first, last):
        boundary = grid.center(cell) + grid.pitch / 2
        reach = 0.15 * grid.pitch
        gaps = [
            x
            for x in np.linspace(boundary - reach, boundary + reach, 13)
            if not any(left < x < right for left, right in strokes)
        ]
        if not gaps:
            return None  # A cell boundary through ink: these are not separate labels.
        cuts.append(float(min(gaps, key=lambda x: abs(x - boundary))))
    return cuts


def _separate(pieces: list[list[TextCharacter]]) -> bool:
    """Separately centered labels break the line's character pitch at every cut.

    A sentence keeps its pitch across any cut (measured 0.8-1.1x); labels centered in their own
    cells step further (measured 1.16x and more), and menus label without sentence punctuation.
    """
    if any(c.text in _SENTENCE for piece in pieces for c in piece):
        return False
    inside = [
        _center(b) - _center(a) for piece in pieces for a, b in zip(piece, piece[1:], strict=False)
    ]
    if not inside:
        return False
    pitch = median(inside)
    return all(
        _center(right[0]) - _center(left[-1]) >= _LABEL_STEP * pitch
        for left, right in zip(pieces, pieces[1:], strict=False)
    )


def _labels(bgr: np.ndarray, line: TextRegion, grid: _Grid) -> tuple[TextRegion, ...] | None:
    cuts = _cuts(bgr, line, grid)
    if cuts is None:
        return None
    pieces: list[list[TextCharacter]] = [[] for _ in range(len(cuts) + 1)]
    for char in line.characters:
        pieces[sum(_center(char) > cut for cut in cuts)].append(char)
    if any(not piece or len(piece) > _MAX_LABEL for piece in pieces):
        return None
    if not _separate(pieces):
        return None
    b = line.bounds
    labels = []
    for piece in pieces:
        bounds = union_bounds(tuple(c.bounds for c in piece))
        middle = bounds.x + bounds.width / 2
        # Each label is centered in its own cell, as menus lay them out.
        if abs(middle - grid.center(grid.cell(middle))) > 0.2 * grid.pitch:
            return None
        labels.append(
            TextRegion(
                "".join(c.text for c in piece),
                Rect(bounds.x, b.y, bounds.width, b.height),
                tuple(piece),
                line_confidence=line.line_confidence,
            )
        )
    return tuple(labels)


def _reads_icons(region: TextRegion, grid: _Grid) -> bool:
    """A row read off the icons themselves: inside their band, most glyphs on an icon, and
    read weakly (measured 0.15-0.47 median; real lettering of that size reads far higher)."""
    b = region.bounds
    middle = b.y + b.height / 2
    chars = region.characters
    if region.vertical or not chars or not grid.top <= middle <= grid.bottom:
        return False
    if median(c.confidence for c in chars) >= _ICON_READING:
        return False
    overlapped = sum(a < b.x + b.width and b.x < z for a, z in grid.icons)
    # The row may extend past the icons seen beside one line; the fitted cells continue.
    on = sum(
        abs(_center(c) - grid.center(grid.cell(_center(c)))) <= 0.3 * grid.pitch for c in chars
    )
    return overlapped >= 2 and on * 2 >= len(chars)


def _same(a: TextRegion, b: TextRegion) -> bool:
    if a.text != b.text:
        return False
    x = min(a.bounds.x + a.bounds.width, b.bounds.x + b.bounds.width) - max(a.bounds.x, b.bounds.x)
    y = min(a.bounds.y + a.bounds.height, b.bounds.y + b.bounds.height) - max(
        a.bounds.y, b.bounds.y
    )
    smaller = min(a.bounds.width * a.bounds.height, b.bounds.width * b.bounds.height)
    return x > 0 and y > 0 and x * y >= 0.6 * smaller


def _candidate(region: TextRegion) -> bool:
    """A line that could pass the split checks; others cost no pixel work or budget.

    Merged labels have no sentence punctuation and their character pitch steps up somewhere.
    """
    chars, b = region.characters, region.bounds
    if not (
        not region.vertical
        and 4 <= len(chars) <= 40
        and "".join(c.text for c in chars) == region.text
        and b.height >= 8
        and b.width >= 4 * b.height
        and not any(c.text in _SENTENCE for c in chars)
    ):
        return False
    steps = [_center(b) - _center(a) for a, b in zip(chars, chars[1:], strict=False)]
    return max(steps) >= _LABEL_STEP * median(steps)


def _weak(region: TextRegion) -> bool:
    return (
        not region.vertical
        and bool(region.characters)
        and median(c.confidence for c in region.characters) < _ICON_READING
    )


def _beside(line: TextRegion, rows: Sequence[TextRegion]) -> bool:
    """A weakly read row within the icon search band of this line, which may read its icons."""
    b, reach = line.bounds, _BAND * line.bounds.height
    return (
        not line.vertical
        and bool(line.characters)
        and b.height >= 8
        and any(
            r is not line
            and r.bounds.x < b.x + b.width + reach
            and b.x - reach < r.bounds.x + r.bounds.width
            and b.y - reach <= r.bounds.y + r.bounds.height / 2 <= b.y + b.height + reach
            for r in rows
        )
    )


def split_icon_grids(bgr: np.ndarray, regions: Sequence[TextRegion]) -> tuple[TextRegion, ...]:
    """Bounded to the first plausible merged-label lines per scan; other lines and their order
    are unchanged."""
    replaced: dict[int, tuple[TextRegion, ...]] = {}
    grids: list[_Grid] = []
    analyzed = 0
    weak = [r for r in regions if _weak(r)]
    for index, region in enumerate(regions):
        # A line that cannot be split is still examined when a weak row beside it may be a
        # reading of icons over it, so that reading can be removed.
        splittable = _candidate(region)
        if not splittable and not _beside(region, weak):
            continue
        if analyzed == _MAX_LINES:
            break
        analyzed += 1
        for above in (True, False):
            grid = _grid(bgr, region.bounds, above)
            if grid is None:
                continue
            # The icon row stands on its own evidence, even when this line stays whole.
            grids.append(grid)
            if splittable and (labels := _labels(bgr, region, grid)) is not None:
                replaced[index] = labels
                break
    if not grids:
        return tuple(regions)
    result: list[TextRegion] = []
    for index, region in enumerate(regions):
        if index in replaced:
            # Overlapping detections of one row split into the same labels; keep each once.
            result.extend(r for r in replaced[index] if not any(_same(r, o) for o in result))
        elif not any(_reads_icons(region, grid) for grid in grids):
            result.append(region)
    return tuple(result)
