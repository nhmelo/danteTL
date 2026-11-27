"""Non-overlapping English placement with bounded, source-anchored candidates.

Qt measures text. Only layouts clear of every higher-priority placement reach
the paint surface. Translation and OCR retain their own lifetimes.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace

from dantetl.domain import MAX_OVERLAY_LABELS, Rect
from dantetl.i18n import tr
from dantetl.ocr.regions import intersects
from dantetl.ocr.text_content import same_label_crop
from dantetl.ui.text_layout import EnglishLayout, layout_english


@dataclass(frozen=True, slots=True)
class EnglishLabel:
    key: tuple[tuple[int, int, str], ...]
    text: str
    source: Rect
    line_height: int
    vertical: bool = False


def _overlap(a: Rect, b: Rect) -> float:
    width = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    height = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    return width * height / max(1, min(a.width * a.height, b.width * b.height))


class _Index:
    """Exact rectangle broad phase; oversized boxes have a bounded all-pairs fallback."""

    def __init__(self, boxes: tuple[Rect, ...] = ()) -> None:
        self.boxes: list[Rect] = []
        self.grid: dict[tuple[int, int], set[int]] = {}
        self.wide: set[int] = set()
        for box in boxes:
            self.add(box)

    def add(self, box: Rect) -> None:
        i = len(self.boxes)
        self.boxes.append(box)
        cells = self._cells(box)
        if cells is None:
            self.wide.add(i)
        else:
            for cell in cells:
                self.grid.setdefault(cell, set()).add(i)

    @staticmethod
    def _cells(box: Rect) -> tuple[tuple[int, int], ...] | None:
        if box.width <= 0 or box.height <= 0:
            return ()
        left, top = box.x // 256, box.y // 256
        right, bottom = (box.x + box.width - 1) // 256, (box.y + box.height - 1) // 256
        if (right - left + 1) * (bottom - top + 1) > 64:
            return None
        return tuple((x, y) for y in range(top, bottom + 1) for x in range(left, right + 1))

    def neighbors(self, box: Rect) -> tuple[int, ...]:
        cells = self._cells(box)
        if cells is None:
            return tuple(
                i
                for i, other in enumerate(self.boxes)
                if min(other.size) > 0 and intersects(box, other)
            )
        if not cells:
            return ()
        candidates = self.wide.copy()
        for cell in cells:
            candidates.update(self.grid.get(cell, ()))
        return tuple(i for i in sorted(candidates) if intersects(box, self.boxes[i]))


def _candidates(
    label: EnglishLabel,
    viewport: tuple[int, int],
    automatic: bool,
    pixels: int,
    base: EnglishLayout,
    rewrap: bool,
) -> tuple[EnglishLayout, ...]:
    layouts = [base]
    for width in (label.source.width, round(base.bounds.width * 1.5)) if rewrap else ():
        alternate = layout_english(
            label.text,
            label.source,
            viewport,
            automatic=automatic,
            manual_pixels=pixels,
            vertical=label.vertical,
            source_line_height=label.line_height,
            wrap_width=width,
        )
        # Avoid solving collisions by throwing away more English or lowering its font.
        if alternate.pixel_size >= base.pixel_size and (
            not alternate.truncated
            or (base.truncated and sum(map(len, alternate.lines)) >= sum(map(len, base.lines)))
        ):
            layouts.append(alternate)
    result = [base]
    for layout in layouts:
        for candidate in _anchors(label.source, viewport, layout):
            if candidate not in result:
                result.append(candidate)
    return tuple(result)


def _anchors(
    source: Rect, viewport: tuple[int, int], layout: EnglishLayout
) -> Iterator[EnglishLayout]:
    width, height = layout.bounds.size
    centered = source.x + (source.width - width) // 2
    for y in (source.y, source.y + source.height - height):
        for x in dict.fromkeys((centered, source.x, source.x + source.width - width)):
            yield replace(
                layout,
                bounds=Rect(
                    max(0, min(x, viewport[0] - width)),
                    max(0, min(y, viewport[1] - height)),
                    width,
                    height,
                ),
            )


def _reserved_overlap(box: Rect, label: EnglishLabel, reserved: _Index) -> float:
    return sum(
        _overlap(box, reserved.boxes[j])
        for j in reserved.neighbors(box)
        if reserved.boxes[j] != label.source
    )


def _best_free_anchor(
    options: tuple[EnglishLayout, ...], label: EnglishLabel, occupied: _Index, reserved: _Index
) -> EnglishLayout | None:
    base = options[0].bounds
    winner = None
    lowest = float("inf")
    for option in options:
        box = option.bounds
        if occupied.neighbors(box):
            continue
        overlap = _reserved_overlap(box, label, reserved)
        distance = abs(box.x - base.x) + abs(box.y - base.y)
        cost = (
            1000 * overlap
            + distance / max(12, label.line_height)
            + abs(box.width - base.width) / max(1, base.width)
        )
        if cost == 0:
            return option
        if cost < lowest:
            winner, lowest = option, cost
    return winner


def _relocate_neighbor(
    options: tuple[EnglishLayout, ...],
    placed: dict[EnglishLabel, EnglishLayout],
    occupied: _Index,
    reserved: _Index,
    viewport: tuple[int, int],
) -> EnglishLayout | None:
    """Try one move with identical text, wrapping and font. Never evict a winner."""
    labels = tuple(placed)
    for option in options:
        blockers = occupied.neighbors(option.bounds)
        if len(blockers) != 1:
            continue
        slot = blockers[0]
        neighbor = labels[slot]
        old = placed[neighbor]
        for move in _anchors(neighbor.source, viewport, old):
            if intersects(option.bounds, move.bounds) or any(
                other != slot for other in occupied.neighbors(move.bounds)
            ):
                continue
            if _reserved_overlap(move.bounds, neighbor, reserved) > _reserved_overlap(
                old.bounds, neighbor, reserved
            ):
                continue
            placed[neighbor] = move
            return option
    return None


class EnglishPlacement:
    """Cache one live scene and at most 128 placements, scoped to exact published content.

    Each ready label gets the best free anchor in priority order. A label can
    return when its competitor disappears. Only the exact scene is cached.
    """

    def __init__(self) -> None:
        self._scene: tuple[object, ...] = ()
        self._placed: dict[EnglishLabel, EnglishLayout] = {}

    def clear(self) -> None:
        self._scene = ()
        self._placed.clear()

    def arrange(
        self,
        labels: tuple[EnglishLabel, ...],
        viewport: tuple[int, int],
        obstacles: tuple[Rect, ...] = (),
        *,
        automatic: bool = True,
        pixels: int = 20,
    ) -> tuple[EnglishLayout | None, ...]:
        """One slot per input label. None means every readable anchor was occupied."""
        if len(labels) > MAX_OVERLAY_LABELS or len(obstacles) > MAX_OVERLAY_LABELS:
            raise ValueError(tr("Translation placement exceeds the live track bound."))
        ordered = tuple(sorted(dict.fromkeys(labels), key=lambda label: label.key))
        obstacles = tuple(sorted(set(obstacles), key=lambda box: (box.y, box.x, *box.size)))
        scene = (ordered, obstacles, viewport, automatic, pixels)
        if scene == self._scene:
            return tuple(self._placed.get(label) for label in labels)

        bases = tuple(
            layout_english(
                label.text,
                label.source,
                viewport,
                automatic=automatic,
                manual_pixels=pixels,
                vertical=label.vertical,
                source_line_height=label.line_height,
            )
            for label in ordered
        )
        sources = _Index(tuple(label.source for label in ordered))
        source_texts = tuple("".join(t for _, _, t in label.key) for label in ordered)

        def priority(i: int):
            label = ordered[i]
            text = source_texts[i]
            # An extra icon can add characters to a short menu read. Prefer the
            # corroborating text-only crop in exactly the cases tracking recognizes.
            decorated = (
                len(label.key) == 1
                and not label.vertical
                and len(text) <= 12
                and any(
                    len(ordered[j].key) == 1
                    and not ordered[j].vertical
                    and ordered[j].source.x - label.source.x >= label.line_height
                    and same_label_crop(label.source, ordered[j].source, text, source_texts[j])
                    for j in sources.neighbors(label.source)
                )
            )
            return (
                label.key[0][0] < 0,
                decorated,
                -sum(c.isalnum() for c in text),
                -len(label.key),
                label.source.width * label.source.height,
                label.source.y,
                label.source.x,
                text,
                label.key,
            )

        base_index = _Index(tuple(base.bounds for base in bases))
        reserved = _Index(tuple(dict.fromkeys((*obstacles, *sources.boxes))))
        occupied = _Index()
        fixed_sources = _Index()
        relocation_attempts = 16
        screen = Rect(0, 0, *viewport)
        placed: dict[EnglishLabel, EnglishLayout] = {}
        for i in sorted(range(len(ordered)), key=priority):
            label, base = ordered[i], bases[i]
            # Every horizontal anchor covers its on-screen source, including
            # after rewrapping or clamping. A winner there rules out all anchors.
            if (
                not label.vertical
                and screen.contains(label.source)
                and fixed_sources.neighbors(label.source)
            ):
                continue
            neighbors = base_index.neighbors(base.bounds)
            best = base
            if (
                len(neighbors) > 1
                or any(reserved.boxes[j] != label.source for j in reserved.neighbors(base.bounds))
                or occupied.neighbors(base.bounds)
            ):
                options = _candidates(
                    label, viewport, automatic, pixels, base, len(neighbors) <= 16
                )
                best = _best_free_anchor(options, label, occupied, reserved)
                if best is None and relocation_attempts and len(neighbors) <= 16:
                    relocation_attempts -= 1
                    best = _relocate_neighbor(options, placed, occupied, reserved, viewport)
                    if best is not None:
                        occupied = _Index(tuple(p.bounds for p in placed.values()))
                if best is None:
                    continue
            occupied.add(best.bounds)
            placed[label] = best
            if not label.vertical and screen.contains(label.source):
                fixed_sources.add(label.source)
        self._scene, self._placed = scene, placed
        return tuple(placed.get(label) for label in labels)
