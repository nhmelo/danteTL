"""Cheap frame evidence, OCR pacing, and reversible occlusion on a bounded worker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import ceil
from typing import TYPE_CHECKING, Any

from dantetl.domain import CapturedFrame, Rect
from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.observation import CandidateReview, MatchingEvidence, TextTrack, TrackPresence
from dantetl.ocr.presence import PixelEvidence, prepare_evidence
from dantetl.ocr.regions import padded, scan_regions
from dantetl.ocr.text_content import cleaned_rows
from dantetl.ocr.tracking import TextActivity, prepare_matching
from dantetl.ocr.values import MAX_OCR_REGIONS, MAX_TRACKS, OcrResult, TextRegion
from dantetl.ocr.visual_values import SettledScan, VisualState

logger = logging.getLogger(__name__)
cv2: Any
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from dantetl.ocr.pop_in import SentenceAppearance

    Bytes = NDArray[np.uint8]
    Gray = NDArray[np.float32]
    # percentile currently promotes normalized ink to float64; preserve that math.
    Ink = NDArray[np.float64]
    Mask = NDArray[np.bool_]
    Indices = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class _Job:
    frame: CapturedFrame
    result: OcrResult | None
    tracks: tuple[TextTrack, ...]
    assignments: tuple[int | None, ...]
    activity: tuple[TextActivity, ...]
    candidate: OcrResult | None = None
    evidence_bounds: tuple[Rect, ...] = ()
    matching_sources: tuple[TextTrack, ...] = ()


def _pixels(frame: CapturedFrame) -> Bytes | None:
    if len(frame.bgra) != frame.width * frame.height * 4:
        return None
    return np.frombuffer(frame.bgra, np.uint8).reshape(frame.height, frame.width, 4)


def _crop(
    pixels: Bytes,
    bounds: Rect,
    *,
    wide: bool = False,
    color: tuple[float, float, float] | None = None,
) -> Gray | None:
    h, w = pixels.shape[:2]
    if min(bounds.size) < 2 or not Rect(0, 0, w, h).contains(bounds):
        return None
    # Every template is at most 256 x 48; no source-sized crop copies are retained.
    size = (256 if wide else min(256, bounds.width), min(48, bounds.height))
    step = max(1, ceil(max(bounds.width / 1024, bounds.height / 192)))
    small = cv2.resize(
        pixels[
            bounds.y : bounds.y + bounds.height : step,
            bounds.x : bounds.x + bounds.width : step,
        ],
        size,
        interpolation=cv2.INTER_AREA,
    )
    return (
        _project_color(small, color)
        if color is not None
        else cv2.cvtColor(small, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    )


def _edges(gray: Gray) -> tuple[Gray, Gray]:
    return (
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1),
    )


def _changed(a: Gray, b: Gray) -> bool:
    if np.array_equal(a, b):
        return False
    # Gradients reject uniform brightness shifts; require more than a few noisy pixels.
    ax, ay = _edges(a)
    bx, by = _edges(b)
    return float(np.mean(np.maximum(np.abs(ax - bx), np.abs(ay - by)) > 35)) > 0.018


def _added_ink(
    a: Gray, b: Gray, *, foreground: bool = False, start: int = 0, min_height: float = 0
) -> bool:
    """Append-like stroke additions, rather than general background replacement."""
    if foreground:
        dark = float(np.median(a)) > float(np.mean(a))
        old, new = _ink(a, dark), _ink(b, dark)
        if old is None or new is None:
            # Faint typewriter text still needs to unlock the accepted shorter source.
            # Smooth lighting contributes little local detail; this only proposes work.
            before, after = (
                (np.abs(value - cv2.GaussianBlur(value, (0, 0), 2)) >= 4).astype(np.uint8)
                for value in (a, b)
            )
        else:
            before, after = (old >= 0.65).astype(np.uint8), (new >= 0.65).astype(np.uint8)
    else:
        ax, ay = _edges(a)
        bx, by = _edges(b)
        before = (np.maximum(np.abs(ax), np.abs(ay)) > 45).astype(np.uint8)
        after = (np.maximum(np.abs(bx), np.abs(by)) > 45).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    additions = (after & ~cv2.dilate(before, kernel))[:, start:]
    if min_height and additions.size:
        # A neutral glint beside uniformly colored text is not a new glyph. Still
        # admit substantial strokes in a different color (e.g. a white appended digit).
        _, labels, stats, _ = cv2.connectedComponentsWithStats(after[:, start:], 8)
        added_per_component = np.bincount(labels.ravel(), weights=additions.ravel())
        keep = (stats[:, cv2.CC_STAT_HEIGHT] >= min_height) & (
            added_per_component >= stats[:, cv2.CC_STAT_AREA] * 0.2
        )
        # A glint touching an existing letter cannot borrow that letter's height.
        keep[0] = False
        additions = additions & keep[labels]
    added = int(np.count_nonzero(additions))
    removed = int(np.count_nonzero((before & ~cv2.dilate(after, kernel))[:, start:]))
    return added > max(12, removed * 1.8) and (foreground or added < after.size * 0.2)


def _ink(gray: Gray, dark: bool) -> Ink | None:
    """High-contrast glyph cores; ignore dim texture behind translucent game labels."""
    gray = 255 - gray if dark else gray
    low, high = np.percentile(gray, (15, 99))
    if high - low < 45:
        return None
    return np.clip((gray - low) / (high - low), 0, 1)


@dataclass(slots=True)
class _InkReference:
    dark: bool
    before: Ink | None
    valid: bool

    @classmethod
    def prepare(cls, reference: Gray) -> _InkReference:
        dark = float(np.median(reference)) > float(np.mean(reference))
        before = _ink(reference, dark)
        return cls(
            dark, before, bool(before is not None and np.count_nonzero(before >= 0.75) >= 24)
        )


def _same_ink(
    reference: Gray,
    current: Gray,
    prepared: _InkReference | None = None,
    *,
    soft_edges: bool = False,
) -> bool | None:
    if np.array_equal(reference, current):
        return True
    prepared = prepared or _InkReference.prepare(reference)
    dark, before = prepared.dark, prepared.before
    if not prepared.valid or before is None:
        return None
    # Hover states may invert colors. Align at most one sampled pixel; never follow
    # detector geometry to decide whether the accepted region changed.
    padded = cv2.copyMakeBorder(current, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    scores = cv2.matchTemplate(padded, reference, cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(np.argmax(np.abs(scores)), scores.shape)
    aligned = padded[y : y + reference.shape[0], x : x + reference.shape[1]]
    after = _ink(aligned, dark if scores[y, x] >= 0 else not dark)
    if after is None:
        return False
    # Local cells catch one changed digit/stroke in a long label. Whole-box averages
    # would conceal a small but meaningful edit among many unchanged characters.
    step = max(6, reference.shape[0] // 2)
    starts = np.arange(0, reference.shape[1], step)
    difference = np.add.reduceat(
        # Color-projected antialiasing can cross a hard threshold over a white
        # particle. Measure stroke-strength change, without dilating away real edits.
        np.sum((before - after) ** 2, axis=0)
        if soft_edges
        else np.count_nonzero(
            ((before >= 0.75) & (after < 0.55)) | ((after >= 0.75) & (before < 0.55)), axis=0
        ),
        starts,
    )
    ink = np.add.reduceat(np.count_nonzero((before >= 0.65) | (after >= 0.65), axis=0), starts)
    return not bool(np.any(difference > np.maximum(4, ink * (0.06 if soft_edges else 0.12))))


@dataclass(slots=True)
class _FaintReference:
    """Local stroke detail for dim text over smooth moving light, at thumbnail size.

    This can add positive presence evidence only. A failed comparison never establishes
    absence, and every informative cell must match so one changed character can unlock OCR.
    """

    detail: Gray
    starts: Indices
    counts: Indices
    energy: Gray
    active: Mask
    valid: bool

    @classmethod
    def prepare(cls, reference: Gray) -> _FaintReference:
        detail = reference - cv2.GaussianBlur(reference, (0, 0), 2)
        starts = np.arange(0, reference.shape[1], max(6, reference.shape[0] // 2))
        counts = np.diff(np.append(starts, reference.shape[1])) * reference.shape[0]
        energy = np.add.reduceat(np.sum(detail * detail, axis=0), starts)
        active = energy / counts >= 1.5**2
        # Do not use a smooth lighting gradient itself as a text template.
        valid = np.count_nonzero(active) >= 2 and np.count_nonzero(np.abs(detail) >= 4) >= 24
        return cls(detail, starts, counts, energy, active, bool(valid))

    def matches(self, current: Gray) -> bool:
        if not self.valid:
            return False
        detail = current - cv2.GaussianBlur(current, (0, 0), 2)
        padded = cv2.copyMakeBorder(detail, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        scores = cv2.matchTemplate(padded, self.detail, cv2.TM_CCOEFF_NORMED)
        y, x = np.unravel_index(np.argmax(np.abs(scores)), scores.shape)
        aligned = padded[y : y + detail.shape[0], x : x + detail.shape[1]]
        energy = np.add.reduceat(np.sum(aligned * aligned, axis=0), self.starts)
        product = np.add.reduceat(np.sum(self.detail * aligned, axis=0), self.starts)
        correlation = np.abs(product) / np.maximum(1e-6, np.sqrt(self.energy * energy))
        return bool(
            np.all(correlation[self.active] >= 0.96)
            and np.all(energy[self.active] / self.counts[self.active] >= 1.0)
            and np.all(energy[self.active] >= self.energy[self.active] * 0.04)
            # Newly appearing strokes in a previously empty cell are real change too.
            and not np.any(energy[~self.active] / self.counts[~self.active] >= 1.5**2)
        )


@dataclass(slots=True)
class _GlyphReference:
    reference: Gray
    ink: _InkReference | None = None
    faint: _FaintReference | None = None
    soft_edges: bool = False

    def matches(self, current: Gray) -> bool | None:
        if np.array_equal(self.reference, current):
            return True
        if self.ink is None:
            self.ink = _InkReference.prepare(self.reference)
        same = _same_ink(self.reference, current, self.ink, soft_edges=self.soft_edges)
        if same is True:
            return True
        if self.faint is None:
            self.faint = _FaintReference.prepare(self.reference)
        return True if self.faint.matches(current) else same


# Coarse thumbnails may propose change or reversible hiding, but cannot prove a
# long line unchanged. Fine references preserve aspect ratio and glyph scale.
_FINE_PIXELS = 1_048_576
_FINE_LINE_PIXELS = 262_144


def _fine_shape(bounds: Rect, vertical: bool) -> tuple[int, int]:
    scale = min(1.0, 48 / (bounds.width if vertical else bounds.height))
    return max(1, round(bounds.width * scale)), max(1, round(bounds.height * scale))


def _fine_pixels(pixels: Bytes, bounds: Rect, vertical: bool, shape: tuple[int, int]) -> Bytes:
    area = pixels[bounds.y : bounds.y + bounds.height, bounds.x : bounds.x + bounds.width]
    image = cv2.resize(area, shape, interpolation=cv2.INTER_AREA)
    if vertical:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _project_color(image: Bytes, direction: tuple[float, float, float]) -> Gray:
    # Require agreement across the foreground's channel contrasts. A dot product
    # alone can mistake subpixel-font fringes over a white glint for new glyph cores.
    channels = image[:, :, :3].astype(np.float32)
    projected = np.full(image.shape[:2], np.inf, np.float32)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        contrast = direction[a] - direction[b]
        if abs(contrast) >= 0.25:
            projected = np.minimum(projected, (channels[:, :, a] - channels[:, :, b]) / contrast)
    return np.maximum(0, projected)


def _foreground_color(
    image: Bytes, track: TextTrack, region: TextRegion
) -> tuple[float, float, float] | None:
    """Optional evidence for uniformly colored glyphs with complete character locations.

    Never let a colored part of mixed text freeze a white/differently colored digit.
    Weak punctuation detections can be background specks; all letters/numbers and reliable
    punctuation must contain the foreground color. This does not alter the OCR reading.
    """
    chars = region.characters
    if (
        track.vertical
        or not chars
        or len(chars) > 512
        or any(len(c.text) != 1 for c in chars)
        or "".join(c.text for c in chars) != track.text
    ):
        return None
    rgb = image[:, :, :3].astype(np.float32)
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    vivid = (high >= 128) & (high - low >= 64)
    if not 24 <= np.count_nonzero(vivid) <= vivid.size * 0.6:
        return None
    chroma = rgb - rgb.mean(axis=2, keepdims=True)
    axis = np.median(chroma[vivid], axis=0)
    axis -= axis.mean()
    length = float(np.linalg.norm(axis))
    if length < 32:
        return None
    direction = tuple(float(n) for n in axis / length)
    direction = (direction[0], direction[1], direction[2])
    projected = _project_color(image, direction)
    threshold = max(32.0, float(np.percentile(projected, 99)) * 0.4)
    h, w = projected.shape
    b = track.bounds
    # A same-pixel OCR correction keeps accepted geometry. Its character boxes can
    # extend a couple of pixels beyond that geometry; sample only the accepted crop.
    metadata_bounds = Rect(b.x - 2, b.y - 2, b.width + 4, b.height + 4)
    for char in chars:
        if char.text.isspace():
            continue
        if not metadata_bounds.contains(char.bounds):
            return None
        c = char.bounds
        left, top = (c.x - b.x) * w // b.width, (c.y - b.y) * h // b.height
        right, bottom = (
            ceil((c.x + c.width - b.x) * w / b.width),
            ceil((c.y + c.height - b.y) * h / b.height),
        )
        patch = projected[max(0, top) : min(h, bottom), max(0, left) : min(w, right)]
        if np.count_nonzero(patch >= threshold) < max(2, patch.size * 0.02) and (
            char.text.isalnum() or not 0 <= char.confidence < 0.65
        ):
            return None
    return direction


@dataclass(slots=True)
class _FineReference:
    shape: tuple[int, int]
    tiles: tuple[tuple[int, _GlyphReference], ...]
    color: tuple[float, float, float] | None = None
    color_tiles: tuple[tuple[int, _GlyphReference], ...] = ()
    color_extension: Gray | None = None

    @property
    def pixels(self) -> int:
        return self.shape[0] * self.shape[1] * (2 if self.color_tiles else 1) + (
            self.color_extension.size if self.color_extension is not None else 0
        )

    @classmethod
    def create(cls, pixels: Bytes, track: TextTrack, budget: int) -> _FineReference | None:
        shape = _fine_shape(track.bounds, track.vertical)
        if shape[0] * shape[1] > min(budget, _FINE_LINE_PIXELS):
            return None
        image = _fine_pixels(pixels, track.bounds, track.vertical, shape)
        reference = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        # Overlap keeps a stroke at a tile boundary in an informative interior.
        tiles = tuple(
            (start, _GlyphReference(reference[:, start : start + 256]))
            for start in range(0, reference.shape[1], 224)
        )
        return cls(shape, tiles)

    def add_color(
        self, pixels: Bytes, track: TextTrack, region: TextRegion, extension: Rect, budget: int
    ) -> None:
        extra = self.shape[0] * self.shape[1] + 256 * min(48, extension.height)
        if extra > min(budget, _FINE_LINE_PIXELS - self.pixels):
            return
        image = _fine_pixels(pixels, track.bounds, track.vertical, self.shape)
        color = _foreground_color(image, track, region)
        colored = _project_color(image, color) if color is not None else None
        color_tiles = (
            tuple(
                (
                    start,
                    _GlyphReference(colored[:, start : start + 256], soft_edges=True),
                )
                for start in range(0, colored.shape[1], 224)
            )
            if colored is not None
            else ()
        )
        self.color, self.color_tiles = color, color_tiles
        self.color_extension = (
            _crop(pixels, extension, wide=True, color=color) if color is not None else None
        )

    def matches(self, pixels: Bytes, track: TextTrack) -> bool | None:
        image = _fine_pixels(pixels, track.bounds, track.vertical, self.shape)
        current = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        same = self._matches(current, self.tiles)
        if same is not True and self.color is not None:
            colored = _project_color(image, self.color)
            if self._matches(colored, self.color_tiles) is True:
                return True
        return same

    @staticmethod
    def _matches(current: Gray, tiles: tuple[tuple[int, _GlyphReference], ...]) -> bool | None:
        certain = True
        for start, tile in tiles:
            same = tile.matches(current[:, start : start + tile.reference.shape[1]])
            if same is False:
                return False
            certain &= same is True
        return True if certain else None


def _backing_colors(pixels: Bytes, track: TextTrack) -> Bytes | None:
    b = track.bounds
    start, end, top, bottom = b.x, b.x + b.width, b.y, b.y + b.height
    if track.vertical:
        pixels = pixels.swapaxes(0, 1)
        start, end, top, bottom = top, bottom, start, end
    if top < 4 or bottom + 3 >= pixels.shape[0]:
        return None
    return pixels[[top - 4, bottom + 3], start : end : max(1, ceil((end - start) / 16)), :3]


def _native_edges(pixels: Bytes, bounds: Rect, limit: int) -> Bytes | None:
    box = padded(bounds, (pixels.shape[1], pixels.shape[0]), 4)
    if box.width * box.height > limit:
        return None
    area = pixels[box.y : box.y + box.height, box.x : box.x + box.width]
    return cv2.morphologyEx(area, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))


def _color_strokes(pixels: Bytes, bounds: Rect) -> Mask:
    area = pixels[bounds.y : bounds.y + bounds.height, bounds.x : bounds.x + bounds.width, :3]
    edges = cv2.morphologyEx(area, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return np.max(edges, axis=2) >= 32


@dataclass(slots=True)
class _Anchor:
    value: TextTrack
    reference: Gray
    layout: Bytes
    backing: Bytes | None
    source_sequence: int
    extension_bounds: Rect | None = None
    extension: Gray | None = None
    x: Gray = field(init=False)
    y: Gray = field(init=False)
    mask: Mask = field(init=False)
    stroke_count: int = field(init=False)
    clear_since: float | None = None
    clear_samples: int = 0
    replacement: Gray | None = None
    cleared: bool = False
    misses: int = 0
    hits: int = 0
    hidden: bool = False
    present: bool = False
    glyphs_match: bool = False
    last_sequence: int = 0
    content: _GlyphReference = field(init=False)
    fine_required: bool = False
    fine: _FineReference | None = None
    color_strokes_required: bool = field(init=False)
    color_strokes: Mask | None = None

    def __post_init__(self) -> None:
        self.content = _GlyphReference(self.reference)
        self.x, self.y = _edges(self.reference)
        strength = np.maximum(np.abs(self.x), np.abs(self.y))
        self.mask = strength >= max(35, float(np.percentile(strength, 80)))
        self.stroke_count = int(self.mask.sum())
        self.color_strokes_required = not self.fine_required and float(np.ptp(self.reference)) <= 8

    @property
    def reference_pixels(self) -> int:
        if self.fine is not None:
            return self.fine.pixels
        return self.color_strokes.size if self.color_strokes is not None else 0

    def prepare_color_strokes(self, source: Bytes, budget: int) -> None:
        if self.reference.size <= budget:
            strokes = _color_strokes(source, self.value.bounds)
            self.color_strokes_required = int(np.count_nonzero(strokes)) >= 12
            if self.color_strokes_required:
                self.color_strokes = strokes

    def _coarse_matches(self, pixels: Bytes, crop: Gray) -> bool | None:
        same = self.content.matches(crop)
        if same is not True or not self.color_strokes_required:
            return same
        # Equal luminance can hide a digit edit. Defer to OCR when the color
        # outline changes or the source mask exceeded the reference budget.
        if self.color_strokes is None:
            return None
        strokes = _color_strokes(pixels, self.value.bounds)
        return True if np.count_nonzero(self.color_strokes ^ strokes) < 8 else None

    def content_matches(self, pixels: Bytes) -> bool | None:
        if self.fine_required:
            same = self.fine.matches(pixels, self.value) if self.fine is not None else None
        else:
            crop = _crop(pixels, self.value.bounds)
            same = None if crop is None else self._coarse_matches(pixels, crop)
        if same is True and self.extension is not None and self.extension_bounds is not None:
            extension = _crop(pixels, self.extension_bounds, wide=True)
            # The accepted box was already checked above. Contrast gains inside it
            # must not masquerade as new characters outside its right edge.
            start = ceil(
                self.value.bounds.width * self.extension.shape[1] / self.extension_bounds.width
            )
            if extension is not None:
                fine = self.fine
                colored = fine is not None and fine.color_extension is not None
                if _added_ink(
                    self.extension,
                    extension,
                    foreground=True,
                    start=start,
                    min_height=extension.shape[0] * 0.5 if colored else 0,
                ):
                    return False
                if colored and fine is not None and fine.color_extension is not None:
                    current = _crop(pixels, self.extension_bounds, wide=True, color=fine.color)
                    if current is not None and _added_ink(
                        fine.color_extension, current, foreground=True, start=start
                    ):
                        return False
        return same

    def _panel_replaced(self, pixels: Bytes, current: Gray, surroundings_changed: bool) -> bool:
        # Losing just the letters leaves most of the old backing pixels in place.
        if (
            not surroundings_changed
            and np.count_nonzero(np.abs(self.reference - current) >= 16) < current.size * 0.9
        ):
            # A panel and the exposed background can have the same luminance.
            # Verify the native colors changed along both sides of the accepted text.
            colors = _backing_colors(pixels, self.value) if self.backing is not None else None
            if colors is None or self.backing is None:
                return False
            different = np.max(cv2.absdiff(self.backing, colors), axis=2) >= 16
            if np.any(np.mean(different, axis=1) < 0.9):
                return False
        # Check the native colors just outside the box too. A shifted glyph or
        # isoluminant lettering can disappear in the reduced grayscale comparison.
        # Smooth shading is consistent with an exposed background. Look for
        # local color edges across a stroke's width, even in very faint text.
        edges = _native_edges(pixels, self.value.bounds, 65_536)
        return edges is not None and cv2.norm(edges, cv2.NORM_INF) <= 4

    def _layout_changed(self, layout: Bytes, size: tuple[int, int]) -> bool:
        # Compare with the accepted source even when a menu closes over several frames.
        # Disregard the line and adjacent text so a blinking dialogue cannot corroborate itself.
        changed = cv2.absdiff(self.layout, layout) >= 24
        x1, y1, x2, y2 = _tile_bounds(self.value.bounds, size)
        changed[max(0, y1 - 1) : y2 + 1, max(0, x1 - 1) : x2 + 1] = False
        if not np.any(changed[max(0, y1 - 4) : y2 + 4, max(0, x1 - 4) : x2 + 4]):
            return False
        ys, xs = np.nonzero(changed)
        return xs.size >= 12 and bool(np.ptp(xs) >= 3 and np.ptp(ys) >= 3)

    def _vacant_after_switch(self, pixels: Bytes, current: Gray) -> bool:
        # A new page can share the old gradient and leave a border in one part of a line.
        # Check native color around the old strokes to catch shifted or isoluminant letters.
        if not self._strokes_replaced(current):
            return False
        edges = _native_edges(pixels, self.value.bounds, _FINE_LINE_PIXELS)
        if edges is None:
            return False
        axis = 0 if self.value.vertical else 1
        clear = 0
        for old, new in zip(
            np.array_split(self.mask, 3, axis=axis),
            np.array_split(edges, 3, axis=axis),
            strict=True,
        ):
            if np.count_nonzero(old) >= 8 and cv2.norm(new, cv2.NORM_INF) <= 8:
                clear += 1
        return clear >= 2

    def _strokes_replaced(self, current: Gray) -> bool:
        """Require loss across the old line, even when the new menu has its own edges."""
        if self.content.faint is None:
            self.content.faint = _FaintReference.prepare(self.reference)
        source = self.content.faint
        if not source.valid:
            return False
        detail = current - cv2.GaussianBlur(current, (0, 0), 2)
        old = source.detail
        if self.value.vertical:
            old, detail = old.T, detail.T
        # Search three separate parts of the line, allowing small positional jitter.
        # An edited character or changing background can spoil a whole-line score.
        padded_detail = cv2.copyMakeBorder(detail, 4, 4, 4, 4, cv2.BORDER_REPLICATE)
        informative = 0
        for index in range(3):
            left, right = index * old.shape[1] // 3, (index + 1) * old.shape[1] // 3
            patch = old[:, left:right]
            if patch.size < 32 or np.std(patch) < 8:
                continue
            scores = cv2.matchTemplate(
                padded_detail[:, left : right + 8], patch, cv2.TM_CCOEFF_NORMED
            )
            if np.max(np.abs(scores)) >= 0.42:
                return False
            informative += 1
        return informative >= 2

    def observe(
        self,
        pixels: Bytes,
        sequence: int,
        now: float,
        changed: bool,
        scene_has_detail: bool,
        surroundings_changed: bool,
        layout: Bytes,
    ) -> bool:
        if sequence <= self.last_sequence:
            return self.hidden
        needs_check = (
            changed
            or not self.last_sequence
            or self.misses == 1
            or (self.clear_samples > 0 and not self.cleared)
            or self.replacement is not None
            or (self.hidden and self.hits == 1)
        )
        self.last_sequence = sequence
        if not needs_check:
            return self.hidden
        current = _crop(pixels, self.value.bounds)
        self.present = False
        self.glyphs_match = False
        if current is None:
            return self.hidden
        if sequence == self.source_sequence:
            glyph_match = True  # The accepted OCR read this exact capture.
        elif self.stroke_count < 24 and not self.cleared:
            return self.hidden
        elif self.cleared and self.fine_required:
            glyph_match = self.fine.matches(pixels, self.value) if self.fine is not None else None
        else:
            glyph_match = self._coarse_matches(pixels, current)
        if glyph_match is True:
            self.glyphs_match = self.present = True
            self.replacement = None
            self.cleared = False
            self.clear_samples = 0
            self.clear_since = None
            self.misses = 0
            self.hits += 1
            if self.hits >= 2:
                self.hidden = False
            return self.hidden
        if self.stroke_count < 24:
            return self.hidden
        x, y = _edges(current)
        # A one-pixel tolerance allows antialiasing and tiny camera/text shifts.
        kernel = np.ones((3, 3), np.uint8)
        matches = np.zeros(self.mask.shape, dtype=bool)
        for old, new in ((self.x, x), (self.y, y)):
            for sign in (-1, 1):
                wanted = old * sign >= 35
                nearby = cv2.dilate(new * sign, kernel)
                matches |= wanted & (nearby >= np.maximum(12, np.abs(old) * 0.3))
        coverage = float(np.mean(matches[self.mask]))
        correlation = float(
            np.max(
                np.abs(
                    cv2.matchTemplate(
                        cv2.copyMakeBorder(current, 1, 1, 1, 1, cv2.BORDER_REPLICATE),
                        self.reference,
                        cv2.TM_CCOEFF_NORMED,
                    )
                )
            )
        )
        # Unrelated textured backgrounds can accidentally match many individual
        # edges. Require the spatial pattern too, with brightness/shift tolerance.
        lost = (coverage < 0.12 and correlation < 0.4) or correlation < 0.1
        # A stationary gradient can correlate even after every letter is gone.
        # Require the original stroke patches to disappear across the line too.
        vacant = (
            glyph_match is False
            and np.count_nonzero(np.maximum(np.abs(x), np.abs(y)) >= 35) < self.stroke_count * 0.15
            and (lost or (coverage < 0.12 and self._strokes_replaced(current)))
        )
        replaced = (
            not vacant
            and not self.cleared
            and scene_has_detail
            and glyph_match is False
            and correlation < 0.35
            and self._strokes_replaced(current)
        )
        if (
            (vacant or replaced)
            and self.fine_required
            and not self.cleared
            and (self.fine is None or self.fine.matches(pixels, self.value) is not False)
        ):
            vacant = replaced = False
        if lost or vacant:
            self.hits = 0
            self.misses += 1
            if self.misses >= 2:
                self.hidden = True
        elif coverage >= 0.4 or correlation >= 0.6:
            # The line can still be present despite a glyph or background change.
            # Only corroborated edge and spatial agreement can veto an OCR miss.
            self.present = not self.cleared and coverage >= 0.4 and correlation >= 0.7
            self.misses = 0
            self.hits += 1
            if self.hits >= 2:
                self.hidden = False
        else:
            self.misses = 0  # Ambiguous evidence cannot accumulate disappearance votes.
        previous_replacement, self.replacement = self.replacement, None
        if replaced:
            # Outside change corroborates a menu transition immediately. An isolated
            # replacement must show the same new pattern on a second distinct capture.
            if (
                surroundings_changed
                or (
                    previous_replacement is not None and not _changed(previous_replacement, current)
                )
                or self._layout_changed(layout, (pixels.shape[1], pixels.shape[0]))
            ):
                self.cleared = True
                self.clear_since = now
            else:
                self.replacement = current
            self.clear_samples = 0
        elif vacant:
            if self.clear_since is None:
                self.clear_since = now
            self.clear_samples += 1
            sustained = self.clear_samples >= 3 and now - self.clear_since >= 0.16
            self.cleared = (
                self.cleared
                or sustained
                or (
                    scene_has_detail and self._panel_replaced(pixels, current, surroundings_changed)
                )
                or (
                    self._layout_changed(layout, (pixels.shape[1], pixels.shape[0]))
                    and self._vacant_after_switch(pixels, current)
                )
            )
        else:
            self.clear_samples = 0
            if not self.cleared:
                self.clear_since = None
        return self.hidden


@dataclass(slots=True)
class _Probe:
    index: int
    track_id: int | None
    bounds: Rect
    reference: Gray
    previous: Gray
    changed_at: float
    dialogue: bool
    source_sequence: int = 0
    source_text: str = ""
    requested_at: float = float("-inf")
    checked: bool = False
    dirty: bool = False
    growth: bool = False
    appearance: SentenceAppearance | None = None


def _tile_bounds(bounds: Rect, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1 = max(0, bounds.x * 32 // w), max(0, bounds.y * 18 // h)
    x2, y2 = (
        min(32, ceil((bounds.x + bounds.width) * 32 / w)),
        min(18, ceil((bounds.y + bounds.height) * 18 / h)),
    )
    return x1, y1, x2, y2


def _touches(tiles: Mask, bounds: Rect, size: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = _tile_bounds(bounds, size)
    return bool(np.any(tiles[y1:y2, x1:x2]))


def _surroundings_changed(tiles: Mask, bounds: Rect, size: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = _tile_bounds(bounds, size)
    inside = tiles[y1:y2, x1:x2]
    around = tiles[max(0, y1 - 1) : y2 + 1, max(0, x1 - 1) : x2 + 1]
    count = around.size - inside.size
    return count >= 4 and bool(np.count_nonzero(around) - np.count_nonzero(inside) >= count * 0.5)


class VisualAnalyzer:
    """Pure, explicitly timed policy. Templates are bounded to the current 128 tracks.

    Coarse novelty discovers new UI; local stroke evidence masks vanished text without
    retiring identity. Expanded reading areas catch typewriter growth beyond old boxes.
    A periodic full OCR refresh covers uncertain changes and missed small/animated text.
    """

    def __init__(self, language: SourceLanguage = DEFAULT_SOURCE) -> None:
        self._language = language
        global cv2, np
        import cv2
        import numpy as np

        from dantetl.ocr.discovery import DiscoveryTiles
        from dantetl.ocr.menu_appearance import MenuAppearance
        from dantetl.ocr.scan_cadence import ScanCadence

        cv2.setNumThreads(1)
        self._tiles = DiscoveryTiles()
        self._menus = MenuAppearance(language)
        self._anchors: dict[TextTrack, _Anchor] = {}
        self._probes: list[_Probe] = []
        self._ocr_sequence = 0
        self._sequence = 0
        self._captured_at = float("-inf")
        self._before: dict[int, Bytes] = {}
        self._size: tuple[int, int] | None = None
        self._dpi = 0
        self._cadence = ScanCadence()
        self._reference_key: tuple[int, tuple[TextTrack, ...]] | None = None
        self._checked_key: tuple[int, tuple[TextTrack, ...]] | None = None
        self._checked: frozenset[TextTrack] = frozenset()
        self._unchanged: frozenset[TextTrack] = frozenset()

    def _references(
        self, job: _Job, pixels: Bytes, layout: Bytes, matching: MatchingEvidence | None
    ) -> None:
        key = (job.result.frame.sequence if job.result else 0, job.tracks)
        if key == self._reference_key:
            return
        self._reference_key = key
        dismissed = {t.track_id: a.clear_since for t, a in self._anchors.items() if a.cleared}
        self._anchors = {t: a for t, a in self._anchors.items() if t in job.tracks}
        result = job.result
        if result is None:
            return
        source = _pixels(result.frame)
        if (
            source is None
            or source.shape != pixels.shape
            or result.frame.geometry.dpi != job.frame.geometry.dpi
        ):
            return
        cleaned = cleaned_rows(result, self._language, matching)
        by_id = {
            key: region
            for key, region in zip(job.assignments, cleaned, strict=False)
            if key is not None
        }
        fine_budget = _FINE_PIXELS - sum(a.reference_pixels for a in self._anchors.values())
        reference_layout = layout if result.frame is job.frame else None
        color_candidates: list[tuple[_FineReference, TextTrack, TextRegion, Rect]] = []
        short_color_candidates: list[_Anchor] = []
        for track in job.tracks[:MAX_TRACKS]:
            region = by_id.get(track.track_id)
            anchor = self._anchors.get(track)
            if (
                anchor is not None
                and anchor.clear_since is not None
                and result.frame.sequence > self._ocr_sequence
                and result.frame.captured_at >= anchor.clear_since
                and region is not None
                and region.text == track.text
            ):
                # A positive reading from inside the clear interval supersedes its template.
                fine_budget += anchor.reference_pixels
                del self._anchors[track]
            if track not in self._anchors and region is not None and region.text == track.text:
                crop = _crop(source, track.bounds)
                if crop is not None:
                    if reference_layout is None:
                        from dantetl.ocr.discovery import source_layout

                        reference_layout = source_layout(source)
                    b = track.bounds
                    width = max(
                        b.width + b.height * 3, b.height * 36 if len(track.text) >= 10 else 0
                    )
                    area = Rect(b.x, b.y, min(width, job.frame.width - b.x), b.height)
                    fine_required = b.width > 256 or b.height > 48
                    fine = (
                        _FineReference.create(source, track, fine_budget) if fine_required else None
                    )
                    if fine is not None:
                        fine_budget -= fine.pixels
                        color_candidates.append((fine, track, region, area))
                    clear_since = dismissed.get(track.track_id)
                    anchor = _Anchor(
                        track,
                        crop,
                        reference_layout,
                        _backing_colors(source, track),
                        result.frame.sequence,
                        area,
                        _crop(source, area, wide=True),
                        fine_required=fine_required,
                        fine=fine,
                        cleared=clear_since is not None,
                        clear_since=clear_since,
                    )
                    self._anchors[track] = anchor
                    if anchor.color_strokes_required:
                        short_color_candidates.append(anchor)
        # Optional color must never consume another line's basic glyph evidence.
        for fine, track, region, area in color_candidates:
            before = fine.pixels
            # Inspect the start of continuation, not a distant unrelated column.
            b = track.bounds
            edge = Rect(b.x, b.y, min(b.width + b.height * 3, area.width), b.height)
            fine.add_color(source, track, region, edge, fine_budget)
            fine_budget -= fine.pixels - before
            if fine.color is not None:
                anchor = self._anchors[track]
                anchor.extension_bounds = edge
                anchor.extension = _crop(source, edge, wide=True)
        for anchor in short_color_candidates:
            anchor.prepare_color_strokes(source, fine_budget)
            fine_budget -= anchor.reference_pixels
        if result.frame.sequence == self._ocr_sequence:
            return
        self._ocr_sequence = result.frame.sequence
        self._menus.observe(source, result, job.tracks, by_id)
        # A partial observation says nothing about the rest of the screen. Keep
        # those bounded dialogue probes and their exact original pixel references.
        live = {t.track_id: t for t in job.tracks}
        prior = [
            p
            for p in self._probes
            if p.track_id in live and p.track_id not in by_id and result.scanned is not None
        ]
        for probe in prior:
            probe.appearance = None  # Only the current accepted reading can authorize dispatch.
        self._probes = []
        from dantetl.ocr.pop_in import SentenceAppearance

        before = self._before.get(result.frame.sequence)
        appearances = 0
        growing = {a.observation.track_id for a in job.activity if a.growing}
        for index, region in enumerate(cleaned):
            if len(self._probes) >= 128:
                break
            key = job.assignments[index] if index < len(job.assignments) else None
            if key is None or not self._language.contains_text(region.text):
                continue
            b = region.bounds
            dialogue = not region.vertical and (len(region.text) >= 10 or key in growing)
            width = max(b.width, b.height * 36) if dialogue else b.width + b.height * 3
            area = Rect(
                b.x,
                b.y,
                min(width, job.frame.width - b.x),
                min(round(b.height * (4 if dialogue else 1.35)), job.frame.height - b.y),
            )
            crop = _crop(source, area, wide=True)
            if crop is not None:
                appearance = None
                if (
                    before is not None
                    and appearances < 2
                    and dialogue
                    and key not in growing
                    and 0.95 <= (region.confidence or 0) <= 1
                    and (not region.characters or region.characters[-1].confidence >= 0.90)
                    and self._language.explicit_sentence_end(region.text)
                    and (track := live.get(key)) is not None
                    and track.text_revision == 1
                    and track.bounds == b
                    and track.text == region.text
                ):
                    appearance = SentenceAppearance.from_transition(before, source, b)
                    appearances += 1
                self._probes.append(
                    _Probe(
                        index,
                        key,
                        area,
                        crop,
                        crop,
                        result.frame.captured_at,
                        dialogue,
                        result.frame.sequence,
                        region.text,
                        appearance=appearance,
                    )
                )
        self._probes.extend(prior[: 128 - len(self._probes)])

    def observe(
        self,
        frame: CapturedFrame,
        result: OcrResult | None = None,
        tracks: tuple[TextTrack, ...] = (),
        assignments: tuple[int | None, ...] = (),
        activity: tuple[TextActivity, ...] = (),
        candidate: OcrResult | None = None,
        matching: MatchingEvidence | None = None,
    ) -> VisualState:
        started = time.perf_counter()
        pixels = _pixels(frame)
        if pixels is None:
            return VisualState(
                frame=frame,
                request_ocr=True,
                message=tr("Visual evidence unavailable; using OCR."),
            )
        if self._size != frame.geometry.client.size or self._dpi != frame.geometry.dpi:
            self.__init__(self._language)
            self._size = frame.geometry.client.size
            self._dpi = frame.geometry.dpi
        size = frame.geometry.client.size
        fresh = frame.sequence > self._sequence
        now = frame.captured_at
        previous = self._tiles.thumbnail
        evidence = self._tiles.observe(pixels, now, fresh=fresh)
        self._references(
            _Job(frame, result, tracks, assignments, activity), pixels, evidence.layout, matching
        )
        checked_key = (candidate.frame.sequence if candidate else 0, tracks)
        if checked_key != self._checked_key:
            self._checked_key = checked_key
            source = _pixels(candidate.frame) if candidate else None
            matches = (
                {t: a.content_matches(source) for t, a in self._anchors.items()}
                if source is not None and source.shape == pixels.shape
                else {}
            )
            self._checked = frozenset(t for t, same in matches.items() if same is not None)
            self._unchanged = frozenset(t for t, same in matches.items() if same is True)
        changed_tiles = evidence.changed
        # A blank capture can be a one-frame flash. Require some surviving scene
        # detail before using a single capture as evidence of a panel closing.
        # An OCR completion can introduce a new anchor after we analyzed the latest
        # capture. It still sees those pixels once, even if WGC sends no more frames.
        scene_has_detail = bool(self._anchors and np.ptp(evidence.gray) >= 16)
        hidden = frozenset(
            key
            for key, anchor in self._anchors.items()
            if anchor.observe(
                pixels,
                frame.sequence,
                now,
                _touches(changed_tiles, key.bounds, size),
                scene_has_detail,
                _surroundings_changed(evidence.strong, key.bounds, size),
                evidence.layout,
            )
        )
        stable, blocked = [], set()
        stable_tracks = []
        complete_lines = set()
        from dantetl.ocr.scan_cadence import ScanChanges

        changes = ScanChanges(now)
        for probe in self._probes:
            if not probe.checked or _touches(changed_tiles, probe.bounds, size):
                current = _crop(pixels, probe.bounds, wide=True)
                if current is None:
                    continue
                if fresh and (
                    _changed(probe.previous, current) or _added_ink(probe.previous, current)
                ):
                    probe.changed_at = now
                probe.dirty = _changed(probe.reference, current)
                probe.growth = _added_ink(probe.reference, current)
                probe.previous, probe.checked = current, True
            dirty, growth = probe.dirty, probe.growth
            if (
                probe.appearance is not None
                and result
                and probe.source_sequence == result.frame.sequence
            ):
                age = now - result.frame.captured_at
                if age >= 0.18 or dirty or growth or not probe.appearance.matches(pixels):
                    probe.appearance = None
                elif frame.sequence > probe.source_sequence and age >= 0.06:
                    complete_lines.update(
                        t
                        for t in tracks
                        if t.track_id == probe.track_id
                        and t.text == probe.source_text
                        and t.text_revision == 1
                    )
            quiet = now - probe.changed_at >= 0.16
            if growth and probe.track_id is not None:
                blocked.add(probe.track_id)
            if not dirty and not growth and quiet and frame.sequence > probe.source_sequence:
                if probe.track_id is not None:
                    stable_tracks.append((probe.track_id, probe.source_sequence, probe.source_text))
                if result and probe.source_sequence == result.frame.sequence:
                    stable.append(probe.index)
            if probe.dialogue and growth and not quiet:
                changes.growing.append(probe.bounds)
            if (dirty or growth) and probe.changed_at > probe.requested_at:
                if quiet:
                    changes.ready.append(padded(probe.bounds, size))
                else:
                    changes.settling.append(padded(probe.bounds, size))
                    changes.deadline = max(changes.deadline, probe.changed_at + 0.16)
        decision = self._cadence.decide(
            self._tiles, changes, size, tuple(t.bounds for t in tracks), now=now, fresh=fresh
        )
        coverage = decision.areas
        if decision.request:
            if (
                previous is not None
                and frame.sequence == self._sequence + 1
                and 0 < now - self._captured_at <= 0.15
                and np.any(evidence.changed)
            ):
                self._before[frame.sequence] = previous
                if len(self._before) > 4:
                    del self._before[next(iter(self._before))]
            for probe in self._probes:
                if coverage is None or any(area.contains(probe.bounds) for area in coverage):
                    probe.requested_at = now
        if fresh:
            self._sequence = frame.sequence
            self._captured_at = now
        return VisualState(
            frame=frame,
            request_ocr=decision.request,
            hidden=hidden,
            blocked=frozenset(blocked),
            stable_regions=tuple(stable),
            ocr_sequence=self._ocr_sequence,
            duration_ms=(time.perf_counter() - started) * 1000,
            avoided=self._cadence.avoided,
            presence=TrackPresence(
                sequence=frame.sequence,
                absent=frozenset(t for t, a in self._anchors.items() if a.hidden and a.misses >= 2),
                pattern_loss=frozenset(t for t, a in self._anchors.items() if a.misses),
                present=frozenset(t for t, a in self._anchors.items() if a.present),
            ),
            cleared=frozenset(t for t, a in self._anchors.items() if a.cleared),
            matching_lines=frozenset(t for t, a in self._anchors.items() if a.glyphs_match),
            review=CandidateReview(
                candidate.frame.sequence if candidate else 0, self._checked, self._unchanged
            ),
            scan_areas=coverage,
            stable_tracks=tuple(stable_tracks),
            complete_lines=frozenset(complete_lines),
            matching_menus=self._menus.matching(pixels),
            settled_scan=decision.settled,
            changed_areas=tuple(
                Rect(
                    int(x) * size[0] // 32,
                    int(y) * size[1] // 18,
                    ceil(int(x + 1) * size[0] / 32) - int(x) * size[0] // 32,
                    ceil(int(y + 1) * size[1] / 18) - int(y) * size[1] // 18,
                )
                for y, x in zip(*np.nonzero(evidence.significant), strict=True)
            ),
        )


class _ObservationReview:
    """One generation's matching and optional pixel evidence for the pending OCR result."""

    def __init__(self, language: SourceLanguage) -> None:
        self.language = language
        self.matching: MatchingEvidence | None = None
        self.pixels: PixelEvidence | None = None
        self.pixel_key: tuple[int, tuple[Rect, ...]] | None = None
        self.rejected_sequence = 0
        self.message = ""
        self.last_visual_sequence = 0

    def apply(self, state: VisualState, job: _Job) -> VisualState:
        if not state.message:
            self.last_visual_sequence = job.frame.sequence
        elif job.result and job.result.frame.sequence > self.last_visual_sequence:
            # In OCR-only fallback, a later accepted reading can restore dismissed
            # English. Unscanned lines and results from before the failure cannot.
            result = job.result
            rows = cleaned_rows(result, self.language, self.matching)
            by_id = dict(zip(job.assignments, rows, strict=False))
            state = replace(
                state,
                matching_lines=frozenset(
                    t
                    for t in job.tracks
                    if (row := by_id.get(t.track_id)) is not None
                    and row.text == t.text
                    and row.vertical == t.vertical
                    and result.covers(t.bounds)
                ),
            )
        candidate = job.candidate
        if candidate is None:
            return replace(state, review_message=self.message) if self.message else state
        sequence = candidate.frame.sequence
        if sequence != self.rejected_sequence and (
            self.matching is None
            or not self.matching.matches(candidate, job.matching_sources, language=self.language)
        ):
            if self.matching is None or not self.matching.describes(candidate, self.language):
                self.pixel_key = None
            try:
                self.matching = prepare_matching(
                    candidate, job.matching_sources, language=self.language
                )
                self.rejected_sequence = 0
            except Exception:
                logger.exception("Cannot match OCR observation %s", sequence)
                self.matching = None
                self.rejected_sequence = sequence
                self.message = tr("Could not match this OCR result. Waiting for the next scan.")
        if self.matching is None:
            return replace(
                state,
                rejected_sequence=sequence,
                review_message=self.message,
                request_ocr=True,
                scan_areas=None,
                settled_scan=None,
            )
        key = sequence, job.evidence_bounds
        if self.pixel_key != key:
            self.pixel_key = key
            try:
                self.pixels = prepare_evidence(
                    candidate, job.evidence_bounds, language=self.language, matching=self.matching
                )
                self.message = ""
            except Exception:
                logger.exception("Pixel evidence unavailable for OCR %s", sequence)
                self.pixels = None
                self.message = tr("Visual evidence unavailable; using OCR.")
        return replace(
            state,
            review=replace(
                state.review, sequence=sequence, matching=self.matching, pixels=self.pixels
            ),
            review_message=self.message,
        )


class VisualSession:
    """One replaceable analysis job. No inference, Qt, full-frame history, or IO."""

    def __init__(
        self,
        *,
        on_change: Callable[[], None] | None = None,
        language: SourceLanguage = DEFAULT_SOURCE,
    ) -> None:
        self._language = language
        self._notify = on_change or (lambda: None)
        self._condition = threading.Condition()
        self._pending: tuple[int, _Job] | None = None
        self._last_key: (
            tuple[int, int, tuple[TextTrack, ...], int, tuple[Rect, ...], tuple[TextTrack, ...]]
            | None
        ) = None
        self._generation = 0
        self._stop = False
        self._done = threading.Event()
        self._snapshot = VisualState()
        self._delivered_sequence = 0
        self._thread = threading.Thread(target=self._run, name="visual-owner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        frame: CapturedFrame,
        result: OcrResult | None,
        tracks: tuple[TextTrack, ...],
        assignments: tuple[int | None, ...],
        activity: tuple[TextActivity, ...],
        candidate: OcrResult | None = None,
        evidence_bounds: tuple[Rect, ...] = (),
        matching_sources: tuple[TextTrack, ...] = (),
    ) -> None:
        tracks, activity = tracks[:MAX_TRACKS], activity[:MAX_TRACKS]
        assignments = assignments[:MAX_OCR_REGIONS]
        evidence_bounds = evidence_bounds[:MAX_OCR_REGIONS]
        matching_sources = matching_sources[:MAX_TRACKS]
        key = (
            frame.sequence,
            result.frame.sequence if result else 0,
            tracks,
            candidate.frame.sequence if candidate else 0,
            evidence_bounds,
            matching_sources,
        )
        with self._condition:
            if (
                self._stop
                or key == self._last_key
                or (self._last_key is not None and frame.sequence < self._last_key[0])
            ):
                return
            self._last_key = key
            self._pending = (
                self._generation,
                _Job(
                    frame,
                    result,
                    tracks,
                    assignments,
                    activity,
                    candidate,
                    evidence_bounds,
                    matching_sources,
                ),
            )
            self._condition.notify()

    def invalidate(self, *, language: SourceLanguage | None = None) -> None:
        with self._condition:
            if language is not None:
                self._language = language
            self._generation += 1
            self._pending, self._last_key = None, None
            self._snapshot = VisualState()
            self._delivered_sequence = 0

    def request_stop(self) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._snapshot = VisualState()
            self._condition.notify()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def poll(self, latest_sequence: int | None = None, now: float | None = None) -> VisualState:
        """Release work once per capture. The desktop refresh also services quiet deadlines."""
        with self._condition:
            value = self._snapshot
            if not value.request_ocr and value.settled_scan is None:
                return value
            frame = value.frame
            if frame is None or frame.sequence <= self._delivered_sequence:
                self._snapshot = replace(value, request_ocr=False, settled_scan=None)
                return self._snapshot
            settled = value.settled_scan
            if settled and (latest_sequence is None or latest_sequence <= frame.sequence):
                if (time.monotonic() if now is None else now) < settled.deadline:
                    return replace(value, request_ocr=False)
                value = replace(value, request_ocr=True, scan_areas=settled.areas)
            if value.request_ocr:
                self._delivered_sequence = frame.sequence
            self._snapshot = replace(value, request_ocr=False, settled_scan=None)
            return replace(value, settled_scan=None)

    def _run(self) -> None:
        analyzer, generation = None, -1
        failed = False
        review = _ObservationReview(self._language)
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._pending is not None)
                    if self._stop:
                        return
                    pending, self._pending = self._pending, None
                    language = self._language
                assert pending is not None
                current, job = pending
                started = time.perf_counter()
                if current != generation:
                    analyzer, generation, failed = None, current, False
                    review = _ObservationReview(language)
                try:
                    if analyzer is None and not failed:
                        analyzer = VisualAnalyzer(language)
                    state = (
                        VisualState(
                            frame=job.frame,
                            request_ocr=True,
                            message=tr("Visual checks failed; using OCR."),
                        )
                        if analyzer is None or failed
                        else analyzer.observe(
                            job.frame,
                            job.result,
                            job.tracks,
                            job.assignments,
                            job.activity,
                            job.candidate,
                            review.matching,
                        )
                    )
                except Exception:
                    logger.exception("Visual checks failed; using ordinary OCR")
                    failed = True
                    state = VisualState(
                        frame=job.frame,
                        request_ocr=True,
                        message=tr("Visual checks failed; using OCR."),
                    )
                # Independent of visual pattern-analysis success. Always hash the OCR
                # source, never the latest capture, and discard cached evidence on reset.
                state = review.apply(state, job)
                with self._condition:
                    published = not self._stop and current == self._generation
                    if published:
                        old = self._snapshot
                        areas = state.scan_areas
                        if old.request_ocr:
                            areas = (
                                None
                                if old.scan_areas is None or (state.request_ocr and areas is None)
                                else scan_regions(
                                    (*old.scan_areas, *(areas or ())),
                                    job.frame.geometry.client.size,
                                )
                            )
                        settled = state.settled_scan
                        if settled is not None and old.request_ocr:
                            settled = SettledScan(
                                settled.deadline,
                                None
                                if areas is None or settled.areas is None
                                else scan_regions(
                                    (*areas, *settled.areas), job.frame.geometry.client.size
                                ),
                            )
                        self._snapshot = replace(
                            state,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            request_ocr=state.request_ocr or old.request_ocr,
                            scan_areas=areas,
                            settled_scan=settled,
                        )
                if published:
                    self._notify()
                del pending, job, state
        finally:
            self._done.set()
            self._notify()
