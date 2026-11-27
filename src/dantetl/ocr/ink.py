"""Connected ink in a small area, separated by color from the area's own border."""

from dataclasses import dataclass

import cv2
import numpy as np

from dantetl.domain import Rect


@dataclass(frozen=True, slots=True)
class Component:
    """Measured in the analyzed (possibly scaled) image; ``depth`` sums distance-to-background."""

    left: int
    right: int
    top: int
    bottom: int
    pixels: int
    depth: float


@dataclass(frozen=True, slots=True)
class Ink:
    """Components ordered by their left edge, with the mapping back to source pixels."""

    x: int
    y: int
    scale: float
    width: int
    height: int
    components: tuple[Component, ...]

    def source_x(self, value: float) -> float:
        return self.x + value / self.scale

    def source_length(self, value: float) -> float:
        return value / self.scale


def ink(bgr: np.ndarray, area: Rect, scale: float) -> Ink | None:
    """Components of at least a speck's size; ``scale`` shrinks the area first (at most 1).

    ``bgr`` may also be BGRA, as captured; only the area is converted.
    """
    x0, y0 = max(0, area.x), max(0, area.y)
    x1, y1 = min(bgr.shape[1], area.x + area.width), min(bgr.shape[0], area.y + area.height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = bgr[y0:y1, x0:x1]
    if crop.shape[2] == 4:  # A captured BGRA frame: convert only this area.
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    if scale < 1:
        size = (max(4, round((x1 - x0) * scale)), max(4, round((y1 - y0) * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        scale = size[0] / (x1 - x0)
    else:
        scale = 1.0
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = np.median(border, axis=0).astype(np.uint8)
    # Per-channel differences, weighted into one plane: colored text on gray still separates.
    distance = cv2.cvtColor(cv2.absdiff(crop, np.full_like(crop, background)), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(distance, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    flat = labels.ravel()
    pixels = np.bincount(flat, minlength=count).tolist()
    depth = np.bincount(
        flat, weights=cv2.distanceTransform(mask, cv2.DIST_L2, 3).ravel(), minlength=count
    ).tolist()
    minimum = max(3, mask.shape[0] ** 2 * 0.004)
    rows = stats.tolist()
    components = tuple(
        Component(left, left + width, top, top + height, pixels[label], depth[label])
        for label in sorted(range(1, count), key=lambda i: rows[i][0])
        for left, top, width, height, area_pixels in (rows[label],)
        if area_pixels >= minimum
    )
    return Ink(x0, y0, scale, mask.shape[1], mask.shape[0], components)
