"""Connected ink is measured against its area's border, at a scale that keeps thin strokes."""

import cv2
import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.ink import ink


def test_flat_noise_and_gentle_gradients_hold_no_ink() -> None:
    noise = np.random.default_rng(7).integers(28, 34, (64, 256, 3), dtype=np.uint8)
    ramp = np.repeat(np.linspace(30, 36, 256, dtype=np.uint8)[None, :, None], 64, 0)
    for image in (noise, np.repeat(ramp, 3, 2)):
        found = ink(image, Rect(0, 0, 256, 64), 1.0)
        assert found is not None and found.components == ()


def test_faint_but_distinct_strokes_are_still_ink() -> None:
    image = np.full((64, 256, 3), 30, np.uint8)
    cv2.rectangle(image, (40, 16), (70, 48), (52, 52, 52), 2)
    found = ink(image, Rect(0, 0, 256, 64), 1.0)
    assert found is not None and len(found.components) == 1


def test_one_pixel_strokes_survive_a_large_reduction_wherever_they_fall() -> None:
    # A 1 px stroke at any phase of a 1/8 reduction must still read as ink.
    for offset in range(8):
        image = np.full((256, 512, 3), 30, np.uint8)
        for x in range(64 + offset, 448, 48):
            image[64:192, x] = 235
        found = ink(image, Rect(0, 0, 512, 256), 1 / 8)
        assert found is not None and len(found.components) == 8, offset
