"""Connected ink is measured against its area's border, at a scale that keeps thin strokes."""

import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.ink import ink


def test_one_pixel_strokes_survive_a_large_reduction_wherever_they_fall() -> None:
    # A 1 px stroke at any phase of a 1/8 reduction must still read as ink.
    for offset in range(8):
        image = np.full((256, 512, 3), 30, np.uint8)
        for x in range(64 + offset, 448, 48):
            image[64:192, x] = 235
        found = ink(image, Rect(0, 0, 512, 256), 1 / 8)
        assert found is not None and len(found.components) == 8, offset
