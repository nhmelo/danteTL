"""Remember line re-reads by their exact pixels, so an unchanged screen is not read again.

Recognizing identical pixels in an identical box gives an identical reading, so a remembered
answer is the same answer. The memory is bounded and owned by one recognizer on its worker.
"""

import hashlib
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.values import TextRegion

Read = Callable[[list[Rect]], tuple[TextRegion, ...]]


def pixel_key(image: np.ndarray, box: Rect) -> tuple[Rect, bytes]:
    pixels = np.ascontiguousarray(image[box.y : box.y + box.height, box.x : box.x + box.width])
    return box, hashlib.blake2b(pixels.tobytes(), digest_size=16).digest()


class ReadMemory:
    def __init__(self, size: int = 128) -> None:
        self._size = size
        self._answers: OrderedDict[tuple[Rect, bytes], TextRegion] = OrderedDict()

    def reader(self, image: np.ndarray, read: Read) -> Read:
        """``read`` for these pixels, reading only boxes not remembered, in one batch.

        An incomplete answer is returned as empty and nothing from it is remembered.
        """

        def remembered(boxes: list[Rect]) -> tuple[TextRegion, ...]:
            keys = [pixel_key(image, box) for box in boxes]
            missing = [key for key in dict.fromkeys(keys) if key not in self._answers]
            if missing:
                answers = read([box for box, _ in missing])
                if len(answers) != len(missing):
                    return ()
                for key, answer in zip(missing, answers, strict=True):
                    self._answers[key] = answer
            for key in keys:
                self._answers.move_to_end(key)
            found = tuple(self._answers[key] for key in keys)
            while len(self._answers) > self._size:
                self._answers.popitem(last=False)
            return found

        return remembered
