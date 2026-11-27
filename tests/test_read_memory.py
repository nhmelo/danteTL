"""Re-reads of unchanged pixels come from memory; changed or incomplete reads do not."""

import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.read_memory import ReadMemory
from dantetl.ocr.values import TextRegion

A, B = Rect(10, 10, 40, 20), Rect(60, 10, 40, 20)


class Reader:
    def __init__(self, *, short: bool = False) -> None:
        self.calls: list[list[Rect]] = []
        self.short = short

    def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
        self.calls.append(boxes)
        answers = tuple(TextRegion(f"read{len(self.calls)}", box) for box in boxes)
        return answers[:-1] if self.short else answers


def test_unchanged_pixels_are_read_once_and_changed_ones_again() -> None:
    memory, reader, image = ReadMemory(), Reader(), np.zeros((50, 120, 3), np.uint8)
    first = memory.reader(image, reader)([A, B])
    assert memory.reader(image, reader)([B, A]) == (first[1], first[0])
    assert reader.calls == [[A, B]]
    image[15, 20] = 255  # Inside A only.
    again = memory.reader(image, reader)([A, B])
    assert reader.calls[1] == [A] and again[1] == first[1] and again[0] != first[0]


def test_an_incomplete_answer_is_not_remembered() -> None:
    memory, image = ReadMemory(), np.zeros((50, 120, 3), np.uint8)
    short = Reader(short=True)
    assert memory.reader(image, short)([A, B]) == ()
    whole = Reader()
    assert len(memory.reader(image, whole)([A, B])) == 2
    assert whole.calls == [[A, B]]


def test_memory_is_bounded() -> None:
    memory, reader, image = ReadMemory(size=2), Reader(), np.zeros((50, 200, 3), np.uint8)
    boxes = [Rect(10 + 40 * i, 10, 30, 20) for i in range(3)]
    for box in boxes:
        memory.reader(image, reader)([box])
    memory.reader(image, reader)([boxes[0]])
    assert reader.calls[-1] == [boxes[0]]  # The oldest answer was forgotten.
