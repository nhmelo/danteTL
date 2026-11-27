"""Doubted detections become lines only when the recognizer reads them firmly; reads are bounded."""

import cv2
import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.unsure_lines import Unsure, UnsureReads
from dantetl.ocr.values import TextCharacter, TextRegion


def line(box: Rect, text: str, *scores: float) -> TextRegion:
    step = box.width // max(1, len(text))
    chars = tuple(
        TextCharacter(c, Rect(box.x + i * step, box.y, step, box.height), s)
        for i, (c, s) in enumerate(zip(text, scores, strict=True))
    )
    return TextRegion(text, box, chars)


class Reader:
    def __init__(self, answers: dict[Rect, TextRegion]) -> None:
        self.answers = answers
        self.calls: list[list[Rect]] = []

    def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
        self.calls.append(boxes)
        return tuple(self.answers.get(b, TextRegion("", b)) for b in boxes)


LABEL = Rect(100, 20, 120, 30)
ICON = Rect(300, 20, 120, 30)
ROW = Rect(100, 20, 330, 30)


def image() -> np.ndarray:
    return np.full((100, 500, 3), 30, np.uint8)


def test_firm_readings_are_kept_and_weak_or_single_ones_are_not() -> None:
    single = Rect(20, 60, 60, 30)
    reader = Reader(
        {
            LABEL: line(LABEL, "観測所", 0.95, 0.94, 0.9),
            ICON: line(ICON, "同.物", 0.3, 0.2, 0.4),
            single: line(single, "Q", 0.94),
        }
    )
    unsure = [Unsure(LABEL, 0.44), Unsure(ICON, 0.4), Unsure(single, 0.35)]
    found = UnsureReads().lines(image(), unsure, [], reader)
    assert [r.text for r in found] == ["観測所"]


def test_one_weak_character_rejects_a_reading() -> None:
    reader = Reader({LABEL: line(LABEL, "x入る", 0.31, 0.95, 0.96)})
    assert not UnsureReads().lines(image(), [Unsure(LABEL, 0.4)], [], reader)


def test_explained_boxes_and_low_scores_are_not_read() -> None:
    reader = Reader({LABEL: line(LABEL, "観測所", 0.95, 0.94, 0.9)})
    reads = UnsureReads()
    assert not reads.lines(image(), [Unsure(LABEL, 0.44)], [Rect(150, 25, 20, 20)], reader)
    assert not reads.lines(image(), [Unsure(LABEL, 0.2)], [], reader)
    assert not reads.lines(image(), [Unsure(Rect(100, 20, 30, 30), 0.44)], [], reader)  # Square.
    assert not reader.calls


def test_the_largest_firm_reading_wins_where_boxes_nest() -> None:
    reader = Reader(
        {
            ROW: line(ROW, "クエストチーム", *([0.92] * 7)),
            LABEL: line(LABEL, "クエス", 0.95, 0.94, 0.93),
        }
    )
    found = UnsureReads().lines(image(), [Unsure(LABEL, 0.4), Unsure(ROW, 0.33)], [], reader)
    assert [r.text for r in found] == ["クエストチーム"]


def test_reads_are_remembered_by_pixels_and_bounded_per_scan() -> None:
    boxes = [Rect(10 + 60 * i, 60, 30, 12) for i in range(8)]  # Separate lines.
    reader = Reader({b: line(b, "文字", 0.95, 0.95) for b in boxes})
    reads, pixels = UnsureReads(), image()
    unsure = [Unsure(b, 0.45 - i * 0.01) for i, b in enumerate(boxes)]
    first = reads.lines(pixels, unsure, [], reader)
    assert len(reader.calls[0]) == 6 and len(first) == 6  # Most confident boxes first.
    second = reads.lines(pixels, unsure, [], reader)
    assert reader.calls[1] == boxes[6:] and len(second) == 8
    reads.lines(pixels, unsure, [], reader)
    assert len(reader.calls) == 2  # Unchanged pixels: nothing is read again.
    pixels[60:72, 10:40] = 200
    reads.lines(pixels, unsure, [], reader)
    assert reader.calls[2] == [boxes[0]]


def test_rejections_are_remembered_and_a_short_answer_is_not_trusted() -> None:
    reader = Reader({ICON: line(ICON, "同.物", 0.3, 0.2, 0.4)})
    reads = UnsureReads()
    reads.lines(image(), [Unsure(ICON, 0.4)], [], reader)
    reads.lines(image(), [Unsure(ICON, 0.4)], [], reader)
    assert len(reader.calls) == 1

    class Short(Reader):
        def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
            self.calls.append(boxes)
            return ()

    short = Short({})
    assert not reads.lines(image(), [Unsure(LABEL, 0.44)], [], short)
    assert not reads.lines(image(), [Unsure(LABEL, 0.44)], [], short)
    assert len(short.calls) == 2  # Nothing was remembered from the mismatched answer.


def glyphs(pixels: np.ndarray, xs: range | list[int], top: int = 20, size: int = 26) -> None:
    for x in xs:
        cv2.rectangle(pixels, (x, top + 2), (x + size - 6, top + size), (240, 240, 240), 2)


def test_a_cut_box_is_grown_to_its_line_and_read_whole() -> None:
    pixels = np.full((100, 700, 3), 30, np.uint8)
    glyphs(pixels, range(100, 260, 32))  # Five glyphs; the doubted box holds three.
    cv2.line(pixels, (264, 34), (286, 34), (240, 240, 240), 2)  # A long-vowel stroke, ー.
    glyphs(pixels, [292])
    cut = Rect(98, 18, 96, 32)

    class Whole(Reader):
        def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
            self.calls.append(boxes)
            return tuple(line(b, "クエストーム", *([0.95] * 6)) for b in boxes)

    reader = Whole({})
    (found,) = UnsureReads().lines(pixels, [Unsure(cut, 0.4)], [], reader)
    (box,) = reader.calls[0]
    assert box.x <= 100 and box.x + box.width >= 312  # Through the stroke to the last glyph.
    assert box.y == cut.y and box.height == cut.height
    assert found.text == "クエストーム"


def test_a_line_that_keeps_going_past_the_reach_is_not_read() -> None:
    pixels = np.full((100, 2000, 3), 30, np.uint8)
    glyphs(pixels, range(40, 1960, 30))
    reader = Reader({})
    assert not UnsureReads().lines(pixels, [Unsure(Rect(900, 18, 96, 32), 0.4)], [], reader)
    assert not reader.calls


def test_accepted_lines_bound_growth_and_fragments_of_a_line_are_read_once() -> None:
    pixels = np.full((100, 700, 3), 30, np.uint8)
    glyphs(pixels, range(100, 420, 32))
    accepted = Rect(290, 18, 130, 32)  # Already read by the coarse pass.

    class Whole(Reader):
        def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
            self.calls.append(boxes)
            return tuple(line(b, "ほげほげ", *([0.95] * 4)) for b in boxes)

    reader = Whole({})
    fragments = [Unsure(Rect(98, 18, 60, 32), 0.4), Unsure(Rect(160, 18, 60, 32), 0.35)]
    UnsureReads().lines(pixels, fragments, [accepted], reader)
    (box,) = reader.calls[0]
    assert box.x <= 100 and box.x + box.width <= accepted.x


def test_growth_follows_a_line_beyond_one_search_width() -> None:
    pixels = np.full((100, 1200, 3), 30, np.uint8)
    glyphs(pixels, range(100, 640, 32))  # About 17 line heights: several searches long.

    class Whole(Reader):
        def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
            self.calls.append(boxes)
            return tuple(line(b, "ながいみだし", *([0.95] * 6)) for b in boxes)

    reader = Whole({})
    UnsureReads().lines(pixels, [Unsure(Rect(98, 18, 96, 32), 0.4)], [], reader)
    (box,) = reader.calls[0]
    assert box.x + box.width >= 630
