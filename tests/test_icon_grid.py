"""Merged menu labels split at the cells of the icon row they sit under; sentences stay whole."""

import cv2
import numpy as np
import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.icon_grid import split_icon_grids
from dantetl.ocr.values import TextCharacter, TextRegion
from dantetl.ocr.worker import recognize_regions

PITCH = 90
ICON_TOP, ICON = 20, 40
LINE_TOP, GLYPH = 70, 22
STEP = 20  # Glyph pitch inside a word; glyph ink is 14 wide, leaving gaps like real text.


def canvas() -> np.ndarray:
    return np.full((130, 520, 3), 25, np.uint8)


def icons(image: np.ndarray, centers: tuple[int, ...], top: int = ICON_TOP) -> None:
    for x in centers:
        cv2.circle(image, (x, top + ICON // 2), ICON // 2, (220, 200, 90), -1)


def glyphs(image: np.ndarray, xs: list[int], top: int = LINE_TOP) -> None:
    for x in xs:
        cv2.rectangle(image, (x + 2, top), (x + 16, top + GLYPH), (240, 240, 240), 2)


def region(text: str, xs: list[int], scores: list[float] | None = None, top: int = LINE_TOP):
    scores = scores or [0.95] * len(xs)
    chars = tuple(
        TextCharacter(c, Rect(x, top - 1, 18, GLYPH + 2), s)
        for c, x, s in zip(text, xs, scores, strict=True)
    )
    left, right = xs[0] - 3, xs[-1] + 21
    return TextRegion(text, Rect(left, top - 3, right - left, GLYPH + 6), chars)


def centered(center: int, count: int) -> list[int]:
    width = count * STEP - 2
    return [round(center - width / 2) + i * STEP for i in range(count)]


CENTERS = (60, 150, 240, 330)
WORDS = ("クエスト", "チム", "イベント", "記録")


def menu(image: np.ndarray, centers=CENTERS, words=WORDS, top: int = LINE_TOP) -> TextRegion:
    xs = [
        x for center, word in zip(centers, words, strict=True) for x in centered(center, len(word))
    ]
    glyphs(image, xs, top)
    return region("".join(words), xs, top=top)


def test_merged_labels_split_into_their_own_cells_with_their_own_glyphs() -> None:
    image = canvas()
    icons(image, CENTERS)
    merged = menu(image)
    labels = split_icon_grids(image, (merged,))
    assert [label.text for label in labels] == list(WORDS)
    for label, center in zip(labels, CENTERS, strict=True):
        assert abs(label.bounds.x + label.bounds.width / 2 - center) <= 3
        assert label.bounds.y == merged.bounds.y and label.bounds.height == merged.bounds.height
        assert "".join(c.text for c in label.characters) == label.text


def test_labels_over_their_icons_split_too() -> None:
    image = canvas()
    merged = menu(image, top=20)
    icons(image, CENTERS, top=48)
    assert [label.text for label in split_icon_grids(image, (merged,))] == list(WORDS)


def test_weak_reading_of_the_icon_row_is_removed_but_lettering_there_is_kept() -> None:
    image = canvas()
    icons(image, CENTERS)
    merged = menu(image)
    reading = region("と*さの", [x - 9 for x in CENTERS], [0.3, 0.3, 0.32, 0.16], top=ICON_TOP + 9)
    heading = region("見出しだ", [x - 9 for x in CENTERS], [0.95] * 4, top=ICON_TOP + 9)
    assert [r.text for r in split_icon_grids(image, (reading, merged))] == list(WORDS)
    assert split_icon_grids(image, (heading, merged))[0] == heading


def test_two_icons_split_only_the_labels_under_them() -> None:
    image = canvas()
    icons(image, (150, 240))
    merged = menu(image, (150, 240), ("チム", "イベント"))
    assert [r.text for r in split_icon_grids(image, (merged,))] == ["チム", "イベント"]
    wider = canvas()
    icons(wider, (150, 240))
    beyond = menu(wider, (150, 240, 330), ("チム", "イベント", "記録"))
    assert split_icon_grids(wider, (beyond,)) == (beyond,)


@pytest.mark.parametrize("text", ["仲間と一緒に森の奥へ進む", "森の奥は危険だ気をつけ進もう"])
def test_a_sentence_under_an_icon_row_keeps_its_pitch_and_stays_whole(text) -> None:
    image = canvas()
    icons(image, CENTERS)
    xs = [40 + i * STEP for i in range(len(text))]
    glyphs(image, xs)
    sentence = region(text, xs)
    assert split_icon_grids(image, (sentence,)) == (sentence,)


def test_sentence_punctuation_keeps_a_line_whole() -> None:
    image = canvas()
    icons(image, CENTERS)
    merged = menu(image, words=("クエスト", "チム", "イベント", "記録。"))
    assert split_icon_grids(image, (merged,)) == (merged,)


def test_irregular_icons_or_lines_past_the_row_are_not_grids() -> None:
    image = canvas()
    icons(image, (60, 150, 285, 330))  # Irregular spacing.
    merged = menu(image)
    assert split_icon_grids(image, (merged,)) == (merged,)
    image = canvas()
    icons(image, CENTERS[:3])
    beyond = menu(image, (60, 150, 240, 330, 420), (*WORDS, "設定"))
    assert split_icon_grids(image, (beyond,)) == (beyond,)


def test_labels_off_the_cell_centers_are_not_icon_labels() -> None:
    image = canvas()
    icons(image, CENTERS)
    # Each word starts at its cell's left edge instead of centering under the icon.
    xs = [center - 42 + i * STEP for center in CENTERS for i in range(2)]
    glyphs(image, xs)
    line = region("ああいいううええ", xs)
    assert split_icon_grids(image, (line,)) == (line,)


def test_cell_boundary_through_ink_keeps_the_line_whole() -> None:
    image = canvas()
    icons(image, CENTERS)
    merged = menu(image)
    for center in CENTERS[:-1]:  # Bridge every boundary with a stroke.
        cv2.line(image, (center + 20, LINE_TOP + 11), (center + 70, LINE_TOP + 11), (240,) * 3, 3)
    assert split_icon_grids(image, (merged,)) == (merged,)


class StripReader:
    """Reads the merged line inside one crop, as a detector would; it never sees the icons."""

    def __init__(self, area: Rect, merged: TextRegion) -> None:
        self.area, self.merged = area, merged

    def recognize(self, frame: CapturedFrame, *, full_size=None) -> tuple[TextRegion, ...]:
        assert (frame.width, frame.height) == self.area.size
        return (self.merged.translated(-self.area.x, -self.area.y),)

    def close(self) -> None:
        pass


def test_a_regional_scan_splits_labels_by_icons_outside_its_crop() -> None:
    image = canvas()
    icons(image, CENTERS)
    merged = menu(image)
    strip = Rect(0, LINE_TOP - 12, image.shape[1], GLYPH + 24)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bounds = Rect(0, 0, image.shape[1], image.shape[0])
    frame = CapturedFrame(1, 0.0, WindowGeometry(bounds, bounds, bounds, 96), bgra.tobytes())
    regions, scanned = recognize_regions(StripReader(strip, merged), frame, (strip,))
    assert scanned is not None
    assert [r.text for r in regions] == list(WORDS)


def test_ordinary_lines_ahead_of_merged_labels_do_not_use_up_the_analysis() -> None:
    image = np.full((400, 520, 3), 25, np.uint8)
    icons(image, CENTERS)
    merged = menu(image)
    words = []
    for i in range(20):  # Single words at an even pitch: nothing in them could split.
        xs = [40 + j * STEP for j in range(7)]
        glyphs(image, xs, top=130 + i * 12)
        words.append(region("ふつうのことば", xs, top=130 + i * 12))
    labels = split_icon_grids(image, (*words, merged))
    assert [r.text for r in labels[len(words) :]] == list(WORDS)


def test_a_reading_of_icons_over_a_sentence_is_removed_and_the_sentence_kept() -> None:
    image = canvas()
    icons(image, CENTERS)
    text = "仲間と一緒に森の奥へ進む"
    xs = [40 + i * STEP for i in range(len(text))]
    glyphs(image, xs)
    sentence = region(text, xs)
    reading = region("と*さの", [x - 9 for x in CENTERS], [0.3, 0.3, 0.32, 0.16], top=ICON_TOP + 9)
    assert split_icon_grids(image, (reading, sentence)) == (sentence,)
