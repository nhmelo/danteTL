"""Edge icons are separated from lettering by pixels, verified by what OCR read over them."""

import cv2
import numpy as np
import pytest

from dantetl.domain import Rect
from dantetl.ocr.edge_icons import trim_edge_icons
from dantetl.ocr.values import TextCharacter, TextRegion

GLYPH = 30  # Glyph ink height; the line box adds a little margin.
TOP = 40


def canvas(width: int = 420) -> np.ndarray:
    return np.full((120, width, 3), 30, np.uint8)


def glyph(image: np.ndarray, x: int, *, thickness: int = 2) -> None:
    """A thin-stroked character-sized shape."""
    cv2.rectangle(image, (x, TOP), (x + 26, TOP + GLYPH), (235, 235, 235), thickness)
    cv2.line(image, (x, TOP + 15), (x + 26, TOP + 15), (235, 235, 235), thickness)


def icon(image: np.ndarray, x: int) -> None:
    """A filled emblem, taller and heavier than the glyphs beside it."""
    cv2.circle(image, (x + 19, TOP + 15), 19, (90, 200, 240), -1)


def line(
    chars: str, scores: tuple[float, ...], xs: tuple[int, ...], left: int | None = None
) -> TextRegion:
    characters = tuple(
        TextCharacter(c, Rect(x, TOP - 2, 34, GLYPH + 4), s)
        for c, s, x in zip(chars, scores, xs, strict=True)
    )
    left = min(xs) - 4 if left is None else left
    right = max(xs) + 38
    return TextRegion(chars, Rect(left, TOP - 4, right - left, GLYPH + 8), characters)


class Reader:
    def __init__(self, *answers: TextRegion) -> None:
        self.answers = answers
        self.boxes: list[list[Rect]] = []

    def __call__(self, boxes: list[Rect]) -> tuple[TextRegion, ...]:
        self.boxes.append(boxes)
        return self.answers


def labelled(image: np.ndarray, *, glyphs: tuple[int, ...] = (70, 104, 138, 172)) -> None:
    icon(image, 20)
    for x in glyphs:
        glyph(image, x)


def test_icon_glyph_is_dropped_and_its_box_no_longer_covers_the_icon() -> None:
    image = canvas()
    labelled(image)
    raw = line("〇戻るよ", (0.37, 0.93, 0.93, 0.9), (20, 70, 104, 138))
    reader = Reader()
    (trimmed,) = trim_edge_icons(image, (raw,), reader)
    assert trimmed.text == "戻るよ"
    assert [c.text for c in trimmed.characters] == ["戻", "る", "よ"]
    assert trimmed.bounds == raw.bounds  # Detector geometry is preserved; characters move.
    assert not reader.boxes  # Every glyph beside the icon was already read.


def test_icon_drawn_in_two_touching_parts_is_dropped_whole() -> None:
    image = canvas()
    icon(image, 20)
    # The emblem's second half is as tall but drawn with a thinner line, just apart.
    cv2.ellipse(image, (72, TOP + 15), (8, 19), 0, 0, 360, (90, 200, 240), 3)
    for x in (96, 130, 164):
        glyph(image, x)
    # OCR read one mark at the second half only.
    raw = line("*戻るよ", (0.67, 0.93, 0.93, 0.9), (56, 96, 130, 164), left=16)
    (trimmed,) = trim_edge_icons(image, (raw,), Reader())
    assert trimmed.text == "戻るよ"


def test_a_weak_unpaired_bracket_over_an_icon_is_dropped() -> None:
    image = canvas()
    labelled(image)
    raw = line("(軌跡よ", (0.26, 0.93, 0.95, 0.9), (20, 70, 104, 138))
    (trimmed,) = trim_edge_icons(image, (raw,), Reader())
    assert trimmed.text == "軌跡よ"


def test_trailing_icon_is_dropped_too() -> None:
    image = canvas()
    for x in (20, 54, 88):
        glyph(image, x)
    icon(image, 130)
    raw = line("設定画8", (0.95, 0.95, 0.9, 0.4), (20, 54, 88, 130))
    (trimmed,) = trim_edge_icons(image, (raw,), Reader())
    assert trimmed.text == "設定画"


def test_displaced_glyph_is_recovered_by_one_verified_reread() -> None:
    image = canvas()
    labelled(image)
    # OCR read the icon as * and lost the first glyph beside it.
    raw = line("*魂魄", (0.67, 0.89, 0.9), (20, 104, 138))
    recovered = line("星魂魄", (0.93, 0.88, 0.9), (70, 104, 138))
    reader = Reader(recovered)
    (trimmed,) = trim_edge_icons(image, (raw,), reader)
    assert trimmed.text == "星魂魄"
    (box,) = reader.boxes[0]
    assert 39 < box.x < 70  # Between the icon and the displaced glyph.


def ring(image: np.ndarray, x: int) -> None:
    """An outlined emblem at glyph height, heavier than the glyphs but not much taller."""
    cv2.circle(image, (x + 15, TOP + 15), 14, (90, 200, 240), 5)


SPACED = (70, 104, 138)


@pytest.mark.parametrize(
    ("x", "score", "glyphs", "dropped"),
    [
        (20, 0.35, SPACED, True),  # Weakly read and set apart from the label: an icon.
        (20, 0.6, SPACED, False),  # Read with word confidence: pixels alone are not enough.
        (35, 0.35, SPACED, False),  # As close as the label's glyphs: it may be one of them.
        (20, 0.35, (70, 120, 170), False),  # The label is spaced at least as widely.
    ],
)
def test_a_weakly_read_heavier_unit_set_apart_from_the_label_is_an_icon(
    x, score, glyphs, dropped
) -> None:
    image = canvas()
    ring(image, x)
    for g in glyphs:
        glyph(image, g)
    raw = line("#戻るよ", (score, 0.93, 0.93, 0.9), (x, *glyphs))
    (trimmed,) = trim_edge_icons(image, (raw,), Reader())
    assert trimmed.text == ("戻るよ" if dropped else raw.text)


@pytest.mark.parametrize(
    "answer",
    [
        line("魂", (0.9,), (104,)),  # Loses the kept text.
        line("・星魂魄", (0.59, 0.93, 0.88, 0.9), (60, 70, 104, 138)),  # Adds a weak mark.
        line("x星魂魄", (0.3, 0.93, 0.88, 0.9), (60, 70, 104, 138)),
    ],
)
def test_reread_that_changes_the_text_side_is_rejected(answer) -> None:
    image = canvas()
    labelled(image)
    raw = line("*魂魄", (0.67, 0.89, 0.9), (20, 104, 138))
    (trimmed,) = trim_edge_icons(image, (raw,), Reader(answer))
    assert trimmed.text == "魂魄"


def test_missing_reread_answers_only_drop_icon_glyphs() -> None:
    image = canvas()
    labelled(image)
    raw = line("*魂魄", (0.67, 0.89, 0.9), (20, 104, 138))
    (trimmed,) = trim_edge_icons(image, (raw,), Reader())
    assert trimmed.text == "魂魄"


@pytest.mark.parametrize(
    ("chars", "scores"),
    [
        ("新着メ", (0.93, 0.94, 0.9)),  # A confidently read letter is never removed by pixels.
        ("【戻る", (0.9, 0.93, 0.93)),  # Confident brackets and bullets carry structure.
        ("(戻)", (0.3, 0.93, 0.93)),  # So does a paired bracket, however faint.
        ("●戻る", (0.4, 0.93, 0.93)),
        ("85%", (0.4, 0.9, 0.9)),  # A number keeps its digits.
        ("8ツ%", (0.4, 0.2, 0.9)),  # Weak letters are no evidence of a word.
    ],
)
def test_structure_numbers_and_confident_letters_are_kept(chars, scores) -> None:
    image = canvas()
    labelled(image)
    raw = line(chars, scores, (20, 70, 104))
    assert trim_edge_icons(image, (raw,), Reader()) == (raw,)


def test_glyph_sized_or_bold_neighbours_are_not_icons() -> None:
    image = canvas()
    glyph(image, 20, thickness=4)  # Bold but not taller.
    for x in (70, 104, 138):
        glyph(image, x)
    raw = line("太戻るよ", (0.4, 0.93, 0.93, 0.9), (20, 70, 104, 138))
    assert trim_edge_icons(image, (raw,), Reader()) == (raw,)
    plain = canvas()
    for x in (20, 70, 104, 138):
        glyph(plain, x)
    raw = line("口戻るよ", (0.4, 0.93, 0.93, 0.9), (20, 70, 104, 138))
    assert trim_edge_icons(plain, (raw,), Reader()) == (raw,)


def test_vertical_short_and_unaligned_lines_are_untouched() -> None:
    image = canvas()
    labelled(image)
    raw = line("〇戻るよ", (0.37, 0.93, 0.93, 0.9), (20, 70, 104, 138))
    vertical = TextRegion(raw.text, raw.bounds, raw.characters, vertical=True)
    unaligned = TextRegion("違う", raw.bounds, raw.characters)
    single = line("〇", (0.37,), (20,))
    regions = (vertical, unaligned, single)
    assert trim_edge_icons(image, regions, Reader()) == regions
