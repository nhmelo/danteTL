"""Button prompts at line ends are found by their outline and dropped from the reading."""

import cv2
import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.key_prompts import trim_key_prompts
from dantetl.ocr.values import TextCharacter, TextRegion

GLYPH = 30
TOP = 40
WHITE, GOLD = (235, 235, 235), (60, 170, 200)


def canvas(width: int = 520) -> np.ndarray:
    return np.full((120, width, 3), 30, np.uint8)


def glyph(image: np.ndarray, x: int) -> None:
    """A character-sized shape whose strokes reach its own edges."""
    cv2.rectangle(image, (x, TOP), (x + 26, TOP + GLYPH), WHITE, 2)
    cv2.line(image, (x, TOP + 15), (x + 26, TOP + 15), WHITE, 2)


def key(image: np.ndarray, x: int, name: str, *, outline: tuple[int, int, int] = GOLD) -> int:
    """A prompt: a short key name in small letters inside its own outline. Returns its right."""
    right = x + 22 + 12 * len(name)
    cv2.rectangle(image, (x, TOP - 3), (right, TOP + GLYPH + 3), outline, 2)
    cv2.putText(image, name, (x + 9, TOP + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return right


def plus(image: np.ndarray, x: int) -> None:
    cv2.line(image, (x, TOP + 15), (x + 10, TOP + 15), WHITE, 2)
    cv2.line(image, (x + 5, TOP + 10), (x + 5, TOP + 20), WHITE, 2)


def line(chars: str, scores: tuple[float, ...], xs: tuple[int, ...], right: int) -> TextRegion:
    characters = tuple(
        TextCharacter(c, Rect(x, TOP - 2, 26, GLYPH + 4), s)
        for c, s, x in zip(chars, scores, xs, strict=True)
    )
    left = min(xs) - 4
    return TextRegion(chars, Rect(left, TOP - 4, right - left, GLYPH + 8), characters)


LABEL = (20, 54, 88, 122)


def labelled(image: np.ndarray) -> None:
    for x in LABEL:
        glyph(image, x)


def test_trailing_prompts_joined_by_a_mark_are_dropped() -> None:
    image = canvas()
    labelled(image)
    first = key(image, 170, "L2")
    plus(image, first + 8)
    second = key(image, first + 26, "R2")
    # OCR boxes the outline it read as ( a little outside the outline itself.
    raw = line(
        "状態効果(L24(R",
        (0.97, 0.99, 0.93, 0.94, 0.14, 0.56, 0.55, 0.19, 0.18, 0.45),
        (*LABEL, 150, 182, 196, first + 6, first + 24, first + 40),
        second + 2,
    )
    (trimmed,) = trim_key_prompts(image, (raw,))
    assert trimmed.text == "状態効果"
    assert trimmed.bounds == raw.bounds  # Detector geometry is preserved; characters move.


def test_a_leading_prompt_the_detector_cut_through_is_dropped() -> None:
    image = canvas()
    right = key(image, 20, "L1")
    xs = tuple(right + 16 + 34 * i for i in range(4))
    for x in xs:
        glyph(image, x)
    # The line starts inside the prompt; OCR read its right half as a weak kana.
    raw = line("ハショート", (0.31, 0.85, 0.88, 0.92, 0.9), (right - 24, *xs), xs[-1] + 30)
    (trimmed,) = trim_key_prompts(image, (raw,))
    assert trimmed.text == "ショート"


def test_a_confidently_read_key_name_is_dropped_too() -> None:
    image = canvas()
    labelled(image)
    right = key(image, 170, "R1", outline=WHITE)  # Outline in the lettering's own color.
    raw = line("戻るよ、R1", (0.93, 0.93, 0.9, 0.84, 0.89, 0.85), (*LABEL, 182, 196), right + 2)
    (trimmed,) = trim_key_prompts(image, (raw,))
    assert trimmed.text == "戻るよ、"


def test_an_enclosing_kanji_read_as_itself_stays() -> None:
    image = canvas()
    for x in LABEL[:3]:
        glyph(image, x)
    # Box within a box, as 回 is drawn: the inner box floats inside the outer one.
    cv2.rectangle(image, (122, TOP - 1), (152, TOP + GLYPH + 1), WHITE, 2)
    cv2.rectangle(image, (132, TOP + 10), (142, TOP + 20), WHITE, 2)
    raw = line("戻るよ回", (0.93, 0.93, 0.9, 0.95), LABEL, 156)
    assert trim_key_prompts(image, (raw,)) == (raw,)


def test_a_glyph_enclosing_space_with_its_own_strokes_stays() -> None:
    image = canvas()
    for x in LABEL[:3]:
        glyph(image, x)
    # A box glyph drawn in two halves, read weakly: its strokes are the plate's own rim.
    top, bottom = TOP - 1, TOP + GLYPH + 1
    halves = (
        [(135, top), (122, top), (122, bottom), (135, bottom)],
        [(140, top), (152, top), (152, bottom), (140, bottom)],
    )
    cv2.polylines(image, [np.array(h) for h in halves], False, WHITE, 2)
    raw = line("戻るよロ", (0.93, 0.93, 0.9, 0.4), LABEL, 156)
    assert trim_key_prompts(image, (raw,)) == (raw,)


def test_small_latin_without_an_outline_stays() -> None:
    image = canvas()
    for x in LABEL[:3]:
        glyph(image, x)
    cv2.putText(image, "L2", (130, TOP + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    raw = line("戻るよL2", (0.93, 0.93, 0.9, 0.4, 0.5), (*LABEL[:3], 126, 140), 160)
    assert trim_key_prompts(image, (raw,)) == (raw,)


def test_a_prompt_alone_or_inside_the_line_is_kept() -> None:
    image = canvas()
    right = key(image, 20, "R1")
    alone = line("R1", (0.89, 0.85), (30, 44), right + 2)
    assert trim_key_prompts(image, (alone,)) == (alone,)
    inner = canvas()
    glyph(inner, 20)
    middle = key(inner, 54, "LB")
    glyph(inner, middle + 10)
    glyph(inner, middle + 44)
    # A glyph lies between the prompt and the line's start, even though OCR read it weakly.
    xs = (20, 54, 66, 80, middle + 10, middle + 44)
    raw = line("ロ(L)るよ", (0.4, 0.3, 0.6, 0.3, 0.93, 0.9), xs, middle + 74)
    assert trim_key_prompts(inner, (raw,)) == (raw,)
