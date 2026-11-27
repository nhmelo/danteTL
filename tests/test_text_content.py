import json
from dataclasses import replace
from pathlib import Path

import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.ocr.text_content import content_region
from dantetl.ocr.tracking import TextTracker
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion


def button_observations() -> tuple[TextRegion, ...]:
    # OCR values only, measured from the supplied button at 1080p/4K. No game pixels in Git.
    records = json.loads((Path(__file__).parent / "fixtures/button_ocr.json").read_text("utf-8"))
    return tuple(
        TextRegion(
            raw["text"],
            Rect(**raw["bounds"]),
            tuple(
                TextCharacter(c["text"], Rect(**c["bounds"]), c["confidence"])
                for c in raw["characters"]
            ),
            raw["vertical"],
        )
        for record in records
        for raw in record["regions"]
    )


def test_real_button_observations_exclude_weak_icon_but_preserve_raw_evidence() -> None:
    observations = button_observations()
    assert any(raw.text.startswith("中") for raw in observations)
    assert any(raw.text.startswith("@") for raw in observations)
    for raw in observations:
        clean = content_region(raw)
        assert clean.text == "ストーリーを振り返る"
        assert 210 <= clean.bounds.x <= 218
        assert clean.bounds.width < 485
        assert len(clean.characters) == 10


def test_alternating_detector_icon_crops_keep_exact_track_geometry_and_revision() -> None:
    tracker = TextTracker()
    bounds = Rect(0, 0, 873, 245)
    winner = None
    for sequence, raw in enumerate(button_observations() * 12, 1):
        now = sequence * 0.25
        frame = CapturedFrame(sequence, now, WindowGeometry(bounds, bounds, bounds, 96), b"")
        tracker.observe(OcrResult(frame, (raw,), 20, now + 0.02))
        current = tracker.advance(frame, now + 0.02).visible[0]
        if winner is None:
            winner = current
        assert current is winner
        assert current.text == "ストーリーを振り返る"


def sample(edge="@", *, gap=25, score=0.3, trailing=False) -> TextRegion:
    core = tuple(
        TextCharacter(c, Rect(100 + i * 25, 10, 25, 30), 0.95)
        for i, c in enumerate("道具を確認する")
    )
    x = core[-1].bounds.x + 25 + gap if trailing else 100 - gap - 25
    char = TextCharacter(edge, Rect(x, 10, 25, 30), score)
    chars = (*core, char) if trailing else (char, *core)
    return TextRegion("".join(c.text for c in chars), Rect(0, 0, 400, 50), chars)


@pytest.mark.parametrize("edge", ["「", "（", "1", "一", "？", "。", "−"])
def test_structural_punctuation_and_numeric_prefixes_are_not_stripped(edge) -> None:
    raw = sample(edge)
    assert content_region(raw).text == raw.text


@pytest.mark.parametrize("trailing", [False, True])
def test_only_spatially_isolated_weak_edges_are_filtered(trailing) -> None:
    assert content_region(sample(trailing=trailing)).text == "道具を確認する"
    for raw in (sample(gap=1, trailing=trailing), sample(score=0.9, trailing=trailing)):
        assert content_region(raw).text == raw.text


def test_interior_weak_character_and_incomplete_metadata_remain_intact() -> None:
    raw = sample(gap=1)
    chars = list(raw.characters)
    chars[3] = replace(chars[3], confidence=0.1)
    assert content_region(replace(raw, characters=tuple(chars))).text == raw.text
    incomplete = replace(raw, text="違う文章")
    assert content_region(incomplete) is incomplete
    assert content_region(replace(raw, vertical=True)).text == raw.text


def test_unrecognized_detector_padding_does_not_become_overlay_width() -> None:
    raw = sample().characters[1:]
    text = "".join(c.text for c in raw)
    small = content_region(TextRegion(text, Rect(95, 8, 185, 35), raw))
    large = content_region(TextRegion(text, Rect(20, 0, 300, 50), raw))
    assert small == large


def test_unreliable_character_metadata_does_not_authorize_removing_text() -> None:
    raw = sample()
    chars = (
        raw.characters[0],
        replace(raw.characters[1], confidence=float("nan")),
        *raw.characters[2:],
    )
    assert content_region(replace(raw, characters=chars)).text == raw.text
    unordered = (raw.characters[0], *raw.characters[1:][::-1])
    mismatched = replace(raw, text="".join(c.text for c in unordered), characters=unordered)
    assert content_region(mismatched).text == mismatched.text


def scored(text: str, *scores: float) -> TextRegion:
    chars = tuple(
        TextCharacter(c, Rect(100 + i * 25, 10, 25, 30), score)
        for i, (c, score) in enumerate(zip(text, scores, strict=True))
    )
    return TextRegion(text, Rect(90, 5, 40 + 25 * len(text), 40), chars)


@pytest.mark.parametrize(
    ("raw", "kept"),
    [
        # Confidences measured from Meiki misreads of a percentage, a plus sign and an icon.
        (scored("396ツ%", 0.87, 0.86, 0.77, 0.17, 0.81), "396%"),
        (scored("千159%", 0.30, 0.81, 0.86, 0.69, 0.18), "159%"),
        (scored("十", 0.34), ""),
    ],
)
def test_weak_letters_are_not_word_evidence_beside_a_read_number_or_alone(raw, kept) -> None:
    clean = content_region(raw)
    assert clean.text == kept
    assert "".join(c.text for c in clean.characters) == kept
    assert not DEFAULT_SOURCE.contains_text(clean.text)
    if kept:
        assert clean.bounds.x == clean.characters[0].bounds.x
    else:
        assert clean.bounds == raw.bounds


@pytest.mark.parametrize(
    "raw",
    [
        scored("天輔", 0.95, 0.22),  # A word keeps a confident letter.
        scored("設定", 0.45, 0.42),  # Faint labels are not filtered by confidence alone.
        scored("5人", 0.40, 0.30),  # An unread number gives no evidence against its counter.
        scored("3回", 0.90, 0.80),
        scored("あいう1", 0.30, 0.30, 0.30, 0.90),  # Three weak letters read as faint text.
        scored("十", float("nan")),
    ],
)
def test_words_faint_labels_and_unread_numbers_keep_weak_letters(raw) -> None:
    assert content_region(raw).text == raw.text


def test_lone_weak_glyph_never_becomes_a_track() -> None:
    tracker = TextTracker()
    bounds = Rect(0, 0, 400, 100)
    for sequence in range(1, 5):
        now = sequence * 0.25
        frame = CapturedFrame(sequence, now, WindowGeometry(bounds, bounds, bounds, 96), b"")
        icon, number = scored("十", 0.34), scored("50ツ%", 0.89, 0.82, 0.22, 0.46)
        number = replace(
            number,
            bounds=replace(number.bounds, y=55),
            characters=tuple(replace(c, bounds=replace(c.bounds, y=60)) for c in number.characters),
        )
        tracker.observe(OcrResult(frame, (icon, number), 20, now + 0.02))
        visible = tracker.advance(frame, now + 0.02).visible
        assert [t.text for t in visible] == ["50%"]
