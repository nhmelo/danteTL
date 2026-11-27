"""Exercise source-matched quote evidence through the live grouping and request path."""

from dataclasses import replace

import numpy as np
import pytest
from live_fixtures import Replay

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.values import TextCharacter, TextRegion


def quote_panel(prefix: str, *, dash: bool = True, detector: bool = True):
    frame_bounds = Rect(0, 0, 600, 230)
    pixels = np.zeros((230, 600, 4), np.uint8)
    # Identical ink whether OCR includes the leading strokes or starts at the bracket.
    if dash:
        pixels[83:85, 60:117, :3] = 220
    pixels[73:94, 121:390:3, :3] = 220
    frame = CapturedFrame(
        1, 1, WindowGeometry(frame_bounds, frame_bounds, frame_bounds, 96), pixels.tobytes()
    )
    text = prefix + "私はこの長い道をこれからも"
    full = Rect(60, 72, 330, 24)
    ink = full if prefix.startswith(("-", "―")) else Rect(121, 72, 269, 24)
    rows = (
        TextRegion("新しい朝がやって来た。", Rect(60, 38, 220, 24)),
        TextRegion(text, full if detector else ink, (TextCharacter(text, ink, 0.98),)),
        TextRegion("一緒に歩いていきます」", Rect(60, 106, 240, 24)),
        TextRegion("- 戻る", Rect(60, 140, 100, 24)),
    )
    return frame, rows, ink


@pytest.mark.parametrize("prefix", ["-----「", "「", "/"])
@pytest.mark.parametrize("detector", [True, False])
def test_only_quote_wraps_with_seen_or_missed_dash(prefix, detector):
    frame, rows, ink = quote_panel(prefix, detector=detector)
    replay = Replay()
    scene = replay.observe(frame, rows)
    assert [[t.text for t in b.lines] for b in scene.blocks] == [
        [rows[0].text],
        [rows[1].text, rows[2].text],
        [rows[3].text],
    ]
    quote = scene.requests[1]
    assert quote.text == rows[1].text + rows[2].text
    assert quote.lines[0].bounds == ink
    assert len(replay.live.scene(frame, 2, grouping=False).blocks) == 4


@pytest.mark.parametrize("prefix", ["-----「", "/"])
@pytest.mark.parametrize("choice", ["- 道を進みます」", "1. 道を進みます」", "設定を変更する？"])
def test_nearby_choice_cannot_supply_quote_continuation(prefix, choice):
    frame, rows, _ = quote_panel(prefix)
    rows = (*rows[:2], replace(rows[2], text=choice), rows[3])
    scene = Replay().observe(frame, rows)
    assert [len(b.lines) for b in scene.blocks] == [1, 1, 1, 1]


@pytest.mark.parametrize("prefix", ["-----「", "―「", "/"])
def test_unmarked_button_directly_below_closing_quote_stays_separate(prefix):
    frame, rows, _ = quote_panel(prefix)
    rows = (*rows[:3], TextRegion("装備を変更する", Rect(60, 140, 230, 24)))
    scene = Replay().observe(frame, rows)
    assert [b.text for b in scene.blocks] == [
        rows[0].text,
        rows[1].text + rows[2].text,
        rows[3].text,
    ]


def test_missing_dash_needs_ink_on_the_exact_observation():
    replay = Replay()
    frame, rows, _ = quote_panel("/")
    assert [len(b.lines) for b in replay.observe(frame, rows).blocks] == [1, 2, 1]
    blank, _, _ = quote_panel("/", dash=False)
    blank = replace(blank, sequence=2, captured_at=2)
    assert [len(b.lines) for b in replay.observe(blank, rows).blocks] == [1, 1, 1, 1]
    # Re-polling the older OCR result cannot restore its pixel evidence.
    assert [len(b.lines) for b in replay.observe(frame, rows).blocks] == [1, 1, 1, 1]


def test_partial_scan_uses_prior_column_but_requires_fresh_dash_pixels():
    replay = Replay()
    frame, rows, ink = quote_panel("/", detector=False)
    assert [len(b.lines) for b in replay.observe(frame, rows).blocks] == [1, 2, 1]
    crop = replace(frame, sequence=2, captured_at=2)
    assert [len(b.lines) for b in replay.observe(crop, rows[1:2], scanned=(ink,)).blocks] == [
        1,
        2,
        1,
    ]
    blank, _, _ = quote_panel("/", dash=False)
    crop = replace(blank, sequence=3, captured_at=3)
    assert [len(b.lines) for b in replay.observe(crop, rows[1:2], scanned=(ink,)).blocks] == [
        1,
        1,
        1,
        1,
    ]


def test_nested_quotation_keeps_its_following_row_until_the_outer_quote_closes():
    frame, rows, _ = quote_panel("-----「")
    middle = TextRegion("あの日の合言葉は『新しい夜明け』", Rect(60, 106, 330, 24))
    end = TextRegion("それを今も覚えています」", Rect(60, 140, 320, 24))
    button = TextRegion("設定を変更する", Rect(60, 174, 260, 24))
    scene = Replay().observe(frame, (*rows[:2], middle, end, button))
    assert [len(b.lines) for b in scene.blocks] == [1, 3, 1]
