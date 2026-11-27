"""Exercise graph admission, its boundaries and source-matched lettering through LiveSession."""

from dataclasses import replace

import numpy as np
import pytest
from live_fixtures import Replay

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.selection import source_language
from dantetl.ocr.values import TextRegion


def panel(regions: tuple[TextRegion, ...], *, accent: bool) -> CapturedFrame:
    bounds = Rect(0, 0, 800, 200)
    pixels = np.full((bounds.height, bounds.width, 4), 24, np.uint8)
    for region in regions:
        b = region.bounds
        pixels[b.y : b.y + b.height, b.x : b.x + b.width : 3, :3] = 224
    if accent:
        b = regions[0].bounds
        pixels[b.y : b.y + b.height, b.x + 99 : b.x + 219 : 3, :3] = (200, 60, 230)
    return CapturedFrame(1, 1, WindowGeometry(bounds, bounds, bounds, 96), pixels.tobytes())


@pytest.mark.parametrize(
    ("code", "first", "tail"),
    [
        ("en", "After looking through the archive, I finally found the letter you", "sent me"),
        ("fr", "En rangeant mes archives, j’ai retrouvé la lettre que tu", "m’avais envoyée"),
        ("es", "Al ordenar mis archivos, encontré la carta que me", "habías enviado"),
        ("ja", "古い手紙を整理していたら、あなたが書いたものを見つけた", "わね"),
    ],
)
@pytest.mark.parametrize("layout", ["bubble", "buttons", "no_edges", "large_panel", "heading"])
def test_unpunctuated_wrap_requires_a_shared_fitted_panel(code, first, tail, layout):
    language = source_language(code)
    speaker = "姫子" if code == "ja" else "Mira"
    if layout == "heading":
        first = "図書館所蔵特別展示資料目録" if code == "ja" else "Mira Valentina Castiglione"
    rows = (
        TextRegion(speaker, Rect(62, 7, 50, 17)),
        TextRegion(first, Rect(90, 52, 640, 25)),
        TextRegion(tail, Rect(89, 89, 110, 19)),
    )
    pixels = np.full((190, 800, 4), (200, 200, 200, 255), np.uint8)
    if layout != "no_edges":
        pixels[36 : 123 if layout != "large_panel" else 180, 65:770, :3] = 234
    if layout == "buttons":
        pixels[79:85, 65:770, :3] = 200
    for row in rows:
        b = row.bounds
        pixels[b.y : b.y + b.height, b.x : b.x + b.width : 3, :3] = 24
    viewport = Rect(0, 0, 800, 190)
    source = CapturedFrame(1, 1, WindowGeometry(viewport, viewport, viewport, 96), pixels.tobytes())
    replay = Replay(language=language)
    scene = replay.observe(source, rows[::-1])
    expected = (
        [speaker, first + language.word_separator + tail]
        if layout == "bubble"
        else [speaker, first, tail]
    )
    assert [block.text for block in scene.requests] == expected
    assert [block.text for block in replay.live.scene(source, 2, grouping=False).requests] == [
        speaker,
        first,
        tail,
    ]
    later = replace(source, sequence=2, captured_at=2)
    scene = replay.observe(later, (rows[-1],), scanned=(Rect(65, 77, 160, 45),))
    assert [block.text for block in scene.requests] == expected


def test_prose_closing_with_an_ocr_comma_leaves_short_button_out_of_request():
    rows = (
        TextRegion("この道は長いけれど、一緒に歩けば", Rect(30, 20, 610, 21)),
        TextRegion("すぐに着く。私もそう思っている、", Rect(30, 48, 485, 20)),
        TextRegion("ここで何を学べる？", Rect(30, 76, 182, 20)),
    )
    source = panel(rows, accent=False)
    replay = Replay()
    scene = replay.observe(source, rows)
    assert [b.text for b in scene.requests] == [rows[0].text + rows[1].text, rows[2].text]
    assert [b.text for b in replay.live.scene(source, 2, grouping=False).blocks] == [
        r.text for r in rows
    ]


@pytest.mark.parametrize(
    ("tail", "width", "joins"),
    [
        ("所についたが、まさか彼が……", 167, True),
        ("設定を開く……", 66, False),
        ("詳細を見る…", 55, False),
    ],
    ids=["todd-paragraph", "open-settings-button", "view-details-button"],
)
def test_short_ellipsis_row_needs_a_clause_break_to_join_prose(tail: str, width: int, joins: bool):
    # Full-frame OCR dropped 場 at the wrap in 待ち合わせ場所.
    first = (
        "トッドはあなたにメッセージを送り、喜びをかち合おうとあなたを招待する。"
        "あなたは約通り待ち合わせ"
    )
    rows = (
        TextRegion(first, Rect(32, 20, 587, 12)),
        TextRegion(tail, Rect(30, 40, width, 11)),
    )
    scene = Replay().observe(panel(rows, accent=False), rows)
    assert [block.text for block in scene.requests] == ([first + tail] if joins else [first, tail])


def test_two_effect_sentences_use_exact_crop_ink_and_survive_unrelated_scans():
    rows = (
        TextRegion("前方に踏み込み、追加の効果を与える。", Rect(30, 20, 610, 21)),
        TextRegion("スキル発動中、無敵効果を得る。", Rect(30, 54, 248, 20)),
    )
    colored = panel(rows, accent=True)
    replay = Replay()
    assert [b.text for b in replay.observe(colored, rows).requests] == [r.text for r in rows]
    later = replace(colored, sequence=2, captured_at=2)
    assert [
        b.text for b in replay.observe(later, (), scanned=(Rect(0, 150, 80, 20),)).requests
    ] == [r.text for r in rows]
    neutral = replace(panel(rows, accent=False), sequence=3, captured_at=3)
    scene = replay.observe(neutral, rows[:1], scanned=(rows[0].bounds,))
    assert [b.text for b in scene.requests] == [rows[0].text + rows[1].text]
