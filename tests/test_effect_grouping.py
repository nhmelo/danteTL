"""Equipment-panel OCR replays: item boundaries, numeric wraps and stable publication."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from live_fixtures import Replay

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.observation import CandidateReview, TextTrack
from dantetl.ocr.tracking import TextActivity
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextRegion
from dantetl.ocr.visual_values import VisualState
from dantetl.translation.grouping import DialogueGrouper
from dantetl.translation.values import (
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)

# Actual full-frame ink boxes/readings, including checkmark and plus-sign OCR errors.
ROWS = (
    ("自転が止まったサルソット", (2980, 695, 578, 49)),
    ("(・2セット:装備キャラの会心率:", (2986, 770, 749, 50)),
    ("8%。装備キャラの会心率が", (3063, 835, 591, 49)),
    ("50%以上の場合、追加攻撃ダ", (3063, 899, 624, 50)),
    ("メージと必殺技によるダメージ", (3063, 963, 671, 49)),
    ("1150%。", (3056, 1038, 147, 37)),
    ("荒海を越える船長", (2978, 1152, 382, 45)),
    ("32セット:会心ダメージャ16%。", (2997, 1226, 708, 49)),
    ("34セット:装備キャラが他の味", (3016, 1295, 697, 50)),
    ("方のスキルターゲットになった", (3059, 1362, 669, 48)),
    ("時、・「助力」を1層獲得し、最", (3061, 1425, 650, 51)),
    ("大で2層累積できる。必殺技を", (3058, 1492, 651, 48)),
    ("発動する時、「助力」を2層所", (3060, 1555, 655, 50)),
    ("持している場合、すべての「助", (3058, 1620, 670, 50)),
    ("カ」を消費し、装備キャラの攻", (3054, 1687, 677, 47)),
    ("撃力ャ48%、1ターン継続。", (3061, 1751, 549, 50)),
)
GROUPS = [(1,), (2, 3, 4, 5, 6), (7,), (8,), tuple(range(9, 17))]
# A panel crop includes different pieces of the checkmark; a tight crop excludes it.
VARIANTS = (
    {},
    {
        1: ("自転が止まったサルソット、", (2982, 696, 583, 50)),
        2: ("ろ2セット:装備キャラの会心率ャ", (3015, 773, 725, 48)),
        6: ("t15%。", (3059, 1038, 142, 39)),
        8: ("2セット:会心ダメージ:16%。", (3058, 1227, 649, 47)),
        9: ("4セット:装備キャラが他の味", (3061, 1297, 650, 48)),
    },
    {
        2: ("2セット:装備キャラの会心率ャ", (3058, 774, 680, 46)),
        6: ("t15%。", (3060, 1039, 142, 37)),
        9: ("4セット:装備キャラが他の味", (3059, 1299, 654, 45)),
    },
)


def tracks_for(variant=0, scale=1):
    return tuple(
        TextTrack(i, text, Rect(*(round(v * scale) for v in bounds)))
        for i, (text, bounds) in enumerate(
            (VARIANTS[variant].get(i, row) for i, row in enumerate(ROWS, 1)), 1
        )
    )


@pytest.mark.parametrize("variant", range(3))
@pytest.mark.parametrize("scale", [0.5, 0.75, 1, 1.5])
def test_headings_and_effects_stay_separate_while_each_effect_stays_whole(variant, scale):
    tracks = tracks_for(variant, scale)
    grouper = DialogueGrouper()
    grouper.update(tracks[::-1], tracks, 0)
    grouped = grouper.update(tracks[::2] + tracks[1::2], tracks, 0.6)
    assert [tuple(t.track_id for t in b.lines) for b in grouped.blocks] == GROUPS
    assert grouped.ready == grouped.blocks
    effect = TranslationRequest.from_block(grouped.ready[1])
    assert effect.source.endswith(tracks[5].text)
    assert len(effect.line_sources) == 5


@pytest.mark.parametrize("prefix", ["◆条件：", "1. 条件：", "- 条件：", "条件：", "Lv.2: "])
def test_marked_item_can_wrap_but_next_item_and_heading_remain_independent(prefix):
    tracks = (
        TextTrack(1, "この装備で利用できる追加効果", Rect(100, 100, 650, 40)),
        TextTrack(2, prefix + "装備キャラの攻撃力を上げて、", Rect(100, 156, 650, 40)),
        TextTrack(3, "次のターンまで効果が持続する。", Rect(160, 212, 590, 40)),
        TextTrack(4, prefix + "仲間の回復効果を高める。", Rect(100, 268, 650, 40)),
    )
    result = DialogueGrouper().update(tracks, tracks, 0)
    assert [b.lines for b in result.blocks] == [(tracks[0],), tracks[1:3], (tracks[3],)]


@pytest.mark.parametrize("tail", ["+15%。", "−15%", "１５％。", "t15%。", "1150%。"])
def test_numeric_tail_requires_an_established_unfinished_paragraph(tail):
    tracks = tracks_for(2)[1:6]
    tracks = (*tracks[:-1], replace(tracks[-1], text=tail))
    assert len(DialogueGrouper().update(tracks, tracks, 0).blocks) == 1
    # Standalone stats, large gaps, other columns and a completed description are not tails.
    for before, after in (
        (tracks[-2], tracks[-1]),
        (tracks[-2], replace(tracks[-1], bounds=Rect(3060, 1150, 142, 37))),
        (tracks[-2], replace(tracks[-1], bounds=Rect(3500, 1039, 142, 37))),
        (replace(tracks[-2], text=tracks[-2].text + "。"), tracks[-1]),
    ):
        rows = (
            (before, after)
            if before == tracks[-2] and after == tracks[-1]
            else (*tracks[:-2], before, after)
        )
        result = DialogueGrouper().update(rows, rows, 0)
        assert result.blocks[-1].lines == (after,)


def source_frame(sequence):
    bounds = Rect(0, 0, 3840, 2160)
    return CapturedFrame(sequence, sequence * 0.5, WindowGeometry(bounds, bounds, bounds, 96), b"")


def regions_for(tracks):
    return tuple(TextRegion(t.text, t.bounds) for t in tracks)


def test_first_cache_preview_uses_the_complete_equipment_description():
    replay = Replay()
    rows = tracks_for(2)[1:6]
    scene = replay.observe(
        source_frame(1),
        tuple(replace(r, line_confidence=0.98) for r in regions_for(rows)),
        settle=0,
    )
    (effect,) = scene.candidates
    assert len(effect.lines) == 5 and not scene.requests
    partial = TranslatedText(TranslationRequest.from_track(effect.lines[0]), "Partial effect")
    complete = TranslatedText(TranslationRequest.from_block(effect), "Complete effect")
    backend = Mock()
    for hits, expected in [((partial,), ()), ((partial, complete), (complete,))]:
        backend.snapshot.return_value = TranslationSnapshot(
            TranslationState.READY, cached_candidates=hits
        )
        english = replay.live.publish(
            backend, scene.requests, scene.blocks, candidates=scene.candidates
        )
        assert english.translations == expected


@pytest.mark.parametrize(
    "tail,box,joins",
    [
        ("*15%。", Rect(3058, 1038, 143, 38), True),  # Full-frame reading on another background.
        ("t15%。", Rect(3058, 1042, 143, 32), True),  # Short ink at the expected row center.
        ("*15%。", Rect(3058, 1015, 143, 38), False),  # Between rows.
        ("*15%。", Rect(3500, 1038, 143, 38), False),  # Neighboring column.
        ("HP15%", Rect(3058, 1038, 143, 38), False),
        ("・15%。", Rect(3058, 1038, 143, 38), False),
        ("¥15000", Rect(3058, 1038, 143, 38), False),
    ],
)
def test_hanging_opening_survives_uncertain_tail_and_only_fitting_value_joins(tail, box, joins):
    rows = (
        ("()2セット:装備キャラの会心率:", Rect(2984, 769, 752, 52)),
        ("8%。装備キャラの会心率が", Rect(3063, 835, 593, 50)),
        ("50%以上の場合、追加攻撃ダ", Rect(3064, 899, 623, 50)),
        ("メージと必殺技によるダメージュ", Rect(3063, 964, 685, 49)),
        (tail, box),
    )
    replay = Replay()
    regions = tuple(TextRegion(text, bounds) for text, bounds in rows)
    for seq in (1, 2):
        scene = replay.observe(source_frame(seq), regions[::-1])
    assert [len(b.lines) for b in scene.blocks] == ([5] if joins else [4, 1])
    expected = rows if joins else rows[:-1]
    assert scene.requests[0].text == "".join(text for text, _ in expected)


@pytest.mark.parametrize(
    "second,tail",
    [
        ("殺技によるダメージャ20%。必", "がさらに:25%。"),  # Full frame and panel crop.
        ("殺技によるダメージ:20%。必", "がさらにゃ25%。"),  # Tight crop of the same effect.
        ("殺技によるダメージ+20%。必", "がさらに＋２５％。"),
    ],
)
def test_wrapped_percentage_request_preserves_complete_effect_and_raw_ocr(second, tail):
    rows = (
        ("()2セット:会心率:8%。", Rect(2983, 1227, 556, 49)),
        ("4セット:戦闘スキルおよび必", Rect(3053, 1297, 660, 47)),
        (second, Rect(3060, 1361, 651, 49)),
        ("殺技を発動した後、次に戦闘ス", Rect(3061, 1426, 668, 49)),
        ("キルを発動する時、与ダメージ", Rect(3058, 1489, 673, 49)),
        (tail, Rect(3054, 1558, 340, 45)),
    )
    replay = Replay()
    regions = tuple(TextRegion(text, box) for text, box in rows)
    for seq in (1, 2):
        scene = replay.observe(source_frame(seq), regions[::-1])
    assert [b.text for b in scene.requests] == [rows[0][0], "".join(text for text, _ in rows[1:])]
    request = TranslationRequest.from_block(scene.requests[1])
    assert [source for _, _, source in request.line_sources] == [text for text, _ in rows[1:]]


@pytest.mark.parametrize(
    "body,tail",
    [
        ("装備すると与ダメージがさらに", "+15%"),
        ("装備すると与ダメージがさらに", "：１５．５％。"),
        ("装備すると与ダメージがさらに", "−15％"),
        ("次に与える攻撃のダメージ", "がさらに:25%"),
        ("次に与える攻撃のダメージ", "がさらに＋25％"),
    ],
)
def test_percentage_can_finish_a_two_line_sentence_with_an_explicit_value_cue(body, tail):
    lead = TextTrack(1, body, Rect(100, 100, 640, 40))
    end = TextTrack(2, tail, Rect(102, 155, 136, 36))
    tracks = (lead, end)
    owner = DialogueGrouper()
    assert not owner.update(tracks, (lead,), 0).ready
    assert [b.lines for b in owner.update(tracks, tracks, 1).ready] == [tracks]


@pytest.mark.parametrize(
    "label",
    [
        "会心率：25%。",
        "攻撃によるダメージ:25%。",
        "別セット：25%。",
        "◆がさらに:25%。",
        "がさらに:25回。",
        "がさらに:25%の条件",
        "がさらに:25%増加の条件",
    ],
)
def test_nearby_stat_or_choice_is_still_separate_from_unfinished_prose(label):
    body = tracks_for(2)[1:5]
    stat = replace(tracks_for(2)[5], text=label, bounds=Rect(3060, 1035, 340, 45))
    tracks = (*body, stat)
    assert DialogueGrouper().update(tracks, tracks, 1).blocks[-1].lines == (stat,)


def translated_backend(blocks):
    backend = Mock()
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY,
        tuple(
            TranslatedText(TranslationRequest.from_block(b), f"Effect {i}")
            for i, b in enumerate(blocks)
        ),
    )
    return backend


def test_pending_ordinary_reading_cannot_repartition_committed_text():
    tracks = tracks_for(2)
    grouper = DialogueGrouper()
    activity = tuple(TextActivity(t, 0, False, False) for t in tracks)
    initial = grouper.update(tracks, tracks, 0.6, activity=activity)
    changed = replace(tracks[2], text="条件：会心率が50%の場合")
    pending = (*activity[:2], TextActivity(changed, 1, False, True), *activity[3:])
    waiting = grouper.update(tracks, tracks, 2, activity=pending)
    assert waiting.blocks == initial.blocks
    assert initial.blocks[1] not in waiting.ready
    committed = (*tracks[:2], replace(changed, text_revision=2), *tracks[3:])
    settled = tuple(TextActivity(t, 2, False, False) for t in committed)
    result = grouper.update(committed, committed, 2.6, activity=settled)
    assert result.blocks != initial.blocks  # Genuine accepted structure changes still apply.


def test_committed_full_and_regional_readings_keep_the_same_item_membership():
    replay = Replay()
    for sequence in range(1, 13):
        tracks = tracks_for((sequence // 2) % 3)
        scene = replay.observe(source_frame(sequence), regions_for(tracks)[::-1])
        assert [len(b.lines) for b in scene.blocks] == [1, 5, 1, 1, 8]
        assert all(b in scene.blocks for b in scene.requests)


def test_uncommitted_ordinary_scan_does_not_revoke_published_paragraph():
    replay = Replay()
    tracks = tracks_for(2)
    for sequence in (1, 2):
        scene = replay.observe(source_frame(sequence), regions_for(tracks))
    backend = translated_backend(scene.blocks)
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    source = source_frame(3)
    changed = TextRegion("条件：会心率が50%の場合", tracks[2].bounds)
    replay.ocr.current = OcrSnapshot(
        OcrState.READY,
        OcrResult(source, (changed,), 25, source.captured_at, (changed.bounds,)),
    )
    replay.visual.current = VisualState(
        review=CandidateReview(sequence=3), frame=source, ocr_sequence=3
    )
    replay.live.reconcile(source, source.captured_at)
    # Capture keeps advancing, so one noisy observation cannot settle as a static frame.
    pending = replay.live.scene(source_frame(4), source.captured_at + 0.1, grouping=True)
    assert pending.blocks == scene.blocks
    assert scene.blocks[1] not in pending.requests
    assert replay.live.publish(backend, pending.requests, pending.blocks) == english
    restored = replay.observe(source_frame(5), regions_for(tracks))
    assert restored.blocks == scene.blocks
    assert replay.live.publish(backend, restored.requests, restored.blocks) == english


@pytest.mark.parametrize("variant", range(3))
def test_partial_and_noisy_observations_keep_groups_and_english_until_a_real_edit(variant):
    replay = Replay()
    tracks = tracks_for(variant)
    for sequence in (1, 2):
        scene = replay.observe(source_frame(sequence), regions_for(tracks))
    assert [len(b.lines) for b in scene.blocks] == [1, 5, 1, 1, 8]
    backend = translated_backend(scene.blocks)
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    assert len(english.translations) == 5
    initial = scene.blocks
    for sequence in range(3, 12):
        alternate = regions_for(tracks_for(sequence % 3))
        scanned = (Rect(2900, 755, 900, 370),) if sequence % 2 else None
        raw = tuple(r for r in alternate if scanned is None or scanned[0].contains(r.bounds))
        scene = replay.observe(
            source_frame(sequence),
            raw[::-1],
            scanned=scanned,
            unchanged=frozenset(scene.tracks),
            checked=frozenset(scene.tracks),
        )
        assert scene.blocks == initial
        assert replay.live.publish(backend, scene.requests, scene.blocks) == english
        assert replay.live.accepted is not None and replay.live.accepted.regions == raw[::-1]
    # A new value is a genuine source change, even in a numeric-only final row.
    changed = TextRegion("+25%。", tracks[5].bounds)
    for sequence in (12, 13, 14):
        scene = replay.observe(
            source_frame(sequence), (changed,), scanned=(Rect(3000, 1020, 300, 100),)
        )
    assert [len(b.lines) for b in scene.blocks] == [1, 5, 1, 1, 8]
    assert scene.blocks[1].text.endswith("+25%。")
    assert scene.blocks[1] in scene.requests
    after = replay.live.publish(backend, scene.requests, scene.blocks)
    assert after.translations == english.translations
