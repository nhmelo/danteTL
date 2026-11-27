"""Profile-panel OCR geometry: keep paragraphs whole and quoted wraps together."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from live_fixtures import Replay

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.selection import source_language
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.tracking import TextActivity
from dantetl.ocr.values import TextRegion
from dantetl.translation.grouping import DialogueGrouper
from dantetl.translation.values import (
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)


@pytest.mark.parametrize("scale", [0.5, 1, 1.5])
@pytest.mark.parametrize("change", ["full", "crop", "off-pitch", "smaller-font", "short-label"])
def test_wrapped_column_compares_ink_heights_to_the_preceding_body_row(scale, change):
    texts = (
        "Each point of resonance",
        "additionally increases the damage",
        "multiplier for the current",
        "instance of the skill by 7%.",
    )
    boxes = [
        (2977, 1204, 746, 49),
        (2980, 1266, 740, 57),
        (2983, 1341, 598, 41),
        (2978, 1403, 556, 49),
    ]
    if change == "crop":
        boxes = [
            (2983, 1210, 738, 42),
            (2981, 1272, 738, 49),
            (2981, 1340, 602, 48),
            (2978, 1403, 556, 49),
        ]
    elif change == "off-pitch":
        boxes[2] = (2983, 1359, 598, 41)
    elif change == "smaller-font":
        boxes[2] = (2983, 1341, 598, 30)
    elif change == "short-label":
        boxes[2] = (2983, 1341, 140, 41)
    rows = tuple(
        TextTrack(i, text, Rect(*(round(v * scale) for v in box)))
        for i, (text, box) in enumerate(zip(texts, boxes, strict=True))
    )
    grouper = DialogueGrouper(language=source_language("en"))
    initial = grouper.update(rows, rows[:-1], 0)
    settled = grouper.update(rows[::-1], rows, 1)
    if change in ("full", "crop"):
        assert not initial.ready
        (block,) = settled.ready
        assert block.lines == rows and block.text == " ".join(texts)
    else:
        assert all(rows[1] not in b.lines or rows[2] not in b.lines for b in settled.blocks)


# Character-bounded rows from the supplied screenshots; policy contains no game names.
ARCHER = (
    ("過去から未来へと絶望は繰り返さ", (2962, 824, 696, 46)),
    ("れ、赤き外套に包まれた無数の理想", (2959, 888, 747, 45)),
    ("は燃え尽きた。だが、残された灰燼", (2960, 950, 750, 46)),
    ("に惑わされてはならない――", (2961, 1013, 609, 45)),
    ("理想を騙り、偽りの美しき夢を紡ぐ", (2957, 1075, 751, 47)),
    ("者に抗い、彼は再び立ち上がる。こ", (2960, 1137, 745, 47)),
    ("の世にはびこる偽善と最後まで戦い", (2952, 1199, 756, 46)),
    ("抜くだろう。", (2955, 1260, 257, 48)),
    ("どこに身を置こうとも、彼が正義の", (2961, 1325, 745, 45)),
    ("味方であることに変わりはないのだ", (2956, 1385, 749, 47)),
    ("から。", (2956, 1452, 117, 40)),
)
SABER = (
    ("運命の長夜を歩んだ孤高の英霊。そ", (2960, 823, 748, 48)),
    ("の夢の中では、円卓の旗が今なお風", (2961, 888, 743, 45)),
    ("に揺れている。キャメロットの騎士", (2962, 950, 744, 47)),
    ("王はこの時、まだ理想郷に辿り着い", (2961, 1012, 747, 46)),
    ("ていない。", (2959, 1080, 206, 39)),
    ("召喚に応じ、少女は異質な「聖杯戦", (2961, 1137, 747, 47)),
    ("争トに参戦する。再び選定の剣を目", (2958, 1198, 744, 48)),
    ("の前にした時、彼女は過去の幻境か", (2960, 1260, 746, 48)),
    ("ら抜け出せるのだろうか?", (2961, 1323, 549, 47)),
    ("―「果たせなかった願いも…今こ", (3002, 1386, 708, 47)),
    ("こで断ち切ります」", (2964, 1449, 392, 46)),
)


def panel(rows, scale=1, jitter=0):
    tracks = (
        TextTrack(0, "キャラクター紹介。", Rect(2954, 750, 395, 46)),
        *(TextTrack(i, text, Rect(*bounds)) for i, (text, bounds) in enumerate(rows, 1)),
    )
    return tuple(
        replace(
            t,
            bounds=Rect(
                round((t.bounds.x + (jitter if t.track_id % 2 else -jitter)) * scale),
                round(t.bounds.y * scale),
                round(t.bounds.width * scale),
                round(t.bounds.height * scale),
            ),
        )
        for t in tracks
    )


@pytest.mark.parametrize("scale", [0.5, 0.75, 1, 1.5])
@pytest.mark.parametrize("jitter", [-2, 0, 2])
@pytest.mark.parametrize(
    "rows,groups",
    [
        (ARCHER, [(0,), (1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 11)]),
        (SABER, [(0,), (1, 2, 3, 4, 5), (6, 7, 8, 9), (10, 11)]),
    ],
)
def test_profile_paragraphs_keep_continuations_and_quote_across_ocr_jitter(
    rows, groups, scale, jitter
):
    tracks = panel(rows, scale, jitter)
    grouper = DialogueGrouper()
    grouper.update(tracks[::-1], tracks, 0)
    settled = grouper.update(tracks[::2] + tracks[1::2], tracks, 0.4)
    assert [tuple(t.track_id for t in b.lines) for b in settled.blocks] == groups
    assert settled.ready == settled.blocks
    assert sorted(t.track_id for b in settled.blocks for t in b.lines) == list(range(12))


def test_incomplete_quoted_wrap_waits_for_exact_members_without_holding_other_paragraphs():
    tracks = panel(SABER)
    grouper = DialogueGrouper()
    grouper.update(tracks, tracks[:-1], 0)
    pending = grouper.update(tracks, tracks[:-1], 1)
    assert [len(b.lines) for b in pending.ready] == [1, 5, 4]
    quote = grouper.update(tracks, tracks, 1.1).ready[-1]
    assert len(quote.lines) == 2
    original = TranslationRequest.from_block(quote)
    edited = (*tracks[:-1], replace(tracks[-1], text="ここで終わりにします」", text_revision=2))
    assert len(grouper.update(edited, tracks, 2).ready) == 3
    following = grouper.update(edited, edited, 2.5).ready[-1]
    assert TranslationRequest.from_block(following) != original


@pytest.mark.parametrize("prefix", ["「", "―「", "――「", "--「", "『"])
def test_quote_indent_is_limited_to_a_wrapped_row_and_does_not_join_the_previous_paragraph(prefix):
    tracks = (
        TextTrack(1, "新しい冒険について考えている。", Rect(100, 100, 800, 40)),
        TextTrack(2, prefix + "私はこの長い道をこれからも", Rect(140, 156, 760, 40)),
        TextTrack(3, "進んでいきます」", Rect(100, 212, 400, 40)),
    )
    result = DialogueGrouper().update(tracks, tracks, 0)
    assert [len(b.lines) for b in result.blocks] == [1, 2]


@pytest.mark.parametrize("prefix", ["「", "―「", "――「", "--「", "『"])
def test_prefixed_quote_keeps_dialogue_readiness_while_short_menu_is_immediate(prefix):
    tracks = (
        TextTrack(1, prefix + "私はこの長い道をこれからも歩いて行きます", Rect(100, 100, 760, 40)),
        TextTrack(2, "- 設定", Rect(100, 300, 200, 40)),
    )
    activity = tuple(TextActivity(t, changed_at=0, growing=False, pending=False) for t in tracks)
    grouper = DialogueGrouper()
    waiting = grouper.update(tracks, tracks, 0, activity=activity)
    assert waiting.blocks[0].dialogue
    assert [b.lines[0].track_id for b in waiting.ready] == [2]
    assert grouper.update(tracks, tracks, 0.6, activity=activity).ready == waiting.blocks


@pytest.mark.parametrize("change", ["alignment", "gap", "font", "choices", "ambiguous"])
def test_quote_leniency_still_respects_other_panel_boundaries(change):
    first = TextTrack(1, "――「私はこの長い道をこれからも", Rect(140, 100, 760, 40))
    second = TextTrack(2, "進んでいきます」", Rect(100, 156, 400, 40))
    tracks = (first, second)
    if change == "alignment":
        tracks = (first, replace(second, bounds=replace(second.bounds, x=20)))
    elif change == "gap":
        tracks = (first, replace(second, bounds=replace(second.bounds, y=220)))
    elif change == "font":
        tracks = (first, replace(second, bounds=replace(second.bounds, height=20)))
    elif change == "choices":
        tracks = (
            replace(first, text="1. " + first.text),
            replace(second, text="2. " + second.text),
        )
    else:
        tracks = (*tracks, replace(second, track_id=3, bounds=replace(second.bounds, x=105)))
    assert len(DialogueGrouper().update(tracks, tracks, 0).blocks) == len(tracks)


@pytest.mark.parametrize("ending", ["。", "――"])
def test_full_width_sentence_end_is_not_a_paragraph_boundary(ending):
    tracks = (
        TextTrack(1, "この長い道の先には、新しい景色が", Rect(100, 100, 800, 40)),
        TextTrack(2, "どこまでも広がっているでしょう" + ending, Rect(100, 156, 780, 40)),
        TextTrack(3, "明日また一緒に歩いて行きましょう。", Rect(100, 212, 800, 40)),
    )
    assert len(DialogueGrouper().update(tracks, tracks, 0).blocks) == 1


@pytest.mark.parametrize("count", [7, 8, 12, 24])
def test_clear_paragraphs_are_not_split_by_line_count(count):
    tracks = tuple(
        TextTrack(
            i,
            "ここから先には、新しい冒険が続き" + ("ます。" if i == count else "ます"),
            Rect(100, i * 54, 800, 40),
        )
        for i in range(1, count + 1)
    )
    grouper = DialogueGrouper()
    pending = grouper.update(tracks, tracks[:-1], 0)
    assert len(pending.blocks) == 1 and not pending.ready
    ready = grouper.update(tracks[::-1], tracks, 0.6)
    assert ready.ready == ready.blocks
    request = TranslationRequest.from_block(ready.ready[0])
    assert len(request.line_sources) == count
    assert request.source == "".join(t.text for t in tracks)


def test_split_detection_in_a_long_paragraph_preserves_all_members():
    tracks = tuple(
        TextTrack(i, "ここから先には、新しい冒険が続きます", Rect(100, i * 54, 800, 40))
        for i in range(1, 8)
    )
    tracks += (
        TextTrack(8, "私たちはいつまでも", Rect(100, 432, 350, 40)),
        TextTrack(9, "一緒に歩いて行こう。", Rect(470, 432, 430, 40)),
    )
    result = DialogueGrouper().update(tracks, tracks, 0)
    assert len(result.blocks) == 1
    assert result.blocks[0].lines == tracks


@pytest.mark.parametrize("rows,count", [(ARCHER, 8), (SABER, 5)])
@pytest.mark.parametrize("split_index", [0, 1])
@pytest.mark.parametrize("scale", [0.5, 0.75, 1, 1.5])
def test_split_opening_rows_recover_complete_source_and_exact_members(
    rows, count, split_index, scale
):
    intact = panel(rows, scale)[1 : count + 1]
    t = intact[split_index]
    cut = len(t.text) // 2
    width = round(t.bounds.width * cut / len(t.text))
    gap = round(12 * scale)
    left = replace(t, text=t.text[:cut], bounds=replace(t.bounds, width=width))
    right = replace(
        t,
        track_id=100,
        text=t.text[cut:],
        bounds=replace(
            t.bounds,
            x=t.bounds.x + width + gap,
            y=t.bounds.y - 1,
            width=t.bounds.width - width - gap,
        ),
    )
    lines = (*intact[:split_index], left, right, *intact[split_index + 1 :])
    owner = DialogueGrouper()
    activity = tuple(TextActivity(line, 0, False, False) for line in lines)
    pending = owner.update(
        lines[::-1], tuple(line for line in lines if line != right), 1, activity=activity
    )
    assert [b.lines for b in pending.blocks] == [lines] and not pending.ready
    ready = owner.update(lines, lines, 1, activity=activity)
    request = TranslationRequest.from_block(ready.ready[0])
    assert request.source == "".join(line.text for line in intact)
    assert request.line_sources == tuple(
        (line.track_id, line.text_revision, line.text) for line in lines
    )
    assert len(owner.update(lines, lines, 2, enabled=False).blocks) == len(lines)


@pytest.mark.parametrize(
    "layout", ["buttons", "buttons_above_body", "close_columns", "marked_choices"]
)
def test_opening_row_reconstruction_keeps_controls_and_columns_separate(layout):
    # The body below the controls occupies exactly their combined width.
    def line(key, text, x, y, width):
        return TextTrack(key, text, Rect(x, y, width, 40))

    a = line(1, "冒険を保存する", 100, 100, 240)
    b = line(2, "冒険を再開する", 346, 100, 240)
    body = (
        line(3, "ここで保存した冒険について、", 100, 156, 486),
        line(4, "下の一覧から選んでください。", 100, 212, 486),
    )
    if layout == "marked_choices":
        a = replace(a, text="◆冒険を保存する")
        b = replace(b, text="◆冒険を再開する", bounds=replace(b.bounds, x=345))
    if layout == "buttons":
        more = (
            replace(a, track_id=3, bounds=replace(a.bounds, y=156)),
            replace(b, track_id=4, bounds=replace(b.bounds, y=156)),
        )
        expected = [(a,), (b,), (more[0],), (more[1],)]
    elif layout == "close_columns":
        # Both paragraphs have their own continuation, with a narrow repeated gutter.
        a = replace(a, text="これは左側の説明文であり、")
        b = replace(b, text="これは右側の説明文であり、", bounds=replace(b.bounds, x=345))
        more = (
            line(3, "この列で終わります。", 100, 156, 220),
            line(4, "ここで説明を終えます。", 345, 156, 220),
        )
        expected = [(a, more[0]), (b, more[1])]
    else:
        more = body
        expected = [(a,), (b,), body]
    tracks = (a, b, *more)
    result = DialogueGrouper().update(tracks[::-1], tracks, 1)
    assert [block.lines for block in result.blocks] == expected


def test_exact_prior_intact_row_supports_ambiguous_split_until_source_or_scene_changes():
    original = TextTrack(1, "ここで冒険を再開します", Rect(100, 100, 400, 40))
    left = replace(original, text="ここで冒険を", bounds=Rect(100, 100, 196, 40), text_revision=2)
    right = TextTrack(2, "再開します", Rect(322, 100, 178, 40))
    tracks = (left, right)
    owner = DialogueGrouper()
    owner.update((original,), (), 0)
    assert len(owner.update(tracks, tracks, 1).blocks) == 2  # Unconfirmed history cannot join.
    owner.update((original,), (original,), 2)
    assert [b.lines for b in owner.update(tracks, tracks, 3).blocks] == [tracks]
    assert [b.lines for b in owner.update(tracks[::-1], tracks, 4).blocks] == [tracks]
    edited = (left, replace(right, text="保存します", text_revision=2))
    assert len(owner.update(edited, edited, 5).blocks) == 2
    owner.update((original,), (original,), 6)
    owner.reset()
    assert len(owner.update(tracks, tracks, 7).blocks) == 2


def test_live_split_opening_beside_second_column_publishes_exact_current_members():
    left = (
        TextRegion("この長い旅の", Rect(100, 100, 300, 40)),
        TextRegion("続きを読みましょう、", Rect(408, 99, 392, 40)),
        TextRegion("新しい景色を一緒に見ながら、", Rect(100, 156, 700, 40)),
        TextRegion("もう少し歩いて行きましょう。", Rect(100, 212, 700, 40)),
    )
    right = (
        TextRegion("こちらに表示される説明は、", Rect(808, 100, 600, 40)),
        TextRegion("別の物語についてのものです。", Rect(808, 156, 580, 40)),
    )
    replay = Replay()
    for seq in (1, 2):
        scene = replay.observe(source_frame(seq), (*right, *left)[::-1])
    expected = ["".join(r.text for r in column) for column in (left, right)]
    assert [block.text for block in scene.requests] == expected
    requests = tuple(TranslationRequest.from_block(b) for b in scene.requests)
    assert [[text for _, _, text in req.line_sources] for req in requests] == [
        [r.text for r in column] for column in (left, right)
    ]
    backend = Mock()
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY,
        tuple(TranslatedText(req, f"Column {i}") for i, req in enumerate(requests)),
    )
    assert len(replay.live.publish(backend, scene.requests, scene.blocks).translations) == 2
    changed = (left[0], replace(left[1], text="別の話を始めましょう、"), *left[2:])
    for seq in (3, 4):
        scene = replay.observe(source_frame(seq), (*changed, *right))
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    assert {t.request for t in english.translations} == set(requests)
    revised = TranslationRequest.from_block(scene.requests[0])
    assert revised != requests[0]
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY, (TranslatedText(revised, "Revised left column"),)
    )
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    assert {t.request for t in english.translations} == {revised, requests[1]}


@pytest.mark.parametrize("ending,separate", [("――", False), ("。", True)])
def test_ragged_dash_is_continuation_but_ragged_sentence_end_starts_a_new_paragraph(
    ending, separate
):
    tracks = (
        TextTrack(1, "この長い道の先には、新しい景色が", Rect(100, 100, 800, 40)),
        TextTrack(2, "どこまでも広がっているでしょう" + ending, Rect(100, 156, 640, 40)),
        TextTrack(3, "明日また一緒に歩いて行きましょう。", Rect(100, 212, 800, 40)),
    )
    result = DialogueGrouper().update(tracks, tracks, 0)
    assert [len(b.lines) for b in result.blocks] == ([2, 1] if separate else [3])


# First two are actual crop readings; the third exercises complete bracket loss.
REGIONAL_QUOTES = (
    (
        ("――/果たせなかった願いも…今こ", (2962, 1388, 747, 46)),
        ("こで断ち切ります/", (2967, 1448, 395, 48)),
    ),
    (
        ("「――/果たせなかった願いも…今こ", (2928, 1389, 784, 45)),
        ("こで断ち切ります!", (2966, 1448, 398, 48)),
    ),
    (
        ("――果たせなかった願いも…今こ", (2962, 1388, 747, 46)),
        ("こで断ち切ります", (2967, 1448, 395, 48)),
    ),
)


def source_frame(sequence):
    bounds = Rect(0, 0, 3840, 2160)
    return CapturedFrame(sequence, sequence * 0.5, WindowGeometry(bounds, bounds, bounds, 96), b"")


@pytest.mark.parametrize("quote", REGIONAL_QUOTES)
@pytest.mark.parametrize("first_observation", [False, True])
def test_regional_quote_through_live_tracking_groups_and_retains_english(quote, first_observation):
    replay = Replay()
    initial = SABER[:-2] + quote if first_observation else SABER
    for sequence in (1, 2):
        replay.observe(
            source_frame(sequence), tuple(TextRegion(text, Rect(*b)) for text, b in initial)
        )
    for sequence in (3, 4, 5):
        scene = replay.observe(
            source_frame(sequence),
            tuple(TextRegion(text, Rect(*b)) for text, b in quote),
            scanned=(Rect(2900, 1370, 850, 150),),
        )
    assert [len(b.lines) for b in scene.blocks] == [5, 4, 2]
    assert scene.requests == scene.blocks
    backend = Mock()
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY,
        tuple(
            TranslatedText(TranslationRequest.from_block(b), f"Paragraph {i}")
            for i, b in enumerate(scene.blocks)
        ),
    )
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    assert len(english.translations) == 3
    for sequence in (6, 7, 8):
        scene = replay.observe(source_frame(sequence), (), scanned=(Rect(0, 0, 100, 100),))
        assert replay.live.publish(backend, scene.requests, scene.blocks) == english
    # Inspection still exposes the actual partial OCR result, never synthesized rows.
    assert replay.live.accepted is not None and replay.live.accepted.regions == ()
