from dataclasses import replace

import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.tracking import TextTracker, TrackingPolicy
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion


def region(
    text: str = "新しい冒険が、ここから始まる。",
    x: int = 100,
    y: int = 700,
    width: int = 400,
    height: int = 40,
    vertical: bool = False,
    confidence: float | None = None,
) -> TextRegion:
    bounds = Rect(x, y, width, height)
    characters = (
        tuple(TextCharacter(char, bounds, confidence) for char in text)
        if confidence is not None
        else ()
    )
    return TextRegion(text, bounds, characters=characters, vertical=vertical)


class Replay:
    def __init__(self, policy: TrackingPolicy | None = None) -> None:
        self.tracker = TextTracker(policy or TrackingPolicy())
        self.sequence = 0
        self.now = 0.0
        self.ids: tuple[int | None, ...] = ()
        self.result: OcrResult | None = None

    def step(self, *regions: TextRegion, dt: float = 0.25) -> tuple[TextTrack, ...]:
        self.sequence += 1
        self.now += dt
        bounds = Rect(0, 0, 1920, 1080)
        frame = CapturedFrame(
            self.sequence, self.now, WindowGeometry(bounds, bounds, bounds, 144), b""
        )
        self.result = OcrResult(frame, regions, 20, self.now + 0.02)
        self.ids = self.tracker.observe(self.result)
        return self.tracker.advance(frame, self.now + 0.02).visible


def test_only_explicit_advancement_settles_a_static_observation():
    replay = Replay()
    first = replay.step(region())
    assert replay.result is not None
    before = replay.tracker.snapshot()
    assert before.visible == first and not before.confirmed
    for _ in range(3):
        assert replay.tracker.snapshot() == before
    after = replay.tracker.advance(replay.result.frame, replay.now + 1)
    assert after.visible == after.confirmed == first
    assert not before.confirmed and after == replay.tracker.snapshot()


def test_pending_text_and_geometry_survive_partial_scan_and_delayed_confirmation():
    replay = Replay()
    original = replay.step(region())[0]
    replay.step(region())
    growth = region(original.text + "一緒に行こう。", width=500)
    assert replay.step(growth) == (original,)
    assert replay.result is not None
    source = replay.result.frame
    # Another region completes OCR while this line's visual confirmation is pending.
    latest = replace(source, sequence=source.sequence + 1, captured_at=source.captured_at + 0.1)
    partial = OcrResult(latest, (), 900, latest.captured_at + 0.9, (Rect(0, 0, 100, 100),))
    replay.tracker.observe(partial)
    for now in (partial.completed_at, partial.completed_at + 10):
        assert replay.tracker.advance(latest, now).confirmed == (original,)
        assert replay.tracker.snapshot().activity[0].pending
    replay.tracker.confirm_observations(((original.track_id, source.sequence, growth.text),), 12)
    committed = replay.tracker.advance(latest, 12).confirmed[0]
    assert committed.text == growth.text and committed.bounds == growth.bounds
    assert committed.track_id == original.track_id
    assert committed.text_revision == original.text_revision + 1
    replay.tracker.reset()
    replay.tracker.confirm_observations(((original.track_id, source.sequence, growth.text),), 13)
    assert replay.tracker.advance(latest, 13).visible == ()


def test_completed_miss_discards_text_and_geometry_candidates_without_revoking_accepted_text():
    replay = Replay()
    original = replay.step(region())[0]
    replay.step(region())
    growth = region(original.text + "一緒に行こう。", width=500)
    assert replay.step(growth) == (original,)
    assert replay.step(dt=0.1) == (original,)
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now + 10).confirmed == (original,)
    assert not replay.tracker.snapshot().activity[0].pending
    # The next positive observation begins fresh evidence, rather than completing
    # either discarded candidate with its pre-miss hit count.
    assert replay.step(growth, dt=0.9) == (original,)
    committed = replay.step(growth, dt=0.9)[0]
    assert committed.text == growth.text and committed.bounds == growth.bounds
    assert committed.track_id == original.track_id
    assert committed.text_revision == original.text_revision + 1


def test_jitter_and_character_score_changes_keep_exact_display_objects() -> None:
    replay = Replay()
    original = replay.step(region())
    for offset in (2, -3, 5, -4, 0, 7, -6) * 5:
        noisy = replace(
            region(x=100 + offset),
            characters=(
                TextCharacter("新", Rect(100 + offset, 700, 20, 40), 0.7 + abs(offset) / 100),
            ),
        )
        current = replay.step(noisy)
        assert current == original
        assert current[0] is original[0]


def test_small_persistent_move_confirms_once_and_slow_drift_cannot_accumulate_unseen() -> None:
    replay = Replay()
    original = replay.step(region())[0]
    assert replay.step(region(x=110))[0] is original  # outside the 8 px deadband
    moved = replay.step(region(x=111))[0]
    assert moved.track_id == original.track_id and moved.bounds.x == 111
    assert moved.text_revision == original.text_revision
    for x in range(112, 131):
        moved = replay.step(region(x=x))[0]
    assert moved.bounds.x >= 120


def test_large_move_follows_immediately_without_changing_identity() -> None:
    replay = Replay()
    original = replay.step(region())[0]
    moved = replay.step(region(x=160, y=720))[0]
    assert moved.track_id == original.track_id
    assert moved.bounds == Rect(160, 720, 400, 40)


def test_one_bad_crop_does_not_resize_display_but_persistent_resize_commits() -> None:
    replay = Replay()
    original = replay.step(region())[0]
    assert replay.step(region(width=500))[0] is original
    assert replay.step(region())[0] is original
    assert replay.step(region(width=500))[0] is original
    assert replay.step(region(width=500))[0].bounds.width == 500


def test_fuzzy_single_frame_error_is_ignored_but_persistent_change_gets_one_revision() -> None:
    replay = Replay()
    original = replay.step(region())[0]
    typo = "新しい冒険が、ここから始まろ。"
    assert replay.step(region(typo, confidence=0.9))[0] is original
    assert replay.step(region())[0] is original
    assert replay.step(region(typo, confidence=0.9))[0] is original
    updated = replay.step(region(typo, confidence=0.9))[0]
    assert updated.track_id == original.track_id
    assert updated.text == typo and updated.text_revision == 2
    assert replay.step(region(typo))[0] is updated


@pytest.mark.parametrize("old,new", [("はい", "いいえ"), ("HP 100", "HP 101"), ("猫", "犬")])
def test_meaningful_short_edits_are_confirmed_not_hidden_as_fuzzy_matches(old, new) -> None:
    replay = Replay()
    original = replay.step(region(old))[0]
    assert replay.step(region(new))[0] is original
    updated = replay.step(region(new))[0]
    assert updated.track_id == original.track_id
    assert updated.text == new and updated.text_revision == 2


def test_width_and_whitespace_normalization_do_not_revise_raw_display_text() -> None:
    replay = Replay()
    original = replay.step(region("準備はいい？"))[0]
    assert replay.step(region("準備 は いい?"))[0] is original
    assert replay.step(region("準備 は いい?"))[0] is original


def test_typewriter_prefix_growth_is_same_track_but_text_waits_until_stable() -> None:
    replay = Replay()
    original = replay.step(region("新しい冒険", width=100))[0]
    typing = replay.step(region("新しい冒険が始まる", width=210))[0]
    assert typing.track_id == original.track_id and typing.text == original.text
    finished = replay.step(region("新しい冒険が始まる", width=210))[0]
    assert finished.text == "新しい冒険が始まる"
    assert finished.text_revision == 2


def test_reordered_duplicate_labels_and_new_detection_keep_existing_ids() -> None:
    replay = Replay()
    original = replay.step(region("はい", y=200), region("はい", y=400))
    current = replay.step(region("設定", y=50), region("はい", y=402), region("はい", y=199))
    assert replay.ids == (3, 2, 1)
    assert current[:2] == original
    assert len({track.track_id for track in current}) == 3


@pytest.mark.parametrize("prefix,label", [("", "軌跡"), ("7", "光円錐"), ("☆", "星魂")])
@pytest.mark.parametrize("reverse", [False, True])
def test_icon_and_label_share_one_track_and_keep_the_tight_reading(prefix, label, reverse):
    replay = Replay()
    clean = region(label, x=434, y=200, width=76, height=32, confidence=0.91)
    wide = region(prefix + label, x=332, y=193, width=179, height=46, confidence=0.98)
    pair = (clean, wide) if reverse else (wide, clean)
    (original,) = replay.step(*pair)
    assert (original.text, original.bounds) == (label, clean.bounds)
    assert replay.ids[pair.index(clean)] == original.track_id
    assert replay.ids[pair.index(wide)] is None
    for rows in (pair[::-1], (wide,), (clean,), pair):
        assert replay.step(*rows) == (original,)
        assert replay.result is not None
        assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == (original,)


def test_icon_only_scan_then_text_only_scan_confirm_one_correction():
    replay = Replay()
    wide = region("7光円錐", x=332, y=193, width=179, height=46)
    clean = region("光円錐", x=434, y=200, width=76, height=32)
    (original,) = replay.step(wide)
    assert replay.step(clean) == (original,)
    (corrected,) = replay.step(clean)
    assert (corrected.track_id, corrected.text, corrected.bounds) == (
        original.track_id,
        clean.text,
        clean.bounds,
    )


def test_overlapping_icon_reads_with_a_dropped_glyph_use_one_slot_in_either_order():
    full = region("*魂", x=334, y=1028, width=197, height=58, confidence=0.82)
    partial = region("*星魂", x=331, y=1027, width=198, height=58, confidence=0.95)
    for rows in ((full, partial), (partial, full)):
        replay = Replay()
        (original,) = replay.step(*rows)
        assert original.text == partial.text
        assert replay.step(*rows[::-1]) == (original,)
        assert sum(key is not None for key in replay.ids) == 1


def test_confident_numeric_prefix_can_change_an_established_label():
    replay = Replay()
    clean = region("遺物", x=434, y=845, width=95, height=44, confidence=0.99)
    wide = TextRegion(
        "3遺物",
        Rect(330, 837, 200, 61),
        (
            TextCharacter("3", Rect(333, 837, 66, 61), 0.96),
            TextCharacter("遺", Rect(428, 837, 50, 61), 0.93),
            TextCharacter("物", Rect(479, 837, 50, 61), 0.95),
        ),
    )
    (original,) = replay.step(clean)
    assert replay.step(wide) == (original,)
    (changed,) = replay.step(wide)
    assert changed.track_id == original.track_id and changed.text == "3遺物"


def test_icon_variants_do_not_absorb_nearby_choices_or_edit_numeric_values():
    replay = Replay()
    clean = region("星魂", x=434, y=200, width=76, height=32)
    wide = region("☆星魂", x=332, y=193, width=179, height=46)
    others = (
        region("星魂", x=518, y=200, width=76, height=32),
        region("星魂", x=434, y=225, width=76, height=32),
        region("10%", x=700, y=200, width=90, height=32),
    )
    original = replay.step(clean, wide, *others)
    assert len(original) == 4 and {t.bounds for t in original} == {
        r.bounds for r in (clean, *others)
    }
    changed = replace(others[-1], text="110%", bounds=Rect(670, 200, 120, 32))
    assert replay.step(changed, *others[:2], wide, clean) == original
    current = {t.track_id: t for t in replay.step(changed, *others[:2], wide, clean)}
    (percent,) = (t for t in original if t.text == "10%")
    assert current.keys() == {t.track_id for t in original}
    assert all(current[t.track_id] == t for t in original if t is not percent)
    assert current[percent.track_id].text == "110%"


@pytest.mark.parametrize("reverse", (False, True))
def test_competing_nearby_labels_preserve_both_ids_instead_of_greedy_replacement(reverse):
    replay = Replay()
    original = replay.step(
        region("同じ表示", y=200, width=180, height=30),
        region("同じ表示", y=210, width=180, height=30),
    )
    # The first observation is the best individual match for track 1, but it is
    # track 2's only compatible observation. Track 1 can match the other row.
    rows = (
        region("同じ表示", y=202, width=180, height=30),
        region("同じ表示", y=112, width=180, height=30),
    )
    if reverse:
        rows = tuple(reversed(rows))
    current = replay.step(*rows)
    assert {t.track_id for t in current} == {t.track_id for t in original}
    assert replay.ids == ((1, 2) if reverse else (2, 1))
    assert all(t.text_revision == 1 for t in current)
    replay.step(*reversed(rows))
    assert replay.ids == ((2, 1) if reverse else (1, 2))
    assert replay.result is not None
    assert len(replay.tracker.advance(replay.result.frame, replay.now).confirmed) == 2


def test_equal_score_ties_use_geometry_instead_of_detector_order():
    assignments = []
    for reverse in (False, True):
        replay = Replay()
        # Two separate rows become two separate columns. All four pairings
        # have the same overlap, edge distance and text.
        replay.step(region("はい", x=200, y=680, width=40), region("はい", x=200, y=720, width=40))
        rows = (region("はい", x=180, width=40), region("はい", x=220, width=40))
        if reverse:
            rows = tuple(reversed(rows))
        replay.step(*rows)
        assignments.append({row.bounds.x: key for row, key in zip(rows, replay.ids, strict=True)})
    assert assignments[0] == assignments[1] == {180: 1, 220: 2}


def test_repetitive_minor_edit_requires_a_better_reading():
    replay = Replay()
    original = replay.step(region("abababababab", confidence=0.99))[0]
    # One deletion, despite a misleading long-block diff alignment on repeated text.
    for _ in range(12):
        assert replay.step(region("ababbababab", confidence=0.6))[0] == original
    replay.step(region("ababbababab", confidence=0.999))
    changed = replay.step(region("ababbababab", confidence=0.999))[0]
    assert changed.track_id == original.track_id and changed.text_revision == 2
    assert changed.text == "ababbababab"


def test_same_text_far_away_does_not_steal_old_identity() -> None:
    replay = Replay()
    first = replay.step(region())[0]
    current = replay.step(region(x=1100, y=100))
    assert replay.ids == (first.track_id + 1,)
    assert len(current) == 2  # Old region has one-frame dropout grace.
    assert len(replay.step(region(x=1100, y=100))) == 2
    assert len(replay.step(region(x=1100, y=100))) == 1


def test_one_missed_detection_or_empty_recognition_does_not_blink() -> None:
    replay = Replay()
    original = replay.step(region())
    assert replay.step(region("")) == original
    assert replay.ids == (None,)
    assert replay.step(region()) == original
    assert replay.step() == original
    assert replay.step() == original
    assert replay.step() == ()
    assert replay.step(region())[0].track_id != original[0].track_id


def test_slow_ocr_and_advancing_capture_do_not_prove_text_disappeared() -> None:
    replay = Replay()
    original = replay.step(region())
    assert replay.result is not None
    source = replay.result.frame
    assert replay.tracker.advance(source, 100).visible == original
    advanced = replace(source, sequence=source.sequence + 1, captured_at=1)
    assert replay.tracker.advance(advanced, 1.1).visible == original
    assert replay.tracker.advance(advanced, 100).visible == original


def test_duplicate_or_out_of_order_results_cannot_confirm_a_text_change() -> None:
    replay = Replay()
    original = replay.step(region("猫"))
    old = replay.result
    replay.step(region("犬"))
    result = replay.result
    assert result is not None and old is not None
    for _ in range(20):
        replay.tracker.observe(result)
        replay.tracker.observe(old)
    assert replay.tracker.advance(result.frame, replay.now).visible == original
    assert replay.step(region("犬"))[0].text == "犬"


def test_reset_and_resize_clear_state_without_reusing_ids() -> None:
    replay = Replay()
    original = replay.step(region())[0]
    assert replay.result is not None
    resized = replace(
        replay.result.frame,
        geometry=replace(replay.result.frame.geometry, client=Rect(0, 0, 1280, 720)),
    )
    assert replay.tracker.advance(resized, replay.now).visible == ()
    second = replay.step(region())[0]
    assert second.track_id > original.track_id
    replay.tracker.reset()
    assert replay.step(region())[0].track_id > second.track_id


def test_scene_replacement_drops_missing_tracks_and_track_count_is_bounded() -> None:
    replay = Replay(TrackingPolicy(max_tracks=4))
    assert len(replay.step(*(region(str(i), y=i * 80) for i in range(8)))) == 4
    # Capacity cannot evict a tracked label on one missed pass. Once absence is
    # confirmed, its slot is available to the new scene.
    replay.step(region("別の場面", y=900), dt=1)
    replay.step(region("別の場面", y=900))
    current = replay.step(region("別の場面", y=900))
    assert len(current) == 1 and current[0].text == "別の場面"


def test_vertical_text_uses_its_width_for_jitter_tolerance() -> None:
    replay = Replay()
    original = replay.step(region("日本語", y=100, width=40, height=400, vertical=True))
    assert (
        replay.step(region("日本語", x=104, y=102, width=40, height=400, vertical=True)) == original
    )


def test_static_window_keeps_identity_when_capture_eventually_resumes() -> None:
    replay = Replay()
    original = replay.step(region())
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, 10).visible == original
    assert replay.step(region(x=102), dt=10) == original


def test_final_static_edit_settles_without_a_second_capture_frame() -> None:
    replay = Replay()
    original = replay.step(region("猫"))[0]
    replay.step(region("犬", width=420))
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now + 0.3).visible[0] is original
    settled = replay.tracker.advance(replay.result.frame, replay.now + 0.4).visible[0]
    assert settled.track_id == original.track_id
    assert settled.text == "犬" and settled.text_revision == 2
    assert settled.bounds.width == 420
    assert replay.tracker.advance(replay.result.frame, replay.now + 5).visible[0] is settled


def test_pending_edit_does_not_settle_on_unprocessed_advancing_capture() -> None:
    replay = Replay()
    original = replay.step(region("猫"))[0]
    replay.step(region("犬"))
    assert replay.result is not None
    advanced = replace(replay.result.frame, sequence=replay.sequence + 1)
    assert replay.tracker.advance(advanced, replay.now + 0.4).visible[0] is original


def test_slow_consecutive_observations_can_confirm_a_new_reading() -> None:
    replay = Replay()
    original = replay.step(region("猫"))[0]
    replay.step(region("犬"))
    changed = replay.step(region("犬"), dt=10)[0]
    assert changed.text == "犬" and changed.track_id == original.track_id


def test_translation_waits_for_confirmation_and_preserves_committed_content_through_noise() -> None:
    replay = Replay()
    replay.step(region("新しい冒険"))
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now + 0.1).confirmed == ()
    stable = replay.step(region("新しい冒険"))
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == stable
    replay.step(region("新しい冒険が始まる"))
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == stable
    settled = replay.tracker.advance(replay.result.frame, replay.now + 0.4).confirmed
    assert settled[0].text == "新しい冒険が始まる"
    replay.step()
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == settled
    replay.step()
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == settled
    replay.step()
    assert replay.result is not None
    assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == ()


@pytest.mark.parametrize(
    "winner,variant",
    [
        ("新しい冒険が、ここから始まる。", "新しい冒険が、ここから始まろ。"),
        ("ポーションを使う", "ボーションを使う"),
        ("アイテムを使う", "◆アイテムを使う"),
        ("冒険に出発", "胃険に出発"),
    ],
)
@pytest.mark.parametrize("weaker_score", [0.65, 0.95])
def test_alternating_weaker_or_equal_variants_keep_the_winning_text_and_box(
    winner, variant, weaker_score
) -> None:
    replay = Replay()
    original = replay.step(region(winner, confidence=0.95))[0]
    replay.step(region(winner, confidence=0.9))
    for _ in range(12):
        for _ in range(3):
            current = replay.step(region(variant, x=80, width=420, confidence=weaker_score))[0]
            assert current is original
            assert replay.result is not None
            assert replay.result.regions[0].text == variant  # Raw inspection is untouched.
            assert replay.tracker.advance(replay.result.frame, replay.now).confirmed == (original,)
        # A weaker observation of the winning spelling must not lower its remembered score.
        assert replay.step(region(winner, confidence=0.5))[0] is original


def test_higher_confidence_variant_wins_once_and_remembers_its_peak_score() -> None:
    replay = Replay()
    typo, correct = "ボーションを使う", "ポーションを使う"
    original = replay.step(region(typo, confidence=0.7))[0]
    assert replay.step(region(correct, confidence=0.96))[0] is original
    winner = replay.step(region(correct, confidence=0.9))[0]
    assert winner.text == correct and winner.text_revision == 2
    assert winner.track_id == original.track_id
    for _ in range(5):
        assert replay.step(region(typo, confidence=0.95))[0] is winner
        assert replay.step(region(typo, confidence=0.95))[0] is winner
        assert replay.step(region(correct, confidence=0.5))[0] is winner


@pytest.mark.parametrize(
    "old,new",
    [
        ("体力が１００残っている", "体力が１０１残っている"),
        ("攻撃力が＋１０％", "攻撃力が−１０％"),
        ("移動速度１．２５倍", "移動速度１２．５倍"),
        ("残りの回数は一回です", "残りの回数は二回です"),
        ("解除", "解放"),
        ("冒険に出発する", "冒険に出発するよ"),
        ("冒険に出発するよ", "冒険に出発する"),
        ("新しい冒険が始まる", "準備ができたら出かけよう"),
    ],
)
def test_real_changes_do_not_need_to_beat_the_old_confidence(old, new) -> None:
    replay = Replay()
    original = replay.step(region(old, confidence=0.99))[0]
    assert replay.step(region(new, confidence=0.6))[0] is original
    current = replay.step(region(new, confidence=0.6))[0]
    assert current.text == new and current.text_revision == 2


def test_repeated_substitutions_cannot_lower_the_confidence_of_a_live_reading() -> None:
    replay = Replay()
    original = replay.step(region("東の門へ向かってください", confidence=0.99))[0]
    for _ in range(15):
        assert replay.step(region("西の門へ向かってください", confidence=0.6))[0] is original
    replay.step(region("西の門へ向かってください", confidence=1.0))
    current = replay.step(region("西の門へ向かってください", confidence=1.0))[0]
    assert current.text == "西の門へ向かってください" and current.text_revision == 2
    assert current.track_id == original.track_id
    assert replay.step(region("南の門へ向かってください", confidence=0.8))[0] is current


def test_elapsed_time_and_visual_confirmation_cannot_promote_a_weaker_spelling() -> None:
    replay = Replay()
    original = replay.step(region("東の門へ向かってください", confidence=0.99))[0]
    replay.step(region("西の門へ向かってください", confidence=0.6))
    assert replay.result is not None
    frame = replay.result.frame
    for elapsed in (0.35, 1.26, 5.0):
        replay.tracker.observe(replay.result)  # Duplicate polls cannot add confirmation evidence.
        replay.tracker.confirm_visual(frame.sequence, (0,), replay.now + elapsed)
        assert replay.tracker.advance(frame, replay.now + elapsed).visible[0] is original


def test_changed_dialogue_resets_confidence_even_when_it_keeps_the_same_track_id() -> None:
    replay = Replay()
    original = replay.step(region("新しい冒険が始まる", confidence=0.99))[0]
    replay.step(region("ポーションを使う", confidence=0.6))
    current = replay.step(region("ポーションを使う", confidence=0.6))[0]
    assert current.track_id == original.track_id
    replay.step(region("ボーションを使う", confidence=0.8))
    assert replay.step(region("ボーションを使う", confidence=0.8))[0].text == "ボーションを使う"


def test_confidence_memory_is_independent_for_identical_labels_in_different_places() -> None:
    replay = Replay()
    high = region("ポーションを使う", y=200, confidence=0.99)
    low = region("ポーションを使う", y=500, confidence=0.6)
    original = replay.step(high, low)
    variants = (
        region("ボーションを使う", y=500, confidence=0.8),
        region("ボーションを使う", y=200, confidence=0.8),
    )
    replay.step(*variants)
    current = replay.step(*variants)
    assert replay.ids == (2, 1)
    assert current[0] is original[0]
    assert current[1].text == "ボーションを使う"


@pytest.mark.parametrize("reset", [True, False])
def test_removed_or_reset_tracks_do_not_inherit_confidence_at_the_same_position(reset) -> None:
    replay = Replay()
    original = replay.step(region("ポーションを使う", confidence=0.99))[0]
    if reset:
        replay.tracker.reset()
    else:
        replay.step()
        replay.step()
        replay.step()
    recreated = replay.step(region("ポーションを使う", confidence=0.6))[0]
    assert recreated.track_id != original.track_id
    replay.step(region("ボーションを使う", confidence=0.8))
    assert replay.step(region("ボーションを使う", confidence=0.8))[0].text == "ボーションを使う"


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -0.5, 1.5])
def test_unavailable_or_invalid_scores_cannot_supersede_a_similar_reading(score) -> None:
    replay = Replay()
    original = replay.step(region("ポーションを使う", confidence=0.99))[0]
    assert replay.step(region("ボーションを使う", confidence=score))[0] is original
    assert replay.step(region("ボーションを使う", confidence=score))[0] is original


def test_substantially_changed_location_uses_normal_confirmation() -> None:
    replay = Replay()
    original = replay.step(region("ポーションを使う", confidence=0.99))[0]
    replay.step(region("ボーションを使う", x=200, confidence=0.7))
    current = replay.step(region("ボーションを使う", x=200, confidence=0.7))[0]
    assert current.track_id == original.track_id and current.text_revision == 2
    assert current.bounds.x == 200


def test_missed_observations_break_a_weaker_candidates_sustained_confirmation() -> None:
    replay = Replay()
    original = replay.step(region("東の門へ向かってください", confidence=0.99))[0]
    for _ in range(4):
        for _ in range(3):
            assert replay.step(region("西の門へ向かってください", confidence=0.6))[0] is original
        assert replay.step()[0] is original


def test_dense_slow_results_and_two_pass_dropouts_keep_all_committed_objects() -> None:
    tracker = TextTracker()
    viewport = Rect(0, 0, 3840, 2160)
    observations = tuple(
        region("設定の確認", x=40 + i % 8 * 450, y=40 + i // 8 * 150, width=260, height=36)
        for i in range(80)
    )
    original = ()
    for sequence in range(1, 15):
        captured = sequence * 1.2
        frame = CapturedFrame(
            sequence, captured, WindowGeometry(viewport, viewport, viewport, 96), b""
        )
        # Ten labels are missed twice in a row, then return; detection order varies.
        detected = observations if sequence % 3 == 0 or sequence <= 2 else observations[10:]
        tracker.observe(OcrResult(frame, detected[::-1], 1100, captured + 1.1))
        latest = replace(frame, sequence=sequence + 1, captured_at=captured + 1)
        for offset in (1.1, 1.4, 1.8, 2.2):
            current = tracker.advance(latest, captured + offset).visible
            if not original:
                original = current
            assert len(current) == 80
            assert all(a is b for a, b in zip(original, current, strict=True))
        if sequence >= 2:
            assert tracker.advance(latest, captured + 2.2).confirmed == original


def test_duplicate_polls_and_skipped_capture_frames_never_count_as_absence() -> None:
    replay = Replay()
    original = replay.step(region())
    replay.step()
    assert replay.result is not None
    latest = replace(replay.result.frame, sequence=1000, captured_at=20)
    for now in range(1, 100):
        replay.tracker.observe(replay.result)
        assert replay.tracker.advance(latest, now).visible == original
    assert replay.step() == original
    assert replay.step() == ()


def test_static_miss_without_pixel_evidence_does_not_expire_on_a_timer() -> None:
    replay = Replay()
    original = replay.step(region())
    replay.step()
    assert replay.result is not None
    frame = replay.result.frame
    assert replay.tracker.advance(frame, replay.result.completed_at + 0.2).visible == original
    assert replay.tracker.advance(frame, replay.result.completed_at + 50).visible == original


def test_fast_missing_passes_need_time_span_and_redetection_resets_absence() -> None:
    replay = Replay()
    original = replay.step(region())
    for _ in range(10):
        assert replay.step(dt=0.01) == original
    assert replay.step(region()) == original
    assert replay.step(dt=2) == original  # A long first miss is still only one miss.
    assert replay.step() == original
    assert replay.step() == ()


def test_dense_detector_reordering_cannot_evict_existing_tracks_at_capacity() -> None:
    replay = Replay(TrackingPolicy(max_tracks=8))
    observations = tuple(region("設定", x=i * 100, y=100, width=60) for i in range(16))
    original = replay.step(*observations)
    for _ in range(6):
        assert replay.step(*observations[::-1]) == original


def test_truncated_observations_are_not_negative_evidence_for_unexamined_regions() -> None:
    replay = Replay(TrackingPolicy(max_tracks=2))
    original = replay.step(region("設定", x=1500, y=900, width=60))
    crowded = tuple(region("別の項目", x=i * 100, y=100, width=60) for i in range(9))
    for _ in range(6):
        current = replay.step(*crowded, region("設定", x=1500, y=900, width=60))
        assert current[0] is original[0]


def test_one_orientation_error_cannot_erase_the_existing_box() -> None:
    replay = Replay()
    original = replay.step(region())
    assert replay.step(region(vertical=True)) == original
    assert replay.step(region()) == original


@pytest.mark.parametrize("reverse", [False, True])
def test_two_new_readings_of_the_same_ink_admit_only_the_more_confident(reverse) -> None:
    weaker = region("新しい冒険がここから始まる", x=104, y=702, width=396, confidence=0.8)
    stronger = region("新しい冒険が、ここから始まる。", confidence=0.9)
    readings = (weaker, stronger)[::-1] if reverse else (weaker, stronger)
    replay = Replay()
    assert [track.text for track in replay.step(*readings)] == [stronger.text]
    assert replay.ids.count(None) == 1


@pytest.mark.parametrize("revealing", [True, False])
def test_lower_confidence_terminal_punctuation_only_gets_fast_confirmation_after_reveal(revealing):
    replay = Replay()
    text = "新しい冒険がここから始まります"
    if revealing:
        replay.step(region("新しい", confidence=0.99))
    replay.step(region(text, confidence=0.99))
    replay.step(region(text, confidence=0.99))
    replay.step(region(text + "。", confidence=0.90))
    values = replay.step(region(text + "。", confidence=0.90))
    assert values[0].text == (text + "。" if revealing else text)


def test_a_settled_typewriter_line_keeps_its_reading_through_weaker_terminal_variants():
    replay = Replay()
    text = "新しい冒険がここから始まります"
    replay.step(region("新しい", confidence=0.99))
    replay.step(region(text, confidence=0.97))
    accepted = replay.step(region(text, confidence=0.97))[0]
    for variant in (text + "）", text + "。", text[:-1]):
        for _ in range(3):
            assert replay.step(region(variant, confidence=0.9), dt=2)[0] is accepted


def test_quiet_pixels_cannot_commit_a_single_weak_disagreement_on_a_locked_label():
    replay = Replay()
    label = region("防御力", width=122, confidence=0.95)
    replay.step(label)
    locked = replay.step(label)
    noise = region("全く違う読み方", x=94, width=136, confidence=0.2)
    for reading in (noise, region("誤った観測", width=138, confidence=0.2), noise):
        assert replay.step(reading) == locked
        assert replay.result is not None
        f = replay.result.frame
        replay.tracker.confirm_visual(f.sequence, (0,), replay.now + 5)
        replay.tracker.confirm_observations(
            ((locked[0].track_id, f.sequence, reading.text),), replay.now + 5
        )
        assert replay.tracker.advance(f, replay.now + 5).confirmed == locked
        assert not replay.tracker.snapshot().activity[0].pending
    assert replay.step(noise)[0].text == noise.text


@pytest.mark.parametrize(
    "first,following",
    [
        (region("新しい冒険―", width=200), region("新しい冒険がここから始まります", width=500)),
        (region("確認", width=60), region("新しい冒険がここから始まります", width=500)),
        (region("二", y=710, width=20, height=18), region("二度目の冒険が始まります", width=500)),
    ],
)
def test_same_line_origin_does_not_spawn_duplicate_partial_tracks(first, following):
    replay = Replay()
    original = replay.step(first)[0]
    pending = replay.step(following)
    assert len(pending) == 1 and pending[0].track_id == original.track_id
    assert replay.tracker.snapshot().activity[0].pending
    final = replay.step(following)
    assert len(final) == 1 and final[0].track_id == original.track_id
    assert final[0].text == following.text


def test_repetitive_long_lines_keep_spatial_identity_without_suppressing_real_edits():
    replay = Replay()
    text = "繰り返し表示" * 70 + "体力3204"
    first = region(text, x=100, y=200, width=1400, confidence=0.98)
    second = region(text, x=100, y=260, width=1400, confidence=0.98)
    original = replay.step(first, second)
    assert replay.step(second, first) == original
    edited = replace(first, text=text[:-1] + "5", characters=())
    tracks = original
    for _ in range(4):
        tracks = replay.step(second, edited)
    assert [(t.track_id, t.text) for t in tracks] == [
        (original[0].track_id, edited.text),
        (original[1].track_id, text),
    ]
    assert tracks[0].text_revision > original[0].text_revision
    # Equal text at an unrelated position cannot borrow the old line's identity.
    elsewhere = region(edited.text, x=100, y=800, width=1400)
    tracks = replay.step(elsewhere, edited, second)
    assert len(tracks) == 3 and tracks[-1].track_id not in {t.track_id for t in original}
