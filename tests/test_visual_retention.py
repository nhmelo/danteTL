"""Replay stationary character-screen labels with hostile OCR, not just clean demos."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.observation import CandidateReview, TextTrack
from dantetl.ocr.presence import prepare_evidence
from dantetl.ocr.tracking import TextTracker
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion
from dantetl.ocr.visual import VisualAnalyzer
from dantetl.translation.grouping import DialogueGrouper, TextBlock
from dantetl.translation.presentation import EnglishPresentation
from dantetl.translation.values import TranslatedText, TranslationRequest

BOX = Rect(20, 30, 220, 28)


@pytest.mark.parametrize("menu_switch", (False, True))
def test_quest_paragraph_clears_over_its_original_gradient_while_ocr_is_delayed(menu_switch):
    viewport = Rect(0, 0, 2480, 400)
    rows = (Rect(30, 40, 2398, 46), Rect(30, 100, 1165, 46))
    gradient = np.linspace(20, 150, viewport.width, dtype=np.uint8)
    backing = np.tile(gradient, (viewport.height, 1))
    cv2.putText(backing, "Menu", (35, 202), cv2.FONT_HERSHEY_SIMPLEX, 1, 225, 2)
    original = backing.copy()
    for row in rows:
        cv2.putText(
            original[:, row.x : row.x + row.width],
            "QUEST DETAILS " * 12,
            (0, row.y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            210,
            2,
            cv2.LINE_AA,
        )
    tracks = (
        TextTrack(1, "物語の始まり", rows[0]),
        TextTrack(2, "次の行です。", rows[1]),
        TextTrack(3, "設定", Rect(30, 172, 100, 36)),
    )
    visible = CapturedFrame(
        1,
        0.0,
        WindowGeometry(viewport, viewport, viewport, 96),
        cv2.cvtColor(original, cv2.COLOR_GRAY2BGRA).tobytes(),
    )
    if menu_switch:
        # The new page keeps the gradient behind the paragraph and gains controls elsewhere.
        for y in range(175, 290, 24):
            cv2.putText(backing, "EQUIPMENT  " * 6, (340, y), 0, 0.8, 235, 2, cv2.LINE_AA)
    bare = replace(visible, bgra=cv2.cvtColor(backing, cv2.COLOR_GRAY2BGRA).tobytes())
    result = OcrResult(visible, tuple(TextRegion(t.text, t.bounds) for t in tracks), 10, 0.01)
    blocks = (TextBlock(tracks[:2]), TextBlock(tracks[2:]))
    replies = tuple(
        TranslatedText(TranslationRequest.from_block(block), text)
        for block, text in zip(blocks, ("Quest description", "Menu"), strict=True)
    )
    visual, presentation = VisualAnalyzer(), EnglishPresentation()
    visual.observe(visible, result, tracks, (1, 2, 3))
    initial = presentation.update(blocks, replies)
    assert len(initial) == 2
    for seq in (2, 2, 3, 4, 5):
        state = visual.observe(
            replace(bare, sequence=seq, captured_at=(seq - 1) / 8), result, tracks, (1, 2, 3)
        )
        assert presentation.update(blocks, (), cleared=state.cleared) == (
            initial if not menu_switch and seq < 4 else initial[1:]
        )
    restored = visual.observe(
        replace(visible, sequence=6, captured_at=0.625), result, tracks, (1, 2, 3)
    )
    assert presentation.update(blocks, (), cleared=restored.cleared) == initial


def picture(sequence, text="HP 3204", *, stars=True, gone=False, extension=False):
    rng = np.random.default_rng(sequence)
    gray = (
        rng.integers(10, 90, (180, 400), dtype=np.uint8)
        if stars
        else np.full((180, 400), 30, np.uint8)
    )
    if not gone:
        cv2.putText(gray, text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 235, 2, cv2.LINE_AA)
    if extension:
        cv2.putText(gray, "MORE", (245, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 235, 2, cv2.LINE_AA)
    # Always animate the rest of the screen, even when local text does not change.
    gray[100:, :] = sequence * 13 % 200
    size = Rect(0, 0, 400, 180)
    return CapturedFrame(
        sequence,
        sequence * 0.25,
        WindowGeometry(size, size, size, 96),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA).tobytes(),
    )


def result(sequence, text="攻撃力", score=0.96, **kwargs):
    frame = picture(sequence, **kwargs)
    return OcrResult(
        frame,
        (TextRegion(text, BOX, (TextCharacter(text, BOX, score),)),),
        250,
        frame.captured_at + 0.25,
    )


class Replay:
    def __init__(self, first):
        self.visual = VisualAnalyzer()
        self.tracker = TextTracker()
        self.result = first
        self.ids = self.tracker.observe(
            first, review=CandidateReview(pixels=prepare_evidence(first))
        )
        self.tracks = self.tracker.advance(first.frame, first.completed_at + 0.5).confirmed
        self.visual.observe(first.frame, first, self.tracks, self.ids, candidate=first)

    def step(self, incoming):
        state = self.visual.observe(
            incoming.frame,
            self.result,
            self.tracks,
            self.ids,
            self.tracker.snapshot().activity,
            incoming,
        )
        self.ids = self.tracker.observe(
            incoming,
            review=replace(
                state.review, pixels=prepare_evidence(incoming, self.tracker.evidence_bounds())
            ),
            presence=state.presence,
        )
        self.tracks = self.tracker.advance(incoming.frame, incoming.completed_at + 0.5).confirmed
        self.result = incoming
        self.visual.observe(
            incoming.frame,
            incoming,
            self.tracks,
            self.ids,
            self.tracker.snapshot().activity,
            incoming,
        )
        return state


@pytest.mark.parametrize(
    "edit, recolor, exhaust_budget",
    [(True, False, False), (True, True, False), (False, True, False), (True, False, True)],
)
def test_equal_luminance_counter_edit_and_hover_through_publication(
    edit, recolor, exhaust_budget, monkeypatch
):
    if exhaust_budget:
        monkeypatch.setattr("dantetl.ocr.visual._FINE_PIXELS", 0)

    def observation(sequence, digit, color, reading):
        bgra = np.full((120, 320, 4), (76, 76, 76, 255), np.uint8)
        mask = np.zeros(bgra.shape[:2], np.uint8)
        cv2.putText(mask, f"HP {digit}", (25, 62), 0, 0.95, 255, 2)
        bgra[mask >= 128] = color
        assert np.ptp(cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)) == 0
        viewport = Rect(0, 0, 320, 120)
        frame = CapturedFrame(
            sequence,
            sequence * 0.25,
            WindowGeometry(viewport, viewport, viewport, 96),
            bgra.tobytes(),
        )
        row = TextRegion(f"残り{reading}回", Rect(20, 30, 160, 40), line_confidence=0.98)
        return OcrResult(frame, (row,), 15, frame.captured_at + 0.015)

    red, green = (0, 0, 254, 255), (0, 129, 0, 255)
    replay = Replay(observation(1, 4, red, 4))
    presentation = EnglishPresentation()
    block = TextBlock(replay.tracks)
    initial_reply = TranslatedText(TranslationRequest.from_block(block), "Remaining 4")
    assert presentation.update((block,), (initial_reply,))[0].translation == initial_reply
    original = replay.tracks[0]
    for sequence in range(2, 5):
        # OCR also reports 5 during a hover-only recolor, where it must be ignored.
        replay.step(observation(sequence, 5 if edit else 4, green if recolor else red, 5))
    assert replay.tracks[0].track_id == original.track_id
    expected = "5" if edit else "4"
    assert replay.tracks[0].text == f"残り{expected}回"
    block = TextBlock(replay.tracks)
    reply = TranslatedText(TranslationRequest.from_block(block), f"Remaining {expected}")
    assert presentation.update((block,), (reply,) if edit else ())[0].translation == reply


def long_line(
    sequence, width, *, reading="体力3204", digit="4", vertical=False, brightness=0, font=0.6
):
    gray = np.full((48, width + 40), 30, np.uint8)
    cv2.putText(
        gray,
        "STAT 320" + digit + " MENU" * (width // 25),
        (20, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        font,
        235,
        1,
        cv2.LINE_AA,
    )
    gray = np.clip(gray.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    box = Rect(20, 10, width, 28)
    if vertical:
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        box = Rect(10, 20, 28, width)
    viewport = Rect(0, 0, gray.shape[1], gray.shape[0])
    frame = CapturedFrame(
        sequence,
        sequence * 0.25,
        WindowGeometry(viewport, viewport, viewport, 96),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA).tobytes(),
    )
    return OcrResult(
        frame,
        (TextRegion(reading, box, (TextCharacter(reading, box, 0.98),), vertical),),
        100,
        frame.captured_at + 0.1,
    )


@pytest.mark.parametrize("width", [768, 1536, 3072])
@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("font", [0.35, 0.6])
def test_fine_evidence_retains_long_lines_but_unlocks_one_changed_digit(width, vertical, font):
    replay = Replay(long_line(1, width, vertical=vertical, font=font))
    original = replay.tracks
    # Brightness and OCR disagreement cannot alone replace the accepted text.
    state = replay.step(
        long_line(2, width, vertical=vertical, font=font, reading="体力3205", brightness=10)
    )
    assert state.review.unchanged == set(original) and replay.tracks == original
    for sequence in (3, 4, 5):
        state = replay.step(
            long_line(sequence, width, vertical=vertical, font=font, reading="体力3205", digit="5")
        )
        if sequence == 3:
            assert not state.review.unchanged
    assert replay.tracks[0].track_id == original[0].track_id
    assert (
        replay.tracks[0].text == "体力3205"
        and replay.tracks[0].text_revision > original[0].text_revision
    )


def test_exhausted_fine_evidence_budget_allows_ordinary_ocr_confirmation():
    # This line exceeds the per-line reference budget. A coarse thumbnail must
    # not freeze its text forever, even if the original reading was confident.
    replay = Replay(long_line(1, 10000))
    original = replay.tracks[0]
    for sequence in (2, 3, 4):
        state = replay.step(long_line(sequence, 10000, reading="体力3205", digit="5"))
        assert not state.review.unchanged
    assert replay.tracks[0].track_id == original.track_id and replay.tracks[0].text == "体力3205"


def test_stationary_labels_keep_exact_identity_layout_and_english_through_noisy_ocr():
    replay = Replay(result(1))
    original = replay.tracks
    presentation, grouper = EnglishPresentation(), DialogueGrouper()
    block = TextBlock(original)
    english = TranslatedText(TranslationRequest.from_block(block), "Attack")
    assert presentation.update((block,), (english,))
    for sequence in range(2, 40):
        incoming = result(sequence, text=("攻撃力", "防御力", "攻擊力")[sequence % 3], score=0.99)
        if sequence % 4 == 0:
            incoming = replace(incoming, regions=())
        elif sequence % 3 == 0:
            jittered = Rect(10, 28, 245, 32)
            raw = incoming.regions[0]
            incoming = replace(
                incoming,
                regions=(
                    replace(
                        raw, bounds=jittered, characters=(TextCharacter(raw.text, jittered, 0.99),)
                    ),
                ),
            )
        state = replay.step(incoming)
        assert state.review.unchanged == set(original)
        assert not state.hidden
        assert replay.tracks == original
        grouped = grouper.update(
            replay.tracks,
            replay.tracks,
            incoming.completed_at,
            activity=replay.tracker.snapshot().activity,
        )
        published = presentation.update(grouped.blocks, ())
        assert len(published) == 1 and published[0].translation == english


def test_low_confidence_reviews_are_bounded_monotonic_and_keep_geometry():
    replay = Replay(result(1, score=0.60))
    original_bounds = replay.tracks[0].bounds
    for sequence, text, score in (
        (2, "攻擊力", 0.70),
        (3, "攻撃力", 0.65),
        (4, "攻撃カ", 0.80),
        (5, "攻撃力", 0.99),
    ):
        replay.step(result(sequence, text=text, score=score))
        assert replay.tracks[0].bounds == original_bounds
    assert replay.tracks[0].text == "攻撃カ"  # Three reviews consumed; no endless score chasing.
    # A genuine local edit starts a new content lifetime, regardless of the old score.
    for sequence in (6, 7):
        replay.step(result(sequence, text="防御力", score=0.50, gone=True))
    assert replay.tracks[0].text == "防御力"


@pytest.mark.parametrize("text", ["HP 3205", "HP 3201", "HP 320", "HP 9204"])
def test_one_changed_digit_unlocks_a_long_otherwise_unchanged_label(text):
    replay = Replay(result(1, text="体力3204"))
    incoming = result(2, text="体力3205")
    incoming = replace(incoming, frame=picture(2, text=text))
    state = replay.step(incoming)
    assert not state.review.unchanged
    assert state.review.checked
    assert replay.tracks[0].text == "体力3205"


def test_extension_outside_old_box_unlocks_typewriter_growth():
    replay = Replay(result(1, text="冒険の始まり"))
    state = replay.step(result(2, text="冒険の始まりです", extension=True))
    assert not state.review.unchanged
    assert replay.tracks[0].text == "冒険の始まりです"


def test_dim_stars_appearing_beside_text_are_not_a_typewriter_extension():
    replay = Replay(result(1, stars=False))
    incoming = result(2, stars=False, text="全く違う読み方")
    pixels = np.frombuffer(incoming.frame.bgra, np.uint8).reshape(180, 400, 4).copy()
    for x in range(244, 316, 9):
        cv2.rectangle(pixels, (x, 34), (x + 3, 42), (100, 100, 100, 255), 1)
    incoming = replace(incoming, frame=replace(incoming.frame, bgra=pixels.tobytes()))
    state = replay.step(incoming)
    assert state.review.unchanged
    assert replay.tracks[0].text == "攻撃力"


def test_large_reveal_after_one_glyph_is_not_mistaken_for_unchanged_content():
    box = Rect(20, 30, 20, 28)
    first = result(1, text="新", stars=False)
    first = replace(
        first, frame=picture(1, text="H", stars=False), regions=(TextRegion("新", box),)
    )
    replay = Replay(first)
    full = result(2, text="新しい文章です", stars=False)
    full = replace(full, frame=picture(2, text="HMMMMMMMMM", stars=False))
    state = replay.step(full)
    assert state.review.checked and not state.review.unchanged


def test_real_removal_still_hides_and_retires_after_visual_and_ocr_evidence():
    replay = Replay(result(1))
    for sequence in (2, 3):
        missing = replace(result(sequence, gone=True), regions=())
        state = replay.step(missing)
    assert state.hidden or not replay.tracks
    assert not replay.tracks


def test_atomic_publication_keeps_english_through_live_edits_and_drops_retired_members():
    store = EnglishPresentation()
    first = TextTrack(1, "初めの文章", BOX)
    second = TextTrack(2, "次の行です。", Rect(20, 65, 220, 28))
    old = TextBlock((first, second))
    english = TranslatedText(TranslationRequest.from_block(old), "Original complete sentence.")
    store.update((old,), (english,))
    correction = replace(second, text="次の行でした。", text_revision=2)
    current = TextBlock((first, correction))
    kept = store.update((current,), ())
    assert kept[0].painted == old and kept[0].translation == english
    next_correction = replace(correction, text="次の行でした！", text_revision=3)
    final = TextBlock((first, next_correction))
    assert store.update((final,), ())[0].translation == english
    replacement = TranslatedText(
        TranslationRequest.from_block(final), "Corrected complete sentence."
    )
    assert store.update((final,), (replacement,))[0].translation == replacement
    changed = TextBlock((replace(first, text="違う文章", text_revision=2), next_correction))
    assert store.update((changed,), (english,))[0].translation == replacement
    store.update((old,), (english,))
    assert not store.update((), ())
    assert not store.update((old,), ())  # Removed entries cannot resurrect old English.
    store.update((old,), (english,))
    store.reset()
    assert not store.update((old,), ())


def test_regrouping_swaps_only_connected_labels_once_every_replacement_is_ready():
    store = EnglishPresentation()
    a, b, c, d, menu = (
        TextTrack(i, "日本語" + str(i), Rect(20, i * 40, 220, 28)) for i in range(1, 6)
    )
    original = (TextBlock((a, b)), TextBlock((c, d)), TextBlock((menu,)))
    proposed = (TextBlock((a,)), TextBlock((b, c)), TextBlock((d,)), original[-1])

    def translate(block):
        return TranslatedText(TranslationRequest.from_block(block), "English " + block.text)

    original_results = tuple(map(translate, original))
    locked = store.update(original, original_results)
    first, middle, last = map(translate, proposed[:3])
    new_menu = TextBlock((replace(menu, text="戻る", text_revision=2),))
    proposed = (*proposed[:3], new_menu)
    for received in ((middle,), (last,), (translate(new_menu),)):
        screen = store.update(proposed, received)
        assert screen[:2] == locked[:2]
    assert screen[-1].translation == translate(new_menu)
    screen = store.update(proposed, (first, original_results[0]))
    assert tuple(p.translation for p in screen) == (first, middle, last, translate(new_menu))
    # Retirement removes exactly the affected publication, including late replies.
    screen = store.update((proposed[0], *proposed[2:]), (middle,))
    assert tuple(p.translation for p in screen) == (first, last, translate(new_menu))


def test_publication_retention_cannot_cross_backend_generations_or_worker_restarts():
    store = EnglishPresentation()
    block = TextBlock((TextTrack(1, "設定", BOX),))
    old = TranslatedText(TranslationRequest.from_block(block), "E4B fixture settings")
    store.update((block,), (old,), scope=("worker1", 0))
    assert store.update((block,), (), scope=("worker1", 0))[0].translation == old
    assert not store.update((block,), (), scope=("worker1", 1))
    new = replace(old, english="Gemma settings")
    assert store.update((block,), (new,), scope=("worker1", 1))[0].translation == new
    assert not store.update((block,), (), scope=("worker2", 0))


def test_reply_for_partially_returned_paragraph_waits_through_a_revision_and_regroup():
    store = EnglishPresentation()
    first = TextTrack(1, "次の冒険に出ましょう。", BOX)
    second = TextTrack(2, "装備が必要です。", replace(BOX, y=66))
    old = TextBlock((first, second))
    reply = TranslatedText(TranslationRequest.from_block(old), "Old paragraph")
    assert store.update((old,), (reply,))
    assert not store.update((old,), (), cleared=frozenset(old.lines))

    # The first row returns. A delayed OCR revision of the missing row forms a
    # different group whose translation is still pending.
    revised = replace(second, text="装備を確認します。", text_revision=2)
    third = replace(first, track_id=3, text="続けますか。", bounds=replace(BOX, y=102))
    new = TextBlock((first, revised, third))
    missing = frozenset((revised,))
    assert not store.update((new,), (), cleared=missing)
    translated = TranslatedText(TranslationRequest.from_block(new), "Revised paragraph")
    assert not store.update((new,), (translated,), cleared=missing)
    assert store.update((new,), ())[0].translation == translated


def test_stale_visual_evidence_cannot_lock_new_source_or_a_new_revision():
    tracker = TextTracker()
    first = result(1)
    tracker.observe(first)
    old = tracker.advance(first.frame, 1).visible[0]
    edit = result(2, text="防御力", gone=True)
    tracker.observe(
        edit,
        review=CandidateReview(unchanged=frozenset((old,)), checked=frozenset((old,)), sequence=1),
    )
    assert tracker.advance(edit.frame, 2).visible[0].text == "防御力"


def faint_result(sequence, *, text="HP 3204", gone=False, extension=False, noise=False):
    y, x = np.mgrid[:180, :400]
    gray = 204 + 12 * np.sin((x + sequence * 7) / 60) + 2 * np.cos((y + sequence) / 17)
    mask = np.zeros((180, 400), np.uint8)
    if not gone:
        cv2.putText(mask, text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    if extension:
        cv2.putText(mask, "MORE", (245, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    gray += (235 - gray) * (mask / 255) * 0.8
    if noise:
        gray += np.random.default_rng(sequence).uniform(-0.4, 0.4, gray.shape)
    original = result(sequence)
    return replace(
        original,
        frame=replace(
            original.frame, bgra=cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGRA).tobytes()
        ),
    )


@pytest.mark.parametrize("noise", [False, True])
def test_faint_stationary_text_retains_exact_english_over_moving_light(noise):
    replay = Replay(faint_result(1, noise=noise))
    original = replay.tracks
    presentation, grouper = EnglishPresentation(), DialogueGrouper()
    block = TextBlock(original)
    english = TranslatedText(TranslationRequest.from_block(block), "Attack")
    presentation.update((block,), (english,))
    for sequence in range(2, 40):
        incoming = faint_result(sequence, noise=noise)
        raw = incoming.regions[0]
        incoming = replace(
            incoming, regions=(replace(raw, text="誤った観測"),) if sequence % 4 else ()
        )
        state = replay.step(incoming)
        assert state.review.unchanged == set(original)
        assert not state.hidden
        assert replay.tracks == original
        grouped = grouper.update(
            replay.tracks,
            replay.tracks,
            incoming.completed_at,
            activity=replay.tracker.snapshot().activity,
        )
        painted = presentation.update(grouped.blocks, ())
        assert len(painted) == 1 and painted[0].translation == english


@pytest.mark.parametrize("text", ["HP 3205", "HP 3201", "HP 320", "HP 9204"])
def test_faint_real_digit_changes_are_not_frozen(text):
    replay = Replay(faint_result(1))
    incoming = faint_result(2, text=text)
    incoming = replace(incoming, regions=(replace(incoming.regions[0], text="防御力"),))
    assert not replay.step(incoming).review.unchanged
    assert replay.tracks[0].text == "防御力"


def test_faint_extension_unlocks_and_real_removal_retires():
    replay = Replay(faint_result(1))
    incoming = faint_result(2, extension=True)
    incoming = replace(incoming, regions=(replace(incoming.regions[0], text="攻撃力を強化"),))
    assert not replay.step(incoming).review.unchanged
    assert replay.tracks[0].text == "攻撃力を強化"
    for sequence in range(3, 7):
        state = replay.step(replace(faint_result(sequence, gone=True), regions=()))
        assert not state.review.unchanged
    assert not replay.tracks


@pytest.mark.parametrize("score", [0.55, 0.96])
def test_detector_splits_cannot_duplicate_or_truncate_visually_unchanged_line(score):
    replay = Replay(result(1, text="新しい冒険が始まります", score=score))
    original = replay.tracks
    first, last = Rect(20, 30, 175, 28), Rect(198, 30, 42, 28)
    for sequence in range(2, 10):
        pieces = (
            TextRegion("新しい冒険が", first, (TextCharacter("新しい冒険が", first, 0.99),)),
            TextRegion("始まります", last, (TextCharacter("始まります", last, 0.99),)),
        )
        incoming = replace(result(sequence), regions=pieces if sequence % 2 else pieces[::-1])
        assert replay.step(incoming).review.unchanged == set(original)
        assert replay.tracks == original
        assert sum(key is not None for key in replay.ids) == 1


def test_fragment_suppression_needs_current_visual_proof_and_allows_outside_content():
    first = result(1, text="新しい冒険が始まります")
    tracker = TextTracker()
    tracker.observe(first)
    track = tracker.advance(first.frame, 1).confirmed[0]
    pieces = (
        TextRegion("新しい冒険が", Rect(20, 30, 105, 28)),
        TextRegion("始まります", Rect(130, 30, 110, 28)),
        TextRegion("設定", Rect(280, 100, 60, 28)),
    )
    incoming = replace(result(2), regions=pieces)
    # Evidence from another source cannot suppress new text tracks.
    assignments = tracker.observe(
        incoming,
        review=CandidateReview(
            unchanged=frozenset((track,)), checked=frozenset((track,)), sequence=1
        ),
    )
    assert all(key is not None for key in assignments)
    replay = Replay(first)
    replay.step(incoming)
    assert replay.tracks[0] == track
    assert {t.text for t in replay.tracks} == {track.text, "設定"}
