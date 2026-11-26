"""Replay the real scene policy without Qt, native capture, or inference threads."""

import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest

from dantetl.application.live_session import LiveSession
from dantetl.domain import (
    CapturedFrame,
    CaptureSnapshot,
    CaptureState,
    Rect,
    WindowGeometry,
    WindowTarget,
)
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.ocr.observation import CandidateReview, TrackPresence
from dantetl.ocr.tracking import prepare_matching
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextCharacter, TextRegion
from dantetl.ocr.visual import VisualSession
from dantetl.ocr.visual_values import MenuSample, VisualState
from dantetl.translation.dialogue.classifier import Classification
from dantetl.translation.values import (
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)


def frame(sequence, now=1.0):
    bounds = Rect(0, 0, 1200, 900)
    return CapturedFrame(sequence, now, WindowGeometry(bounds, bounds, bounds, 96), b"")


class Worker:
    finished = False

    def __init__(self):
        self.stop_requested = False
        self.invalidations = 0

    def start(self):
        pass

    def request_stop(self):
        self.stop_requested = True

    def invalidate(self):
        self.invalidations += 1


class Capture(Worker):
    def set_border(self, enabled):
        self.show_border = enabled

    target = WindowTarget(1, 2, "Fixture")
    current = CaptureSnapshot(CaptureState.RUNNING)

    def snapshot(self):
        return self.current


class Ocr(Worker):
    current = OcrSnapshot(OcrState.READY)

    def __init__(self):
        super().__init__()
        self.acknowledged = []
        self.submitted = []

    def snapshot(self):
        return self.current

    def acknowledge(self, sequence):
        self.acknowledged.append(sequence)

    def submit(self, frame, areas=None):
        self.submitted.append((frame, areas))


class Visual(Worker):
    current = VisualState()

    def __init__(self, ocr=None, *, language=DEFAULT_SOURCE):
        super().__init__()
        self.ocr = ocr
        self.language = language
        self.sources = ()

    def invalidate(self, *, language=DEFAULT_SOURCE):
        super().invalidate()
        self.language = language

    def poll(self, latest_sequence=None, now=None):
        if (
            self.ocr
            and (result := self.ocr.current.result)
            and self.current.review.sequence == result.frame.sequence
        ):
            self.current = replace(
                self.current,
                review=replace(
                    self.current.review,
                    matching=prepare_matching(result, self.sources, language=self.language),
                ),
            )
        return self.current

    def submit(
        self,
        frame,
        result,
        tracks,
        assignments,
        activity,
        candidate=None,
        evidence_bounds=(),
        matching_sources=(),
    ):
        self.sources = matching_sources
        self.submitted = (frame, result, tracks, assignments, activity, candidate)


class Replay:
    def __init__(self, *, language=DEFAULT_SOURCE):
        self.ocr, self.capture = Ocr(), Capture()
        self.visual = Visual(self.ocr, language=language)
        self.live = LiveSession(
            ocr_factory=lambda _: self.ocr, visual_factory=lambda: self.visual, language=language
        )
        self.live.start(self.capture, Mock())

    def observe(
        self,
        source,
        regions,
        *,
        scanned=None,
        checked=frozenset(),
        unchanged=frozenset(),
        absent=frozenset(),
        settle=0.6,
        **evidence,
    ):
        self.ocr.current = OcrSnapshot(
            OcrState.READY, OcrResult(source, regions, 25, source.captured_at, scanned)
        )
        self.visual.current = VisualState(
            frame=source,
            review=CandidateReview(source.sequence, checked, unchanged),
            presence=TrackPresence(source.sequence, absent),
            ocr_sequence=source.sequence,
            **evidence,
        )
        self.live.reconcile(source, source.captured_at)
        if self.live.accepted != self.ocr.current.result:
            # Static settling may have advanced a revision since the worker's
            # previous matching snapshot. Return evidence for the refreshed tracks.
            self.live.reconcile(source, source.captured_at)
        self.live.reconcile(source, source.captured_at + settle)
        return self.live.scene(source, source.captured_at + settle, grouping=True)


def test_first_cached_label_withdraws_on_visual_growth_before_ocr_reads_the_suffix():
    replay = Replay()
    source = frame(1)
    rows = (
        TextRegion("設定", Rect(40, 40, 70, 30), line_confidence=0.99),
        TextRegion("装備", Rect(300, 600, 70, 30), line_confidence=0.45),
    )
    scene = replay.observe(source, rows, settle=0)
    assert not scene.requests
    (label,) = scene.candidates
    assert label.text == "設定"
    hit = TranslatedText(TranslationRequest.from_block(label), "Settings")
    backend = Mock()
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY, cached_candidates=(hit,)
    )

    def paint(scene):
        return replay.live.publish(
            backend, scene.requests, scene.blocks, candidates=scene.candidates
        )

    assert paint(scene).translations == (hit,)
    for seq, blocked in [(2, {label.lines[0].track_id}), (3, set())]:
        latest = frame(seq, 1 + seq * 0.05)
        replay.visual.current = replace(
            replay.visual.current, frame=latest, blocked=frozenset(blocked)
        )
        replay.live.reconcile(latest, latest.captured_at)
        scene = replay.live.scene(latest, latest.captured_at, grouping=True)
        assert not scene.candidates and not scene.requests
        assert not paint(scene).translations


def test_wrapping_invalidates_the_initial_candidate_until_the_whole_group_is_ready():
    replay = Replay()
    first = TextRegion(
        "新しい冒険がここから始まりますが、", Rect(40, 400, 460, 30), line_confidence=0.99
    )
    second = TextRegion(
        "その前に旅の準備をしてください。", Rect(40, 440, 420, 30), line_confidence=0.99
    )
    initial = replay.observe(frame(1), (first,), settle=0)
    assert tuple(b.text for b in initial.candidates) == (first.text,)
    wrapped = replay.observe(frame(2, 1.1), (first, second), settle=0)
    assert len(wrapped.blocks) == 1 and not wrapped.requests and not wrapped.candidates
    done = replay.observe(frame(3, 1.5), (first, second), settle=0)
    assert tuple(b.text for b in done.requests) == (first.text + second.text,)


def test_reconcile_waits_for_source_evidence_then_preserves_unscanned_tracks():
    replay = Replay()
    first = frame(1)
    label = TextRegion("設定", Rect(50, 40, 100, 30))
    replay.ocr.current = OcrSnapshot(OcrState.READY, OcrResult(first, (label,), 25, 1.0))
    replay.visual.current = VisualState(review=CandidateReview(sequence=0), frame=first)
    replay.live.reconcile(first, 1.0)
    assert replay.live.accepted is None and not replay.ocr.acknowledged
    scene = replay.observe(first, (label,))
    assert replay.live.accepted is replay.ocr.current.result
    assert replay.visual.submitted[1] is replay.ocr.current.result
    assert replay.ocr.acknowledged[-1] == 1
    old = scene.tracks[0]
    second = frame(2, 1.5)
    new = TextRegion("戻る", Rect(600, 40, 100, 30))
    scene = replay.observe(second, (new,), scanned=(Rect(580, 20, 160, 80),))
    assert old in scene.tracks and len(scene.tracks) == 2
    assert replay.live.accepted and replay.live.accepted.regions == (new,)
    # A noisy full pass is held by source-matched visual evidence, not inspector state.
    scene = replay.observe(
        frame(3, 2.0), (), unchanged=frozenset(scene.tracks), checked=frozenset(scene.tracks)
    )
    assert len(scene.tracks) == 2


def test_completion_bursts_do_not_count_as_new_observations_or_change_english():
    replay = Replay()
    source = frame(1)
    regions = (TextRegion("設定", Rect(50, 40, 100, 30)),)
    scene = replay.observe(source, regions)
    backend = Mock()
    result = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Settings")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (result,))
    english = replay.live.publish(backend, scene.requests, scene.blocks)
    for _ in range(100):
        replay.live.reconcile(source, 1.6)
        assert replay.live.scene(source, 1.6, grouping=True) == scene
        assert replay.live.publish(backend, scene.requests, scene.blocks) == english
    assert replay.live.accepted is not None and replay.live.accepted.frame is source
    assert replay.live._tracker._sequence == 1


def test_regional_icon_and_full_menu_reads_publish_one_light_cone_label():
    replay = Replay()
    backend = Mock()
    clean = TextRegion("光円錐", Rect(434, 200, 76, 32))
    wide = TextRegion("7光円錐", Rect(332, 193, 179, 46))
    scene = replay.observe(frame(1), (clean,))
    (block,) = scene.blocks
    english = TranslatedText(TranslationRequest.from_block(block), "Light Cone")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
    for seq, rows in enumerate(((wide, clean), (clean,), (clean, wide)), 2):
        scene = replay.observe(
            frame(seq, float(seq)), rows, scanned=(wide.bounds,) if seq % 2 else None
        )
        assert replay.live.accepted and replay.live.accepted.regions == rows
        assert scene.requests == (block,)
        painted = replay.live.publish(backend, scene.requests, scene.blocks)
        assert painted.blocks == (block,)
        assert painted.translations == (english,)


def test_full_and_cropped_soul_star_readings_share_one_publication():
    # Actual character boxes and scores from full and sidebar OCR of the supplied 4K frame.
    damaged = TextRegion(
        "*魂",
        Rect(332, 1028, 202, 58),
        (
            TextCharacter("*", Rect(334, 1028, 73, 58), 0.823),
            TextCharacter("魂", Rect(476, 1028, 55, 58), 0.946),
        ),
    )
    clean = TextRegion(
        "星魂",
        Rect(432, 1032, 97, 45),
        (
            TextCharacter("星", Rect(434, 1032, 49, 45), 0.979),
            TextCharacter("魂", Rect(481, 1032, 48, 45), 0.968),
        ),
    )
    screen = Rect(0, 0, 3840, 2160)
    geometry = WindowGeometry(screen, screen, screen, 168)
    replay = Replay()
    scene = replay.observe(CapturedFrame(1, 1, geometry, b""), (damaged,))
    (old,) = scene.tracks
    backend = Mock()
    old_english = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Soul")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (old_english,))
    assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (old_english,)
    for seq, rows in enumerate(
        ((clean,), (clean, damaged), (damaged,), (damaged,), (damaged, clean)), 2
    ):
        # The clean partial scan cannot cover the old icon. Its pixels can still
        # be unchanged, the very evidence that used to keep the second track alive.
        scene = replay.observe(
            CapturedFrame(seq, seq, geometry, b""),
            rows,
            scanned=(Rect(410, 980, 210, 200),) if seq == 2 else None,
            unchanged=frozenset((old,)) if seq == 2 else frozenset(),
            checked=frozenset((old,)) if seq == 2 else frozenset(),
        )
        (track,) = scene.tracks
        assert (track.track_id, track.text, track.bounds) == (
            old.track_id,
            "星魂",
            Rect(434, 1032, 95, 45),
        )
        (block,) = scene.requests
        english = TranslatedText(TranslationRequest.from_block(block), "Soul Star")
        if seq == 2:
            assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (
                old_english,
            )
        backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
        painted = replay.live.publish(backend, scene.requests, scene.blocks)
        assert painted.blocks == (block,) and painted.translations == (english,)


def test_exact_menu_reappearance_paints_before_ocr_and_never_supplies_a_translation_job():
    replay = Replay()
    source = frame(1)
    region = TextRegion("設定", Rect(50, 40, 100, 30))
    scene = replay.observe(source, (region,))
    sample = MenuSample(scene.tracks[0], 1, Rect(10, 5, 180, 110))
    backend = Mock()
    english = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Settings")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
    replay.visual.current = replace(replay.visual.current, matching_menus=(sample,))
    replay.live.reconcile(source, 1.6)
    replay.live.publish(backend, scene.requests, scene.blocks)
    # Completed misses retire the original track before reopening.
    for seq in range(2, 7):
        scene = replay.observe(frame(seq, float(seq)), ())
    assert not scene.tracks
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY)
    returned = frame(7, 7)
    replay.visual.current = VisualState(frame=returned, matching_menus=(sample,))
    replay.live.reconcile(returned, 7)
    scene = replay.live.scene(returned, 7, grouping=True)
    assert not scene.requests and not scene.blocks
    preview = replay.live.publish(backend, scene.requests, scene.blocks)
    assert len(preview.translations) == 1 and preview.translations[0].english == "Settings"
    assert preview.blocks[0].bounds == region.bounds and preview.blocks[0].lines[0].track_id < 0
    assert not backend.update.call_args.args[0]
    replay.live.scene(frame(8, 7.01), 7.01, grouping=True)
    assert not replay.live.publish(backend, (), ()).blocks  # New capture awaiting its pixel check.
    replay.live.scene(returned, 7, grouping=True)
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY, backend_generation=1
    )
    assert not replay.live.publish(backend, (), ()).blocks


@pytest.mark.parametrize("failure", ["prepare_matching", "prepare_evidence"])
def test_review_failure_releases_ocr_and_recovers_on_the_next_result(monkeypatch, failure):
    from dantetl.ocr import visual

    original = getattr(visual, failure)

    def fault(result, *args, **kwargs):
        if result.frame.sequence == 2:
            raise ValueError("fixture review failure")
        return original(result, *args, **kwargs)

    monkeypatch.setattr(visual, failure, fault)
    ocr, capture, analysis = Ocr(), Capture(), VisualSession()
    live = LiveSession(ocr_factory=lambda _: ocr, visual_factory=lambda: analysis)
    live.start(capture, Mock())
    label = TextRegion("設定", Rect(50, 40, 100, 30))
    try:
        for seq in (1, 2, 3):
            source = frame(seq, float(seq))
            ocr.current = OcrSnapshot(OcrState.READY, OcrResult(source, (label,), 25, seq))
            deadline = time.monotonic() + 5
            while seq not in ocr.acknowledged:
                assert not analysis.finished and time.monotonic() < deadline
                live.reconcile(source, source.captured_at)
                time.sleep(0.002)
            assert live.accepted is not None
            rejected = seq == 2 and failure == "prepare_matching"
            assert live.accepted.frame.sequence == (1 if rejected else seq)
            assert bool(live.visual_state.review_message) == (seq == 2)
            if rejected:
                assert (source, None) in ocr.submitted
            assert live.scene(source, source.captured_at, grouping=True).tracks
            live.reconcile(source, source.captured_at)
        assert ocr.acknowledged == [1, 2, 3]
    finally:
        live.stop()
        capture.finished = ocr.finished = True
        deadline = time.monotonic() + 5
        while not live.drain():
            assert time.monotonic() < deadline
            time.sleep(0.002)


def test_visual_loss_between_ocr_passes_keeps_locked_dialogue_and_backend_scope():
    replay = Replay()
    regions = (
        TextRegion("ここから先は危険だから、", Rect(100, 650, 380, 35)),
        TextRegion("私と一緒に進んでください。", Rect(100, 695, 380, 35)),
    )
    replay.observe(frame(1), regions)
    scene = replay.observe(frame(2, 1.5), regions)
    assert len(scene.blocks) == 1 and len(scene.blocks[0].lines) == 2
    assert scene.requests == scene.blocks
    backend = Mock()
    result = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Come with me.")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (result,))
    published = replay.live.publish(backend, scene.requests, scene.blocks)
    assert published.translations == (result,)
    replay.visual.current = replace(replay.visual.current, hidden=frozenset(scene.tracks))
    replay.live.reconcile(frame(2, 1.5), 2.0)
    scene = replay.live.scene(frame(2, 1.5), 2.0, grouping=True)
    assert len(scene.tracks) == 2
    assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (result,)
    replay.visual.current = replace(replay.visual.current, hidden=frozenset())
    replay.live.reconcile(frame(2, 1.5), 2.1)
    assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (result,)
    backend.snapshot.return_value = TranslationSnapshot(
        TranslationState.READY, backend_generation=1
    )
    assert not replay.live.publish(backend, scene.requests, scene.blocks).translations
    assert replay.live.accepted and replay.live.accepted.regions == regions


def test_vacated_group_hides_locally_and_restores_cached_english_without_new_ocr():
    replay, backend = Replay(), Mock()
    regions = (
        TextRegion("ここから先は危険だから、", Rect(100, 650, 380, 35)),
        TextRegion("私と一緒に進んでください。", Rect(100, 695, 380, 35)),
        TextRegion("設定", Rect(850, 100, 100, 35)),
    )
    replay.observe(frame(1), regions)
    scene = replay.observe(frame(2, 1.5), regions)
    dialogue = next(b for b in scene.blocks if len(b.lines) == 2)
    label = next(b for b in scene.blocks if len(b.lines) == 1)
    english = tuple(
        TranslatedText(TranslationRequest.from_block(b), text)
        for b, text in ((dialogue, "Come with me."), (label, "Settings"))
    )
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, english)
    locked = replay.live.publish(backend, scene.requests, scene.blocks)
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY)
    a, b = dialogue.lines
    for seq, cleared, matches, visible in (
        (3, (a,), (b,), locked.blocks),
        (4, (a, replace(b, text_revision=b.text_revision + 1)), (), locked.blocks),
        (5, (a, b), (), (label,)),
        (6, (b,), (a,), (label,)),
        (7, (), (), (label,)),
        (8, (), (a, replace(b, bounds=replace(b.bounds, x=b.bounds.x + 1))), (label,)),
        (9, (), (a, b), locked.blocks),
    ):
        current = frame(seq, seq / 8 + 1.5)
        replay.visual.current = replace(
            replay.visual.current,
            frame=current,
            cleared=frozenset(cleared),
            matching_lines=frozenset(matches),
        )
        replay.live.reconcile(current, current.captured_at)
        observed = replay.live.scene(current, current.captured_at, grouping=True)
        assert observed.blocks == scene.blocks
        hidden_lines = (b,) if seq in (7, 8) else cleared
        assert observed.tracks == tuple(t for t in scene.tracks if t not in hidden_lines)
        shown = replay.live.publish(backend, observed.requests, observed.blocks)
        assert shown.blocks == visible
        assert shown.translations == ((english[1],) if visible == (label,) else locked.translations)
        assert backend.update.call_args.kwargs["retain"] == scene.blocks
    assert replay.ocr.acknowledged == [1, 2]


def test_cached_translation_for_a_revised_dismissed_line_waits_for_its_current_glyphs():
    replay, backend = Replay(), Mock()
    bounds = Rect(50, 40, 100, 30)
    first = replay.observe(frame(1), (TextRegion("設定", bounds),))
    old = first.tracks[0]
    english = TranslatedText(TranslationRequest.from_block(first.blocks[0]), "Settings")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
    assert replay.live.publish(backend, first.requests, first.blocks).translations == (english,)

    closed = frame(4, 2.0)
    replay.visual.current = replace(replay.visual.current, frame=closed, cleared=frozenset((old,)))
    replay.live.reconcile(closed, closed.captured_at)
    assert not replay.live.publish(backend, first.requests, first.blocks).translations

    # An edit from before closure finishes now. The new translation is already cached.
    for sequence in (2, 3):
        edited = OcrResult(
            frame(sequence, 1.5 + sequence / 10), (TextRegion("装備", bounds),), 400, 2.2
        )
        replay.ocr.current = OcrSnapshot(OcrState.READY, edited)
        replay.visual.current = replace(replay.visual.current, review=CandidateReview(sequence))
        replay.live.reconcile(closed, 2.2)
    scene = replay.live.scene(closed, 2.2, grouping=True)
    revised = scene.blocks[0].lines[0]
    assert revised.track_id == old.track_id and revised.text == "装備"
    new = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Equipment")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (new,))
    assert (
        not scene.tracks
        and not replay.live.publish(backend, scene.requests, scene.blocks).translations
    )
    for seq, matches, visible in ((5, (old,), False), (6, (revised,), True)):
        current = frame(seq, 2 + seq / 10)
        replay.visual.current = replace(
            replay.visual.current,
            frame=current,
            cleared=frozenset(),
            matching_lines=frozenset(matches),
        )
        replay.live.reconcile(current, current.captured_at)
        scene = replay.live.scene(current, current.captured_at, grouping=True)
        assert bool(scene.tracks) == visible
        assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (
            (new,) if visible else ()
        )


@pytest.mark.parametrize("verified_pixels", [False, True])
def test_similar_ocr_keeps_group_english_until_exact_replacement_arrives(verified_pixels):
    replay, backend = Replay(), Mock()
    top = Rect(100, 650, 380, 35)
    bottom = Rect(100, 695, 380, 35)

    def row(text, bounds, score):
        return TextRegion(text, bounds, (TextCharacter(text, bounds, score),))

    first = row("ここから先は危険だから、", top, 0.98)
    source = "私と一緒に（奥まで）進んでください。"
    typo = source.replace("）", "")
    original = row(typo, bottom, 0.94)
    scene = replay.observe(frame(1), (first, original))
    assert len(scene.blocks) == 1 and len(scene.blocks[0].lines) == 2
    old = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Come with me.")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (old,))

    def publish():
        return replay.live.publish(backend, scene.requests, scene.blocks)

    assert publish().translations == (old,)
    original_tracks = scene.tracks
    same_pixels = frozenset(original_tracks) if verified_pixels else frozenset()
    for seq, score in enumerate((0.90, 0.94, 0.85), 2):
        scene = replay.observe(
            frame(seq, float(seq)),
            (first, row(source, bottom, score)),
            checked=frozenset(original_tracks),
            unchanged=same_pixels,
        )
        assert publish().translations == (old,)
        assert TranslationRequest.from_block(scene.requests[0]) == old.request
    scene = replay.observe(
        frame(5, 5.0),
        (first, row(source, bottom, 0.943)),
        checked=frozenset(original_tracks),
        unchanged=same_pixels,
    )
    assert publish().translations == (old,)
    request = TranslationRequest.from_block(scene.requests[0])
    assert request.source == first.text + source and request != old.request

    # The backend can drop the obsolete request immediately. A later geometry
    # update still moves the old English, with its original exact source attached.
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, pending=1)
    scene = replay.observe(
        frame(6, 8.0),
        (row(first.text, replace(top, x=125), 0.98), row(source, replace(bottom, x=125), 0.941)),
        hidden=frozenset(original_tracks),
    )
    retained = publish()
    assert retained.translations == (old,) and retained.blocks[0].bounds.x == 125
    assert TranslationRequest.from_block(retained.blocks[0]) == old.request
    assert TranslationRequest.from_block(scene.requests[0]) == request
    replacement = TranslatedText(request, "Come with me all the way inside.")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (replacement,))
    assert publish().translations == (replacement,)
    scene = replay.observe(frame(7, 10), (first, row("ここが目的地です。", bottom, 0.71)))
    assert publish().translations == (replacement,)


@pytest.mark.parametrize("stale", [False, True])
def test_only_current_pattern_loss_can_start_a_new_weaker_similar_reading(stale):
    replay = Replay()
    box = Rect(100, 650, 380, 35)
    east, west = "東の門へ向かってください", "西の門へ向かってください"
    original = TextRegion(east, box, (TextCharacter(east, box, 0.99),))
    changed = TextRegion(west, box, (TextCharacter(west, box, 0.8),))
    scene = replay.observe(frame(1), (original,))
    old = scene.tracks[0]
    lost = replace(old, text_revision=0) if stale else old
    scene = replay.observe(frame(2, 2), (changed,), absent=frozenset((lost,)))
    assert scene.tracks[0].text == (east if stale else west)


def test_history_observes_complete_publications_without_changing_model_work():
    replay = Replay()
    replay.live.configure_history(True, "layout", layout_only=True)
    speaker = Rect(100, 605, 60, 27)
    regions = (
        TextRegion("旅人", speaker, (TextCharacter("旅人", speaker, 0.98),)),
        TextRegion("ここから先は危険だから、", Rect(100, 650, 380, 35)),
        TextRegion("私と一緒に進んでください。", Rect(100, 695, 380, 35)),
    )
    replay.observe(frame(1), regions)
    scene = replay.observe(frame(2, 1.5), regions)
    backend = Mock()
    results = tuple(
        TranslatedText(
            TranslationRequest.from_block(b), "Traveler" if b.text == "旅人" else "Fixture English"
        )
        for b in scene.requests
    )
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, results)
    replay.live.publish(backend, scene.requests, scene.blocks)
    replay.live.scene(frame(2, 1.5), 2.7, grouping=True)
    published = replay.live.publish(backend, scene.requests, scene.blocks)
    assert len(replay.live.history.turns) == 1
    turn = replay.live.history.turns[0]
    assert turn.speaker == "旅人" and turn.source == regions[1].text + regions[2].text
    assert turn.speaker_english == "Traveler"
    expected = backend.update.call_args
    replay.live.history.clear()
    assert replay.live.publish(backend, scene.requests, scene.blocks) == published
    assert backend.update.call_args == expected  # No extra sources or prompt-context reset.
    assert replay.live.history.turns == ()
    replay.live.stop(backend)
    assert replay.live.history.turns == ()


def test_classifier_never_holds_translation_and_swap_disable_drain_the_owner():
    replay = Replay()
    entered, release = threading.Event(), threading.Event()
    events = []

    class Engine:
        def __init__(self, key):
            self.key = key
            events.append((key, "load"))

        def classify(self, value, cancelled):
            if self.key == "old":
                entered.set()
                assert release.wait(3)
                assert cancelled.is_set()
            return Classification("dialogue", "no")

        def close(self):
            events.append((self.key, "close"))

    regions = (
        TextRegion("ここから先は危険です。私と一緒に進んでください。", Rect(100, 650, 700, 35)),
    )
    replay.observe(frame(1), regions)
    scene = replay.observe(frame(2, 1.5), regions)
    backend = Mock()
    english = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Come with me.")
    backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
    assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (english,)
    assert not replay.live.history.turns and not events  # default-off really does no work
    replay.live.configure_history(True, "old", lambda: Engine("old"))
    replay.live.scene(frame(2, 1.5), 2.1, grouping=True)
    replay.live.scene(frame(2, 1.5), 2.7, grouping=True)
    assert entered.wait(2)
    try:
        for _ in range(30):
            assert replay.live.publish(backend, scene.requests, scene.blocks).translations == (
                english,
            )
        assert not replay.live.history.turns
        replay.live.configure_history(True, "new", lambda: Engine("new"))
        replay.live.scene(frame(2, 1.5), 2.8, grouping=True)
        assert events == [("old", "load")]
        release.set()
        deadline = time.monotonic() + 3
        while not replay.live.history.turns:
            assert time.monotonic() < deadline
            replay.live.scene(frame(2, 1.5), 3.5, grouping=True)
            replay.live.publish(backend, scene.requests, scene.blocks)
            time.sleep(0.005)
        assert events[:3] == [("old", "load"), ("old", "close"), ("new", "load")]
        replay.live.configure_history(False, "new", lambda: Engine("new"))
        assert replay.live.history_status.startswith("Disabled")
    finally:
        release.set()
        replay.live.stop()
        replay.capture.finished = replay.ocr.finished = replay.visual.finished = True
        deadline = time.monotonic() + 3
        while not replay.live.drain():
            assert time.monotonic() < deadline
            time.sleep(0.005)
    assert events[-1] == ("new", "close")
    assert len(replay.live.history.turns) == 1  # Disabling keeps the readable transcript.


def test_resize_reset_and_stop_wait_for_workers_before_another_capture():
    replay = Replay()
    replay.observe(frame(1), (TextRegion("設定", Rect(50, 40, 100, 30)),))
    backend = Mock()
    replay.live.accept_size((1200, 900), backend)
    assert replay.live.accepted is None
    backend.invalidate.assert_called_once()
    assert not replay.live.accept_size((1200, 900), backend)
    replay.live.stop(backend)
    assert all(w.stop_requested for w in (replay.capture, replay.ocr, replay.visual))
    replay.capture.finished = True
    assert not replay.live.drain()
    with pytest.raises(RuntimeError, match="previous capture"):
        replay.live.start(Capture(), None)
    replay.ocr.finished = True
    assert not replay.live.drain()
    replay.visual.finished = True
    assert replay.live.drain() and replay.live.capture is None
    replay.live.start(Capture(), None)
    assert (
        replay.live.accepted is None and not replay.live.scene(frame(10), 5, grouping=True).tracks
    )


def test_scene_change_resets_dialogue_once_and_stopping_rejects_late_ocr_setup():
    replay = Replay()
    source = frame(1)
    replay.observe(source, (), changed_areas=(Rect(0, 0, 1200, 900),))
    translator = Mock()
    translator.snapshot.return_value = TranslationSnapshot(TranslationState.READY)
    replay.live.publish(translator, (), ())
    assert translator.update.call_args.kwargs["reset_context"]
    replay.live.publish(translator, (), ())
    assert not translator.update.call_args.kwargs["reset_context"]
    capture = Capture()
    capture.current = CaptureSnapshot(CaptureState.STOPPING)
    factory = Mock()
    live = LiveSession(ocr_factory=factory)
    live.start(capture, None)
    live.add_recognition(Mock())
    factory.assert_not_called()


def test_compatible_source_switch_starts_fresh_history_for_same_visible_line():
    from dantetl.languages.selection import source_language

    replay = Replay(language=source_language("en"))
    replay.live.configure_history(True, "layout", layout_only=True)
    box = Rect(100, 605, 70, 27)
    regions = (
        TextRegion("Alex", box, (TextCharacter("Alex", box, 0.98),)),
        TextRegion("Attention!", Rect(100, 650, 380, 35)),
    )
    backend = Mock()
    for index, code in enumerate(("en", "fr")):
        if index:
            replay.live.change_language(source_language(code), backend)
            assert not replay.live.history.turns
        for tick in range(3):
            seq = index * 3 + tick + 1
            scene = replay.observe(frame(seq, float(seq)), regions)
            backend.snapshot.return_value = TranslationSnapshot(
                TranslationState.READY,
                tuple(
                    TranslatedText(TranslationRequest.from_block(b), b.text) for b in scene.requests
                ),
            )
            replay.live.publish(backend, scene.requests, scene.blocks)
        (turn,) = replay.live.history.turns
        assert turn.source == "Attention!" and turn.language == source_language(code)
