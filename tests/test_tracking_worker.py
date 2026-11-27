"""Completed dense observations must not perform hashing/scoring on the GUI owner."""

import threading
import time
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QTimer
from test_live_session import Capture, Ocr

from dantetl.application.live_session import LiveSession
from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr import presence, visual
from dantetl.ocr.observation import CandidateReview
from dantetl.ocr.tracking import TextTracker, prepare_matching
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextRegion
from dantetl.ocr.visual import VisualSession
from dantetl.ocr.visual_values import VisualState
from dantetl.translation.values import (
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)


def observation(sequence, *, changed=False):
    bounds = Rect(0, 0, 3840, 2160)
    source = CapturedFrame(
        sequence,
        sequence * 0.5,
        WindowGeometry(bounds, bounds, bounds, 96),
        bytes((30, 30, 30, 255)) * (3840 * 2160),
    )
    rows = tuple(
        TextRegion(
            ("日本語の文章です。" * 24) + str(i) + ("次" if changed else ""),
            Rect(30 + (i % 4) * 950, 30 + (i // 4) * 60, 850, 30),
        )
        for i in range(128)
    )
    return OcrResult(source, rows, 900, source.captured_at + 0.9)


def test_stale_matching_cannot_mutate_tracks_or_consume_the_observation():
    tracker = TextTracker()
    first, second = observation(1), observation(2, changed=True)
    stale = prepare_matching(second, ())
    tracker.observe(first)
    previous = tracker.advance(first.frame, 2).visible
    with pytest.raises(ValueError, match="current source/state"):
        tracker.observe(second, review=CandidateReview(matching=stale))
    assert tracker.advance(first.frame, 2).visible == previous
    fresh = prepare_matching(second, tracker.matching_sources(second.frame))
    assert tracker.observe(second, review=CandidateReview(matching=fresh)) == tuple(
        t.track_id for t in previous
    )
    # A reset also invalidates a previously prepared state, even with the same frame.
    tracker.reset()
    with pytest.raises(ValueError, match="current source/state"):
        tracker.observe(second, review=CandidateReview(matching=fresh))


def test_completed_dense_ocr_waits_off_qt_and_retains_english(app, monkeypatch):
    owner = threading.get_ident()
    entered, release, block = threading.Event(), threading.Event(), threading.Event()
    scored, hashed = [], []
    original_score, original_hash = visual.prepare_matching, presence.blake2b

    def score(*args, **kwargs):
        scored.append(threading.get_ident())
        assert scored[-1] != owner
        if block.is_set():
            entered.set()
            assert release.wait(5)
        return original_score(*args, **kwargs)

    def digest(*args, **kwargs):
        hashed.append(threading.get_ident())
        assert hashed[-1] != owner
        return original_hash(*args, **kwargs)

    class Analyzer:
        def observe(self, frame, *args):
            return VisualState(frame=frame, ocr_sequence=args[0].frame.sequence if args[0] else 0)

    monkeypatch.setattr(visual, "VisualAnalyzer", lambda language: Analyzer())
    monkeypatch.setattr(visual, "prepare_matching", score)
    monkeypatch.setattr(presence, "blake2b", digest)
    ocr, worker = Ocr(), VisualSession()
    live = LiveSession(ocr_factory=lambda _: ocr, visual_factory=lambda: worker)
    live.start(Capture(), Mock())
    ticks = []
    timer = QTimer()
    timer.setInterval(1)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()

    def advance(result, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            app.processEvents()
            live.reconcile(result.frame, result.completed_at)
            assert time.monotonic() < deadline
            time.sleep(0.001)

    try:
        for sequence in (1, 2):
            result = observation(sequence)
            ocr.current = OcrSnapshot(OcrState.READY, result)
            advance(result, lambda result=result: live.accepted is result)
        scene = live.scene(result.frame, result.completed_at + 0.6, grouping=False)
        assert len(scene.tracks) == 128
        backend = Mock()
        english = TranslatedText(TranslationRequest.from_block(scene.blocks[0]), "Retained English")
        backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, (english,))
        assert live.publish(backend, scene.requests, scene.blocks).translations == (english,)
        block.set()
        incoming = observation(3, changed=True)
        ocr.current = OcrSnapshot(OcrState.READY, incoming)
        advance(incoming, entered.is_set)
        before = len(ticks)
        for _ in range(30):
            app.processEvents()
            live.reconcile(incoming.frame, incoming.completed_at)
            waiting = live.scene(incoming.frame, incoming.completed_at, grouping=False)
            assert waiting.tracks == scene.tracks
            assert live.publish(backend, waiting.requests, waiting.blocks).translations == (
                english,
            )
            assert live.accepted is result and ocr.acknowledged[-1] == 2
            time.sleep(0.002)
        assert len(ticks) > before + 5
        release.set()
        advance(incoming, lambda: live.accepted is incoming)
        assert ocr.acknowledged[-1] == 3 and scored and hashed
        assert all(thread != owner for thread in (*scored, *hashed))
    finally:
        release.set()
        timer.stop()
        worker.request_stop()
        deadline = time.monotonic() + 5
        while not worker.finished:
            assert time.monotonic() < deadline
            time.sleep(0.001)
