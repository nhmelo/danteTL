"""Controlled capture, OCR and visual workers for explicit-time LiveSession replays."""

from dataclasses import replace
from unittest.mock import Mock

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
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState
from dantetl.ocr.visual_values import VisualState


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
