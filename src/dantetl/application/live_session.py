"""Live pipeline ownership and ordered reconciliation, independent of Qt and Win32.

Model preparation/selection belongs to the application lifecycle. An active translator
is supplied explicitly; this coordinator owns which sources it sees and what may paint.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from dantetl.application.values import Scene
from dantetl.domain import CapturedFrame, CaptureSnapshot, CaptureState, Rect, WindowTarget
from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.tracking import TextActivity, TextTracker, TrackingSnapshot
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState
from dantetl.ocr.visual import VisualSession
from dantetl.ocr.visual_values import VisualState
from dantetl.ocr.worker import OcrSession, Recognizer, overlay_result_ready
from dantetl.translation.dialogue.classifier import ClassifierSession
from dantetl.translation.dialogue.history import DialogueHistory
from dantetl.translation.first_reading import FirstReadings
from dantetl.translation.grouping import DialogueGrouper, TextBlock
from dantetl.translation.presentation import EnglishPresentation
from dantetl.translation.values import (
    TranslatedText,
    TranslationSnapshot,
    TranslationState,
    TranslationWorker,
)


class CaptureWorker(Protocol):
    target: WindowTarget

    @property
    def finished(self) -> bool: ...
    def start(self) -> None: ...
    def request_stop(self) -> None: ...
    def set_border(self, enabled: bool) -> None: ...
    def snapshot(self) -> CaptureSnapshot: ...


class OcrWorker(Protocol):
    @property
    def finished(self) -> bool: ...
    def start(self) -> None: ...
    def request_stop(self) -> None: ...
    def invalidate(self) -> None: ...
    def snapshot(self) -> OcrSnapshot: ...
    def acknowledge(self, sequence: int) -> None: ...
    def submit(self, frame: CapturedFrame, areas: tuple[Rect, ...] | None = None) -> None: ...


class VisualWorker(Protocol):
    @property
    def finished(self) -> bool: ...
    def start(self) -> None: ...
    def request_stop(self) -> None: ...
    def invalidate(self, *, language: SourceLanguage) -> None: ...
    def poll(self, latest_sequence: int, now: float) -> VisualState: ...
    def submit(
        self,
        frame: CapturedFrame,
        result: OcrResult | None,
        tracks: tuple[TextTrack, ...],
        assignments: tuple[int | None, ...],
        activity: tuple[TextActivity, ...],
        candidate: OcrResult | None = None,
        evidence_bounds: tuple[Rect, ...] = (),
        matching_sources: tuple[TextTrack, ...] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EnglishScene:
    snapshot: TranslationSnapshot
    blocks: tuple[TextBlock, ...]
    translations: tuple[TranslatedText, ...]


class LiveSession:
    """One bounded live scene, with commands hiding reset/reconcile/ack ordering."""

    def __init__(
        self,
        *,
        ocr_factory: Callable[[Callable[[], Recognizer]], OcrWorker] | None = None,
        visual_factory: Callable[[], VisualWorker] | None = None,
        language: SourceLanguage = DEFAULT_SOURCE,
    ) -> None:
        self._ocr_factory = ocr_factory or (lambda factory: OcrSession(factory, require_ack=True))
        self._language = language
        self._visual_factory = visual_factory or VisualSession
        self.capture: CaptureWorker | None = None
        self.ocr: OcrWorker | None = None
        self.visual: VisualWorker | None = None
        self.visual_state = VisualState()
        self.accepted: OcrResult | None = None
        self._consumed_sequence = 0
        self._ocr_failed = False
        self.assignments: tuple[int | None, ...] = ()
        self.valid_size: tuple[int, int] | None = None
        self._tracker = TextTracker(language=language)
        self.tracking = TrackingSnapshot()
        self._grouper = DialogueGrouper(language=language)
        self._first_readings = FirstReadings()
        self._presentation = EnglishPresentation()
        self._context_reset_sequence = 0
        self._scene_sequence = 0
        self._early_lines: frozenset[TextTrack] = frozenset()
        self.history = DialogueHistory()
        self.history_enabled = False
        self._history_language = language
        self._history_key: object = None
        self._history_factory: Callable[[], object] | None = None
        self._classifier: ClassifierSession | None = None
        self._classifier_retiring = False
        self._history_layout_only = False
        self._history_suspended = False

    def suspend_history(self, suspended: bool) -> bool:
        if suspended and not self._history_suspended:
            self.history.reset_scene()
        self._history_suspended = suspended
        if suspended and self._classifier and not self._classifier_retiring:
            self._classifier.request_stop()
            self._classifier_retiring = True
        self._advance_history()
        return self._classifier is None

    def configure_history(
        self,
        enabled: bool,
        key: object = "",
        factory: Callable[[], object] | None = None,
        *,
        layout_only: bool = False,
    ) -> None:
        if (enabled, key, layout_only) == (
            self.history_enabled,
            self._history_key,
            self._history_layout_only,
        ):
            if self._history_language != self._language:
                self._history_language = self._language
                self._history_factory = factory
                if self._classifier and not self._classifier_retiring and factory:
                    self._classifier.set_source(self._language, factory)
            return
        self._history_language = self._language
        self.history_enabled, self._history_key = enabled, key
        self._history_factory, self._history_layout_only = factory, layout_only
        self.history.reset_scene()
        if self._classifier:
            self._classifier.request_stop()
            self._classifier_retiring = True

    @property
    def history_status(self) -> str:
        if not self.history_enabled:
            return tr("Disabled · No history collection or classification.")
        if self._classifier_retiring:
            return tr("Finishing the previous classifier…")
        if self._history_layout_only:
            return tr("Layout heuristics only · No classifier calls.")
        if not self._history_factory:
            return tr(
                "Choose an available classifier model. For first use, select the local model "
                "once in a main-screen translation selector to prepare it."
            )
        if self._classifier:
            state = self._classifier.snapshot()
            return state.message + (
                tr(" · {pending} pending", pending=state.pending) if state.pending else ""
            )
        return tr("Ready · Start capture with translation enabled.")

    def _advance_history(self) -> None:
        if self._classifier_retiring and self._classifier and self._classifier.finished:
            self._classifier = None
            self._classifier_retiring = False
        if (
            self.history_enabled
            and self._history_factory
            and self.capture
            and not self._classifier
            and not self._history_suspended
        ):
            self._classifier = ClassifierSession(self._history_factory, language=self._language)

    def change_language(
        self, language: SourceLanguage, translation: TranslationWorker | None
    ) -> None:
        self._language = language
        self._tracker = TextTracker(language=language)
        self._grouper = DialogueGrouper(language=language)
        self.history.start()
        if self.ocr:
            self.ocr.invalidate()
        self.valid_size = None
        self.reset(translation)

    def start(
        self,
        capture: CaptureWorker,
        recognizer: Callable[[], Recognizer] | None,
        translation: TranslationWorker | None = None,
    ) -> None:
        if self.capture is not None:
            raise RuntimeError(tr("The previous capture must finish before starting another."))
        self.reset(translation)
        self.history.start()
        self.valid_size = None
        self.capture = capture
        capture.start()
        if recognizer:
            self.add_recognition(recognizer)

    def add_recognition(self, factory: Callable[[], Recognizer]) -> None:
        if (
            self.ocr
            or not self.capture
            or self.capture.finished
            or self.capture.snapshot().state
            in (CaptureState.STOPPING, CaptureState.STOPPED, CaptureState.FAILED)
        ):
            return
        self.ocr, self.visual = self._ocr_factory(factory), self._visual_factory()
        self.ocr.start()
        self.visual.invalidate(language=self._language)
        self.visual.start()

    def stop(self, translation: TranslationWorker | None = None) -> None:
        self.invalidate(translation)
        for worker in (self.ocr, self.visual, self.capture, self._classifier):
            if worker:
                worker.request_stop()

    def drain(self) -> bool:
        """A stopped source cannot be replaced while its workers still own resources."""
        if not any((self.capture, self.ocr, self.visual, self._classifier)):
            return True
        if self.capture and not self.capture.finished:
            return False
        for worker in (self.visual, self.ocr, self._classifier):
            if worker:
                worker.request_stop()
        if any(
            worker and not worker.finished for worker in (self.visual, self.ocr, self._classifier)
        ):
            return False
        self.capture = self.ocr = self.visual = None
        self._classifier = None
        self._classifier_retiring = False
        self.valid_size = None
        self.reset()
        return True

    def invalidate(self, translation: TranslationWorker | None = None) -> None:
        if self.valid_size is not None and self.ocr:
            self.ocr.invalidate()
        self.valid_size = None
        self.reset(translation)

    def accept_size(self, size: tuple[int, int], translation: TranslationWorker | None) -> bool:
        if self.valid_size == size:
            return False
        self.invalidate(translation)
        self.valid_size = size
        return True

    def reset(self, translation: TranslationWorker | None = None) -> None:
        self._first_readings = FirstReadings()
        self._scene_sequence = 0
        self._early_lines = frozenset()
        self.accepted = None
        self._ocr_failed = False
        self._consumed_sequence = 0
        self.assignments = ()
        if self.visual:
            self.visual.invalidate(language=self._language)
        self.visual_state = VisualState()
        self._tracker.reset()
        self.tracking = TrackingSnapshot()
        self._grouper.reset()
        self._presentation.reset()
        self._context_reset_sequence = 0
        self.history.reset_scene()
        if self._classifier:
            self._classifier.invalidate()
        if translation:
            translation.invalidate()

    def change_grouping(self, translation: TranslationWorker | None) -> None:
        self._first_readings = FirstReadings()
        self._grouper.reset()
        self.clear_english()
        if translation:
            translation.invalidate()

    def clear_english(self) -> None:
        self._presentation.reset()

    def reconcile(
        self, frame: CapturedFrame, now: float, translation: TranslationWorker | None = None
    ) -> OcrSnapshot | None:
        if not self.ocr:
            return None
        snapshot = self.ocr.snapshot()
        if snapshot.state == OcrState.FAILED:
            if not self._ocr_failed:
                self.reset(translation)
                self._ocr_failed = True
            return snapshot
        self._ocr_failed = False
        if self.visual:
            state = self.visual.poll(frame.sequence, now)
            candidate = (
                snapshot.result
                if snapshot.result and snapshot.result.frame.sequence > self._consumed_sequence
                else None
            )
            if candidate and state.rejected_sequence == candidate.frame.sequence:
                # A failed review releases backpressure without voting against live tracks.
                self._consumed_sequence = candidate.frame.sequence
                self.ocr.acknowledge(self._consumed_sequence)
                candidate = None
            elif (
                candidate
                and (
                    assignments := self._tracker.observe_reviewed(
                        candidate, state.review, state.presence
                    )
                )
                is not None
            ):
                self.assignments = assignments
                self.accepted = candidate
                self._consumed_sequence = candidate.frame.sequence
                self.ocr.acknowledge(self._consumed_sequence)
                candidate = None
            if state.frame is not None:
                if state.request_ocr:
                    self.ocr.submit(state.frame, state.scan_areas)
                self._tracker.confirm_observations(state.stable_tracks, state.frame.captured_at)
            self.tracking = self._tracker.advance(frame, now)
            if state.frame is not None:
                self.visual_state = self._display_evidence(state)
            self.visual.submit(
                frame,
                self.accepted,
                self.tracking.visible,
                self.assignments,
                self.tracking.activity,
                candidate,
                self._tracker.evidence_bounds(),
                self._tracker.matching_sources(snapshot.result.frame) if snapshot.result else (),
            )
        return snapshot

    def _display_evidence(self, state: VisualState) -> VisualState:
        # Tracking may adopt delayed OCR before the visual worker has examined
        # its revised bounds or reading. Keep a previously dismissed ID hidden
        # until the current live line itself has positive evidence.
        if not (state.cleared or self.visual_state.cleared):
            return state
        dismissed = {line.track_id for line in self.visual_state.cleared}
        cleared = frozenset(
            line
            for line in self.tracking.visible
            if line in state.cleared
            or (line.track_id in dismissed and line not in state.matching_lines)
        )
        return state if cleared == state.cleared else replace(state, cleared=cleared)

    def scene(
        self, frame: CapturedFrame | None, now: float, *, grouping: bool, stabilize: bool = True
    ) -> Scene:
        tracking = (
            self.tracking
            if frame and frame.geometry.client.size == self.tracking.client_size
            else TrackingSnapshot()
        )
        tracks = tracking.visible
        self._scene_sequence = frame.sequence if frame else 0
        proof = self.visual_state.complete_lines
        self._early_lines = proof if frame is self.visual_state.frame else self._early_lines & proof
        grouped = self._grouper.update(
            tracks,
            tracking.confirmed,
            now,
            enabled=grouping,
            activity=tracking.activity,
            complete_lines=self._early_lines,
            dash_edges=dict(tracking.dash_edges) if grouping else None,
            lettering=dict(tracking.lettering) if grouping else None,
        )
        requests = tuple(
            block
            for block in grouped.ready
            if not any(t.track_id in self.visual_state.blocked for t in block.lines)
            and (
                self.visual_state.message
                or (
                    self.accepted and self.visual_state.ocr_sequence == self.accepted.frame.sequence
                )
            )
        )
        # Exact current names may gain confidence from partial observations. Unscanned
        # names retain their evidence only while the identical live track remains.
        scores = {}
        if self.accepted:
            by_id = {t.track_id: t for t in tracks}
            for key, region in zip(self.assignments, self.accepted.regions, strict=True):
                track = by_id.get(key) if key is not None else None
                if (
                    track
                    and region.text.strip() == track.text.strip()
                    and region.confidence is not None
                ):
                    scores[track] = region.confidence
        self._advance_history()
        if (
            self.history_enabled
            and not self._history_suspended
            and (self._history_layout_only or self._history_factory)
        ):
            self.history.observe(
                grouped.blocks,
                requests,
                (frame.width, frame.height) if frame else (0, 0),
                now,
                scores=scores,
                corrections=tracking.corrections,
                assisted=not self._history_layout_only,
            )
            if self._classifier and not self._classifier_retiring:
                self._classifier.update(self.history.classification_requests())
                self.history.accept_classifications(self._classifier.snapshot().results)
        if not stabilize:
            tracks = (
                tuple(
                    TextTrack(key, region.text, region.bounds, region.vertical)
                    for key, region in zip(self.assignments, self.accepted.regions, strict=True)
                    if key is not None
                )
                if self.accepted and frame and overlay_result_ready(self.accepted, frame, now)
                else ()
            )
        if self.visual_state.cleared:
            tracks = tuple(t for t in tracks if t not in self.visual_state.cleared)
        state = self.visual_state
        blocked = set(state.blocked)
        blocked.update(t.track_id for t in state.hidden | state.cleared)
        if state.message:
            blocked.update(t.track_id for t in tracking.visible)
        candidates = self._first_readings.update(
            grouped.blocks,
            requests,
            tracking.activity,
            blocked,
            reviewed=bool(self.accepted and state.ocr_sequence == self.accepted.frame.sequence),
        )
        return Scene(
            tracks, grouped.blocks, requests, len(grouped.blocks) - len(grouped.ready), candidates
        )

    def publish(
        self,
        session: TranslationWorker,
        requests: tuple[TextBlock, ...],
        retained: tuple[TextBlock, ...],
        *,
        candidates: tuple[TextBlock, ...] = (),
        target: TargetLanguage = ENGLISH,
        game: str = "",
    ) -> EnglishScene:
        state = self.visual_state
        broad_change = bool(
            state.frame
            and state.frame.sequence != self._context_reset_sequence
            and sum(b.width * b.height for b in state.changed_areas)
            > state.frame.width * state.frame.height * 0.65
        )
        if broad_change and state.frame:
            self._context_reset_sequence = state.frame.sequence
        session.update(
            requests,
            retain=retained,
            candidates=candidates,
            reset_context=broad_change,
            target=target,
            game=game,
            corrections=self.tracking.corrections,
        )
        snapshot = session.snapshot()
        published = self._presentation.update(
            retained,
            snapshot.results,
            eligible=snapshot.routed_requests,
            scope=(session, snapshot.backend_generation, target),
            cleared=state.cleared,
        )
        if self.history_enabled:
            self.history.published(tuple(p.translation for p in published))
        menus = ()
        if (
            state.frame
            and state.frame.sequence == self._scene_sequence
            and snapshot.state == TranslationState.READY
            and not state.message
        ):
            menus = self._presentation.menu_previews(state.matching_menus, retained)
            published += menus
        if snapshot.cached_candidates:
            published += self._presentation.cached_previews(
                candidates + requests,
                snapshot.cached_candidates,
                state.hidden | state.cleared,
                menus,
            )
        return EnglishScene(
            snapshot, tuple(p.painted for p in published), tuple(p.translation for p in published)
        )
