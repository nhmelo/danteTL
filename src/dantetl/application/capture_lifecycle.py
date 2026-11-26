"""Capture transitions and frozen device settings, independent of desktop presentation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from dantetl.application.live_session import CaptureWorker, LiveSession
from dantetl.application.preference_values import Preferences
from dantetl.domain import CaptureSnapshot, CaptureState, WindowGeometry, WindowTarget
from dantetl.i18n import CatalogText, tr
from dantetl.ocr.worker import OcrChoice
from dantetl.translation.selection import OcrOption
from dantetl.translation.values import TranslationWorker


class WindowAccess(Protocol):
    @property
    def supports_background_overlay(self) -> bool: ...
    def is_target(self, target: WindowTarget) -> bool: ...
    def geometry(self, target: WindowTarget) -> WindowGeometry: ...
    def is_foreground(self, target: WindowTarget) -> bool: ...
    def list_targets(self) -> Sequence[WindowTarget]: ...
    def foreground_target(self) -> WindowTarget: ...


class CapturePhase(StrEnum):
    IDLE = "idle"
    STOPPING = "stopping"
    DRAINING = "draining"
    MINIMIZED = "minimized"
    WAITING_FRAME = "waiting_frame"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class CaptureStep:
    phase: CapturePhase
    snapshot: CaptureSnapshot | None = None
    geometry: WindowGeometry | None = None
    foreground: bool = False
    target: WindowTarget | None = None


class CaptureLifecycle:
    """Own capture intent until every capture/OCR/visual worker has drained.

    LiveSession retains sole ownership of the workers and their source ordering. This
    owner freezes device/rate choices and validates the target before scene processing.
    """

    def __init__(
        self,
        live: LiveSession,
        windows: WindowAccess,
        choices: tuple[OcrChoice, ...],
        create: Callable[[WindowTarget, int, bool], CaptureWorker],
    ) -> None:
        self.live, self.windows = live, windows
        self._create = create
        self.set_choices(choices)
        self.ocr_choice: OcrChoice | None = None
        self.rate = 8
        self.started_at = 0.0
        self._invalid_through = -1
        self.stopping = False
        self.stop_message = CatalogText("Capture stopped.")

    @property
    def can_stop(self) -> bool:
        capture = self.live.capture
        return bool(
            capture and not self.stopping and capture.snapshot().state != CaptureState.STOPPING
        )

    def set_choices(self, choices: tuple[OcrChoice, ...]) -> None:
        self._choices = choices
        self.options = tuple(OcrOption(c.name, c.key, c.capture_fps) for c in choices)

    def choice_for(self, values: Preferences) -> OcrChoice:
        return next((c for c in self._choices if c.key == values.ocr_device), self._choices[0])

    def prepare(self, target: WindowTarget, now: float, values: Preferences) -> CaptureWorker:
        if not self.windows.is_target(target):
            raise RuntimeError(tr("The selected window closed. Refresh the window list."))
        self.rate, self.started_at = values.capture_rate, now
        self.ocr_choice = self.choice_for(values)
        capture = self._create(target, self.rate, values.show_border)
        self._invalid_through = -1
        self.stop_message = CatalogText("Capture source closed.")
        self.stopping = False
        return capture

    def recognize_when_ready(self, ready: bool, *, closing: bool) -> None:
        capture = self.live.capture
        if (
            ready
            and not closing
            and not self.stopping
            and capture
            and not capture.finished
            and self.live.ocr is None
            and self.ocr_choice
        ):
            self.live.add_recognition(self.ocr_choice.create)

    def stop(self, message: str, translation: TranslationWorker | None = None) -> None:
        self.stopping = self.live.capture is not None
        self.stop_message = message
        self.live.stop(translation)

    def poll(self) -> CaptureStep:
        capture = self.live.capture
        if capture is None:
            return CaptureStep(CapturePhase.IDLE)
        snapshot = capture.snapshot()
        if capture.finished:
            return CaptureStep(CapturePhase.DRAINING, snapshot)
        if self.stopping or snapshot.state in (CaptureState.STOPPING, CaptureState.FAILED):
            if self.live.ocr:
                self.live.ocr.request_stop()
            return CaptureStep(CapturePhase.STOPPING, snapshot)
        geometry = self.windows.geometry(capture.target)
        foreground = self.windows.is_foreground(capture.target)
        frame = snapshot.frame
        phase = CapturePhase.WAITING_FRAME
        if frame is not None:
            if geometry.minimized or frame.geometry.client.size != geometry.client.size:
                self._invalid_through = max(self._invalid_through, frame.sequence)
            elif frame.sequence > self._invalid_through:
                phase = CapturePhase.ACTIVE
        if geometry.minimized:
            phase = CapturePhase.MINIMIZED
        return CaptureStep(phase, snapshot, geometry, foreground, capture.target)

    def drain(self, *, translation_drained: bool, snapshot: CaptureSnapshot) -> str:
        if not translation_drained:
            return tr("Waiting for the current translation operation to finish…")
        if not self.live.drain():
            return tr("Waiting for OCR and visual workers to finish…")
        return snapshot.message if snapshot.state == CaptureState.FAILED else self.stop_message
