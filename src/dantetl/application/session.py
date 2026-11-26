"""Headless application commands and ordered processing; desktop views only present results."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Unpack

from dantetl.application.cache_management import CacheManagement
from dantetl.application.capture_lifecycle import CaptureLifecycle, CapturePhase, WindowAccess
from dantetl.application.cloud_services import CloudServices
from dantetl.application.live_session import CaptureWorker, LiveSession
from dantetl.application.model_inventory import ModelInventory, SourceModels
from dantetl.application.notice import NoticeInput, Notices, project_notices
from dantetl.application.ollama_services import OllamaServices
from dantetl.application.preference_values import SettingsEdit
from dantetl.application.preferences import PreferenceStore
from dantetl.application.presentation import (
    ocr_status,
    processing_progress,
    reset_scene,
    translation_status,
)
from dantetl.application.setup_coordinator import SetupCoordinator
from dantetl.application.shutdown import ShutdownCoordinator
from dantetl.application.translation_activity import TranslationMode
from dantetl.application.translation_coordinator import TranslationCoordinator
from dantetl.application.values import (
    ApplicationControls,
    ApplicationView,
    DesktopAction,
    ModelConsent,
    ShutdownPhase,
    ShutdownSnapshot,
)
from dantetl.domain import (
    CapturedFrame,
    CaptureState,
    WindowTarget,
    overlay_ready,
)
from dantetl.i18n import CatalogText, tr
from dantetl.languages.selection import source_language
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.ocr.models import ocr_model_for
from dantetl.ocr.values import OcrState
from dantetl.ocr.worker import OcrChoice
from dantetl.runtime.hardware_values import HardwareStatus
from dantetl.translation.cloud.values import CloudProfile
from dantetl.translation.dialogue.values import HistoryTurn
from dantetl.translation.selection import AssignmentChange, OcrOption
from dantetl.translation.values import TranslationState


@dataclass(frozen=True, slots=True)
class SourceChange:
    target: WindowTarget | None
    drain_capture: bool


class ApplicationSession:
    def __init__(
        self,
        *,
        windows: WindowAccess,
        live: LiveSession,
        translation: TranslationCoordinator,
        ocr_choices: tuple[OcrChoice, ...],
        source_models: Callable[[SourceLanguage], SourceModels],
        inventory: ModelInventory,
        cloud: CloudServices,
        ollama: OllamaServices,
        setup: SetupCoordinator,
        cache: CacheManagement,
        preferences: PreferenceStore,
        shutdown: ShutdownCoordinator,
        capture: Callable[[WindowTarget, int, bool], CaptureWorker],
        notify: Callable[[], None] = lambda: None,
        hardware: Callable[[], HardwareStatus | None] = lambda: None,
    ) -> None:
        self.windows = windows
        self._source_models = source_models
        self.inventory, self.cloud, self.setup = inventory, cloud, setup
        self.preferences, self.shutdown = preferences, shutdown
        self.ollama = ollama
        self.cache = cache
        self._notify = notify
        self.live, self.translation = live, translation
        self.capture = CaptureLifecycle(live, windows, ocr_choices, capture)
        self._hardware = hardware
        self.values = preferences.current
        self._ocr_model = ocr_model_for(source_language(self.values.source_language))
        self.cloud.test_target = TARGET_LANGUAGES[self.values.output_language]
        self.view = ApplicationView()
        self.raw_boxes = False
        self.debug = False
        self._desktop_action: DesktopAction | None = None
        self.error = ""
        self.error_revision = 0
        self._setup_consent: int | None = None
        self._inventory_revision = -1
        self._cloud_revision: tuple[tuple[CloudProfile, ...], frozenset[str], int] | None = None
        self._advancing = self._again = False
        self._first_run_requested = False
        self._source_change: SourceChange | None = None
        self.refresh_routes()

    @property
    def ocr_options(self) -> tuple[OcrOption, ...]:
        return self.capture.options

    @property
    def can_change_source(self) -> bool:
        return not (
            self._source_change
            or self.closing
            or self.setup.busy
            or self.cloud.busy
            or self.translation.lifecycle.assignment_busy
        )

    def source_requires_restart(self, code: str) -> bool:
        capture = self.live.capture
        ocr = self.live.ocr
        return bool(
            capture
            and not self.capture.stopping
            and not capture.finished
            and capture.snapshot().state in (CaptureState.STARTING, CaptureState.RUNNING)
            and (
                ocr_model_for(source_language(code)) != self._ocr_model
                or not self.capture.ocr_choice
                or self.capture.ocr_choice.key != self.capture.choice_for(self.values).key
                or (ocr and ocr.snapshot().state not in (OcrState.LOADING, OcrState.READY))
            )
        )

    def change_source(self, code: str) -> bool:
        language = source_language(code)
        if not self.can_change_source or code == self.values.source_language:
            return False
        models = self._source_models(language)
        restart = self.source_requires_restart(code)
        target = self.live.capture.target if restart and self.live.capture else None
        self._source_change = SourceChange(target, restart or self.capture.stopping)
        if restart:
            self.capture.stop(tr("Changing game text language…"))
        self.live.change_language(language, self.translation.lifecycle.publication)
        self.values = replace(self.values, source_language=code)
        self.persist()
        self.cloud.language = language
        self.translation.language = language
        self.capture.set_choices(models.ocr)
        self.setup.set_models(models.bundles)
        self.inventory.set_inspector(models.inspect)
        self._ocr_model = ocr_model_for(language)
        self.view = ApplicationView(
            status=tr("Switching to {name}…", name=tr(language.name)), reset=self.view.reset + 1
        )
        return True

    def _finish_source_change(self, change: SourceChange, now: float) -> None:
        if change.drain_capture and not self.live.drain():
            return
        inventory = self.inventory.snapshot()
        if not inventory.loaded and not inventory.problem:
            return
        self._source_change = None
        self._inventory_revision = -1
        self.refresh_routes()
        self.view = replace(
            self.view,
            status=CatalogText(
                "Game text language changed to {name}.",
                name=CatalogText(self.translation.language.name),
            ),
        )
        if change.target:
            self._start_capture(change.target, now)

    def configure(self, **changes: Unpack[SettingsEdit]) -> None:
        if unknown := changes.keys() - SettingsEdit.__optional_keys__:
            raise TypeError(f"Unsupported settings: {sorted(unknown)}")
        prior = self.values
        values = replace(prior, **changes)
        if values == prior:
            return
        if values.output_language not in TARGET_LANGUAGES:
            raise ValueError(
                tr(
                    "Unknown output language: {output_language}",
                    output_language=values.output_language,
                )
            )
        self.values = values
        if prior.output_language != values.output_language:
            self.cloud.test_target = TARGET_LANGUAGES[values.output_language]
            if worker := self.translation.lifecycle.publication:
                worker.invalidate()
            self._invalidate_english()
        if prior.group_dialogue != values.group_dialogue:
            self.live.change_grouping(self.translation.lifecycle.publication)
            self._invalidate_english()
        if prior.show_border != values.show_border and self.live.capture:
            self.live.capture.set_border(values.show_border)
        if (prior.game_hint, prior.game_from_window) != (values.game_hint, values.game_from_window):
            self.translation.set_game(
                automatic=values.game_from_window,
                manual=values.game_hint,
                title=self.translation.window_title,
            )
            self.refresh_routes()
        if not self._source_change:
            self._configure_history()
        self.persist()

    def persist(self) -> None:
        if not self.translation.restored:
            # A gesture before background discovery completes must not erase saved IDs.
            self.preferences.save(self.values)
            return
        plan = self.translation.plan
        self.values = replace(
            self.values,
            short_model=plan.short,
            long_model=plan.long,
            translate_short=plan.short_enabled,
            translate_long=plan.long_enabled,
        )
        self.preferences.save(self.values)

    def remember_model(self) -> None:
        choice = self.translation.lifecycle.selected
        key = choice.preference_key if choice else "none"
        if key != self.values.model:
            self.values = replace(self.values, model=key)
            self.persist()

    def set_window_title(self, title: str) -> None:
        self.translation.set_game(
            automatic=self.values.game_from_window, manual=self.values.game_hint, title=title
        )
        self.refresh_routes()

    def refresh_routes(self) -> None:
        if self._source_change is not None:
            return  # The new source inventory determines which workers remain compatible.
        if self.translation.refresh_route():
            self._invalidate_english()
        self._configure_history()

    def _configure_history(self) -> None:
        self.translation.configure_history(
            self.live, enabled=self.values.history_enabled, reference=self.values.history_model
        )
        self.live.suspend_history(self.setup.probing or not self.values.translation)

    @property
    def _can_assign(self) -> bool:
        return not (
            self.shutdown.active
            or self._source_change is not None
            or not self.translation.restored
            or self.translation.lifecycle.assignment_busy
            or self.setup.probing
        )

    def assign(self, change: AssignmentChange) -> ModelConsent | None:
        if not self._can_assign:
            return None
        outcome = self._apply_assignment(change)
        if isinstance(outcome, ModelConsent):
            return outcome
        if outcome:
            self._desktop_action = outcome
        elif self.values.setup_model and change.reference:
            self.values = replace(self.values, setup_model="")
            self.persist()
        return None

    def _apply_assignment(self, change: AssignmentChange) -> DesktopAction | None:
        outcome = self.translation.assign(change)
        if outcome == "unchanged":
            return None
        if outcome != "committed":
            return outcome
        self.refresh_routes()
        if not change.plan.short and not change.plan.long:
            self.remember_model()
        self.persist()
        return None

    def consent(self, token: int, accepted: bool) -> None:
        if self.shutdown.active:
            return
        choice = self.translation.lifecycle.consent(token, accepted)
        if choice is None:
            return
        if token == self._setup_consent:
            if self.values.setup_model == choice.preference_key:
                self.values = replace(self.values, setup_model="")
            self._setup_consent = None
        elif accepted and self.values.setup_model:
            self.values = replace(self.values, setup_model="")
        if accepted:
            self._invalidate_english()
        else:
            self.refresh_routes()
        self.persist()

    def _invalidate_english(self) -> None:
        self.live.clear_english()
        self.view = replace(
            self.view, blocks=(), english=(), results=(), english_reset=self.view.english_reset + 1
        )

    def _lose_capture_scene(self) -> None:
        if self.live.valid_size is not None:
            self.live.invalidate(self.translation.lifecycle.publication)
            self.view = reset_scene(self.view)

    def start_capture(self, target: WindowTarget, now: float) -> bool:
        if (
            self._source_change
            or self.shutdown.active
            or self.live.capture
            or (self.live.ocr and not self.live.drain())
        ):
            return False
        return self._start_capture(target, now)

    def _start_capture(self, target: WindowTarget, now: float) -> bool:
        try:
            capture = self.capture.prepare(target, now, self.values)
        except (OSError, RuntimeError) as error:
            self.view = replace(self.view, status=str(error))
            return False
        inventory = self.inventory.snapshot()
        choice = self.capture.ocr_choice
        recognizer = (
            choice.create if choice and inventory.loaded and inventory.data.ocr_ready else None
        )
        self.view = replace(reset_scene(self.view), status=tr("Starting capture…"))
        self.set_window_title(target.title)
        self.live.start(capture, recognizer, self.translation.lifecycle.publication)
        return True

    def stop(self, message: str = "Capture stopped.") -> None:
        if self._source_change:
            self._source_change = SourceChange(None, True)
        self.translation.lifecycle.stop_capture(closing=self.shutdown.active)
        self.capture.stop(tr(message), self.translation.lifecycle.publication)
        self.view = reset_scene(self.view)
        if self.live.capture:
            self.view = replace(self.view, status=tr("Stopping capture, OCR and translation…"))

    def refresh_models(self) -> None:
        if not self.shutdown.active:
            self.inventory.invalidate()

    def offer_first_run(self) -> None:
        self._first_run_requested = True

    def setup_skipped(self) -> None:
        self.values = replace(self.values, setup_seen=True)
        self._first_run_requested = False
        self.persist()
        self.refresh_models()

    def setup_completed(
        self, model: str, target: Path | None, *, defer_preparation: bool = False
    ) -> None:
        self.values = replace(self.values, setup_seen=True, setup_model=model)
        self.persist()
        self.refresh_models()
        if target:
            self.request_close(target)
        else:
            # A postponed guide restart should prepare the model in the chosen runtime.
            self._first_run_requested = not defer_preparation

    def _first_run(self) -> None:
        if (
            not self._first_run_requested
            or not self.inventory.snapshot().loaded
            or self.shutdown.active
        ):
            return
        if self.values.setup_model:
            key = "local:" + self.values.setup_model
            if not self._can_assign:
                return
            current = self.translation.plan
            # Completing a local download preserves independently configured cloud routes.
            plan = replace(
                current,
                short=current.short if current.short.startswith(("cloud:", "ollama:")) else key,
                long=current.long if current.long.startswith(("cloud:", "ollama:")) else key,
            )
            if key not in (plan.short, plan.long):
                self.values = replace(self.values, setup_model="")
                self.persist()
                self._first_run_requested = False
                return
            # A completed download may still be waiting for inventory discovery.
            if self.translation.resolve(key) is None:
                return
            outcome = self._apply_assignment(AssignmentChange(plan, key))
            if outcome:
                self._desktop_action = outcome
                if isinstance(outcome, ModelConsent):
                    self._setup_consent = outcome.token
            else:
                self.values = replace(self.values, setup_model="")
                self.persist()
        elif not self.values.setup_seen:
            if self.inventory.snapshot().data.ocr_ready and any(
                f.present for f in self.setup.files if f.key != "ocr"
            ):
                self.setup_skipped()
            elif self.can_setup:
                self._desktop_action = "guide"
            else:
                return
        self._first_run_requested = False

    def request_close(self, target: Path | None = None) -> None:
        if self.shutdown.active:
            return
        self._setup_consent = None
        self._desktop_action = None
        self._source_change = None
        self.cloud.close()
        self.ollama.discard()
        self.shutdown.request(target)
        self.view = ApplicationView(reset=self.view.reset + 1)

    @property
    def controls(self) -> ApplicationControls:
        capture = self.live.capture
        preparation = self.translation.lifecycle.preparation_message
        return ApplicationControls(
            self.translation.options(),
            self.translation.plan,
            preparation is not None,
            preparation or "",
            capture is not None,
            self.capture.can_stop,
            self.capture.started_at,
            self.capture.rate,
            self._source_change is not None,
        )

    @property
    def closing(self) -> bool:
        return self.shutdown.active

    @property
    def close_status(self) -> ShutdownSnapshot:
        return self.shutdown.snapshot()

    @property
    def history_turns(self) -> tuple[HistoryTurn, ...]:
        return self.live.history.turns

    @property
    def history_status(self) -> str:
        return self.live.history_status

    def clear_history(self) -> None:
        self.live.history.clear()

    def take_desktop_action(self) -> DesktopAction | None:
        action, self._desktop_action = self._desktop_action, None
        return action

    def foreground_target(self) -> WindowTarget | None:
        if self.closing or self.live.capture:
            return None
        try:
            target = self.windows.foreground_target()
            if not self.windows.is_foreground(target):
                raise RuntimeError(tr("Focus changed. Press the capture hotkey again."))
            return target
        except (OSError, RuntimeError) as error:
            self.view = replace(self.view, status=str(error))
            return None

    def set_diagnostics(self, *, boxes: bool, debug: bool) -> None:
        self.raw_boxes, self.debug = boxes, debug

    @property
    def can_setup(self) -> bool:
        return not (
            self.shutdown.active
            or self._source_change is not None
            or not self.inventory.snapshot().loaded
            or self.live.capture
            or self.setup.busy
            or self.cloud.busy
            or self.translation.lifecycle.preparation_message is not None
        )

    def notices(self) -> Notices:
        ocr = self.view.ocr
        translation = self.view.translation
        cloud = self.cloud.snapshot()
        discovering = not self.translation.restored or (cloud.busy and not cloud.completion)
        return project_notices(
            NoticeInput(
                self.translation.options(),
                self.translation.plan,
                self.inventory.snapshot().data.ocr_ready,
                self.preferences.error,
                self.setup.problem or cloud.problem or self.inventory.snapshot().problem,
                self.setup.busy or (cloud.busy and not discovering),
                self.translation.lifecycle.preparation_message is not None,
                str(ocr.state) if ocr else "",
                self.view.ocr_status,
                str(translation.state) if translation else "",
                self.view.translation_status,
                bool(translation and translation.rejected),
                self._hardware(),
                self.capture.ocr_choice.key
                if self.live.capture and self.capture.ocr_choice
                else self.values.ocr_device,
                ocr.provider if ocr else "",
                ocr.message if ocr else "",
                target=TARGET_LANGUAGES[self.values.output_language],
                ocr_model=self._ocr_model,
                interface_language=self.values.interface_language,
                discovering=discovering,
                installed_runtime=self.setup.installed_runtime,
            )
        )

    def _advance_preparation(self) -> None:
        outcome = self.translation.lifecycle.advance_preparation()
        if outcome is None:
            return
        if outcome.committed:
            self.remember_model()
        elif outcome.error:
            self.error, self.error_revision = outcome.error, self.error_revision + 1
            self.view = replace(
                self.view, translation_status=tr("Preparation failed: {error}", error=outcome.error)
            )
        self.refresh_routes()
        self.persist()

    def _translate(self, running: bool, *, now: float) -> None:
        activity = self.translation.advance_activity(
            running=running,
            enabled=self.values.translation,
            probing=self.setup.probing,
        )
        history_ready = self.live.suspend_history(self.setup.probing or not self.values.translation)
        if activity.probe_ready and history_ready:
            self.setup.permit_probe()
        if activity.clears_english:
            if self.view.english or self.view.results:
                self._invalidate_english()
            else:
                self.live.clear_english()
        if activity.mode == TranslationMode.SWITCHING:
            self.view = replace(
                self.view,
                translation_mode=activity.mode,
                translation_status=translation_status(activity, None, now),
            )
            return
        snapshot = activity.snapshot
        if activity.worker is not None:
            scene = self.view.scene
            published = self.live.publish(
                activity.worker,
                scene.requests,
                scene.blocks,
                candidates=scene.candidates,
                target=TARGET_LANGUAGES[self.values.output_language],
                game=self.translation.game or self.translation.window_title,
            )
            snapshot = published.snapshot
            self.view = replace(
                self.view,
                blocks=published.blocks,
                english=published.translations,
                results=snapshot.results,
            )
        if (
            activity.mode in (TranslationMode.ACTIVE, TranslationMode.STANDBY)
            and snapshot
            and snapshot.state == TranslationState.READY
        ):
            self.remember_model()
        self.view = replace(
            self.view,
            translation=snapshot
            if activity.mode not in (TranslationMode.PROBING, TranslationMode.PREPARING)
            else self.view.translation,
            translation_status=translation_status(activity, snapshot, now),
            translation_mode=activity.mode,
        )

    def _update_ocr(self, frame: CapturedFrame, now: float) -> None:
        if self.live.ocr is None:
            self.view = replace(
                self.view,
                frame=frame,
                ocr=None,
                ocr_status=tr(
                    "Capture only. Open Settings → Models to download {_ocr_model}.",
                    _ocr_model=self._ocr_model,
                ),
            )
            return
        snapshot = self.live.reconcile(frame, now, self.translation.lifecycle.publication)
        assert snapshot is not None
        if snapshot.state == OcrState.FAILED and (
            self.view.ocr is None or self.view.ocr.state != OcrState.FAILED
        ):
            self.view = reset_scene(self.view)
        self.view = replace(
            self.view,
            ocr=snapshot,
            ocr_status=self.live.visual_state.review_message
            or ocr_status(snapshot, self.live.visual_state.message, now, self._ocr_model),
            frame=frame,
            observation=self.live.accepted,
            assignments=self.live.assignments,
        )

    def advance(self, now: float) -> ApplicationView:
        if self._advancing:
            self._again = True
            return self.view
        self._advancing = True
        try:
            self._advance(now)
            self.view = replace(self.view, visual=self.live.visual_state)
            self.view = replace(
                self.view,
                progress=processing_progress(
                    self.view,
                    translation_enabled=self.values.translation,
                    ocr_model=self._ocr_model,
                ),
            )
            return self.view
        finally:
            self._advancing = False
            if self._again:
                self._again = False
                self._notify()

    def _advance(self, now: float) -> None:
        self.setup.advance()
        self.inventory.advance(now)
        self.cloud.advance()
        if self.ollama.advance():
            config = self.ollama.snapshot().config
            self.translation.update_ollama(config, self.ollama.unavailable)
            self.refresh_routes()
            if config != self.values.ollama:
                self.values = replace(self.values, ollama=config)
                self.persist()
        if self._source_change:
            self._finish_source_change(self._source_change, now)
            return
        inventory = self.inventory.snapshot()
        cloud = self.cloud.snapshot()
        cloud_key = (cloud.profiles, cloud.available, cloud.revision)
        if inventory.revision != self._inventory_revision or cloud_key != self._cloud_revision:
            self._inventory_revision, self._cloud_revision = inventory.revision, cloud_key
            self.refresh_routes()
            self.capture.recognize_when_ready(
                inventory.loaded and inventory.data.ocr_ready,
                closing=self.shutdown.active,
            )
        if self.shutdown.active:
            self.shutdown.advance()
            if self.shutdown.snapshot().phase == ShutdownPhase.FAILED:
                self.inventory.resume()
                self.setup.resume()
                self.cloud.resume()
            return
        self._first_run()
        self._advance_preparation()
        try:
            self._advance_capture(now)
        except (OSError, RuntimeError) as error:
            self.stop(str(error))

    def _advance_capture(self, now: float) -> None:
        step = self.capture.poll()
        if step.phase == CapturePhase.IDLE:
            self.view = replace(self.view, capture=None, overlay_bounds=None)
            self._translate(False, now=now)
            return
        if step.phase in (CapturePhase.STOPPING, CapturePhase.DRAINING):
            self._lose_capture_scene()
            self.view = replace(self.view, capture=None)
            self._translate(False, now=now)
            if step.phase == CapturePhase.DRAINING:
                assert step.snapshot is not None
                message = self.capture.drain(
                    translation_drained=self.translation.lifecycle.capture_drained,
                    snapshot=step.snapshot,
                )
                self.view = replace(
                    self.view,
                    status=message,
                    ocr_status=CatalogText(
                        "OCR stopped. Start capture to load the local models again."
                    )
                    if self.live.capture is None
                    else self.view.ocr_status,
                )
            return
        assert step.snapshot is not None and step.geometry is not None
        snapshot, geometry, foreground = step.snapshot, step.geometry, step.foreground
        frame = snapshot.frame if step.phase == CapturePhase.ACTIVE else None
        if step.phase == CapturePhase.ACTIVE:
            assert frame is not None
            if self.live.accept_size(geometry.client.size, self.translation.lifecycle.publication):
                self.view = replace(
                    self.view,
                    observation=None,
                    frame=None,
                    reset=self.view.reset + 1,
                )
            self._update_ocr(frame, now)
            message = (
                tr("Capturing. The overlay appears on the selected window when ready.")
                if foreground and self.values.show_overlay
                else tr("Capturing. Switch to the selected window to see the overlay.")
                if self.values.show_overlay
                else tr("Capturing. Game overlay is hidden.")
            )
        else:
            self._lose_capture_scene()
            message = (
                tr("Window minimized. Restore it to resume.")
                if step.phase == CapturePhase.MINIMIZED
                else tr("Waiting for a frame matching the window size…")
            )
        scene = self.live.scene(
            frame,
            now,
            grouping=self.values.group_dialogue,
            stabilize=self.values.stabilize or self.values.translation,
        )
        # Publish the snapshot whose geometry we checked, or the initial waiting state.
        capture = (
            snapshot
            if step.phase == CapturePhase.ACTIVE
            or (step.phase == CapturePhase.WAITING_FRAME and snapshot.frame is None)
            else None
        )
        self.view = replace(self.view, capture=capture, scene=scene, status=message)
        if not self.values.translation:
            self.view = replace(self.view, blocks=scene.blocks)
        self._translate(True, now=now)
        bounds = (
            geometry.client
            if (
                self.values.show_overlay
                and (self.values.translation or self.raw_boxes or self.debug)
                and (scene.tracks or self.debug)
                and frame is not None
                and overlay_ready(snapshot, geometry, foreground)
            )
            else None
        )
        self.view = replace(self.view, overlay_bounds=bounds)
