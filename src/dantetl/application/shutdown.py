"""Application close/restart ordering, independent of Qt and native process ownership."""

from collections.abc import Callable
from typing import Protocol

from dantetl.application.live_session import LiveSession
from dantetl.application.model_lifecycle import ModelLifecycle
from dantetl.application.values import ShutdownPhase, ShutdownSnapshot
from dantetl.i18n import tr
from dantetl.runtime.restart import RestartSnapshot
from dantetl.runtime.setup_values import RuntimeTarget


class RestartTask(Protocol):
    @property
    def finished(self) -> bool: ...
    def snapshot(self) -> RestartSnapshot: ...


class ShutdownCoordinator:
    """Cancel, drain, flush, release shortcuts, then hand off an acknowledged restart.

    Polling never waits. Resource owners decide when cleanup actually finishes.
    Setup adapters only expose busy/cancel; they do not decide shutdown ordering.
    """

    def __init__(
        self,
        live: LiveSession,
        models: ModelLifecycle,
        *,
        setup_busy: Callable[[], bool],
        cancel_setup: Callable[[], None],
        preferences_idle: Callable[[], bool],
        hardware_finished: Callable[[], bool],
        release_hotkeys: Callable[[], None],
        restart: Callable[[RuntimeTarget], RestartTask],
    ) -> None:
        self._live, self._models = live, models
        self._setup_busy, self._cancel_setup = setup_busy, cancel_setup
        self._preferences_idle, self._hardware_finished = preferences_idle, hardware_finished
        self._release_hotkeys, self._start_restart = release_hotkeys, restart
        self._snapshot = ShutdownSnapshot()
        self._target: RuntimeTarget | None = None
        self._restart: RestartTask | None = None

    @property
    def active(self) -> bool:
        return self._snapshot.phase not in (ShutdownPhase.IDLE, ShutdownPhase.FAILED)

    def snapshot(self) -> ShutdownSnapshot:
        return self._snapshot

    def request(self, target: RuntimeTarget | None = None) -> None:
        if self.active:
            return
        self._target, self._restart = target, None
        self._snapshot = ShutdownSnapshot(ShutdownPhase.DRAINING, tr("Finishing active work…"))
        self._cancel_setup()
        self._models.shutdown()
        self._live.stop()

    def advance(self) -> ShutdownSnapshot:
        phase = self._snapshot.phase
        if phase == ShutdownPhase.DRAINING:
            self._cancel_setup()
            models_done = self._models.finish_shutdown()
            live_done = self._live.drain()
            message = (
                tr("Cancelling setup…")
                if self._setup_busy()
                else tr("Waiting for translation to finish unloading…")
                if not models_done
                else tr("Waiting for capture, OCR and visual workers…")
                if not live_done
                else tr("Saving preferences…")
                if not self._preferences_idle()
                else tr("Finishing the graphics hardware check…")
                if not self._hardware_finished()
                else ""
            )
            if message:
                self._snapshot = ShutdownSnapshot(phase, message)
                return self._snapshot
            try:
                self._release_hotkeys()
                if self._target is None:
                    self._snapshot = ShutdownSnapshot(ShutdownPhase.READY)
                else:
                    self._restart = self._start_restart(self._target)
                    self._snapshot = ShutdownSnapshot(ShutdownPhase.RESTARTING)
            except (OSError, ValueError, RuntimeError) as error:
                self._failed(str(error))
        if self._snapshot.phase == ShutdownPhase.RESTARTING:
            assert self._restart is not None
            progress = self._restart.snapshot()
            if not self._restart.finished:
                self._snapshot = ShutdownSnapshot(ShutdownPhase.RESTARTING, progress.message)
            else:
                progress = self._restart.snapshot()
                if progress.phase == "ready":
                    self._snapshot = ShutdownSnapshot(ShutdownPhase.READY)
                    self._restart = None
                else:
                    self._failed(progress.message)
        return self._snapshot

    def _failed(self, message: str) -> None:
        self._target, self._restart = None, None
        self._models.resume_after_failed_restart()
        self._snapshot = ShutdownSnapshot(ShutdownPhase.FAILED, message)
