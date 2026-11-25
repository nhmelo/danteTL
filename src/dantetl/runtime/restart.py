"""Launch a verified runtime and acknowledge a real GUI startup before closing the old app."""

import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dantetl.i18n import mark, tr
from dantetl.paths import is_standalone, module_command, resource_root
from dantetl.runtime.application import prepare_application
from dantetl.runtime.hardware import require_profile
from dantetl.runtime.owned_process import OwnedProcess
from dantetl.runtime.setup import (
    project_root,
    select_runtime,
    selected_runtime,
    verified_runtime,
)


def acknowledge_startup() -> None:
    token = os.environ.pop("DANTETL_RESTART_TOKEN", "")
    if re.fullmatch("[a-f0-9]{32}", token):
        path = project_root() / "data/restarts" / token
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(os.getpid()), encoding="ascii")
        os.replace(temporary, path)


class RuntimeRestart:
    def __init__(
        self,
        root: Path,
        target: Path,
        *,
        on_cleanup: Callable[[], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        key = target.name.split("-", 1)[0]
        if verified_runtime(root, key, target.name) != target.resolve():
            raise ValueError(tr("Runtime is no longer ready. Open Runtime setup and retry."))
        require_profile(key, cancel)
        self._on_cleanup = on_cleanup or (lambda: None)
        self.root, self.target = root, target
        token = uuid4().hex
        self.receipt = root / "data/restarts" / token
        environment = prepare_application(
            root, target / "Scripts/python.exe", cancel or threading.Event()
        )
        environment["DANTETL_RESTART_TOKEN"] = token
        if is_standalone():
            environment["DANTETL_DATA_DIR"] = str(root)
            environment["DANTETL_RESOURCE_DIR"] = str(resource_root())
        self.process = OwnedProcess(
            module_command(target / "Scripts/pythonw.exe", "dantetl"),
            root,
            environment,
            show_window=True,
        )
        self.started_pid: int | None = None
        self.deadline = time.monotonic() + 30
        self._released = self._closed = False

    def poll(self) -> bool:
        try:
            return self._poll()
        except BaseException:
            self.cancel()
            raise

    def _poll(self) -> bool:
        if self.receipt.is_file() and self.process.poll() is None:
            previous = selected_runtime(self.root)
            try:
                self.started_pid = int(self.receipt.read_text("ascii"))
                select_runtime(self.root, self.target)
                self.process.release()
                self._released = True
            except (OSError, ValueError):
                self._close()
                if previous:
                    select_runtime(self.root, previous)
                else:
                    (self.root / "data/runtime.json").unlink(missing_ok=True)
                raise
            # Handoff is committed; stale acknowledgement cannot undo it.
            with suppress(OSError):
                self.receipt.unlink(missing_ok=True)
            return True
        if self.process.poll() is not None:
            self._close()
            self.receipt.unlink(missing_ok=True)
            raise RuntimeError(
                tr("The new app could not start. Your current runtime is still selected.")
            )
        if time.monotonic() > self.deadline:
            # This process is ours and has not acknowledged startup; never kill another app.
            self._close()
            self.receipt.unlink(missing_ok=True)
            raise RuntimeError(
                tr("Restart timed out. Your current runtime is still selected; retry setup.")
            )
        return False

    def _close(self) -> None:
        if not self._released and not self._closed:
            self._on_cleanup()
            self.process.close()
            self._closed = True

    def cancel(self) -> None:
        self._close()
        # Owned processes are already drained; stale receipts cannot acknowledge another launch.
        with suppress(OSError):
            self.receipt.unlink(missing_ok=True)


@dataclass(frozen=True)
class RestartSnapshot:
    phase: Literal["starting", "draining", "ready", "failed"] = "starting"
    message: str = mark("Opening DanteTL with the installed runtime…")
    started_pid: int | None = None


class RestartOperation:
    """Own launch, acknowledgement and all native draining away from the GUI thread."""

    def __init__(
        self, root: Path, target: Path, *, create: Callable[..., RuntimeRestart] = RuntimeRestart
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot = RestartSnapshot()
        self._cancelled = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(root, target, create), name="runtime-restart", daemon=True
        )
        self._thread.start()

    def snapshot(self) -> RestartSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive()

    def cancel(self) -> None:
        self._cancelled.set()

    def _set(self, value: RestartSnapshot) -> None:
        with self._lock:
            self._snapshot = value

    def _draining(self) -> None:
        self._set(RestartSnapshot("draining", tr("Stopping the unsuccessful replacement app…")))

    def _run(self, root: Path, target: Path, create: Callable[..., RuntimeRestart]) -> None:
        try:
            restart = create(root, target, on_cleanup=self._draining, cancel=self._cancelled)
            while True:
                if self._cancelled.is_set():
                    restart.cancel()
                    self._set(
                        RestartSnapshot(
                            "failed", tr("Restart cancelled. Current runtime retained.")
                        )
                    )
                    return
                if restart.poll():
                    self._set(RestartSnapshot("ready", started_pid=restart.started_pid))
                    return
                self._cancelled.wait(0.05)
        except Exception as error:
            self._set(RestartSnapshot("failed", str(error) or tr("Runtime restart failed.")))
