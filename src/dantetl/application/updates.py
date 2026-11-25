"""Single-flight update commands, durable prompt decisions and immutable desktop snapshots."""

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from packaging.version import Version

from dantetl.file_lock import file_lock
from dantetl.runtime.release_values import Release, UpdateProgress

Decision = Literal["yes", "later", "no"]
Phase = Literal[
    "idle", "checking", "current", "available", "saving", "installing", "restart", "error"
]
WEEK = 7 * 24 * 3600


@dataclass(frozen=True)
class UpdateView:
    phase: Phase = "idle"
    release: Release | None = None
    prompt: bool = False
    progress: UpdateProgress | None = None
    error: str = ""


@dataclass
class _History:
    checked: str = ""
    offered: str = ""
    choice: str = ""
    remind_at: float = 0


class Updates:
    def __init__(
        self,
        installed: str,
        state: Path,
        *,
        fetch: Callable[[threading.Event], Release | None],
        install: Callable[[Release, threading.Event, Callable[[UpdateProgress], None]], None]
        | None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.installed = installed
        self._path = state / "data/updates.json"
        self._fetch, self._install, self._clock = fetch, install, clock
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="release-update")
        self._cancel = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._view = UpdateView()
        self._job: Future[UpdateView] | None = None

    def can_install(self, release: Release) -> bool:
        return self._install is not None and release.signed_assets

    def _record(self, edit: Callable[[_History], None]) -> None:
        with file_lock(self._path.with_suffix(".lock")):
            try:
                history = _History(**json.loads(self._path.read_text("utf-8")))
                Version(history.offered or "0")
                history.remind_at = float(history.remind_at)
            except (FileNotFoundError, ValueError, TypeError):
                history = _History()
            edit(history)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(history)), "utf-8")
            tmp.replace(self._path)

    def snapshot(self) -> UpdateView:
        if self._job is not None and self._job.done():
            try:
                self._view = self._job.result()
            except Exception as error:
                logging.getLogger(__name__).warning("Release update failed: %s", error)
                self._view = UpdateView("error", self._view.release, error=str(error))
            self._job = None
        with self._lock:
            return self._view

    def check(self, *, automatic: bool = False) -> None:
        if self._job is not None or self._closed or self._view.phase == "restart":
            return
        self._cancel.clear()
        self._view = UpdateView("checking")

        def run() -> UpdateView:
            release = self._fetch(self._cancel)
            newer = release is not None and Version(release.version) > Version(self.installed)
            prompt = False

            def record(history: _History) -> None:
                nonlocal prompt
                history.checked = release.version if release else ""
                if not newer or release is None:
                    return
                if not automatic:
                    prompt = True
                    return
                now = self._clock()
                prompt = Version(release.version) > Version(history.offered or "0") or (
                    release.version == history.offered
                    and history.choice == "later"
                    and now >= history.remind_at
                )
                if prompt:
                    # Reserve the offer across concurrent launches. Closing the dialog snoozes it.
                    history.offered, history.choice, history.remind_at = (
                        release.version,
                        "later",
                        now + WEEK,
                    )

            self._record(record)
            return UpdateView("available", release, prompt) if newer else UpdateView("current")

        self._job = self._pool.submit(run)

    def take_offer(self) -> Release | None:
        if self._view.prompt:
            self._view = replace(self._view, prompt=False)
            return self._view.release
        return None

    def decide(self, release: Release, choice: Decision) -> None:
        if (
            self._job is not None
            or self._closed
            or self._view.release != release
            or self._view.phase not in ("available", "error")
            or (choice == "yes" and not self.can_install(release))
        ):
            return
        self._cancel.clear()
        self._view = (
            UpdateView("installing", release, progress=UpdateProgress("preparing"))
            if choice == "yes"
            else UpdateView("saving", release)
        )

        def run() -> UpdateView:
            def record(history: _History) -> None:
                if Version(release.version) >= Version(history.offered or "0"):
                    history.offered, history.choice = release.version, choice
                    history.remind_at = self._clock() + WEEK if choice == "later" else 0

            self._record(record)
            if choice == "yes":
                assert self._install is not None
                self._install(release, self._cancel, self._progress)
                return UpdateView("restart", release)
            return UpdateView("available", release)

        self._job = self._pool.submit(run)

    def _progress(self, progress: UpdateProgress) -> None:
        with self._lock:
            self._view = replace(self._view, progress=progress)

    def cancel(self) -> None:
        self._cancel.set()

    def finish(self) -> None:
        self._closed = True
        self.cancel()
        self._pool.shutdown(wait=True)
