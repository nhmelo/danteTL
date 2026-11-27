"""On-demand Cache tab maintenance: translation cache inspection/deletion and unused runtime files.

Each operation runs on its own worker; translation-cache work uses its own SQLite connection.
"""

import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from dantetl.runtime.setup_values import RuntimeFiles, RuntimeFilesView
from dantetl.translation.cache import TranslationCache
from dantetl.translation.cache_values import CacheView

logger = logging.getLogger(__name__)

# (remove, downloads) -> what is unused afterwards. Raises OSError while a runtime installs.
RuntimeFilesPass = Callable[[bool, bool], RuntimeFiles]


class CacheManagement:
    def __init__(
        self,
        path: Path,
        notify: Callable[[], None],
        runtime_files: RuntimeFilesPass | None = None,
    ) -> None:
        self._path, self._notify = path, notify
        self._runtime_pass = runtime_files
        self._lock = threading.Lock()
        self._view = CacheView()
        self._runtime = RuntimeFilesView()

    def snapshot(self) -> CacheView:
        with self._lock:
            return self._view

    def runtime_files(self) -> RuntimeFilesView:
        with self._lock:
            return self._runtime

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._view.busy or self._runtime.busy

    def refresh(self) -> None:
        self._start(False, None)
        self._start_runtime(remove=False, downloads=False)

    def clear(self, game: str | None) -> None:
        self._start(True, game)

    def clear_runtime_files(self) -> None:
        self._start_runtime(remove=True, downloads=True)

    def tidy_runtime_files(self) -> None:
        """Automatic pass: keeps setup downloads, which an interrupted install resumes from."""
        self._start_runtime(remove=True, downloads=False)

    def _start(self, clear: bool, game: str | None) -> None:
        with self._lock:
            if self._view.busy:
                return
            self._view = replace(self._view, busy=True, error="")
        threading.Thread(target=self._run, args=(clear, game), name="cache-maintenance").start()

    def _run(self, clear: bool, game: str | None) -> None:
        try:
            cache = TranslationCache(self._path)
            try:
                view = CacheView(cache.inventory(clear=clear, game=game))
            finally:
                cache.close()
        except (sqlite3.Error, OSError) as error:
            view = CacheView(error=str(error))
        with self._lock:
            self._view = view
        self._notify()

    def _start_runtime(self, *, remove: bool, downloads: bool) -> None:
        if self._runtime_pass is None:
            return
        with self._lock:
            if self._runtime.busy:
                return
            self._runtime = replace(self._runtime, busy=True, error="")
        threading.Thread(
            target=self._run_runtime,
            args=(self._runtime_pass, remove, downloads),
            name="runtime-files",
        ).start()

    def _run_runtime(self, work: RuntimeFilesPass, remove: bool, downloads: bool) -> None:
        try:
            view = RuntimeFilesView(work(remove, downloads))
        except OSError as error:
            logger.warning("Runtime file maintenance failed", exc_info=True)
            view = RuntimeFilesView(error=str(error))
        with self._lock:
            self._runtime = view
        self._notify()
