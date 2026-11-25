"""One cancellable editor operation. Only verified, current Save results change routing options."""

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import replace

import httpx2

from dantetl.i18n import tr
from dantetl.translation.ollama.client import OllamaClient
from dantetl.translation.ollama.values import (
    MAX_ENABLED,
    OllamaConfig,
    OllamaModel,
    OllamaSnapshot,
    server_address,
)


class OllamaServices:
    def __init__(
        self,
        config: OllamaConfig,
        notify: Callable[[], None],
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self._view = OllamaSnapshot(config)
        self.unavailable: frozenset[OllamaModel] = frozenset()
        self._notify, self._transport = notify, transport
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._result: OllamaSnapshot | None = None
        self.begin()

    def snapshot(self) -> OllamaSnapshot:
        return self._view

    def begin(self) -> None:
        self.discard()
        config = self._view.config
        if config.models:
            self._view = replace(self._view, server=config.server, listings=config.listings)

    def discard(self) -> None:
        self._cancel.set()
        self._view = OllamaSnapshot(self._view.config, busy=self._thread is not None)

    def find(self, address: str) -> None:
        if self._thread is not None:
            return
        try:
            server = server_address(address)
        except ValueError as error:
            self._view = OllamaSnapshot(self._view.config, error=str(error))
            return
        self._start(OllamaSnapshot(self._view.config, server), None)

    def save(self, server: str, models: tuple[OllamaModel, ...]) -> None:
        view = self._view
        if self._thread or server != view.server or not server:
            return
        if (
            len(models) > MAX_ENABLED
            or len(set(models)) != len(models)
            or not set(models).issubset(view.models)
        ):
            self._view = replace(view, error=tr("Choose up to {count} models.", count=MAX_ENABLED))
            return
        self._start(view, models)

    def _start(self, view: OllamaSnapshot, selected: tuple[OllamaModel, ...] | None) -> None:
        self._cancel = cancel = threading.Event()
        self._view = replace(view, busy=True, saved=False, error="")
        client = OllamaClient(view.server, self._transport)

        def run() -> None:
            try:
                deadline = time.monotonic() + 30
                if selected is None:
                    listings = asyncio.run(client.catalog(cancel, deadline))
                    self._result = replace(view, listings=listings, error="", saved=False)
                else:
                    if selected:
                        asyncio.run(client.verify(selected, cancel, deadline))
                    enabled = set(selected)
                    entries = tuple(entry for entry in view.listings if entry.model in enabled)
                    self._result = replace(
                        view, config=OllamaConfig(view.server, entries), saved=True, error=""
                    )
            except InterruptedError:
                pass
            except ValueError as error:
                self._result = replace(
                    view,
                    server="" if selected is None else view.server,
                    saved=False,
                    error=str(error),
                )
            finally:
                self._notify()

        self._thread = threading.Thread(target=run, name="ollama-settings", daemon=True)
        self._thread.start()

    def advance(self) -> bool:
        if self._thread is None or self._thread.is_alive():
            return False
        result = self._result if not self._cancel.is_set() else None
        observed = False
        if result is not None:
            if result.saved:
                self.unavailable = frozenset()
                observed = True
            elif self._view.server == result.config.server and (
                not result.error or not result.server
            ):
                self.unavailable = frozenset(result.config.models).difference(result.models)
                observed = True
            self._view = result
        self._view = replace(self._view, busy=False)
        self._result = self._thread = None
        return observed
