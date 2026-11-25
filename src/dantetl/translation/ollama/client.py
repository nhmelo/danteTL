"""Bounded metadata and chat calls to a single explicit Ollama server."""

import json
import threading
import time

import httpx2

from dantetl.i18n import tr
from dantetl.translation.http_request import read_response
from dantetl.translation.ollama.values import OllamaListing, OllamaModel


class OllamaClient:
    def __init__(self, server: str, transport: httpx2.AsyncBaseTransport | None = None):
        self.server, self.transport = server, transport

    async def request(
        self, path: str, payload: dict | None, cancel: threading.Event, deadline: float, limit: int
    ) -> dict:
        try:
            status, data = await read_response(
                "GET" if payload is None else "POST",
                self.server + "/api/" + path,
                transport=self.transport,
                cancel=cancel,
                deadline=deadline,
                max_bytes=limit,
                skip_error_body=True,
                read_timeout=max(1, deadline - time.monotonic()),
                **({"json": payload} if payload is not None else {}),
            )
            if status != 200:
                raise ValueError
            value = json.loads(data)
            if not isinstance(value, dict) or value.get("error"):
                raise ValueError
            return value
        except (ValueError, RecursionError, TimeoutError, httpx2.HTTPError):
            raise ValueError(
                tr(
                    "Ollama request failed at {server}. Check the address and model in Settings.",
                    server=self.server,
                )
            ) from None

    async def catalog(self, cancel: threading.Event, deadline: float) -> tuple[OllamaListing, ...]:
        value = await self.request("tags", None, cancel, deadline, 1024 * 1024)
        rows = value.get("models")
        if not isinstance(rows, list) or len(rows) > 1024:
            raise ValueError(tr("Invalid Ollama model list."))
        models = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("remote_model") or row.get("remote_host"):
                continue
            caps = row.get("capabilities")
            if caps is not None and (not isinstance(caps, list) or "completion" not in caps):
                continue
            name, digest = row.get("name"), row.get("digest")
            if not isinstance(name, str) or not isinstance(digest, str):
                continue
            try:
                model = OllamaModel(name, digest)
            except ValueError:
                continue
            if model.name in models:
                raise ValueError(tr("Invalid Ollama model list."))
            models[model.name] = OllamaListing.from_metadata(
                model, row.get("details"), row.get("size")
            )
        return tuple(models[key] for key in sorted(models))

    async def models(self, cancel: threading.Event, deadline: float) -> frozenset[OllamaModel]:
        return frozenset(entry.model for entry in await self.catalog(cancel, deadline))

    async def verify(
        self, models: tuple[OllamaModel, ...], cancel: threading.Event, deadline: float
    ) -> None:
        available = await self.models(cancel, deadline)
        for model in models:
            if model not in available:
                raise ValueError(
                    tr("Refresh the Ollama model list for {server}.", server=self.server)
                )
            info = await self.request(
                "show", {"model": model.name}, cancel, deadline, 2 * 1024 * 1024
            )
            caps = info.get("capabilities")
            if (
                not isinstance(caps, list)
                or "completion" not in caps
                or info.get("remote_host")
                or info.get("remote_model")
            ):
                raise ValueError(
                    tr("{model} is not a server-hosted text generation model.", model=model.name)
                )
