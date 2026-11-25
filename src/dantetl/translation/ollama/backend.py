"""Cancellable, single-flight Ollama translation on an owned I/O loop."""

import asyncio
import hashlib
import json
import threading
import time
from concurrent.futures import Future

import httpx2

from dantetl.i18n import tr
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.translation.backend import BatchResult, RequestRejected, Source
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.ollama.client import OllamaClient
from dantetl.translation.ollama.values import OllamaIdentity
from dantetl.translation.prompt import PROMPT_VERSION, translation_instruction, translation_payload


class OllamaBackend:
    batch_size = 1
    max_in_flight = 1

    def __init__(
        self,
        identity: OllamaIdentity,
        language: SourceLanguage,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self._identity, self._language = identity, language
        self._client = OllamaClient(identity.server, transport)
        self._cancel = threading.Event()
        # Construction runs on the translation worker, before cache lookup.
        asyncio.run(self._client.verify((identity.model,), self._cancel, time.monotonic() + 8))
        key = [
            identity.server,
            identity.model.name,
            identity.model.digest,
            identity.game,
            language.code,
            PROMPT_VERSION,
            "ollama-chat-v1",
        ]
        self.cache_identity = "ollama:" + hashlib.sha256(json.dumps(key).encode()).hexdigest()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ollama-chat", daemon=True)
        self._pending: Future[BatchResult] | None = None
        self._thread.start()

    def _run(self) -> None:
        with asyncio.Runner(loop_factory=lambda: self._loop):
            self._loop.run_forever()

    async def _translate(self, source: Source, target: TargetLanguage) -> BatchResult:
        identity = self._identity
        try:
            user = translation_payload(identity.game, source)
            if identity.model not in await self._client.models(self._cancel, time.monotonic() + 8):
                raise ValueError(
                    tr("Refresh the Ollama model list for {server}.", server=identity.server)
                )
            value = await self._client.request(
                "chat",
                {
                    "model": identity.model.name,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 768},
                    "messages": [
                        {
                            "role": "system",
                            "content": translation_instruction(self._language, target),
                        },
                        {"role": "user", "content": user},
                    ],
                },
                self._cancel,
                time.monotonic() + 120,
                65536,
            )
            message = value.get("message")
            if (
                value.get("model") != identity.model.name
                or value.get("done") is not True
                or value.get("done_reason") != "stop"
                or value.get("remote_host")
                or value.get("remote_model")
                or not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("tool_calls")
            ):
                raise ValueError(tr("Invalid Ollama answer."))
            answer = message.get("content")
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > 4096
                or "<think" in answer.lower()
                or "</think" in answer.lower()
            ):
                raise ValueError(tr("Invalid Ollama answer."))
            return {source: answer.strip()}, {}
        except ValueError as error:
            return {}, {source: str(error)}

    def submit_translation(
        self, sources: tuple[Source, ...], *, target: TargetLanguage = ENGLISH
    ) -> Future[BatchResult]:
        if len(sources) != 1 or self._pending is not None and not self._pending.done():
            raise RuntimeError("Ollama accepts one request at a time")
        self._pending = asyncio.run_coroutine_threadsafe(
            self._translate(sources[0], target), self._loop
        )
        return self._pending

    def translate_with_context(
        self, text: str, context: tuple[DialogueTurn, ...], *, target: TargetLanguage = ENGLISH
    ) -> str:
        source = (text, context)
        good, bad = self.submit_translation((source,), target=target).result()
        if source in bad:
            raise RequestRejected(bad[source])
        return good[source]

    def translate(
        self, texts: tuple[str, ...], *, target: TargetLanguage = ENGLISH
    ) -> tuple[str, ...]:
        return tuple(self.translate_with_context(text, (), target=target) for text in texts)

    def cancel_pending(self) -> None:
        if self._pending:
            self._pending.cancel()

    def close(self) -> None:
        self._cancel.set()
        self.cancel_pending()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
