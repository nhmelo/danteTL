"""Native Ollama contract, persistence and cancellation with an in-memory server."""

import asyncio
import json
import threading
from dataclasses import asdict, replace

import httpx2
import pytest

from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.translation.backend import RequestRejected
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.ollama.backend import OllamaBackend
from dantetl.translation.ollama.values import (
    DEFAULT_SERVER,
    OllamaIdentity,
    OllamaModel,
    server_address,
)

MODEL = OllamaModel("my-model:7b", "a" * 64)
REMOTE = "http://game-pc:11434"


class Server:
    def __init__(self):
        self.rows = [asdict(MODEL)]
        self.info: dict[str, object] = {"capabilities": ["completion", "vision"]}
        self.answer = {
            "model": MODEL.name,
            "message": {"role": "assistant", "content": "  Carte "},
            "done": True,
            "done_reason": "stop",
        }
        self.calls = []
        self.hold = ""
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.transport = httpx2.MockTransport(self.handle)

    async def handle(self, request):
        path = request.url.path
        self.calls.append((path, json.loads(request.content) if request.content else None))
        if self.hold == path:
            self.entered.set()
            try:
                await asyncio.sleep(60)
            finally:
                self.cancelled.set()
        value = {
            "/api/tags": {"models": self.rows},
            "/api/show": self.info,
            "/api/chat": self.answer,
        }[path]
        return httpx2.Response(200, json=value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://pc",
        "https://user:secret@pc",
        "https://pc:99999",
        "https://pc/?token=x",
        "https://ollama.com",
    ],
)
def test_address_rejects_invalid_or_credentialed_endpoint(url):
    with pytest.raises(ValueError):
        server_address(url)


def test_chat_checks_model_context_target_completion_and_cancels():
    server = Server()
    identity = OllamaIdentity(REMOTE, MODEL, "Adventure")
    backend = OllamaBackend(identity, DEFAULT_SOURCE, server.transport)
    try:
        result = backend.translate_with_context(
            "地図", (DialogueTurn("次へ", "Next"),), target=TARGET_LANGUAGES["fr"]
        )
        assert result == "Carte"
        path, body = server.calls[-1]
        assert path == "/api/chat" and body["stream"] is False and body["model"] == MODEL.name
        system, user = body["messages"]
        assert "French" in system["content"]
        assert json.loads(user["content"]) == {
            "CURRENT": "地図",
            "RECENT": [["次へ", "Next"]],
            "GAME": "Adventure",
        }
        server.answer["done_reason"] = "length"
        with pytest.raises(RequestRejected):
            backend.translate(("地図",))
        server.rows[0]["digest"] = "b" * 64
        with pytest.raises(RequestRejected, match="Refresh"):
            backend.translate(("設定",))
        assert server.calls[-1][0] == "/api/tags"
        server.rows[0]["digest"] = MODEL.digest
        server.hold = "/api/chat"
        future = backend.submit_translation((("設定", ()),))
        assert server.entered.wait(2)
        assert future.cancel() and server.cancelled.wait(1)
    finally:
        backend.close()


def test_changed_or_cloud_model_fails_before_worker_can_reuse_cache():
    server = Server()
    identity = OllamaIdentity(REMOTE, MODEL, "")
    server.info["remote_host"] = "https://ollama.com"
    with pytest.raises(ValueError):
        OllamaBackend(identity, DEFAULT_SOURCE, server.transport)
    server.rows.clear()
    with pytest.raises(ValueError, match="Refresh"):
        OllamaBackend(identity, DEFAULT_SOURCE, server.transport)


def test_ollama_reuses_persistent_cache_only_for_matching_translation_inputs(tmp_path):
    from test_translation import stop, track, until

    from dantetl.languages.selection import source_language
    from dantetl.translation.grouping import TextBlock
    from dantetl.translation.worker import TranslationSession

    server = Server()
    identity = OllamaIdentity(REMOTE, MODEL, "Adventure")

    def translate(route=identity, *, source="es", target="en", text="Menu", prior=""):
        language = source_language(source)
        server.rows = [asdict(route.model)]
        server.answer["model"] = route.model.name
        session = TranslationSession(
            lambda: OllamaBackend(route, language, server.transport),
            tmp_path / "cache.sqlite3",
            language=language,
        )
        session.start()
        try:
            for i, line in enumerate([prior, text] if prior else [text]):
                item = TextBlock((track(line, key=i + 1),), dialogue=True)
                session.update((item,), target=TARGET_LANGUAGES[target])
                until(
                    lambda expected=line: any(
                        r.request.source == expected for r in session.snapshot().results
                    )
                )
            return session.snapshot().cache_hits
        finally:
            stop(session)

    assert translate() == 0
    assert translate(replace(identity, attempt=1)) == 1
    assert len([p for p, _ in server.calls if p == "/api/chat"]) == 1
    for route in (
        replace(identity, server=DEFAULT_SERVER),
        replace(identity, model=replace(MODEL, name="other:7b")),
        replace(identity, model=replace(MODEL, digest="b" * 64)),
        replace(identity, game="Different game"),
    ):
        assert translate(route) == 0
    assert translate(source="fr") == 0
    assert translate(target="pt-BR") == 0
    assert translate(text="Mapa") == 0
    assert translate(prior="Otra vez.") == 0
    chats = [body for path, body in server.calls if path == "/api/chat"]
    assert json.loads(chats[-1]["messages"][1]["content"])["RECENT"] == [["Otra vez.", "Carte"]]
    assert translate(prior="Otra vez.") > 0
    assert [body for path, body in server.calls if path == "/api/chat"] == chats
