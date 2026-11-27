"""Native Ollama contract, persistence and cancellation with an in-memory server."""

import json
import threading
import time
from contextlib import suppress
from dataclasses import asdict, replace

import pytest
from remote_fixtures import MODEL, REMOTE, Server, settle

from dantetl.application.ollama_services import OllamaServices
from dantetl.application.preference_values import Preferences
from dantetl.application.preferences import PreferenceStore
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.translation.backend import RequestRejected
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.ollama.backend import OllamaBackend
from dantetl.translation.ollama.values import (
    DEFAULT_SERVER,
    OllamaConfig,
    OllamaIdentity,
    OllamaListing,
    server_address,
)


def test_catalog_save_is_explicit_and_stale_completion_cannot_commit(tmp_path):
    server = Server()
    server.rows.extend(
        [
            {**asdict(MODEL), "name": "vector", "capabilities": ["embedding"]},
            {**asdict(MODEL), "name": "aliased", "remote_model": "cloud"},
            {**asdict(MODEL), "name": "alias:cloud"},
        ]
    )
    service = OllamaServices(OllamaConfig(), lambda: None, server.transport)
    service.begin()
    assert service.snapshot().config.server == DEFAULT_SERVER and not server.calls
    service.find(REMOTE + "/")
    found = settle(service)
    assert found.server == REMOTE and found.models == (MODEL,) and not found.config.models
    server.info = {"capabilities": ["embedding"]}
    service.save(REMOTE, found.models)
    assert settle(service).error and not service.snapshot().config.models
    server.info = {"capabilities": ["completion"]}
    service.save(REMOTE, found.models)
    config = settle(service).config
    ref = MODEL.reference(REMOTE)
    prefs = replace(Preferences(), ollama=config, short_model=ref, long_model=ref)
    path = tmp_path / "settings.json"
    store = PreferenceStore(path)
    store.save(prefs)
    deadline = time.monotonic() + 3
    while not store.idle:
        time.sleep(0.002)
        assert time.monotonic() < deadline
    assert PreferenceStore(path).current == prefs
    assert ref != MODEL.reference(DEFAULT_SERVER)
    service.find(DEFAULT_SERVER)
    assert settle(service).models == (MODEL,)
    server.hold = "/api/show"
    service.save(DEFAULT_SERVER, (MODEL,))
    assert server.entered.wait(2)
    service.discard()
    assert settle(service).config == config and server.cancelled.is_set()
    assert all(path != "/api/chat" for path, _ in server.calls)
    server.rows.clear()
    service.find(REMOTE)
    assert settle(service).models == () and service.snapshot().config == config
    assert service.unavailable == {MODEL}
    server.rows.append(asdict(MODEL))
    service.find(REMOTE)
    assert settle(service).models == (MODEL,) and not service.unavailable


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


def test_live_ollama_lane_uses_saved_server_and_retires_only_its_worker(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from translation_fixtures import Routes, block, remote_ref, until

    from dantetl.translation.plan import TranslationPlan

    entered, release = threading.Event(), threading.Event()
    healthy = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/tags"
            self.reply({"models": [asdict(MODEL)]}, 200 if healthy.is_set() else 503)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == MODEL.name
            if self.path == "/api/show":
                self.reply({"capabilities": ["completion"]})
                return
            assert self.path == "/api/chat"
            entered.set()
            release.wait(5)
            self.reply(
                {
                    "model": MODEL.name,
                    "done": True,
                    "done_reason": "stop",
                    "message": {"role": "assistant", "content": "Retired answer"},
                }
            )

        def reply(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(ConnectionError):
                self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        routes = Routes(tmp_path)
        server = f"http://127.0.0.1:{http.server_port}"
        config = OllamaConfig(server, (OllamaListing(MODEL),))
        ref = MODEL.reference(server)
        try:
            routes.owner.update_ollama(config, frozenset())
            routes.assign(TranslationPlan("local:local", ref), ref)

            def failed():
                routes.worker.advance()
                return any(
                    lane.reference == ref and lane.state.value == "failed"
                    for lane in routes.worker.snapshot().lanes
                )

            until(failed)
            healthy.set()
            routes.owner.update_ollama(config, frozenset())
            routes.owner.refresh_route()
            until(routes.ready)
            items = (block(1, "設定"), block(2, "一緒に行こう。", dialogue=True))
            routes.worker.update(items)
            assert entered.wait(3)
            until(lambda: len(routes.worker.snapshot().results) == 1)
            assert routes.worker.snapshot().results[0].english == "local: 設定"
            routes.assign(TranslationPlan("local:local", remote_ref("b")), remote_ref("b"))
            release.set()
            until(routes.ready)
            routes.worker.update(items)
            until(lambda: len(routes.worker.snapshot().results) == 2)
            assert {r.english for r in routes.worker.snapshot().results} == {
                "local: 設定",
                "b: 一緒に行こう。",
            }
            assert routes.local.call_count == 1
        finally:
            release.set()
            routes.close()
            http.shutdown()
            thread.join()


def test_ollama_reuses_persistent_cache_only_for_matching_translation_inputs(tmp_path):
    from translation_fixtures import stop, track, until

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
