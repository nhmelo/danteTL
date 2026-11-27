import asyncio
import threading
import time

import httpx2
import pytest

from dantetl.domain import Rect
from dantetl.ocr.observation import TextTrack
from dantetl.translation.cloud.auth_work import AuthInterrupted, AuthWork
from dantetl.translation.cloud.backend import CloudBackend
from dantetl.translation.cloud.chatgpt_auth import PLAN_SCOPE, ChatGPTConnections
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.values import TranslationState
from dantetl.translation.worker import TranslationSession


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def account(tmp_path):
    service = ChatGPTConnections(ProtectedStore(tmp_path / "credentials"))
    key = "account-" + "a" * 32
    service.store.write(
        key,
        {
            "access_token": "fixture-old",
            "refresh_token": "fixture-refresh",
            "client_id": "oaiapp_fixture",
            "scopes": [PLAN_SCOPE],
        },
    )
    return service, key


@pytest.mark.parametrize("phase", ["headers", "body"])
@pytest.mark.parametrize("action", ["stop", "switch"])
def test_translation_owner_drains_slow_auth_without_losing_saved_tokens(tmp_path, phase, action):
    service, key = account(tmp_path)
    before = service.store.read(key)
    entered, closed = threading.Event(), threading.Event()

    class SlowBody(httpx2.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            while True:
                yield b" "
                await asyncio.sleep(0.02)  # Always below a socket timeout; still cancellable.

        async def aclose(self):
            closed.set()

    async def handle(request):
        if phase == "headers":
            entered.set()
            try:
                await asyncio.sleep(60)
            finally:
                closed.set()
        return httpx2.Response(200, stream=SlowBody())

    def token(work):
        work.transport = httpx2.MockTransport(handle)
        return service.access_token(key, work)

    profile = CloudProfile(
        "cloud-" + "b" * 32, "Plan", CloudProvider.CHATGPT, "gpt-6-luna", account=key
    )
    backend = CloudBackend(profile, "plan", access_token=token)
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()
    try:
        session.update((TextTrack(1, "設定", Rect(0, 0, 100, 30)),))
        assert entered.wait(2)
        if action == "stop":
            session.request_stop()
            until(lambda: session.finished)
            assert not session.snapshot().results
        else:

            class Local:
                cache_identity = "replacement-fixture"
                batch_size = 1

                def translate(self, texts, *, target=None):
                    return tuple("Settings" for _ in texts)

                def close(self):
                    pass

            session.switch_backend(Local)
            until(lambda: bool(session.snapshot().results))
            assert session.snapshot().results[0].english == "Settings"
        assert closed.is_set() and not backend._thread.is_alive()
        assert all(not thread.is_alive() for thread in backend._auth_owner._threads)
        assert service.store.read(key) == before
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_total_auth_deadline_rejects_only_request_and_preserves_credentials(tmp_path):
    service, key = account(tmp_path)
    closed = threading.Event()

    async def handle(request):
        try:
            await asyncio.sleep(60)
        finally:
            closed.set()
        return httpx2.Response(500)

    def token(work):
        work.transport = httpx2.MockTransport(handle)
        return service.access_token(key, work)

    profile = CloudProfile(
        "cloud-" + "b" * 32, "Plan", CloudProvider.CHATGPT, "gpt-6-luna", account=key
    )
    backend = CloudBackend(profile, "plan", access_token=token, timeout=0.15)
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()
    try:
        session.update((TextTrack(1, "設定", Rect(0, 0, 100, 30)),))
        until(lambda: bool(session.snapshot().rejected))
        assert session.snapshot().state == TranslationState.READY
        assert "timed out" in session.snapshot().message
        until(closed.is_set)
        assert service.store.read(key)["refresh_token"] == "fixture-refresh"
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_refresh_and_model_discovery_share_one_absolute_budget(tmp_path):
    service, key = account(tmp_path)
    seen = []

    async def handle(request):
        seen.append(request.url.path)
        await asyncio.sleep(0.09)
        return httpx2.Response(
            200,
            json={
                "access_token": "fixture-new",
                "refresh_token": "rotated",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    work = AuthWork(deadline=time.monotonic() + 0.15, transport=httpx2.MockTransport(handle))
    with pytest.raises(AuthInterrupted, match="timed out"):
        service.refresh_models(key, work)
    assert len(seen) == 2
    # A successfully rotated token must be persisted even when later catalog discovery times out.
    assert service.store.read(key)["refresh_token"] == "rotated"
    assert not service.store.read(key).get("models")
