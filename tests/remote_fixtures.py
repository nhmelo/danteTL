"""Fake cloud credentials and a local Ollama server; no network or accounts."""

import asyncio
import json
import threading
import time
from dataclasses import asdict

import httpx2

from dantetl.application.cloud_services import CloudServices
from dantetl.translation.cloud.chatgpt_auth import ChatGPTConnections
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.ollama.values import (
    OllamaModel,
)


class Secrets:
    def __init__(self):
        self.values = {}
        self.threads = []

    def read(self, key):
        return self.values.get(key)

    def write(self, key, secret):
        self.threads.append(threading.get_ident())
        self.values[key] = secret

    def delete(self, key):
        self.values.pop(key, None)


def finish(service):
    deadline = time.monotonic() + 3
    while service.busy:
        service.advance()
        assert time.monotonic() < deadline
        time.sleep(0.001)
    return service.snapshot()


def create(tmp_path):
    secrets = Secrets()
    service = CloudServices(
        CloudProfiles(tmp_path / "cloud.json"),
        lambda: secrets,
        ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    service.start()
    finish(service)
    return service, secrets


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


def settle(service):
    deadline = time.monotonic() + 3
    while service.snapshot().busy:
        service.advance()
        assert time.monotonic() < deadline
        time.sleep(0.002)
    return service.snapshot()
