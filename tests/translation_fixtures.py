"""Fake translation engines, routes and requests with bounded waits."""

import threading
import time
from dataclasses import replace
from unittest.mock import Mock

from dantetl.application.cloud_services import CloudServices
from dantetl.application.model_inventory import InventoryData, InventorySnapshot, ModelInventory
from dantetl.application.preference_values import Preferences
from dantetl.application.translation_coordinator import TranslationCoordinator
from dantetl.domain import Rect
from dantetl.languages.target import ENGLISH
from dantetl.ocr.observation import TextTrack
from dantetl.translation.backend import BackendChoice
from dantetl.translation.cloud.values import CloudProfile
from dantetl.translation.grouping import TextBlock
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.selection import AssignmentChange
from dantetl.translation.values import TranslationState
from dantetl.translation.worker import TranslationSession


def track(text="日本語", revision=1, key=1) -> TextTrack:
    return TextTrack(key, text, Rect(100, 100, 200, 40), text_revision=revision)


def until(predicate) -> None:
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline, "Translation worker did not reach expected state"
        time.sleep(0.005)


class Backend:
    batch_size = 4

    def __init__(self, identity="fake-v1") -> None:
        self.cache_identity = identity
        self.calls: list[tuple[str, ...]] = []
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def translate(self, texts: tuple[str, ...], *, target=ENGLISH) -> tuple[str, ...]:
        self.calls.append(texts)
        self.entered.set()
        assert self.release.wait(3)
        return tuple(f"{self.cache_identity}: {text}" for text in texts)

    def close(self) -> None:
        self.closed = True


def stop(session: TranslationSession) -> None:
    session.request_stop()
    until(lambda: session.finished)


def block(key, text, *, dialogue=False):
    return TextBlock((TextTrack(key, text, Rect(0, key * 40, 200, 30)),), dialogue=dialogue)


class Engine:
    batch_size = 1

    def __init__(self, identity, release=None):
        self.cache_identity = identity
        self.release = release
        self.entered = threading.Event()
        self.closed = threading.Event()
        self.calls = []

    def translate(self, texts, *, target=ENGLISH):
        self.calls.append(texts)
        self.entered.set()
        if self.release:
            assert self.release.wait(3)
        return tuple(self.cache_identity + ": " + text for text in texts)

    def close(self):
        self.closed.set()


def remote_ref(key):
    return "cloud:cloud-" + key * 32


class Routes:
    def __init__(self, root, *, blocked=False):
        self.release = threading.Event()
        self.local = Mock(side_effect=lambda: Engine("local"))
        self.remote = {
            key: Mock(
                side_effect=lambda key=key: Engine(
                    key, self.release if key == "a" and blocked else None
                )
            )
            for key in ("a", "b", "c")
        }
        self.profiles = tuple(
            replace(CloudProfile.new("deepseek"), key="cloud-" + key * 32) for key in self.remote
        )
        self.choice = BackendChoice(
            "Local labels", self.local, "local-cuda", "weights-v1", selection_id="local"
        )
        self.inventory = Mock(spec=ModelInventory)
        self.inventory.snapshot.return_value = InventorySnapshot(
            loaded=True, data=InventoryData((self.choice,))
        )
        self.inventory.prepared.return_value = True
        self.cloud = Mock(spec=CloudServices)
        self.cloud.profiles = self.profiles
        self.cloud.revision = 0
        self.cloud.credential_revision.return_value = 0
        self.cloud.available.return_value = True
        self.cloud.create_backend.side_effect = lambda profile, game, language: self.remote[
            profile.key[-1]
        ]()
        self.owner = TranslationCoordinator(
            self.inventory,
            self.cloud,
            lambda choice, language: TranslationSession(
                choice.create, root / "cache", language=language
            ),
            RoutedTranslationSession,
            root / "cache",
            Preferences(model="none", short_model="local:local", long_model=remote_ref("a")),
        )
        self.owner.refresh_route()
        worker = self.owner.advance_activity(running=True, enabled=True).worker
        assert isinstance(worker, RoutedTranslationSession)
        self.worker = worker

    def assign(self, plan, reference):
        assert self.owner.assign(AssignmentChange(plan, reference)) in ("committed", "unchanged")
        self.owner.refresh_route()
        assert self.owner.advance_activity(running=True, enabled=True).worker is self.worker

    def ready(self):
        self.owner.advance_activity(running=True, enabled=True)
        return all(lane.state == TranslationState.READY for lane in self.worker.snapshot().lanes)

    def close(self):
        self.release.set()
        self.owner.lifecycle.shutdown()
        until(self.owner.lifecycle.finish_shutdown)
