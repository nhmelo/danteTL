"""Explicit non-Qt composition with controlled completions, no providers or model files."""

import time
from unittest.mock import Mock

from test_live_session import Capture, Ocr, Visual, frame
from test_model_lifecycle import Worker

from dantetl.application.cache_management import CacheManagement
from dantetl.application.cloud_services import CloudServices
from dantetl.application.live_session import LiveSession
from dantetl.application.model_inventory import InventoryData, ModelInventory, SourceModels
from dantetl.application.ollama_services import OllamaServices
from dantetl.application.preference_values import Preferences
from dantetl.application.preferences import PreferenceStore
from dantetl.application.session import ApplicationSession
from dantetl.application.setup_coordinator import SetupCoordinator, SetupDependencies
from dantetl.application.shutdown import ShutdownCoordinator
from dantetl.application.translation_coordinator import TranslationCoordinator
from dantetl.languages.selection import source_language
from dantetl.ocr.worker import OcrChoice
from dantetl.runtime.model_store import ModelOperation
from dantetl.runtime.setup_values import ModelProgress
from dantetl.runtime.values import PROFILES
from dantetl.translation.cloud.chatgpt_auth import ChatGPTConnections
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.values import TranslationSnapshot, TranslationState


class InventorySource:
    def __init__(self, data):
        self.data = data
        self.inspect = lambda: self.data

    def __call__(self):
        return self.inspect()


class Route(RoutedTranslationSession):
    """Controlled completion at the worker boundary, without changing owner state."""

    def __init__(self):
        self.ended = False
        self.state = TranslationState.READY
        self.results = ()
        self.updates = []
        self.stops = 0
        self.changes = []
        self.generation = 0

    def start(self):
        pass

    @property
    def finished(self):
        return self.ended

    def update(self, tracks, **kwargs):
        self.updates.append((tracks, kwargs))

    def invalidate(self):
        self.results = ()

    def snapshot(self):
        return TranslationSnapshot(self.state, self.results, backend_generation=self.generation)

    @property
    def stopping(self):
        return self.state == TranslationState.STOPPING

    def reconfigure(self, plan, factories, **kwargs):
        self.changes.append((plan, factories, kwargs))
        self.results = ()
        self.generation += 1

    def advance(self):
        pass

    def request_stop(self):
        self.stops += 1
        self.state = TranslationState.STOPPING

    def finish(self):
        self.ended = True
        self.state = TranslationState.STOPPED


class PreparedWorker(Worker):
    def invalidate(self):
        pass


class Operation(ModelOperation):
    def __init__(self, work):
        self.work = work
        self.value = ModelProgress()
        self.cancelled = False

    def start(self):
        pass

    def snapshot(self):
        return self.value

    def cancel(self):
        self.cancelled = True

    def finish(self, **kwargs):
        self.value = ModelProgress(done=True, **kwargs)


class Replay:
    live: LiveSession
    translation: TranslationCoordinator
    ocr: Ocr
    capture: Capture
    visual: Visual

    def __init__(
        self,
        root,
        *,
        preferences=None,
        data=None,
        loaded=True,
        ocr_choices=None,
        source_models=None,
    ):
        preferences = preferences or Preferences()
        data = data or InventoryData()
        self.preferences = PreferenceStore(root / "preferences.json")
        self.preferences.initial = preferences
        self.receipts, self.operations, self.workers, self.routes = [], [], [], []
        self.source = InventorySource(data)
        self.inventory = ModelInventory(
            self.source,
            self.receipts.append,
            initial=data if loaded else None,
            interval=float("inf"),
        )
        if loaded:
            self.settle_inventory()
        self.cloud = CloudServices(
            CloudProfiles(root / "cloud.json"),
            Mock(side_effect=AssertionError("credentials")),
            ChatGPTConnections(ProtectedStore(root / "accounts")),
        )

        def operation(work):
            value = Operation(work)
            self.operations.append(value)
            return value

        self.dependencies = SetupDependencies(
            operation, Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()
        )
        self.setup = SetupCoordinator(self.inventory, (), PROFILES, self.dependencies)

        def create(choice, language):
            worker = PreparedWorker(choice)
            self.workers.append(worker)
            return worker

        def route(*args, **kwargs):
            self.routes.append((args, kwargs))
            worker = Route()
            self.route_workers.append(worker)
            return worker

        self.route_workers = []
        self.recognizer = Mock()
        self.recognizers = []
        choices = ocr_choices or (OcrChoice("CPU", self.recognizer, key="cpu"),)

        language = source_language(preferences.source_language)
        self.translation = TranslationCoordinator(
            self.inventory, self.cloud, create, route, root / "cache", preferences
        )
        self.capture, self.ocr = Capture(), Ocr()
        self.visual = Visual(self.ocr, language=language)

        def recognize(factory):
            if self.recognizers:
                self.ocr = Ocr()
                self.visual = Visual(self.ocr)
            self.recognizers.append(factory)
            return self.ocr

        self.live = LiveSession(
            ocr_factory=recognize, visual_factory=lambda: self.visual, language=language
        )
        self.restart = Mock()
        self.restart.finished = False

        def cancel():
            self.setup.close()
            self.inventory.close()
            self.cloud.cancel()

        self.shutdown = ShutdownCoordinator(
            self.live,
            self.translation.lifecycle,
            setup_busy=lambda: self.setup.busy or self.cloud.busy or not self.inventory.finished,
            cancel_setup=cancel,
            preferences_idle=lambda: self.preferences.idle,
            hardware_finished=lambda: True,
            release_hotkeys=Mock(),
            restart=lambda _: self.restart,
        )
        self.windows = Mock()
        self.windows.is_target.return_value = True
        self.windows.is_foreground.return_value = True
        self.windows.geometry.return_value = frame(1).geometry
        self.capture_calls = []

        def create_capture(*args):
            if self.capture_calls:
                self.capture = Capture()
            self.capture_calls.append(args)
            return self.capture

        self.app = ApplicationSession(
            windows=self.windows,
            live=self.live,
            translation=self.translation,
            ocr_choices=choices,
            source_models=source_models or (lambda _: SourceModels((), choices, self.source)),
            inventory=self.inventory,
            cloud=self.cloud,
            ollama=OllamaServices(preferences.ollama, lambda: None),
            setup=self.setup,
            preferences=self.preferences,
            shutdown=self.shutdown,
            cache=CacheManagement(root / "cache", lambda: None),
            capture=create_capture,
        )

    def discover(self, data):
        self.source.data = data
        self.inventory.invalidate()
        self.settle_inventory()

    def settle_inventory(self):
        self.inventory.advance(0)
        deadline = time.monotonic() + 3
        while not self.inventory.finished:
            assert time.monotonic() < deadline
            self.inventory.advance(0)
            time.sleep(0.001)

    def close(self):
        self.app.request_close()
        self.ocr.finished = self.visual.finished = self.capture.finished = True
        for worker in self.workers:
            worker.ended = True
        for worker in self.route_workers:
            worker.finish()
        if self.translation.lifecycle.publication:
            # Route fakes and local worker fakes expose distinct completion fields.
            if isinstance(self.translation.lifecycle.publication, Worker):
                self.translation.lifecycle.publication.ended = True
            elif isinstance(self.translation.lifecycle.publication, Mock):
                self.translation.lifecycle.publication.finished = True
        for operation in self.operations:
            operation.finish(cancelled=True)
        self.setup.advance()
        self.inventory.close()
        deadline = time.monotonic() + 3
        while not self.inventory.finished or not self.preferences.idle:
            self.inventory.advance(time.monotonic())
            assert time.monotonic() < deadline
            time.sleep(0.001)
