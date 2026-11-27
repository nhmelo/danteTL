"""Explicit, disposable composition for GUI tests; no personal account discovery."""

import time
import weakref
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from live_fixtures import Capture, frame

from dantetl.application.cloud_services import CloudServices
from dantetl.application.live_session import LiveSession
from dantetl.application.values import Scene
from dantetl.domain import CaptureSnapshot, CaptureState
from dantetl.translation.cloud.chatgpt_auth import ChatGPTConnections
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.grouping import TextBlock

_applications = weakref.WeakKeyDictionary()
_sources = weakref.WeakKeyDictionary()


class SceneFeed(LiveSession):
    """Scripted pipeline output for UI wiring; real pipeline has headless replays."""

    def __init__(self):
        super().__init__()
        self.blocks: tuple[TextBlock, ...] = ()

    def scene(self, *args, **kwargs):
        return Scene(
            tuple(t for block in self.blocks for t in block.lines), self.blocks, self.blocks
        )


class ViewCapture(Capture):
    def request_stop(self):
        super().request_stop()
        self.finished = True


class InventoryFeed:
    def __init__(self, inspect):
        self.inspect = inspect
        self.backends = None
        self.ocr = None

    def __call__(self):
        data = self.inspect()
        if self.backends is not None:
            data = replace(data, backends=self.backends)
        if self.ocr is not None:
            data = replace(data, files=tuple(f for f in data.files if f.key != "ocr") + (self.ocr,))
        return data


class NoCredentials:
    def read(self, key):
        raise AssertionError("Unexpected credential read in fixture")

    def write(self, key, value):
        raise AssertionError("Unexpected credential write in fixture")

    def delete(self, key):
        raise AssertionError("Unexpected credential deletion in fixture")


def cloud_service(*, catalog, secrets, connections):
    service = CloudServices(catalog, secrets, connections)
    service.start()
    deadline = time.monotonic() + 3
    while service.busy:
        service.advance()
        assert time.monotonic() < deadline
        time.sleep(0.001)
    return service


def create_window(windows, recognizer, *backends_args, cloud=None, updates=None, **kwargs):
    if kwargs.pop("scene_feed", False):
        kwargs["live_session"] = SceneFeed()
    args = (windows, recognizer, *backends_args)
    from dantetl.ui.main_window import MainWindow

    temporary = TemporaryDirectory(prefix="dantetl-test-")
    root = Path(temporary.name)
    kwargs.setdefault("preferences_path", root / "preferences.json")
    kwargs.setdefault("cache_path", root / "translations.sqlite3")
    cloud = cloud or cloud_service(
        catalog=CloudProfiles(root / "cloud.json"),
        secrets=NoCredentials,
        connections=ChatGPTConnections(ProtectedStore(root / "accounts")),
    )
    from functools import partial

    from dantetl.application.model_discovery import inspect_models
    from dantetl.application.model_inventory import ModelInventory, SourceModels
    from dantetl.ocr.worker import OcrChoice
    from dantetl.runtime.model_store import mark_prepared

    backends = args[2] if len(args) > 2 else kwargs.get("backends", ())
    bundles = kwargs.get("models", ())
    initial_probe = partial(inspect_models, bundles, lambda: backends, include_imports=False)
    probe = partial(
        inspect_models,
        bundles,
        kwargs.get("backend_provider") or (lambda: backends),
        include_imports=False,
    )
    source = InventoryFeed(probe)
    inventory = ModelInventory(source, mark_prepared, initial=initial_probe())
    from dantetl.composition import application_session, setup_coordinator
    from dantetl.ui.desktop_driver import DesktopDriver

    setup = setup_coordinator(inventory, bundles, root)
    driver = DesktopDriver()
    options = {
        k: v for k, v in kwargs.items() if k not in {"models", "backends", "backend_provider"}
    }
    choices = options.get("ocr_backends") or (OcrChoice("CPU", recognizer, key="cpu"),)
    options.setdefault("source_models", lambda _: SourceModels(bundles, choices, source))
    application = application_session(
        *args[:2], cloud=cloud, inventory=inventory, setup=setup, notify=driver.notify, **options
    )
    window = MainWindow(
        application,
        args[0],
        driver,
        hardware=kwargs.get("hardware"),
        hotkeys=kwargs.get("hotkeys"),
        updates=updates,
    )
    weakref.finalize(cloud, temporary.cleanup)
    _applications[window], _sources[window] = weakref.ref(application), source
    return window


def cloud_panel(**kwargs):
    from PySide6.QtCore import QTimer

    from dantetl.ui.cloud_settings import CloudSettings

    panel = CloudSettings(cloud_service(**kwargs))
    timer = QTimer(panel)
    timer.timeout.connect(cloud_owner(panel).advance)
    timer.timeout.connect(panel.refresh_view)
    timer.timeout.connect(panel.handle_desktop_actions)
    timer.start(10)
    return panel


def model_manager(*args, **kwargs):
    from dantetl.ui.model_manager import ModelManager

    temporary = TemporaryDirectory(prefix="dantetl-test-")
    root = Path(temporary.name)
    service = cloud_service(
        catalog=CloudProfiles(root / "cloud.json"),
        secrets=NoCredentials,
        connections=ChatGPTConnections(ProtectedStore(root / "accounts")),
    )
    from dantetl.application.model_discovery import inspect_models
    from dantetl.application.model_inventory import ModelInventory
    from dantetl.composition import setup_coordinator

    inventory = ModelInventory(
        lambda: inspect_models(args[0], lambda: ()),
        lambda key: None,
        initial=inspect_models(args[0], lambda: ()),
    )
    setup = setup_coordinator(inventory, args[0], root)
    manager = ModelManager(setup, *args[1:], cloud=service, **kwargs)
    weakref.finalize(service, temporary.cleanup)
    return manager


def find_model(combo, choice):
    from dantetl.application.translation_coordinator import model_reference
    from dantetl.translation.selection import ModelOption

    reference = choice.reference if isinstance(choice, ModelOption) else model_reference(choice)
    return next(
        (
            i
            for i in range(combo.count())
            if isinstance(combo.itemData(i), ModelOption)
            and combo.itemData(i).reference == reference
        ),
        -1,
    )


def settle_setup(service):
    deadline = time.monotonic() + 3
    service.advance()
    service.inventory.advance(time.monotonic())
    while service.busy or not service.inventory.finished:
        service.advance()
        service.inventory.advance(time.monotonic())
        assert time.monotonic() < deadline
        time.sleep(0.001)


def replace_backends(window, choices):
    _sources[window].backends = choices
    refresh_inventory(window)


def refresh_inventory(window):
    application = app_owner(window)
    inventory = application.inventory
    application.refresh_models()
    deadline = time.monotonic() + 3
    inventory.advance(time.monotonic())
    while not inventory.finished:
        inventory.advance(time.monotonic())
        assert time.monotonic() < deadline
        time.sleep(0.001)
    application.advance(time.monotonic())


def show_scene(window, blocks=()):
    application = app_owner(window)
    live = application.live
    assert isinstance(live, SceneFeed), "Inject scene_feed=True for a presentation test"
    if live.capture is None:
        capture = ViewCapture()
        source = frame(1)
        source = replace(source, bgra=b"\xff" * (1200 * 900 * 4))
        capture.current = CaptureSnapshot(CaptureState.RUNNING, source)
        live.start(capture, None)
    application.windows.geometry.return_value = frame(1).geometry
    application.windows.is_foreground.return_value = True
    live.blocks = blocks
    window.driver.refresh(inspect=True)


def app_owner(window):
    application = _applications[window]()
    assert application is not None
    return application


def settings(window):
    from dantetl.ui.settings_window import SettingsWindow

    panel = window.findChild(SettingsWindow)
    assert panel is not None
    return panel


def cloud_owner(panel):
    from typing import cast

    from dantetl.application.cloud_services import CloudServices

    return cast(CloudServices, panel.service)


def setup_owner(panel):
    from typing import cast

    from dantetl.application.setup_coordinator import SetupCoordinator

    return cast(SetupCoordinator, panel.service)


def set_ocr_available(window, ready):

    from dantetl.application.model_values import ModelFile

    ocr = ModelFile(
        "ocr",
        "OCR fixture",
        "",
        "recognition",
        0,
        ready,
        False,
        False,
        False,
        "",
        "",
        "",
        "",
        None,
        False,
    )
    _sources[window].ocr = ocr
    refresh_inventory(window)
