"""Production dependency assembly. Views never discover accounts or construct services."""

import os
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from dantetl.application.cache_management import CacheManagement
from dantetl.application.cloud_services import CloudServices
from dantetl.application.hotkeys import Hotkeys
from dantetl.application.live_session import LiveSession
from dantetl.application.model_discovery import inspect_models
from dantetl.application.model_factories import ocr_backends as source_ocr_backends
from dantetl.application.model_inventory import ModelInventory, SourceModels
from dantetl.application.ollama_services import OllamaServices
from dantetl.application.preferences import PreferenceStore
from dantetl.application.session import ApplicationSession
from dantetl.application.setup_coordinator import SetupCoordinator
from dantetl.application.shutdown import ShutdownCoordinator
from dantetl.application.translation_coordinator import TranslationCoordinator
from dantetl.application.updates import Updates
from dantetl.capture.windows import Windows
from dantetl.capture.worker import CaptureSession
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.visual import VisualSession
from dantetl.ocr.worker import OcrChoice, OcrSession, Recognizer
from dantetl.paths import state_root
from dantetl.runtime.hardware import HardwareCheck
from dantetl.runtime.model_store import ModelBundle, mark_prepared, model_catalog
from dantetl.runtime.restart import RestartOperation, RuntimeRestart
from dantetl.runtime.setup_values import RuntimeFiles
from dantetl.translation.cloud.chatgpt_auth import ChatGPTConnections
from dantetl.translation.cloud.credentials import WindowsSecrets
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.worker import TranslationSession


def source_models_for(language: SourceLanguage) -> SourceModels:
    bundles = model_catalog(language)
    return SourceModels(
        bundles, source_ocr_backends(language), partial(inspect_models, bundles, language=language)
    )


def application_session(
    windows: Windows,
    recognizer_factory: Callable[[], Recognizer],
    *,
    cloud: CloudServices,
    inventory: ModelInventory,
    setup: SetupCoordinator,
    notify: Callable[[], None],
    preferences_path: Path | None = None,
    preferences: PreferenceStore | None = None,
    cache_path: Path | None = None,
    runtime_root: Path | None = None,
    ocr_backends: tuple[OcrChoice, ...] = (),
    source_models: Callable[[SourceLanguage], SourceModels],
    hardware: HardwareCheck | None = None,
    hotkeys: Hotkeys | None = None,
    live_session: LiveSession | None = None,
    language: SourceLanguage = DEFAULT_SOURCE,
) -> ApplicationSession:
    preferences = preferences or PreferenceStore(preferences_path)
    cache = cache_path or state_root() / "data/game-translations.sqlite3"
    ollama = OllamaServices(preferences.current.ollama, notify)
    cache_management = CacheManagement(
        cache, notify, runtime_files_pass(runtime_root) if runtime_root else None
    )
    initial_choices = ocr_backends or (OcrChoice("CPU", recognizer_factory, key="cpu"),)

    live = live_session or LiveSession(
        ocr_factory=lambda factory: OcrSession(factory, require_ack=True, on_change=notify),
        visual_factory=lambda: VisualSession(on_change=notify),
        language=language,
    )
    translation = TranslationCoordinator(
        inventory,
        cloud,
        lambda choice, source: TranslationSession(
            choice.create, cache, on_change=notify, language=source
        ),
        RoutedTranslationSession,
        cache,
        preferences.current,
        on_change=notify,
    )

    def cancel_setup() -> None:
        inventory.close()
        setup.close()
        cloud.cancel()

    shutdown = ShutdownCoordinator(
        live,
        translation.lifecycle,
        setup_busy=lambda: (
            setup.busy or cloud.busy or ollama.snapshot().busy or not inventory.finished
        ),
        cancel_setup=cancel_setup,
        preferences_idle=lambda: preferences.idle and not cache_management.busy,
        hardware_finished=lambda: hardware is None or hardware.finished,
        release_hotkeys=lambda: hotkeys.release() if hotkeys else None,
        restart=lambda target: RestartOperation(state_root(), target, create=RuntimeRestart),
    )
    return ApplicationSession(
        windows=windows,
        live=live,
        translation=translation,
        ocr_choices=initial_choices,
        source_models=source_models,
        inventory=inventory,
        cloud=cloud,
        ollama=ollama,
        setup=setup,
        preferences=preferences,
        shutdown=shutdown,
        cache=cache_management,
        capture=lambda target, fps, border: CaptureSession(
            target, windows, fps=fps, show_border=border, on_change=notify
        ),
        notify=notify,
        hardware=hardware.snapshot if hardware else lambda: None,
    )


def runtime_files_pass(root: Path) -> Callable[[bool, bool], RuntimeFiles]:
    from dantetl.runtime.setup import unused_runtime_files

    return lambda remove, downloads: unused_runtime_files(root, remove=remove, downloads=downloads)


def setup_coordinator(inventory: ModelInventory, models: tuple[ModelBundle, ...], root: Path):
    from dantetl.application.setup_coordinator import SetupCoordinator, SetupDependencies
    from dantetl.runtime.first_run import needs_runtime, run_setup
    from dantetl.runtime.hardware import detect_adapters
    from dantetl.runtime.hardware_values import HardwareStatus, recommended_profile
    from dantetl.runtime.llama import installed_device
    from dantetl.runtime.model_store import ModelOperation, delete_model, install_model
    from dantetl.runtime.ocr import gpu_runtime_installed
    from dantetl.runtime.setup import (
        install_runtime,
        ready_runtime,
        runtime_selection,
        select_runtime,
    )
    from dantetl.runtime.setup_values import RuntimeStatus, RuntimeTarget
    from dantetl.runtime.values import PROFILES
    from dantetl.translation.models.gguf import save_registration
    from dantetl.translation.models.probe import ImportTest

    def status() -> RuntimeStatus:
        adapters = detect_adapters()
        hardware = HardwareStatus(adapters, gpu_runtime_installed(), installed_device())
        saved = runtime_selection(root)
        return RuntimeStatus(
            adapters,
            tuple((p.key, ready_runtime(root, p.key)) for p in PROFILES),
            Path(sys.prefix).resolve(),
            recommended_profile(adapters),
            next((p.key for p in PROFILES if not needs_runtime(p, hardware, upgrade=False)), ""),
            os.environ.get("DANTETL_GPU_UUID", ""),
            RuntimeTarget(saved.target, saved.gpu) if saved.target else None,
        )

    return SetupCoordinator(
        inventory,
        models,
        PROFILES,
        SetupDependencies(
            ModelOperation,
            install_model,
            delete_model,
            partial(save_registration, remove=True),
            save_registration,
            ImportTest,
            status,
            lambda key, gpu, progress, cancel: install_runtime(
                root, next(p for p in PROFILES if p.key == key), progress, cancel, gpu
            ),
            lambda target: select_runtime(root, target.path, target.gpu),
            lambda plan, progress, cancel: run_setup(plan, root, progress, cancel),
        ),
    )


def cloud_services(*, language: SourceLanguage = DEFAULT_SOURCE) -> CloudServices:
    service = CloudServices(
        CloudProfiles(), WindowsSecrets, ChatGPTConnections(), language=language
    )
    service.start()
    return service


def model_inventory(models: SourceModels) -> ModelInventory:
    inventory = ModelInventory(models.inspect, mark_prepared)
    inventory.advance(0)
    return inventory


def release_updates(installed: str) -> Updates:
    from dantetl.paths import is_standalone, resource_root
    from dantetl.runtime.release_feed import fetch_release
    from dantetl.runtime.update_install import publisher_key
    from dantetl.runtime.update_launcher import install_release

    project, state = resource_root(), state_root()
    key = publisher_key(project) if is_standalone() else None
    install = partial(install_release, project=project, state=state, key=key) if key else None
    return Updates(installed, state, fetch=fetch_release, install=install)
