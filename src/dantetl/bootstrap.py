"""Application composition and startup diagnostics."""

import logging
import os
import sys
from contextlib import nullcontext
from functools import partial
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from dantetl.application.model_factories import create_recognizer
from dantetl.application.preferences import PreferenceStore
from dantetl.i18n import tr
from dantetl.languages.selection import source_language
from dantetl.paths import is_standalone, resource_root, state_root


def configure_logging() -> None:
    directory = (
        state_root()
        if is_standalone()
        else Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DanteTL"
    ) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / "app.log", maxBytes=500_000, backupCount=2, encoding="utf-8"
    )
    # Cloud adapters supply sanitized diagnostics. Transport debug logs may contain
    # headers/token request bodies and must never enter the application log.
    handler.addFilter(
        lambda record: (
            record.name.split(".", 1)[0] not in ("httpx", "httpx2", "httpcore", "httpcore2")
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler],
    )


def main(*, cuda_update: bool = False) -> None:
    if sys.platform != "win32":
        raise SystemExit(tr("DanteTL currently requires Windows."))
    configure_logging()
    import shiboken6
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dantetl.application.hotkeys import Hotkeys
    from dantetl.capture.windows import Windows
    from dantetl.composition import (
        application_session,
        cloud_services,
        model_inventory,
        release_updates,
        setup_coordinator,
        source_models_for,
    )
    from dantetl.runtime.hardware import HardwareCheck
    from dantetl.runtime.restart import acknowledge_startup
    from dantetl.runtime.update_swap import acknowledge_update, bundle_lease
    from dantetl.ui.bug_reports import error_dialog
    from dantetl.ui.desktop_driver import DesktopDriver
    from dantetl.ui.hotkeys_windows import WindowsHotkeyEvents, WindowsHotkeyRegistrar
    from dantetl.ui.icons import APP_ICON
    from dantetl.ui.localization import install_language
    from dantetl.ui.main_window import MainWindow

    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication(sys.argv)
    app.setApplicationName("DanteTL")
    app.setWindowIcon(QIcon(str(APP_ICON)))
    app.setOrganizationName("DanteTL")
    preferences = PreferenceStore(state_root() / "data/preferences.json")
    install_language(app, preferences.current.interface_language)

    def report_exception(
        kind: type[BaseException], error: BaseException, traceback: TracebackType | None
    ) -> None:
        logging.getLogger(__name__).error(
            tr("Unhandled application error"), exc_info=(kind, error, traceback)
        )
        error_dialog(
            None,
            "DanteTL",
            tr(
                "An unexpected error occurred:\n{error}\n\nDetails are in "
                "%LOCALAPPDATA%/DanteTL/logs/app.log.",
                error=error,
            ),
            icon=QMessageBox.Icon.Critical,
        ).exec()

    sys.excepthook = report_exception
    hardware = HardwareCheck()
    hardware.start()
    hotkeys = Hotkeys(WindowsHotkeyRegistrar())
    hotkey_events = WindowsHotkeyEvents(app, hotkeys)

    driver = DesktopDriver()
    language = source_language(preferences.current.source_language)
    models = source_models_for(language)
    inventory = model_inventory(models)
    setup = setup_coordinator(inventory, models.bundles, state_root())

    application = application_session(
        windows,
        partial(create_recognizer, language=language),
        cloud=cloud_services(language=language),
        language=language,
        inventory=inventory,
        setup=setup,
        ocr_backends=models.ocr,
        source_models=source_models_for,
        notify=driver.notify,
        preferences=preferences,
        hardware=hardware,
        hotkeys=hotkeys,
    )
    updates = release_updates(version("dantetl"))
    window = MainWindow(
        application, windows, driver, hardware=hardware, hotkeys=hotkeys, updates=updates
    )
    window.show()
    QTimer.singleShot(0, window.offer_cuda_update if cuda_update else window.offer_first_run)
    if notice := os.environ.pop("DANTETL_STARTUP_NOTICE", ""):
        window.status.setText(notice)
    QTimer.singleShot(0, acknowledge_startup)
    QTimer.singleShot(0, lambda: acknowledge_update(state_root(), updates.installed))
    if is_standalone() and os.environ.get("DANTETL_DISABLE_UPDATE_CHECK") != "1":
        QTimer.singleShot(0, lambda: updates.check(automatic=True))
    lease = (
        bundle_lease(resource_root().parent.parent, state_root())
        if is_standalone()
        else nullcontext()
    )
    with lease:
        try:
            exit_code = app.exec()
        finally:
            updates.finish()
    hotkey_events.close()
    # Tear down Qt while its native dependencies are still loaded, before Python finalization.
    shiboken6.delete(app)
    sys.exit(exit_code)
