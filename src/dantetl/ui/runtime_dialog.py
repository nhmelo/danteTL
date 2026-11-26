"""Explicit runtime installation UI; detection, processes and disk work stay on a worker."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.services import SetupCommands
from dantetl.i18n import mark, tr
from dantetl.runtime.hardware_values import compatibility_issue
from dantetl.runtime.setup_values import SetupLane
from dantetl.ui.localization import localized, localized_item, on_language_change
from dantetl.ui.runtime_gpus import GraphicsList
from dantetl.ui.runtime_restart_prompt import ask_restart
from dantetl.ui.setup_progress import SetupProgressView


class RuntimeSetupDialog(QDialog):
    restart_requested = Signal(object)

    def __init__(
        self, service: SetupCommands, parent: QWidget | None = None, *, embedded: bool = False
    ) -> None:
        super().__init__(parent)
        self._embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.error_message = ""
        self.service = service
        self._completion = 0
        self._install_after_detection: str | None = None
        self._pending_restart: Path | None = None
        self._install_token = 0
        localized(self, windowTitle=mark("Runtime setup"))
        self.resize(660, 460)
        self._closing = False
        if not service.job(SetupLane.RUNTIME).token:
            service.detect_runtime()
        layout = QVBoxLayout(self)
        self.hardware = GraphicsList()
        layout.addWidget(self.hardware)
        layout.addSpacing(12)
        runtime_label = localized(QLabel(), text=mark("Runtime"))
        runtime_label.setObjectName("sectionTitle")
        layout.addWidget(runtime_label)
        self.choice = QComboBox()
        runtime_label.setBuddy(self.choice)
        for profile in service.profiles:
            localized_item(self.choice, profile.name, profile)
        self.choice.setEnabled(False)
        layout.addWidget(self.choice)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        explanation = localized(
            QLabel(), text=mark("Switching runtimes keeps your models and requires a restart.")
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.progress_view = SetupProgressView()
        self.status = self.progress_view.status
        self.progress = self.progress_view.overall
        layout.addWidget(self.progress_view)
        self.install = localized(QPushButton(), text=mark("Install runtime"))
        self.install.setEnabled(False)
        self.install.clicked.connect(self._install)
        self.restart = localized(QPushButton(), text=mark("Restart now"))
        self.restart.setObjectName("primary")
        localized(
            self.restart,
            toolTip=mark("Stops capture and waits for active workers, then opens the new runtime."),
        )
        self.restart.setEnabled(False)
        self.restart.clicked.connect(lambda: self._restart())
        layout.addWidget(self.restart)
        self.later = localized(QPushButton(), text=mark("Use on next launch"))
        self.later.setEnabled(False)
        self.later.clicked.connect(self._later)
        layout.addWidget(self.later)
        layout.addWidget(self.install)
        self.cancel_button = localized(QPushButton(), text=mark("Cancel"))
        self.cancel_button.clicked.connect(self.cancel_operation if embedded else self.close)
        if embedded:
            localized(self.cancel_button, text=mark("Cancel setup"))
        layout.addWidget(self.cancel_button)
        self.choice.currentIndexChanged.connect(self._selected)
        self.choice.activated.connect(self._offer_restart)
        active = service.job(SetupLane.RUNTIME)
        if active.key:
            self._select_profile(active.key)
        self._selected()
        self.refresh_view()
        on_language_change(self, self._refresh_language)

    def _select_profile(self, key: str) -> None:
        self.choice.setCurrentIndex(
            next(i for i, p in enumerate(self.service.profiles) if p.key == key)
        )

    def start_install(self, key: str) -> None:
        job = self.service.job(SetupLane.RUNTIME)
        if job.action == "detect" and job.token != self._completion:
            self._select_profile(key)
            self._install_after_detection = key
            self.refresh_view()
        elif not job.busy:
            self._select_profile(key)
            self._install()

    def _show_hardware(self) -> None:
        adapters = self.service.runtime.adapters
        checking = self.service.job(SetupLane.RUNTIME)
        self.hardware.set_adapters(
            adapters,
            tr("Checking graphics hardware…")
            if checking.action == "detect" and checking.busy
            else tr("No supported GPU detected. CPU is available."),
        )

    def _refresh_language(self) -> None:
        if self.choice.isEnabled():
            self._show_hardware()
        self._selected()

    @property
    def busy(self) -> bool:
        return self.service.job(SetupLane.RUNTIME).busy

    def _selected(self) -> None:
        profile = self.choice.currentData()
        if profile is None:
            return
        self.description.setText(tr(profile.description))
        ready = self.service.runtime.target(profile.key)
        current = (
            ready == self.service.runtime.current
            if ready
            else profile.key == self.service.runtime.active_profile
        )
        idle = not self.busy
        issue = compatibility_issue(profile.key, self.service.runtime.adapters)
        self.install.setEnabled(idle and not issue)
        self.install.setText(
            tr("Verify / repair runtime") if ready or current else tr("Install runtime")
        )
        self.restart.setEnabled(idle and bool(ready) and not current)
        self.later.setEnabled(self.restart.isEnabled())
        self.restart.setVisible(self.restart.isEnabled())
        self.later.setVisible(self.later.isEnabled())
        self.progress.setVisible(not idle)
        if idle:
            self.progress_view.download.hide()
            self.progress_view.download_label.hide()
        if self._embedded:
            self.cancel_button.setEnabled(not idle)
            self.cancel_button.setVisible(
                not idle and self.service.job(SetupLane.RUNTIME).action != "detect"
            )
        if idle:
            self.status.setText(
                issue
                if issue
                else tr("✓ Supported · In use")
                if current
                else tr("Restart DanteTL to use {name}.", name=tr(profile.name))
                if ready
                else tr("Ready to install")
            )

    def _install(self) -> None:
        profile = self.choice.currentData()
        if (
            profile is None
            or compatibility_issue(profile.key, self.service.runtime.adapters)
            or not self.service.install_runtime(profile.key)
        ):
            return
        self._pending_restart = None
        self._install_token = self.service.job(SetupLane.RUNTIME).token
        self._closing = False
        self.error_message = ""
        self.choice.setEnabled(False)
        self.install.setEnabled(False)
        self.restart.setEnabled(False)
        self.later.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()

    def refresh_view(self) -> None:
        job = self.service.job(SetupLane.RUNTIME)
        if not job.token or self._completion == job.token:
            self._prompt_when_ready()
            return
        value = job.progress
        self.progress_view.update_progress(value)
        self.status.setText(
            tr("Cancelling… waiting for installer processes to exit.")
            if job.cancelling and job.busy
            else (value.error or value.message)[-1800:]
        )
        if job.busy:
            return
        self._completion = job.token
        self.error_message = value.error
        self.progress.setRange(0, 1)
        self.progress.setValue(0 if value.error or value.cancelled else 1)
        self.choice.setEnabled(True)
        if job.action == "detect":
            self._show_hardware()
            requested = self._install_after_detection
            self._install_after_detection = None
            runtime = self.service.runtime
            active = runtime.active_profile
            if compatibility_issue(active, runtime.adapters):
                active = ""
            self._select_profile(requested or active or runtime.recommendation)
            if requested and not (value.cancelled or value.error or self._closing):
                self._install()
                if self.busy:
                    return
        self._selected()
        if value.error or value.cancelled or job.action == "select":
            self.status.setText(
                (
                    value.error
                    or (
                        tr("Setup cancelled. Your current runtime was kept.")
                        if value.cancelled
                        else value.message
                    )
                )[-1800:]
            )
        if self._closing:
            self.close()
        elif job.token == self._install_token and not (value.cancelled or value.error):
            self._offer_restart()

    def _offer_restart(self) -> None:
        self._pending_restart = self.service.runtime.target(self.choice.currentData().key)
        self._prompt_when_ready()

    def _prompt_when_ready(self) -> None:
        target = self._pending_restart
        modal = QApplication.activeModalWidget()
        if target is None or self.service.busy or modal not in (None, self.window()):
            return
        self._pending_restart = None
        profile = self.choice.currentData()
        if (
            target == self.service.runtime.target(profile.key)
            and target != self.service.runtime.current
        ):
            ask_restart(self, profile.name, lambda: self._restart(target))

    def _restart(self, expected: Path | None = None) -> None:
        profile = self.choice.currentData()
        ready = self.service.runtime.target(profile.key)
        if ready and not self.busy and (expected is None or expected == ready):
            self._pending_restart = None
            if not self._embedded:
                self.hide()
            self.restart_requested.emit(ready)

    def _later(self) -> None:
        profile = self.choice.currentData()
        if profile is not None and self.service.use_next_launch(profile.key):
            self._pending_restart = None
            self._selected()

    def reject(self) -> None:
        if self._embedded:
            self.window().close()
        else:
            self.close()

    def cancel_operation(self) -> None:
        if self.busy:
            self.service.cancel(SetupLane.RUNTIME)
            self.cancel_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            self._closing = True
            self.cancel_operation()
            event.ignore()
        else:
            event.accept()
