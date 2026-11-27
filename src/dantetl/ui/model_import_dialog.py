"""Local import review. All model inspection, testing and registration runs off Qt."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.services import SetupCommands
from dantetl.i18n import mark, tr
from dantetl.runtime.setup_values import SetupAction, SetupLane
from dantetl.translation.models.values import PROFILES, ImportedModel
from dantetl.ui.localization import localized, localized_item


class ModelImportDialog(QDialog):
    def __init__(
        self,
        service: SetupCommands,
        parent: QWidget | None = None,
        *,
        expected: ImportedModel | None = None,
    ) -> None:
        super().__init__(parent)
        localized(
            self,
            windowTitle=mark("Locate imported model")
            if expected
            else mark("Import local GGUF model"),
        )
        self.resize(670, 600)
        self.setMinimumSize(540, 450)
        self.expected = expected
        self.service = service
        service.begin_import(expected)
        self._completion = 0
        self._closing = False
        layout = QVBoxLayout(self)
        intro = localized(
            QLabel(),
            text=mark(
                "Choose a single-file GGUF chat model. The file stays in its current folder; "
                "DanteTL saves only a reference. The import test uses Japanese and English."
            ),
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.file = QLineEdit(expected.path if expected else "")
        localized(self.file, placeholderText=mark("Path to a local .gguf file"))
        localized(self.file, accessibleName=mark("Model file"))
        row.addWidget(self.file, 1)
        self.browse = localized(QPushButton(), text=mark("Browse…"))
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.name = QLineEdit(expected.name if expected else "")
        self.name.setMaxLength(100)
        localized(self.name, placeholderText=mark("Use the model filename"))
        localized(self.name, accessibleName=mark("Display name"))
        layout.addWidget(localized(QLabel(), text=mark("Display name (optional)")))
        layout.addWidget(self.name)
        self.profile = QComboBox()
        for key, label in PROFILES.items():
            localized_item(self.profile, label, key)
        if expected:
            self.profile.setCurrentIndex(self.profile.findData(expected.profile))
        localized(
            self.profile,
            toolTip=mark(
                "Automatic uses Gemma 4's native format or the GGUF's embedded chat template. "
                "Only choose another format when the model's documentation specifies it."
            ),
        )
        layout.addWidget(localized(QLabel(), text=mark("Prompt format")))
        layout.addWidget(self.profile)
        self.status = localized(
            QLabel(),
            text=mark("Test the model to check this app's runtime and view sample translations."),
        )
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        localized(
            self.results,
            placeholderText=mark(
                "Test translations and timings appear here. "
                "Passing a test does not guarantee translation quality."
            ),
        )
        layout.addWidget(self.results, 1)
        note = localized(
            QLabel(),
            text=mark(
                "Testing temporarily pauses translation to free model memory; capture continues. "
                "Your previous selection resumes after testing. First GPU use may take a minute."
            ),
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.test_button = localized(QPushButton(), text=mark("Test compatibility"))
        self.test_button.clicked.connect(self.start_test)
        self.add_button = localized(
            QPushButton(), text=mark("Update location") if expected else mark("Add model")
        )
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self._add)
        self.cancel_button = localized(QPushButton(), text=mark("Close"))
        self.cancel_button.clicked.connect(self._cancel_or_close)
        for button in (self.test_button, self.add_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.file.textChanged.connect(self._invalidate)
        self.profile.currentIndexChanged.connect(self._invalidate)
        result = service.probe_result
        if result is not None:
            self.file.blockSignals(True)
            self.profile.blockSignals(True)
            self.file.setText(result.model.path)
            self.name.setText(result.model.name)
            self.profile.setCurrentIndex(self.profile.findData(result.model.profile))
            self.file.blockSignals(False)
            self.profile.blockSignals(False)
        self.refresh_view()

    def _browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, tr("Choose local model"), self.file.text(), tr("GGUF models (*.gguf)")
        )
        if filename:
            self.file.setText(filename)

    def _invalidate(self) -> None:
        self.service.invalidate_import()
        self.add_button.setEnabled(False)
        self.results.clear()

    def start_test(self) -> None:
        if self.service.test_import(self.file.text(), self.profile.currentData()):
            self._set_busy(True)
            self.results.clear()
        self.refresh_view()

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.file, self.name, self.browse, self.profile, self.test_button):
            widget.setEnabled(not busy)
        self.add_button.setEnabled(False)
        self.cancel_button.setText(tr("Cancel") if busy else tr("Close"))
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0)

    def _add(self) -> None:
        if self.service.save_import(self.name.text()):
            self._set_busy(True)

    def cancel(self) -> None:
        if self.busy:
            self.service.cancel(SetupLane.MODELS)
            self.status.setText(tr("Cancelling… waiting for the test process to stop."))

    @property
    def busy(self) -> bool:
        return self.service.job(SetupLane.MODELS).busy

    def _cancel_or_close(self) -> None:
        if self.busy:
            self.cancel()
        else:
            self.close()

    def refresh_view(self) -> None:
        job = self.service.job(SetupLane.MODELS)
        if not job.token:
            return
        value = job.progress
        if job.busy:
            self._set_busy(True)
        self.status.setText(value.error or value.message)
        self.progress.setRange(0, 1000 if value.total else 0)
        if value.total:
            self.progress.setValue(round(value.received / value.total * 1000))
        if job.busy or self._completion == job.token:
            return
        self._completion = job.token
        self._set_busy(False)
        succeeded = not value.error and not value.cancelled
        if job.action == SetupAction.REGISTER and succeeded:
            self.accept()
        elif job.action == SetupAction.PROBE and succeeded and self.service.probe_result:
            result = self.service.probe_result
            self.results.setPlainText(
                tr(
                    "{name}\nArchitecture: {architecture} · Format: {value}\n{value2:.2f} "
                    "GiB on disk · Tested: {value3}",
                    name=result.model.name,
                    architecture=result.model.architecture,
                    value=tr(PROFILES[result.model.profile]),
                    value2=result.model.size / 2**30,
                    value3=result.device.upper(),
                )
                + (
                    tr(" · {offloaded_layers} GPU layers", offloaded_layers=result.offloaded_layers)
                    if result.offloaded_layers
                    else ""
                )
                + tr("\nLoad + warm-up: {value:.1f} s\n\n", value=result.load_ms / 1000)
                + "\n\n".join(
                    f"{source}\n{english}\n{ms:.0f} ms" for source, english, ms in result.samples
                )
            )
            self.add_button.setEnabled(True)
        if self._closing:
            self.close()

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            self._closing = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
