"""Small model download dialog; all I/O and hashing belong to its worker."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.model_advice import ADVICE, COMPARISON_NOTE
from dantetl.application.model_values import ModelFile
from dantetl.application.services import CloudCommands, OllamaCommands, SetupCommands
from dantetl.i18n import display, mark, tr
from dantetl.runtime.setup_values import SetupAction, SetupLane
from dantetl.translation.models.values import ImportedModel
from dantetl.ui.cloud_settings import CloudSettings
from dantetl.ui.controls import InfoButton, set_tone
from dantetl.ui.localization import localized, localized_item, on_language_change
from dantetl.ui.model_import_dialog import ModelImportDialog
from dantetl.ui.model_summary import ModelSummary, file_size
from dantetl.ui.ocr_model_settings import OcrModelSettings
from dantetl.ui.ollama_settings import OllamaSettings


class ModelManager(QDialog):
    runtime_requested = Signal()
    operation_started = Signal()

    def __init__(
        self,
        service: SetupCommands,
        parent: QWidget | None = None,
        *,
        cloud: CloudCommands,
        ollama: OllamaCommands | None = None,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        localized(self, windowTitle=mark("Local models"))
        if not embedded:
            self.resize(640, 460)
            self.setMinimumSize(580, 420)
        self.error_message = ""
        self.service = service
        self._completion = 0
        self._inventory_revision = -1
        self.import_dialog: ModelImportDialog | None = None
        self._closing = False
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        category_row = QHBoxLayout()
        category_row.addWidget(localized(QLabel(), text=mark("Model type")))
        self.category = QComboBox()
        localized(self.category, accessibleName=mark("Model type"))
        localized_item(self.category, mark("Local translation"), "translation")
        localized_item(self.category, mark("Text recognition (OCR)"), "recognition")
        localized_item(self.category, mark("Cloud translation"), "cloud")
        category_row.addWidget(self.category, 1)
        category_row.addWidget(
            InfoButton(
                mark("Model types"),
                mark("OCR reads text from game images. Translation models translate that text."),
            )
        )
        root.addLayout(category_row)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.translation_page, layout = self._page()
        self.category.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.choice = QComboBox()
        localized(self.choice, accessibleName=mark("Translation model files"))
        self.choice.setMinimumContentsLength(20)
        self.choice.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.choice.setCurrentIndex(self.choice.count() - 1)
        layout.addWidget(self.choice)
        self.download_badge = QLabel()
        self.download_badge.setObjectName("modelBadge")
        self.download_badge.setWordWrap(True)
        badges = QHBoxLayout()
        badges.addWidget(self.download_badge, 1)
        self.runtime_badge = QLabel()
        self.runtime_badge.setObjectName("modelBadge")
        self.runtime_badge.setWordWrap(True)
        set_tone(self.runtime_badge, "warning")
        badges.addWidget(self.runtime_badge, 1)
        layout.addLayout(badges)
        self.description = QLabel()
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.summary = ModelSummary()
        layout.addWidget(self.summary)
        self.file_details = QWidget()
        file_layout = QVBoxLayout(self.file_details)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_layout.addWidget(self.status)
        self.location = QLabel()
        self.location.setTextFormat(Qt.TextFormat.PlainText)
        self.location.setWordWrap(True)
        self.location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_layout.addWidget(self.location)
        self.license = QLabel()
        self.license.setWordWrap(True)
        file_layout.addWidget(self.license)
        comparison = localized(QLabel(), text=COMPARISON_NOTE)
        comparison.setWordWrap(True)
        file_layout.addWidget(comparison)
        self.download = localized(QPushButton(), text=mark("Download"))
        self.download.setEnabled(bool(service.files))
        self.download.clicked.connect(self._download)
        self.delete_button = localized(QPushButton(), text=mark("Delete model files…"))
        self.delete_button.clicked.connect(self._delete)
        downloads = QHBoxLayout()
        # Equal stretch aligns with the row below; a hidden Delete leaves Download full width.
        downloads.addWidget(self.download, 1)
        downloads.addWidget(self.delete_button, 1)
        layout.addLayout(downloads)
        self.runtime_button = localized(QPushButton(), text=mark("Set up acceleration…"))
        localized(
            self.runtime_button,
            toolTip=mark("Detect graphics cards and install acceleration inside the app."),
        )
        self.runtime_button.clicked.connect(self.runtime_requested.emit)
        imports = QHBoxLayout()
        imports.addWidget(self.runtime_button, 1)
        self.import_button = localized(QPushButton(), text=mark("Import GGUF…"))
        self.import_button.clicked.connect(lambda: self._import())
        imports.addWidget(self.import_button, 1)
        layout.addLayout(imports)
        imported_actions = QHBoxLayout()
        self.locate_button = localized(QPushButton(), text=mark("Locate file…"))
        self.locate_button.clicked.connect(self._locate)
        self.remove_button = localized(QPushButton(), text=mark("Remove from list"))
        localized(
            self.remove_button,
            toolTip=mark("Removes the registration only. Your model file is never deleted."),
        )
        self.remove_button.clicked.connect(self._remove)
        for button in (self.locate_button, self.remove_button):
            imported_actions.addWidget(button)
        layout.addLayout(imported_actions)
        self.details_toggle = QToolButton()
        localized(self.details_toggle, text=mark("File and runtime details"))
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.details_toggle.toggled.connect(self._details_toggled)
        layout.addWidget(self.details_toggle)
        layout.addWidget(self.file_details)
        self.file_details.hide()
        layout.addStretch()
        self.ocr_page, ocr_layout = self._page()
        self.ocr = OcrModelSettings()
        self.ocr.install_requested.connect(self._start_download)
        ocr_layout.addWidget(self.ocr)
        ocr_layout.addStretch()
        self.cloud_page, cloud_layout = self._page()
        self.cloud = CloudSettings(cloud)
        cloud_layout.addWidget(self.cloud)
        self.ollama = OllamaSettings(ollama) if ollama is not None else None
        if self.ollama is not None:
            self.category.addItem("Ollama API", "ollama")
            self.pages.addWidget(self.ollama)
        self.operation_status = QLabel()
        self.operation_status.setTextFormat(Qt.TextFormat.PlainText)
        self.operation_status.setWordWrap(True)
        self.operation_status.hide()
        root.addWidget(self.operation_status)
        self.progress = QProgressBar()
        self.progress.hide()
        root.addWidget(self.progress)
        self.cancel_button = localized(QPushButton(), text=mark("Cancel download"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self._cancel)
        root.addWidget(self.cancel_button)
        self.choice.currentIndexChanged.connect(self._selected)
        self.refresh()
        on_language_change(self, self._refresh_language)

    def _refresh_language(self) -> None:
        if not self.busy:
            self._selected()
            self.ocr.refresh(self.service.files)

    def _page(self) -> tuple[QScrollArea, QVBoxLayout]:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(2, 8, 12, 8)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        self.pages.addWidget(scroll)
        return scroll, layout

    def select_translation_file(self, key: str) -> None:
        self.category.setCurrentIndex(self.category.findData("translation"))
        for index in range(self.choice.count()):
            item = self.choice.itemData(index)
            if isinstance(item, ModelFile) and item.key == key:
                self.choice.setCurrentIndex(index)
                break

    def _details_toggled(self, expanded: bool) -> None:
        self.file_details.setVisible(expanded)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def refresh(self) -> None:
        revision = self.service.inventory_revision
        if self.busy or revision == self._inventory_revision:
            return
        self._inventory_revision = revision
        files = self.service.files
        self.ocr.refresh(files)
        selected = self.choice.currentData()
        self.choice.blockSignals(True)
        self.choice.clear()
        for item in files:
            if item.category != "translation":
                continue
            self.choice.addItem(display(item.name), item)
            if selected and item.key == selected.key or selected is None and item.key == "gemma":
                self.choice.setCurrentIndex(self.choice.count() - 1)
        self.choice.blockSignals(False)
        self._selected()

    @property
    def busy(self) -> bool:
        return self.service.job(SetupLane.MODELS).busy

    def _import(self, expected: ImportedModel | None = None) -> None:
        if self.busy:
            return
        self.error_message = ""
        if self.import_dialog and self.import_dialog.isVisible():
            self.import_dialog.raise_()
            return
        if self.import_dialog:
            self.import_dialog.deleteLater()
        self.import_dialog = ModelImportDialog(self.service, self, expected=expected)
        self.import_dialog.show()

    def _locate(self) -> None:
        value = self.choice.currentData()
        if isinstance(value, ModelFile) and value.imported:
            self._import(value.imported)

    def _remove(self) -> None:
        value = self.choice.currentData()
        if isinstance(value, ModelFile) and self.service.remove(value.key):
            self.refresh_view()

    def _delete(self) -> None:
        value = self.choice.currentData()
        if not isinstance(value, ModelFile) or self.busy:
            return
        prompt = QMessageBox(self)
        prompt.setWindowTitle(tr("Local models"))
        prompt.setTextFormat(Qt.TextFormat.PlainText)
        prompt.setText(
            tr(
                "Delete the downloaded files for {name} ({size})? "
                "You can download them again later.",
                name=display(value.name),
                size=file_size(value.size),
            )
        )
        prompt.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        prompt.setDefaultButton(QMessageBox.StandardButton.Cancel)
        answer = prompt.exec()
        prompt.deleteLater()
        if answer != QMessageBox.StandardButton.Yes or not self.service.delete(value.key):
            return
        self.error_message = ""
        self.operation_status.setText(tr("Deleting model files…"))
        self.operation_status.show()
        self.refresh_view()

    def _selected(self) -> None:
        self._refresh_badges()
        bundle = self.choice.currentData()
        imported = isinstance(bundle, ModelFile) and bundle.imported is not None
        self.locate_button.setVisible(imported)
        self.remove_button.setVisible(imported)
        self.download.setVisible(not imported)
        self.delete_button.setVisible(
            isinstance(bundle, ModelFile)
            and not imported
            and bundle.present
            and not bundle.included
        )
        if isinstance(bundle, ModelFile) and bundle.imported:
            self.summary.set_model(bundle.key, bundle.size, True, imported=True)
            self.license.setText(tr("License: check the imported model's source."))
            self.description.setText(display(bundle.description))
            self.location.setText(tr("Referenced file: {directory}", directory=bundle.directory))
            self.status.setText(display(bundle.status))
            return
        if not isinstance(bundle, ModelFile):
            return
        self.description.setText(display(bundle.description))
        self.summary.set_model(bundle.key, bundle.size, bundle.language_model)
        advice = ADVICE.get(bundle.key)
        self.license.setText(
            (tr("License: {license}.", license=advice.license) if advice else "")
            + (
                tr(
                    " Personal/non-commercial use is free; review the model card for "
                    "commercial use."
                )
                if bundle.needs_plamo
                else ""
            )
        )
        self.status.setText(display(bundle.status))
        self.location.setText(
            tr("Model folder: {local_directory}", local_directory=bundle.local_directory)
            + (
                tr("\nRepairs folder: {directory}", directory=bundle.directory)
                if bundle.bundled_directory
                else ""
            )
        )
        self.download.setText(tr("Verify / repair") if bundle.present else tr("Download"))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        self.cancel_button.hide()

    def _refresh_badges(self) -> None:
        for i in range(self.choice.count()):
            candidate = self.choice.itemData(i)
            if isinstance(candidate, ModelFile):
                self.choice.setItemText(
                    i,
                    f"{'✓' if candidate.present else '↓'}  {display(candidate.name)}"
                    + (tr(" · Imported") if candidate.imported else ""),
                )
        bundle = self.choice.currentData()
        if not isinstance(bundle, ModelFile):
            self.download_badge.hide()
            self.runtime_badge.hide()
            return
        downloaded = bundle.present
        self.download_badge.setText(
            (tr("✓  Local file registered") if downloaded else tr("⚠  Locate model file"))
            if bundle.imported
            else tr("✓  Included with app")
            if bundle.included
            else tr("✓  Downloaded")
            if downloaded
            else tr("↓  Download required")
        )
        self.download_badge.setAccessibleName(self.download_badge.text())
        self.download_badge.setToolTip(
            tr("Model files are present. Integrity is checked before use.")
        )
        set_tone(self.download_badge, "success" if downloaded else "warning")
        self.download_badge.show()
        device, upgrade = bundle.device, bundle.upgrade
        self.runtime_badge.setText(
            tr("⚠  Engine update required")
            if upgrade
            else tr("✓  CPU ready · GPU optional")
            if device == "cpu"
            else tr("⚠  Engine setup required")
        )
        self.runtime_badge.setToolTip(display(bundle.status))
        set_tone(self.runtime_badge, "info" if device == "cpu" and not upgrade else "warning")
        self.runtime_badge.setVisible(
            not bundle.imported
            and downloaded
            and bundle.language_model
            and (upgrade or device in (None, "cpu"))
        )

    def _download(self) -> None:
        self._start_download(self.choice.currentData())

    def _start_download(self, bundle: ModelFile) -> None:
        if not isinstance(bundle, ModelFile) or not self.service.download(bundle.key):
            return
        self._closing = False
        self.error_message = ""
        self.choice.setEnabled(False)
        self.ocr.set_busy(True)
        self.download.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.progress.show()
        self.operation_status.setText(tr("{name}: starting…", name=bundle.name))
        self.operation_status.show()
        self.operation_started.emit()

    def _cancel(self) -> None:
        if self.busy:
            self.service.cancel(SetupLane.MODELS)
            self.cancel_button.setEnabled(False)
            self.operation_status.setText(tr("Cancelling… finishing the current network read."))

    def refresh_view(self) -> None:
        self.refresh()
        self.cloud.refresh_view()
        if self.ollama is not None:
            self.ollama.refresh_view()
        if self.import_dialog:
            self.import_dialog.refresh_view()
        job = self.service.job(SetupLane.MODELS)
        busy = job.busy
        self.import_button.setEnabled(not busy)
        self.locate_button.setEnabled(not busy)
        self.remove_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        value = self.choice.currentData()
        self.download.setEnabled(not busy and isinstance(value, ModelFile) and not value.imported)
        self.choice.setEnabled(not busy)
        self.ocr.set_busy(busy)
        if (
            job.action in (SetupAction.DOWNLOAD, SetupAction.DELETE, SetupAction.REMOVE)
            and job.token
        ):
            progress = job.progress
            self.operation_status.setText(
                tr("Cancelling… finishing the current network read.")
                if job.cancelling and busy
                else progress.error or progress.message
            )
            self.progress.setRange(0, 1000 if progress.total else 0)
            if progress.total:
                self.progress.setValue(round(progress.received / progress.total * 1000))
                self.progress.setFormat(
                    tr(
                        "{value:.1f} / {value2:.1f} MB",
                        value=progress.received / 1000000.0,
                        value2=progress.total / 1000000.0,
                    )
                )
            if not busy and self._completion != job.token:
                self._completion = job.token
                self.error_message = progress.error
                self.cancel_button.hide()
                self.progress.hide()
        if self._closing and not busy and not self.cloud.busy:
            self.close()

    def reject(self) -> None:
        if self._embedded:
            self.window().close()
        else:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy or self.cloud.busy:
            self._closing = True
            self.cloud.cancel()
            self._cancel()
            event.ignore()
        else:
            event.accept()
