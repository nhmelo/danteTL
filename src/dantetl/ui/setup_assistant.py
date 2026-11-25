"""Skippable guided setup; one explicit action owns one cancellable background operation."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.model_advice import RECOMMENDED_TRANSLATION
from dantetl.application.services import SetupCommands
from dantetl.i18n import display, mark, tr
from dantetl.runtime.hardware_values import (
    HardwareReader,
    HardwareStatus,
    compatibility_issue,
    recommended_profile,
)
from dantetl.runtime.setup_values import ModelProgress, SetupLane, SetupReview
from dantetl.ui.cloud_settings import CloudSettings
from dantetl.ui.controls import Switch
from dantetl.ui.localization import (
    InterfaceLanguageCombo,
    localized,
    localized_item,
    on_language_change,
)
from dantetl.ui.model_summary import ModelSummary
from dantetl.ui.runtime_restart_prompt import ask_restart
from dantetl.ui.setup_progress import SetupProgressView


class SetupAssistant(QDialog):
    completed = Signal(str, object, bool)  # model ID, restart target, preparation deferred
    skipped = Signal()

    def __init__(
        self,
        service: SetupCommands,
        hardware: HardwareReader | None,
        parent: QWidget,
        *,
        cloud: CloudSettings,
        interface_language: str = "en",
        backdrop_blur: bool = True,
    ) -> None:
        super().__init__(parent)
        localized(self, windowTitle=mark("DanteTL · Welcome"))
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(640, 510)
        self.service = service
        self.cloud = cloud
        self._pending_restart: Path | None = None
        self.models = {model.key: model for model in service.files}
        ocr = self.models["ocr"]
        self._completion = 0
        self._started_token = 0
        self.hardware = hardware
        self.hardware_status: HardwareStatus | None = None
        self._closing = self._finished = self._reported = False
        self._runtime_chosen = False
        self._review: SetupReview | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        self.heading = localized(QLabel(), text=mark("Welcome to DanteTL"))
        self.heading.setObjectName("heading")
        layout.addWidget(self.heading)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        page = self._page()
        self.interface_language = InterfaceLanguageCombo(interface_language)
        language_label = localized(QLabel(), text=mark("App language"))
        language_label.setBuddy(self.interface_language)
        page.addWidget(language_label)
        page.addWidget(self.interface_language)
        self.backdrop_blur = localized(Switch(), text=mark("Blur behind translated text"))
        self.backdrop_blur.setChecked(backdrop_blur)
        page.addWidget(self.backdrop_blur)
        page.addWidget(
            localized(
                self._label(""),
                text=lambda: (
                    (
                        tr("Text recognition is included with DanteTL.")
                        if self.models["ocr"].included
                        else tr("This guide can set up local text recognition.")
                    )
                    + "\n\n"
                    + tr(
                        "Add a cloud model here, or choose Next to set up a local model and "
                        "optional GPU acceleration. You can also add cloud models while "
                        "files download."
                    )
                ),
            )
        )
        page.addStretch()
        page = self._page()
        page.addWidget(
            localized(
                self._label(""),
                text=lambda: (
                    tr("Choose a translation model.")
                    if self.models["ocr"].included
                    else tr(
                        "{name} ({value:.0f} MB). Choose a translator:",
                        name=ocr.name,
                        value=ocr.size / 1000000.0,
                    )
                ),
            )
        )
        self.model = QComboBox()
        for bundle in service.guide_models:
            self.model.addItem("", bundle.key)
        self._model_labels()
        recommended = self.model.findData(RECOMMENDED_TRANSLATION)
        if recommended >= 0:
            self.model.setCurrentIndex(recommended)
        page.addWidget(self.model)
        self.model_help = self._label("")
        page.addWidget(self.model_help)
        self.model_summary = ModelSummary()
        page.addWidget(self.model_summary)
        self.model.currentIndexChanged.connect(self._model_changed)
        self._model_changed()
        page.addStretch()
        page = self._page()
        self.hardware_label = self._label(tr("Checking graphics hardware…"))
        page.addWidget(self.hardware_label)
        self.runtime = QComboBox()
        for profile in sorted(service.profiles, key=lambda p: p.key != "cpu"):
            localized_item(self.runtime, profile.name, profile)
        page.addWidget(self.runtime)
        self.runtime_help = self._label("")
        page.addWidget(self.runtime_help)
        self.runtime.currentIndexChanged.connect(self._runtime_changed)
        self.runtime.activated.connect(self._runtime_changed)
        self._runtime_changed()
        self._runtime_chosen = False
        page.addWidget(
            localized(
                self._label(""),
                text=mark("Setup checks the runtime before restart. Your drivers stay unchanged."),
            )
        )
        page.addStretch()
        page = self._page()
        self.review = self._label("")
        page.addWidget(self.review)
        page.addStretch()
        page = self._page()
        self.progress_view = SetupProgressView()
        page.addWidget(self.progress_view)
        self.runtime_progress = SetupProgressView()
        page.addWidget(self.runtime_progress)
        self.runtime_progress.hide()
        self.result_note = self._label("")
        page.addWidget(self.result_note)
        page.addStretch()
        self.add_cloud = localized(QPushButton(), text=mark("Add cloud model…"))
        self.add_cloud.clicked.connect(self._add_cloud)
        layout.addWidget(self.add_cloud, 0, Qt.AlignmentFlag.AlignLeading)
        self.cloud_note = localized(
            self._label(""), text=mark("Model saved. Select it on the main screen to use it.")
        )
        self.cloud_note.hide()
        layout.addWidget(self.cloud_note)
        buttons = QHBoxLayout()
        self.skip = localized(QPushButton(), text=mark("Skip for now"))
        self.back = localized(QPushButton(), text=mark("Back"))
        self.next = localized(QPushButton(), text=mark("Next"))
        self.next.setObjectName("primary")
        buttons.addWidget(self.skip)
        buttons.addStretch()
        buttons.addWidget(self.back)
        buttons.addWidget(self.next)
        layout.addLayout(buttons)
        self.skip.clicked.connect(self._skip)
        self.back.clicked.connect(lambda: self._show(self.pages.currentIndex() - 1))
        self.next.clicked.connect(self._next)
        on_language_change(self, self._refresh_language)
        self._show(0)
        job = service.job(SetupLane.GUIDE)
        if job.token and service.reviewed is not None:
            self._review = service.reviewed
            self._runtime_chosen = True
            self.model.setCurrentIndex(self.model.findData(self._review.model))
            for index in range(self.runtime.count()):
                if self.runtime.itemData(index).key == self._review.runtime:
                    self.runtime.setCurrentIndex(index)
                    break
            self._show(4)
            self.next.setEnabled(False)
            localized(self.skip, text=mark("Cancel setup"))
        self.refresh_view()

    def _add_cloud(self) -> None:
        editor = self.cloud.open_editor(parent=self)
        if editor is not None:
            editor.accepted.connect(self._cloud_saved)

    def _cloud_saved(self) -> None:
        self.cloud_note.show()
        if self.pages.currentIndex() < 4:
            self.skip.setText(tr("Finish"))

    def _model_labels(self) -> None:
        for i in range(self.model.count()):
            bundle = self.models[self.model.itemData(i)]
            suffix = (
                tr(" · downloaded")
                if bundle.present
                else tr(" · {value:.2f} GB", value=bundle.size / 1000000000.0)
            )
            self.model.setItemText(i, display(bundle.name) + suffix)

    def _show_hardware(self) -> None:
        status = self.hardware_status
        if status is None:
            self.hardware_label.setText(tr("Checking graphics hardware…"))
            return
        self.hardware_label.setText(
            tr("Hardware detection failed. CPU is available; you can choose manually.")
            if status.error
            else tr("Detected: ") + "; ".join(a.name for a in status.adapters)
            if status.adapters
            else tr("No supported GPU detected. CPU is available.")
        )

    def _refresh_language(self) -> None:
        self._model_labels()
        self._show_hardware()
        self._model_changed()
        self._show_runtime()
        if self.pages.currentIndex() == 0:
            self._show(0)

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    def _page(self) -> QVBoxLayout:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        self.pages.addWidget(widget)
        return layout

    def _model_changed(self) -> None:
        bundle = self.models.get(self.model.currentData())
        self.model_help.setText(
            display(bundle.description) if bundle else tr("Choose a translation model.")
        )
        if bundle:
            self.model_summary.set_model(bundle.key, bundle.size, bundle.language_model)

    def _runtime_changed(self) -> None:
        self._runtime_chosen = True
        self._show_runtime()

    def _show_runtime(self) -> None:
        profile = self.runtime.currentData()
        status = self.hardware_status
        issue = compatibility_issue(profile.key, status.adapters) if status else ""
        self.runtime_help.setText(issue or tr(profile.description))
        if self.pages.currentIndex() == 2:
            self.next.setEnabled(status is not None and not issue)

    def _show(self, page: int) -> None:
        self.pages.setCurrentIndex(page)
        self.add_cloud.setVisible(page in (0, 1, 4))
        if page < 4:
            self.skip.setText(tr("Skip for now") if self.cloud_note.isHidden() else tr("Finish"))
            self.skip.setEnabled(True)
        self.heading.setText(
            (
                tr("Welcome to DanteTL"),
                tr("Choose your translation model"),
                tr("Choose acceleration"),
                tr("Review setup"),
                tr("Setting up DanteTL"),
            )[page]
        )
        self.back.setVisible(0 < page < 4)
        self.next.setText(tr("Download and set up") if page == 3 else tr("Next"))
        self.next.setEnabled(True)
        if page == 2:
            self._show_runtime()
        if page == 3:
            assert self.hardware_status is not None
            profile = self.runtime.currentData()
            ocr = self.models["ocr"]
            self._review = self.service.review(
                self.model.currentData(), profile.key, self.hardware_status
            )
            pending = self._review.download_bytes
            self.review.setText(
                tr(
                    "Recognition: {ocr}{included}\nTranslation: {model}\nRuntime: "
                    "{runtime}\n\nModel downloads: up to {size:.2f} GB; existing files are "
                    "verified and reused.\n",
                    ocr=ocr.name.split(" · ")[0],
                    included=tr(" · included, no download") if ocr.included else "",
                    model=self.model.currentText(),
                    runtime=tr(profile.name),
                    size=pending / 1e9,
                )
                + (
                    tr(
                        "Runtime libraries and app dependencies will also be downloaded. "
                        "NVIDIA libraries add about 1.3 GB; CPU and Vulkan downloads are smaller. "
                        "Restart is offered when ready.\n"
                    )
                    if self._review.install_runtime
                    else tr(
                        "Your current runtime provides this support; no runtime install needed.\n"
                    )
                )
                + tr(
                    "\nNothing has downloaded yet. Download and set up starts the selected work. "
                    "You can cancel; completed verified files are kept."
                )
            )

    def _next(self) -> None:
        page = self.pages.currentIndex()
        if page < 3:
            self._show(page + 1)
        elif page == 3 or (page == 4 and not self._finished):
            self._start()
        else:
            self._finish(self.target)

    def _start(self) -> None:
        if self._review is None or not self.service.start_review(self._review.token):
            return
        self._started_token = self.service.job(SetupLane.GUIDE).token
        self._show(4)
        self.next.setEnabled(False)
        self.skip.setText(tr("Cancel setup"))
        self.result_note.setText("")
        self.runtime_progress.hide()
        self.progress_view.update_progress(ModelProgress(tr("Starting selected setup…")))

    @property
    def busy(self) -> bool:
        return self.service.job(SetupLane.GUIDE).busy

    @property
    def target(self) -> Path | None:
        return self.service.guide_target

    def cancel(self) -> None:
        if self.busy:
            self.service.cancel(SetupLane.GUIDE)
            self.skip.setEnabled(False)
            self.result_note.setText(
                tr("Cancelling… waiting for the current read or installer to stop.")
            )

    def _prompt_if_ready(self) -> None:
        if (
            self._pending_restart
            and self.isVisible()
            and self.cloud.editor is None
            and not self.cloud.busy
            and QApplication.activeModalWidget() in (None, self)
        ):
            target, self._pending_restart = self._pending_restart, None
            if target == self.target:
                ask_restart(
                    self,
                    self.runtime.currentData().name,
                    lambda: self._finish(target) if self.target == target else None,
                )

    def refresh_view(self) -> None:
        self.add_cloud.setEnabled(not self.cloud.busy and self.cloud.editor is None)
        self._prompt_if_ready()
        if self.hardware_status is None:
            status = (
                self.hardware.snapshot()
                if self.hardware
                else HardwareStatus(error=tr("Unavailable"))
            )
            if status is not None:
                self.hardware_status = status
                self._show_hardware()
                if not self._runtime_chosen:
                    key = recommended_profile(status.adapters)
                    self.runtime.setCurrentIndex(
                        next(
                            i
                            for i in range(self.runtime.count())
                            if self.runtime.itemData(i).key == key
                        )
                    )
                self._show_runtime()
        job = self.service.job(SetupLane.GUIDE)
        if not job.token or self._completion == job.token:
            return
        value = job.progress
        if value.parallel and job.busy:
            self.progress_view.update_progress(value.parallel[0], compact=True)
            self.runtime_progress.update_progress(value.parallel[1], compact=True)
            self.runtime_progress.show()
        else:
            self.progress_view.update_progress(value)
            self.runtime_progress.hide()
        if job.busy:
            return
        self._completion = job.token
        self.skip.setEnabled(True)
        self.next.setEnabled(True)
        self._finished = not (value.error or value.cancelled)
        self.skip.setText(
            tr("Finish without restart") if self._finished and self.target else tr("Close")
        )
        self.next.setText(
            tr("Restart now")
            if self._finished and self.target
            else tr("Finish")
            if self._finished
            else tr("Retry setup")
        )
        self.back.setVisible(not self._finished)
        self.result_note.setText(
            tr("Restart DanteTL to use {name}.", name=tr(self.runtime.currentData().name))
            if self._finished and self.target
            else tr(
                "Your chosen model will be selected next. Language models ask before preparation."
            )
            if self._finished
            else tr("Verified files and your previous runtime were kept.")
        )
        if self._closing:
            self.close()
        elif job.token == self._started_token and self._finished and (target := self.target):
            self._pending_restart = target
            self._prompt_if_ready()

    def _finish(self, target: Path | None) -> None:
        self._report_completion(target)
        self.hide()
        self.accept()

    def _report_completion(self, target: Path | None) -> None:
        self._reported = True
        if self._review is not None:
            deferred = self.target is not None and target is None
            key = self.service.acknowledge_review(self._review.token)
            if key is not None:
                self.completed.emit(key, target, deferred)

    def _skip(self) -> None:
        if self.busy:
            self.cancel()
        elif self._finished:
            self._finish(None)
        else:
            self.close()

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            self._closing = True
            self.cancel()
            event.ignore()
            return
        if not self._reported:
            if self._finished:
                self._report_completion(None)
            else:
                self._reported = True
                self.skipped.emit()
        event.accept()
