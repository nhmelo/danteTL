"""One nonmodal settings window owns appearance, recognition and background setup views."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.hotkeys import HotkeyAction, Hotkeys
from dantetl.application.preference_values import MAX_CAPTURE_RATE, EditSettings, Preferences
from dantetl.application.services import CacheCommands, CloudCommands, OllamaCommands, SetupCommands
from dantetl.i18n import display, mark, tr
from dantetl.runtime.setup_values import SetupLane
from dantetl.translation.selection import OcrOption
from dantetl.ui.cache_settings import CacheSettings
from dantetl.ui.controls import (
    InfoButton,
    SourceLanguageCombo,
    StatusBanner,
    Switch,
)
from dantetl.ui.hotkey_settings import HotkeySettings
from dantetl.ui.localization import InterfaceLanguageCombo, localized, on_language_change
from dantetl.ui.model_manager import ModelManager
from dantetl.ui.runtime_dialog import RuntimeSetupDialog


class SettingsWindow(QDialog):
    restart_requested = Signal(object)
    runtime_created = Signal(object)

    def __init__(
        self,
        preferences: Preferences,
        ocr_choices: tuple[OcrOption, ...],
        parent: QWidget,
        *,
        edit: EditSettings,
        cloud: CloudCommands,
        ollama: OllamaCommands,
        setup: SetupCommands,
        cache: CacheCommands,
        hotkeys: Hotkeys | None = None,
    ) -> None:
        super().__init__(parent)
        localized(self, windowTitle=mark("DanteTL · Settings"))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(680, 620)
        self.setMinimumSize(540, 440)
        self._edit = edit
        self._setup = setup
        self._cloud = cloud
        self._ocr_key = preferences.ocr_device
        self.runtime: RuntimeSetupDialog | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        self.model_notice = StatusBanner()
        self.acceleration_notice = StatusBanner()
        layout.addWidget(self.model_notice)
        layout.addWidget(self.acceleration_notice)
        self.tabs = QTabWidget()
        self._pages: dict[str, QWidget] = {}
        layout.addWidget(self.tabs, 1)
        appearance = self._page(mark("Appearance"))
        self.interface_language = InterfaceLanguageCombo(preferences.interface_language)
        self._setting(
            appearance,
            self.interface_language,
            mark("App language"),
            "",
        )
        self.language_notice = StatusBanner()
        appearance.addWidget(self.language_notice)
        self.stabilize = localized(Switch(), text=mark("Stabilize game boxes"))
        self.stabilize.setChecked(preferences.stabilize)
        self._setting(
            appearance,
            self.stabilize,
            mark("Stabilize game boxes"),
            mark(
                "Keeps detected boxes steady when OCR jitters. Turn off only to inspect "
                "raw OCR boxes. "
                "Translation always uses stable tracking."
            ),
        )
        self.backdrop_blur = localized(Switch(), text=mark("Blur behind translated text"))
        self.backdrop_blur.setChecked(preferences.backdrop_blur)
        self._setting(
            appearance,
            self.backdrop_blur,
            mark("Blur behind translated text"),
            mark(
                "Blur the game behind each translated label. If Windows declines the effect, "
                "the labels use the configured background opacity."
            ),
        )
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(20, 100)
        self.opacity.setValue(preferences.opacity)
        self.opacity_value = QLabel(f"{preferences.opacity}%")
        self._setting(
            appearance,
            self.opacity,
            mark("Background opacity"),
            mark(
                "Higher values hide more of the original text. Lower values reveal more of "
                "the game "
                "behind the translation."
            ),
            self.opacity_value,
        )
        self.automatic_size = localized(Switch(), text=mark("Automatic text size"))
        self.automatic_size.setChecked(preferences.automatic_size)
        self._setting(
            appearance,
            self.automatic_size,
            mark("Automatic text size"),
            mark(
                "Starts near the original text height, then wraps and shrinks within limits "
                "to fit the translation."
            ),
        )
        self.text_size = QSpinBox()
        self.text_size.setRange(10, 60)
        self.text_size.setValue(preferences.text_size)
        localized(self.text_size, suffix=mark(" px"))
        self.text_size.setMaximumWidth(160)
        self.text_size.setEnabled(not preferences.automatic_size)
        self._setting(
            appearance,
            self.text_size,
            mark("Text size"),
            mark(
                "Fixed translation font size in physical pixels. "
                "Used only when Automatic text size is off."
            ),
        )
        appearance.addStretch()

        recognition = self._page(mark("Recognition"))
        self.source_picker = SourceLanguageCombo(preferences.source_language)
        source_row = QHBoxLayout()
        source_label = localized(QLabel(), text=mark("Game text language"))
        source_label.setBuddy(self.source_picker)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_picker, 1)
        recognition.addLayout(source_row)
        self.group_dialogue = localized(Switch(), text=mark("Group dialogue lines"))
        self.group_dialogue.setChecked(preferences.group_dialogue)
        self._setting(
            recognition,
            self.group_dialogue,
            mark("Group dialogue lines"),
            mark(
                "Translates nearby lines from the same dialogue panel together. Turn off "
                "if unrelated "
                "text is being joined."
            ),
        )
        self.ocr_device = QComboBox()
        self._ocr_choices: tuple[OcrOption, ...] = ()
        self.set_ocr_choices(ocr_choices)
        self._setting(
            recognition,
            self.ocr_device,
            mark("OCR device"),
            mark(
                "Chooses the device that reads game text, separately from the translation model. "
                "Automatic reports CPU fallback when acceleration cannot be used. Install "
                "acceleration "
                "in the Acceleration tab. AMD currently uses CPU OCR."
            ),
        )
        self.capture_rate = QSpinBox()
        self.capture_rate.setRange(0, MAX_CAPTURE_RATE)
        localized(self.capture_rate, suffix=mark(" samples/s"), specialValueText=mark("Unlimited"))
        self.capture_rate.setMaximumWidth(200)
        self.capture_rate.setValue(preferences.capture_rate)
        self._setting(
            recognition,
            self.capture_rate,
            mark("Capture rate"),
            mark(
                "How often the app checks the game image for changes, not OCR calls per second. "
                "Higher rates use more resources and cannot speed up a busy OCR worker."
            ),
        )
        self.capture_note = localized(
            QLabel(), text=mark("Device and capture rate apply when capture starts.")
        )
        self.capture_note.setWordWrap(True)
        self.capture_note.setObjectName("muted")
        recognition.addWidget(self.capture_note)
        recognition.addStretch()

        self.models = ModelManager(setup, self, cloud=cloud, ollama=ollama, embedded=True)
        self.models.runtime_requested.connect(self.show_runtime)
        self._pages["Models"] = self.models
        self.tabs.addTab(self.models, tr("Models"))
        self.runtime_layout = self._page(mark("Acceleration"))
        self.runtime_layout.addStretch()
        shortcuts = self._page(mark("Hotkeys"))
        self.hotkeys = HotkeySettings(
            preferences.overlay_hotkey,
            preferences.capture_hotkey,
            preferences.capture_focused_window,
            hotkeys,
        )
        self.hotkeys.changed.connect(self._save_hotkeys)
        shortcuts.addWidget(self.hotkeys)
        self.cache = CacheSettings(cache)
        self._pages["Cache"] = self.cache
        self.tabs.addTab(self.cache, tr("Cache"))
        experimental = self._page(mark("Experimental"))
        note = localized(
            QLabel(),
            text=mark("Features under development. Results may be inaccurate or incomplete."),
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        experimental.addWidget(note)
        self.history_enabled = localized(Switch(), text=mark("Dialogue history"))
        self.history_enabled.setChecked(preferences.history_enabled)
        self._setting(
            experimental,
            self.history_enabled,
            mark("Dialogue history"),
            mark(
                "Collect recent dialogue in a separate reader. Dialogue and speaker detection may "
                "be inaccurate. Disabled by default; your choice is saved."
            ),
        )
        self.inspection_enabled = localized(Switch(), text=mark("Inspection and diagnostics"))
        self.inspection_enabled.setChecked(preferences.inspection_enabled)
        experimental.addWidget(self.inspection_enabled)
        experimental.addStretch()

        def refresh_tabs() -> None:
            for idx, choice in enumerate(self._ocr_choices):
                self.ocr_device.setItemText(idx, display(choice.name))
            for source, page in self._pages.items():
                self.tabs.setTabText(self.tabs.indexOf(page), tr(source))

        on_language_change(self.tabs, refresh_tabs)
        self.tabs.currentChanged.connect(self._tab_changed)
        footer = QHBoxLayout()
        self.setup_assistant = localized(QPushButton(), text=mark("Setup assistant…"))
        localized(
            self.setup_assistant,
            toolTip=mark("Guided model downloads and optional GPU setup. Stop capture first."),
        )
        footer.addWidget(self.setup_assistant)
        self.save_note = QLabel()
        self.save_note.setWordWrap(True)
        self.save_note.setObjectName("muted")
        footer.addWidget(self.save_note, 1)
        close = localized(QPushButton(), text=mark("Close settings"))
        close.clicked.connect(self.close)
        footer.addStretch()
        footer.addWidget(close)
        layout.addLayout(footer)
        self.stabilize.toggled.connect(lambda value: edit(stabilize=value))
        self.automatic_size.toggled.connect(lambda value: edit(automatic_size=value))
        self.group_dialogue.toggled.connect(lambda value: edit(group_dialogue=value))
        self.history_enabled.toggled.connect(lambda value: edit(history_enabled=value))
        self.inspection_enabled.toggled.connect(lambda value: edit(inspection_enabled=value))
        self.backdrop_blur.toggled.connect(lambda value: edit(backdrop_blur=value))
        self.opacity.valueChanged.connect(self._opacity_changed)
        self.text_size.valueChanged.connect(lambda value: edit(text_size=value))
        self.capture_rate.valueChanged.connect(lambda value: edit(capture_rate=value))
        self.automatic_size.toggled.connect(lambda enabled: self.text_size.setEnabled(not enabled))
        self.ocr_device.currentIndexChanged.connect(self._ocr_changed)

    def _page(self, name: str) -> QVBoxLayout:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(18)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(page)
        self._pages[name] = scroll
        self.tabs.addTab(scroll, tr(name))
        return layout

    def _setting(
        self,
        layout: QVBoxLayout,
        control: QWidget,
        name: str,
        help: str,
        suffix: QWidget | None = None,
    ) -> None:
        localized(control, accessibleName=name, toolTip=help)
        header = QHBoxLayout()
        if isinstance(control, Switch):
            header.addWidget(control)
        else:
            label = localized(QLabel(), text=name)
            label.setBuddy(control)
            header.addWidget(label)
        if help:
            header.addWidget(InfoButton(name, help))
        header.addStretch()
        if suffix:
            header.addWidget(suffix)
        layout.addLayout(header)
        if not isinstance(control, Switch):
            layout.addWidget(control)

    def _opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value}%")
        self._edit(opacity=value)

    def set_ocr_choices(self, choices: tuple[OcrOption, ...]) -> None:
        if choices == self._ocr_choices:
            return
        self._ocr_choices = choices
        with QSignalBlocker(self.ocr_device):
            self.ocr_device.clear()
            for choice in choices:
                self.ocr_device.addItem(display(choice.name), choice)
                if choice.key == self._ocr_key:
                    self.ocr_device.setCurrentIndex(self.ocr_device.count() - 1)

    def _ocr_changed(self) -> None:
        choice = self.ocr_device.currentData()
        self._ocr_key = choice.key or "cpu"
        self._edit(ocr_device=self._ocr_key)

    def _save_hotkeys(self) -> None:
        self._edit(
            overlay_hotkey=self.hotkeys.rows[HotkeyAction.OVERLAY].committed,
            capture_hotkey=self.hotkeys.rows[HotkeyAction.CAPTURE].committed,
            capture_focused_window=self.hotkeys.capture_focused.isChecked(),
        )

    def refresh_view(
        self,
        preferences: Preferences,
        choices: tuple[OcrOption, ...],
        *,
        active: bool,
        can_change_source: bool,
    ) -> None:
        self.source_picker.setCurrentIndex(self.source_picker.findData(preferences.source_language))
        self.source_picker.setEnabled(can_change_source)
        self.set_ocr_choices(choices)
        self.set_capture_active(active)
        self.models.refresh_view()
        if self.cache.isVisible():
            self.cache.refresh_view()
        if self.runtime:
            self.runtime.refresh_view()

    def show_hotkeys(self) -> None:
        self.tabs.setCurrentWidget(self._pages["Hotkeys"])
        self.show()
        self.raise_()

    def set_capture_active(self, active: bool) -> None:
        self.capture_note.setText(
            tr(
                "Capture is running. Device and capture rate changes apply next time you "
                "start capture."
            )
            if active
            else tr("Device and capture rate apply when capture starts.")
        )

    def show_models(self) -> None:
        self.models.refresh()
        self.models.show()
        self.tabs.setCurrentWidget(self._pages["Models"])
        self.show()
        self.raise_()

    def show_recognition(self) -> None:
        self.tabs.setCurrentWidget(self._pages["Recognition"])
        self.show()
        self.raise_()

    def show_runtime(self, *, install_profile: str | None = None) -> None:
        self.tabs.setCurrentWidget(self._pages["Acceleration"])
        if self.runtime:
            self.runtime.show()
        self.show()
        self.raise_()
        if self.runtime and install_profile:
            self.runtime.start_install(install_profile)

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self._pages["Models"]:
            self.models.refresh()
        if self.tabs.widget(index) is self._pages["Acceleration"] and self.runtime is None:
            self.runtime = RuntimeSetupDialog(self._setup, self, embedded=True)
            self.runtime.restart_requested.connect(self.restart_requested.emit)
            self.runtime_layout.insertWidget(0, self.runtime)
            self.runtime_created.emit(self.runtime)

    @property
    def busy(self) -> bool:
        return bool(
            self._setup.job(SetupLane.MODELS).busy
            or self._setup.job(SetupLane.RUNTIME).busy
            or self._cloud.snapshot().busy
        )

    @property
    def problem(self) -> str:
        return (
            self._setup.job(SetupLane.MODELS).progress.error
            or self._cloud.snapshot().problem
            or self._setup.job(SetupLane.RUNTIME).progress.error
        )

    def cancel_operations(self) -> None:
        self._setup.cancel(SetupLane.MODELS)
        self._setup.cancel(SetupLane.RUNTIME)
        self._cloud.cancel()

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            question = QMessageBox(self)
            question.setWindowTitle(tr("Setup is still running"))
            question.setText(tr("Keep setup running after closing Settings?"))
            background = question.addButton(
                tr("Continue in background"), QMessageBox.ButtonRole.AcceptRole
            )
            question.addButton(tr("Cancel operation"), QMessageBox.ButtonRole.DestructiveRole)
            question.setDefaultButton(background)
            question.exec()
            if question.clickedButton() != background:
                self.cancel_operations()
        event.accept()
