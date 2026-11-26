"""Qt presentation for live processing, model selection and application shutdown owners."""

import logging
import subprocess
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Unpack

from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon, QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.api import ApplicationCommands
from dantetl.application.hotkeys import HotkeyAction, Hotkeys
from dantetl.application.preference_values import SettingsEdit
from dantetl.application.updates import Updates
from dantetl.application.values import ApplicationControls, ModelConsent, ShutdownPhase
from dantetl.capture.windows import Windows
from dantetl.domain import WindowTarget
from dantetl.i18n import display, mark, tr
from dantetl.paths import gui_executable, module_command
from dantetl.runtime.hardware_values import HardwareReader
from dantetl.translation.selection import ModelOption
from dantetl.ui.about import AboutDialog
from dantetl.ui.bug_reports import error_dialog
from dantetl.ui.controls import (
    CaptureButton,
    OutputLanguageCombo,
    OverlayToggleLayout,
    SourceLanguageCombo,
    StatusBanner,
    Switch,
)
from dantetl.ui.desktop_driver import DesktopDriver
from dantetl.ui.diagnostics import debug_display
from dantetl.ui.game_window_combo import GameWindowCombo
from dantetl.ui.history_panel import HistoryPanel
from dantetl.ui.icons import APP_ICON, ICON_DIR
from dantetl.ui.inspection import OcrInspector
from dantetl.ui.localization import install_language, localized, on_language_change
from dantetl.ui.main_sections import MainSections
from dantetl.ui.model_manager import ModelManager
from dantetl.ui.overlay import OverlayWindow
from dantetl.ui.processing_panel import ProcessingPanel
from dantetl.ui.routing_controls import AssignmentChange, RoutingControls
from dantetl.ui.runtime_dialog import RuntimeSetupDialog
from dantetl.ui.settings_window import SettingsWindow
from dantetl.ui.setup_assistant import SetupAssistant
from dantetl.ui.theme import STYLE

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Present the live session and manage explicit model/setup actions."""

    def __init__(
        self,
        application: ApplicationCommands,
        windows: Windows,
        driver: DesktopDriver,
        *,
        hardware: HardwareReader | None = None,
        hotkeys: Hotkeys | None = None,
        updates: Updates | None = None,
    ) -> None:
        super().__init__()
        self.application, self.driver = application, driver
        self._updates = updates
        self.about = AboutDialog(version("dantetl"), updates, self)
        self.about.restart.connect(self.close)
        self._windows, self._hardware, self._hotkeys = windows, hardware, hotkeys
        self._notice_destination = "models"
        self._model_manager: ModelManager | None = None
        self._runtime_dialog: RuntimeSetupDialog | None = None
        self._setup_assistant: SetupAssistant | None = None
        self._render_reset = self._render_english_reset = -1
        self._render_error = 0
        self._restart_failed = False
        area = self.screen().availableGeometry()
        self._compact_size = QSize(min(1160, area.width() - 60), min(680, area.height() - 80))
        self._progress_dialog: QProgressDialog | None = None
        self._overlay = OverlayWindow(windows)
        self.destroyed.connect(self._overlay.deleteLater)  # Keep its native window independent.
        self._overlay.set_scene(())
        self._demo_processes: list[subprocess.Popen[bytes]] = []
        self.setWindowTitle("DanteTL")
        self.resize(self._compact_size)
        self._build_ui()
        self._refresh_history_models(
            self.application.controls.options, self.application.values.history_model
        )
        self.history_panel.changed.connect(self._history_changed)
        self._history_changed()
        self._routing_changed()
        driver.bind(application, self._render_application, self)
        self.refresh_windows()
        if self._hotkeys:
            self._hotkeys.activated = self._hotkey_activated
        self._settings.hotkeys.activate()
        self._update_notice()

    def _build_ui(self) -> None:
        preferences = self.application.values
        self._settings = SettingsWindow(
            preferences,
            self.application.ocr_options,
            self,
            edit=self._edit_settings,
            cloud=self.application.cloud,
            ollama=self.application.ollama,
            setup=self.application.setup,
            cache=self.application.cache,
            hotkeys=self._hotkeys,
        )
        self._model_manager = self._settings.models
        self._settings.restart_requested.connect(self._request_restart)
        self._settings.runtime_created.connect(self._runtime_created)
        self._settings.model_notice.action.clicked.connect(self._choose_or_download_model)
        self._settings.acceleration_notice.action.clicked.connect(self._show_acceleration_advice)
        self._settings.setup_assistant.clicked.connect(self._show_setup_assistant)
        # Aliases for presentation wiring; application commands receive plain values.
        self.stabilize_check = self._settings.stabilize
        self.ocr_device = self._settings.ocr_device
        self.capture_rate = self._settings.capture_rate
        self.group_dialogue = self._settings.group_dialogue
        self.automatic_font = self._settings.automatic_size
        self.font_size = self._settings.text_size
        self.automatic_font.toggled.connect(self._font_style_changed)
        self.font_size.valueChanged.connect(self._font_style_changed)
        self._settings.opacity.valueChanged.connect(self._overlay.set_opacity)
        self._settings.backdrop_blur.toggled.connect(self._overlay.set_backdrop_blur)
        central = QWidget()
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        body = QVBoxLayout(central)
        body.setContentsMargins(24, 20, 24, 20)
        body.setSpacing(14)
        self.control_panel = QWidget()
        self.control_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        controls_row = QHBoxLayout()
        controls_row.addWidget(self.control_panel, 1)
        body.addLayout(controls_row)
        layout = QVBoxLayout(self.control_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        header = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(QIcon(str(APP_ICON)).pixmap(QSize(40, 40), self.devicePixelRatioF()))
        header.addWidget(logo)
        heading = QLabel("DanteTL")
        heading.setObjectName("heading")
        header.addWidget(heading, 1)
        self.history_panel = HistoryPanel(
            lambda: self.application.history_turns, self.application.clear_history, self
        )
        self.history_button = localized(
            QPushButton(QIcon(str(ICON_DIR / "dialogue.svg")), ""), text=mark("Dialogue")
        )
        localized(self.history_button, accessibleName=mark("Dialogue history"))
        localized(
            self.history_button,
            toolTip=mark(
                "Experimental: read the last 20 dialogue turns. This does not change model context."
            ),
        )
        self.history_button.clicked.connect(self.history_panel.open_reader)
        header.addWidget(self.history_button)
        self.settings_button = QPushButton(QIcon(str(ICON_DIR / "settings.svg")), "")
        localized(self.settings_button, accessibleName=mark("Settings"))
        localized(self.settings_button, toolTip=mark("Settings"))
        self.settings_button.clicked.connect(self._show_settings)
        header.addWidget(self.settings_button)
        self.about_button = localized(
            QPushButton(QIcon(str(ICON_DIR / "about.svg")), ""),
            accessibleName=mark("About"),
            toolTip=mark("About"),
        )
        self.about_button.clicked.connect(self.about.open_about)
        header.addWidget(self.about_button)
        layout.addLayout(header)
        self.sections = MainSections()
        layout.addWidget(self.sections)
        game_layout = self.sections.game_layout
        game_layout.addWidget(localized(QLabel(), text=mark("Game window")))
        selection = QHBoxLayout()
        self.windows_combo = GameWindowCombo(
            partial(self._windows.read_process_details, copy=getattr(QImage, "fromHICON", None)),
            preferences.favorite_executables,
        )
        localized(self.windows_combo, accessibleName=mark("Game window"))
        self.windows_combo.setMinimumContentsLength(16)
        self.windows_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.windows_combo.opening.connect(lambda: self._list_windows(refresh=False))
        self.favorite_button = QPushButton()
        self.favorite_button.setObjectName("favoriteWindow")
        self.favorite_button.setCheckable(True)
        self.favorite_button.setFixedWidth(38)
        self.favorite_button.clicked.connect(self._toggle_favorite)
        self.windows_combo.favorite_state_changed.connect(self._show_favorite)
        self.refresh_button = localized(
            QPushButton(QIcon(str(ICON_DIR / "refresh.svg")), ""),
            accessibleName=mark("Refresh"),
            toolTip=mark("Refresh"),
        )
        self.refresh_button.clicked.connect(self.refresh_windows)
        selection.addWidget(self.windows_combo, 1)
        selection.addWidget(self.favorite_button)
        selection.addWidget(self.refresh_button)
        game_layout.addLayout(selection)
        self.start_button = CaptureButton()
        self.start_button.clicked.connect(self._toggle_capture)
        game_layout.addWidget(self.start_button)
        game_layout.addSpacing(4)
        self.overlay_check = localized(Switch(), text=mark("Show game overlay"))
        self.overlay_check.setChecked(preferences.show_overlay)
        localized(
            self.overlay_check,
            toolTip=mark("Show or hide game rendering. Capture and translation keep their state."),
        )
        self.translate_check = localized(Switch(), text=mark("Translation overlay"))
        self.translate_check.setChecked(preferences.translation)
        localized(
            self.translate_check,
            toolTip=mark(
                "Translate and cover recognized text. Turn off to leave the original game "
                "text visible."
            ),
        )
        self.border_check = localized(Switch(), text=mark("Show border"))
        self.border_check.setChecked(preferences.show_border)
        localized(
            self.border_check,
            toolTip=mark(
                "Show Windows' yellow capture indicator. Changes apply during capture without "
                "clearing translations. Windows or another capture app may still require "
                "the border."
            ),
        )
        self.capture_overlay_check = localized(Switch(), text=mark("Include in screen capture"))
        self.unfocused_overlay_check = localized(
            Switch(),
            text=mark("Show overlay when unfocused"),
            toolTip=mark(
                "Keep translations visible on the game when you switch to another window. "
                "Windows in front of the game cover its translations."
            ),
        )
        self.unfocused_overlay_check.setChecked(preferences.show_unfocused_overlay)
        self.capture_overlay_check.setChecked(preferences.include_overlay_in_capture)
        localized(
            self.capture_overlay_check,
            toolTip=mark(
                "Include translations in screenshots, display recordings and screen sharing. "
                "Turn off to ask Windows to exclude the overlay. Game/window-only capture "
                "may omit it either way; this does not affect DanteTL's OCR."
            ),
        )
        game_layout.addLayout(
            OverlayToggleLayout(
                self.overlay_check,
                self.translate_check,
                self.border_check,
                self.capture_overlay_check,
                self.unfocused_overlay_check,
            )
        )
        self.capture_status = StatusBanner()
        self.capture_status.set_message(tr("Select a window, then start capture."), tone="info")
        self.status = self.capture_status.label
        game_layout.addWidget(self.capture_status)
        self._overlay.set_capture_included(preferences.include_overlay_in_capture)
        self.source_picker = SourceLanguageCombo(preferences.source_language)
        self.output_language = OutputLanguageCombo(preferences.output_language)
        languages = QGridLayout()
        languages.setHorizontalSpacing(16)
        languages.setVerticalSpacing(10)
        languages.setColumnStretch(1, 1)
        for row, (combo, source) in enumerate(
            (
                (self.source_picker, mark("Game text language")),
                (self.output_language, mark("Output language")),
            )
        ):
            label = localized(QLabel(), text=source)
            label.setBuddy(combo)
            languages.addWidget(label, row, 0)
            languages.addWidget(combo, row, 1)
        self.sections.translation_layout.addLayout(languages)
        for picker in (self.source_picker, self._settings.source_picker):
            picker.activated.connect(lambda _, p=picker: self._change_source(p))
        self.language_notice = StatusBanner()
        self.sections.translation_layout.addWidget(self.language_notice)
        self.output_language.language_changed.connect(self._change_output_language)
        for banner in (self.language_notice, self._settings.language_notice):
            banner.action.clicked.connect(self._resolve_language_notice)
        self.routing = RoutingControls(preferences)
        self.sections.translation_layout.addWidget(self.routing)
        controls = self.application.controls
        self.routing.set_models(controls.options, controls.assignment)
        self.backend_combo = self.routing.simple
        self.routing.changed.connect(self._routing_changed)
        self.routing.model_requested.connect(self._request_assignment)
        self.routing.disclosure_changed.connect(
            lambda: self.application.configure(advanced_routing=self.routing.advanced.isChecked())
        )
        self.windows_combo.currentIndexChanged.connect(self._game_window_changed)
        self.onboarding = StatusBanner()
        self.notice = self.onboarding.label
        self.models_button = self.onboarding.action
        self.models_button.clicked.connect(self._show_setup_status)
        layout.addWidget(self.onboarding)
        self.acceleration_notice = StatusBanner()
        self.acceleration_notice.action.clicked.connect(self._show_acceleration_advice)
        layout.addWidget(self.acceleration_notice)
        self.hotkey_notice = StatusBanner()
        self.hotkey_notice.action.clicked.connect(self._settings.show_hotkeys)
        layout.addWidget(self.hotkey_notice)
        self.advanced = QToolButton()
        localized(self.advanced, text=mark("Inspection and diagnostics"))
        self.advanced.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.advanced)
        self.advanced_panel = QWidget()
        inspection = QVBoxLayout(self.advanced_panel)
        inspection.setContentsMargins(0, 0, 0, 0)
        diagnostics = QHBoxLayout()
        self.raw_boxes_check = localized(Switch(), text=mark("Show OCR boxes"))
        localized(
            self.raw_boxes_check,
            toolTip=mark(
                "Numbered OCR boxes on the game. Turn Translation overlay off to compare them."
            ),
        )
        self.raw_boxes_check.setEnabled(not self.translate_check.isChecked())
        self.debug_check = localized(Switch(), text=mark("Debug overlay"))
        localized(
            self.debug_check,
            toolTip=mark(
                "Amber: visual changes. Cyan: completed OCR coverage. "
                "Timings and diagnostics do not change processing."
            ),
        )
        diagnostics.addWidget(self.raw_boxes_check)
        diagnostics.addWidget(self.debug_check)
        diagnostics.addStretch()
        self.demo_button = localized(QPushButton(), text=mark("Open demo window"))
        self.demo_button.clicked.connect(self._open_demo)
        diagnostics.addWidget(self.demo_button)
        inspection.addLayout(diagnostics)
        label = localized(QLabel(), text=mark("RAW FRAME + DETECTED TEXT · PERSISTENT IDS"))
        label.setObjectName("eyebrow")
        inspection.addWidget(label)
        self.inspector = OcrInspector()
        self.inspector.set_active(False)
        inspection.addWidget(self.inspector, 1)
        self.processing = ProcessingPanel()
        self.ocr_status = self.processing.ocr.status
        self.translation_status = self.processing.translation.status
        inspection.addWidget(self.processing)
        self.advanced_panel.hide()
        body.addWidget(self.advanced_panel, 1)
        self._compact_space = QWidget()
        body.addWidget(self._compact_space, 1)
        self.advanced.toggled.connect(self._advanced_changed)
        self._set_inspection_enabled(preferences.inspection_enabled)
        self.overlay_check.toggled.connect(self._overlay_visibility_changed)
        self.unfocused_overlay_check.toggled.connect(
            lambda enabled: self._edit_settings(show_unfocused_overlay=enabled)
        )
        self.border_check.toggled.connect(self._border_changed)
        self.capture_overlay_check.toggled.connect(self._capture_visibility_changed)
        self.translate_check.toggled.connect(lambda value: self._edit_settings(translation=value))
        self.translate_check.toggled.connect(self.raw_boxes_check.setDisabled)
        self.raw_boxes_check.toggled.connect(self._tick)
        self.debug_check.toggled.connect(self._tick)
        self._settings.interface_language.currentIndexChanged.connect(
            self._change_interface_language
        )
        self.setStyleSheet(STYLE)
        # Both forms use the widest styled label, including the routing switches.
        grids = (languages, self.routing.model_grid)

        def align_fields() -> None:
            label_width = max(
                label.sizeHint().width()
                for grid in grids
                for row in range(grid.rowCount())
                if (item := grid.itemAtPosition(row, 0)) and (label := item.widget())
            )
            for grid in grids:
                grid.setColumnMinimumWidth(0, label_width)

        align_fields()
        on_language_change(self, align_fields)
        self._overlay.set_opacity(preferences.opacity)
        self._overlay.set_backdrop_blur(preferences.backdrop_blur)
        self._font_style_changed()
        self._update_notice()

    def _show_favorite(self) -> None:
        executable = self.windows_combo.selected_executable
        checked = executable in self.application.values.favorite_executables
        self.favorite_button.setEnabled(bool(executable))
        self.favorite_button.setChecked(checked)
        self.favorite_button.setText("★" if checked else "☆")
        action = tr("Remove from favorites") if checked else tr("Add to favorites")
        self.favorite_button.setAccessibleName(action)
        self.favorite_button.setToolTip(action)

    def _toggle_favorite(self, checked: bool) -> None:
        executable = self.windows_combo.selected_executable
        if executable:
            self.application.set_window_favorite(executable, checked)
            self.windows_combo.set_favorites(self.application.values.favorite_executables)

    def _game_window_changed(self) -> None:
        self._show_favorite()
        target = self.windows_combo.currentData()
        if not self.application.controls.capture_active:
            self.start_button.setEnabled(
                isinstance(target, WindowTarget) and not self.application.controls.changing_source
            )
        title = target.title if isinstance(target, WindowTarget) else ""
        self.routing.set_window_title(title)
        self.application.set_window_title(title)

    def _change_interface_language(self) -> None:
        app = QApplication.instance()
        assert isinstance(app, QApplication)
        code = self._settings.interface_language.currentData()
        self.application.configure(interface_language=code)
        install_language(app, code)
        self._history_options = None
        self.driver.refresh(inspect=True)

    def _change_output_language(self, code: str) -> None:
        self.application.configure(output_language=code)
        self._tick()

    def _resolve_language_notice(self) -> None:
        advice = self.application.notices().language
        if advice.change:
            self._request_assignment(advice.change)
        elif advice.message:
            self._settings.show_models()
            self._settings.models.select_translation_file("gemma")

    def _refresh_history_models(
        self, options: tuple[ModelOption, ...], selected: str | None = None
    ) -> None:
        if options is not getattr(self, "_history_options", None):
            self._history_options = options
            self._history_choices = tuple(
                (c.reference, display(c.name).removesuffix(tr(" · Recommended")), c.prepared)
                for c in options
                if c.classifies
            )
        choices = self._history_choices
        self.history_panel.set_models(
            choices, self.history_panel.model_key if selected is None else selected
        )

    def _history_changed(self) -> None:
        self.application.configure(history_model=self.history_panel.model_key)
        self._render_history()

    def _render_history(self) -> None:
        enabled = self.application.values.history_enabled
        self.history_button.setVisible(enabled)
        self.history_panel.setEnabled(enabled)
        if not enabled:
            self.history_panel.hide()
        self.history_panel.status.setText(self.application.history_status)
        self.history_panel.refresh()

    def _request_assignment(self, change: AssignmentChange) -> None:
        prompt = self.application.assign(change)
        if prompt:
            self._ask_preparation(prompt)
        self.routing.set_selection(self.application.controls.assignment)
        self._render_application(True)

    def _routing_changed(self) -> None:
        self.application.configure(
            game_hint=self.routing.manual_game_hint,
            game_from_window=self.routing.game_from_window.isChecked(),
        )
        self._render_scene()
        controls = self.application.controls
        self.routing.set_models(controls.options, controls.assignment)
        self._update_notice()

    def offer_first_run(self) -> None:
        self.application.offer_first_run()
        self.application.advance(time.monotonic())
        self._handle_desktop_action()

    def offer_cuda_update(self) -> None:
        dialog = QMessageBox(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle(tr("Update CUDA"))
        dialog.setText(tr("DanteTL needs to update CUDA to use your NVIDIA graphics card again."))
        dialog.setInformativeText(
            tr(
                "Setup keeps your downloaded models. "
                "You can also update later in Settings → Acceleration."
            )
        )
        update = dialog.addButton(tr("Update CUDA"), QMessageBox.ButtonRole.AcceptRole)
        cpu = dialog.addButton(tr("Continue on CPU"), QMessageBox.ButtonRole.RejectRole)
        assert isinstance(update, QPushButton)
        dialog.setDefaultButton(update)
        dialog.setEscapeButton(cpu)
        dialog.finished.connect(
            lambda _: (
                self._settings.show_runtime(install_profile="nvidia")
                if dialog.clickedButton() is update
                else None
            )
        )
        dialog.open()

    def _show_setup_assistant(self) -> None:
        if not self.application.can_setup:
            return
        if self._setup_assistant and (
            self._setup_assistant.isVisible() or self._setup_assistant.busy
        ):
            self._setup_assistant.show()
            self._setup_assistant.raise_()
            return
        if self._setup_assistant:
            self._setup_assistant.deleteLater()
        self._settings.hide()
        self._setup_assistant = SetupAssistant(
            self.application.setup,
            self._hardware,
            self,
            cloud=self._settings.models.cloud,
            ollama=self.application.ollama,
            interface_language=self.application.values.interface_language,
            backdrop_blur=self.application.values.backdrop_blur,
        )
        self._setup_assistant.interface_language.currentIndexChanged.connect(
            self._settings.interface_language.setCurrentIndex
        )
        self._setup_assistant.backdrop_blur.toggled.connect(self._settings.backdrop_blur.setChecked)
        self._setup_assistant.skipped.connect(self._setup_skipped)
        self._setup_assistant.completed.connect(self._setup_completed)
        self._setup_assistant.show()

    def _setup_skipped(self) -> None:
        self.application.setup_skipped()

    def _setup_completed(self, model: str, target: Path | None, deferred: bool) -> None:
        self.application.setup_completed(model, target, defer_preparation=deferred)
        self.driver.refresh(inspect=False)

    def _edit_settings(self, **changes: Unpack[SettingsEdit]) -> None:
        self.application.configure(**changes)
        if "inspection_enabled" in changes:
            self._set_inspection_enabled(self.application.values.inspection_enabled)
        self._tick()

    def _border_changed(self, enabled: bool) -> None:
        self.application.configure(show_border=enabled)

    def _overlay_visibility_changed(self, enabled: bool) -> None:
        if enabled:
            self._overlay.retry_configuration()
        self._edit_settings(show_overlay=enabled)

    def _capture_visibility_changed(self, included: bool) -> None:
        try:
            self._overlay.set_capture_included(included)
        except (OSError, RuntimeError) as error:
            with QSignalBlocker(self.capture_overlay_check):
                self.capture_overlay_check.setChecked(not included)
            self._open_warning(tr("Screen capture setting"), str(error))
            return
        self.application.configure(include_overlay_in_capture=included)

    def _show_settings(self) -> None:
        if not self.application.closing:
            self._settings.show()
            self._settings.raise_()

    def _hotkey_activated(self, action: HotkeyAction) -> None:
        if self.application.closing or QApplication.activeModalWidget() is not None:
            return
        if action == HotkeyAction.OVERLAY:
            self.overlay_check.setChecked(not self.overlay_check.isChecked())
        elif action == HotkeyAction.CAPTURE:
            if self.application.controls.changing_source:
                return
            if (
                not self.application.controls.capture_active
                and self._settings.hotkeys.capture_focused.isChecked()
            ):
                target = self.application.foreground_target()
                if target is None:
                    self.status.setText(self.application.view.status)
                    return
                self._select_target(target)
            # Use the button's existing lifecycle guard, including asynchronous stop/drain.
            self.start_button.click()

    def _set_inspection_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.advanced.setChecked(False)
            self.inspector.pause.setChecked(False)
            with QSignalBlocker(self.raw_boxes_check), QSignalBlocker(self.debug_check):
                self.raw_boxes_check.setChecked(False)
                self.debug_check.setChecked(False)
        self.advanced.setCheckable(enabled)
        self.advanced.setVisible(enabled)

    def _advanced_changed(self, expanded: bool) -> None:
        maximized = self.isMaximized() or self.isFullScreen()
        if expanded and not maximized:
            self._compact_size = self.size()
        self.advanced.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.advanced_panel.setVisible(expanded)
        self._compact_space.setVisible(not expanded)
        self.inspector.set_active(expanded)
        # Recompute minimum sizes before restoring the compact geometry.
        if (central := self.centralWidget()) and (layout := central.layout()):
            layout.activate()
        if layout := self.layout():
            layout.activate()
        if maximized:
            return
        if expanded:
            screen = self.screen()
            area = screen.availableGeometry() if screen else self.geometry()
            self.resize(min(1380, area.width() - 40), min(900, area.height() - 80))
        else:
            self.resize(self._compact_size)

    def _show_setup_status(self) -> None:
        if self._notice_destination == "usage":
            QDesktopServices.openUrl(QUrl("https://chatgpt.com/settings/usage"))
        elif self._notice_destination in ("ollama", "cloud"):
            self._settings.show_models()
            self._settings.models.category.setCurrentIndex(
                self._settings.models.category.findData(self._notice_destination)
            )
        elif self._notice_destination == "prepare":
            notice = self.application.notices()
            if notice.prepare_reference:
                self._request_assignment(
                    AssignmentChange(self.application.controls.assignment, notice.prepare_reference)
                )
        elif self._notice_destination == "selection":
            self._settings.hide()
            self.routing.focus_model()
        elif self._notice_destination == "settings":
            self._show_settings()
        elif self._notice_destination == "runtime" or (
            self._settings.runtime
            and (self._settings.runtime.busy or self._settings.runtime.error_message)
        ):
            self._manage_runtime()
        else:
            self._manage_models()

    def _show_acceleration_advice(self) -> None:
        advice = self.application.notices().acceleration
        if advice != self._last_notices.acceleration:
            self._update_notice()
            return
        if target := advice.restart_target:
            self._request_restart(target)
        else:
            self._manage_runtime()

    def _choose_or_download_model(self) -> None:
        self._notice_destination = self.application.notices().model_destination
        self._show_setup_status()

    def _update_notice(self) -> None:
        self._settings.setup_assistant.setEnabled(self.application.can_setup)
        self.hotkey_notice.set_message(self._settings.hotkeys.problem, action=tr("Hotkeys"))
        if self._hotkeys:
            capture_key = self._hotkeys.shortcut(HotkeyAction.CAPTURE) or "disabled"
            overlay_key = self._hotkeys.shortcut(HotkeyAction.OVERLAY) or "disabled"
            source = (
                tr("focused window")
                if self._settings.hotkeys.capture_focused.isChecked()
                else tr("selected window")
            )
            self.start_button.setToolTip(
                tr(
                    "Button: start / stop the selected window. Hotkey {capture_key}: start "
                    "the {source_text}, or stop the current capture.",
                    capture_key=capture_key,
                    source_text=source,
                )
            )
            self.overlay_check.setToolTip(
                tr(
                    "Show or hide game rendering. Capture and translation keep their "
                    "state. Hotkey: {overlay_key}",
                    overlay_key=overlay_key,
                )
            )
        value = self.application.notices()
        self._notice_destination = value.destination
        if value == getattr(self, "_last_notices", None):
            return
        self._last_notices = value
        for banner in (self.language_notice, self._settings.language_notice):
            banner.set_message(value.language.message, action=value.language.action)
        self._settings.model_notice.set_message(value.model_warning, action=value.model_action)
        self._settings.save_note.setText(value.save_note)
        self._settings.save_note.setVisible(bool(value.save_note))
        self.onboarding.set_message(value.message, tone=value.tone, action=value.action)
        for banner in (self.acceleration_notice, self._settings.acceleration_notice):
            banner.set_message(
                value.acceleration.message,
                tone="info" if value.acceleration.restart_target else "warning",
                action=tr(value.acceleration.action),
            )

    def refresh_windows(self) -> None:
        self._list_windows(refresh=True)

    def _list_windows(self, *, refresh: bool) -> None:
        if self.application.controls.capture_active:
            return
        try:
            targets = self._windows.list_targets()
        except OSError as error:
            targets = []
            self.status.setText(tr("Could not list windows: {error}", error=error))
        else:
            if not targets:
                self.status.setText(tr("No windows found. Open a game or the demo, then refresh."))
        previous = self.windows_combo.currentData()
        self.windows_combo.set_targets(targets, refresh=refresh)
        if refresh or self.windows_combo.currentData() != previous:
            self._game_window_changed()

    def _toggle_capture(self) -> None:
        if self.application.closing or not self.start_button.isEnabled():
            return
        if self.application.controls.capture_active:
            self._stop(tr("Capture stopped."))
            return
        self.windows_combo.keep_selection()
        target = self.windows_combo.currentData()
        if isinstance(target, WindowTarget) and self.application.start_capture(
            target, time.monotonic()
        ):
            self.inspector.pause.setChecked(False)
            self.routing.set_window_title(target.title)
        self._render_application(False)

    def _stop(self, message: str) -> None:
        self.application.stop(message)
        self._render_application(False)

    def _clear_scene_display(self) -> None:
        self._overlay.set_scene(())
        self._overlay.set_debug(None)

    def _font_style_changed(self) -> None:
        automatic = self.automatic_font.isChecked()
        self.font_size.setEnabled(not automatic)
        self._overlay.set_text_style(automatic, self.font_size.value())

    def _ask_preparation(self, prompt: ModelConsent) -> None:
        choice = prompt.option
        dialog = QMessageBox(
            QMessageBox.Icon.Question,
            tr("Prepare translation model for this PC?"),
            tr(
                "The model files will be checked and a short test translation will run. "
                "First GPU use may take about a minute. Capture can continue while it prepares.\n\n"
            )
            + (display(choice.performance_notice) + "\n\n" if choice.performance_notice else "")
            + tr("Prepare now? Choosing No keeps your previous assignments."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self,
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(
            lambda result: self.application.consent(
                prompt.token, result == QMessageBox.StandardButton.Yes
            )
        )
        dialog.open()

    def _show_preparation(self, message: str) -> None:
        if self._progress_dialog is None:
            dialog = QProgressDialog(message, "", 0, 0, self)
            dialog.setWindowTitle(tr("Preparing local translation"))
            dialog.setCancelButton(None)
            dialog.setMinimumDuration(0)
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.setMinimumWidth(470)
            self._progress_dialog = dialog
        self._progress_dialog.setLabelText(message)
        self._progress_dialog.show()

    def _render_preparation(self, controls: ApplicationControls) -> None:
        preparing = controls.preparing
        for banner in (self.language_notice, self._settings.language_notice):
            banner.action.setEnabled(not preparing)
        self.routing.setEnabled(not preparing)
        self.backend_combo.setEnabled(not preparing)
        if preparing:
            self._show_preparation(controls.preparation_message)
        elif self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        if self._render_error != self.application.error_revision:
            self._render_error = self.application.error_revision
            self._open_warning(tr("Model preparation failed"), self.application.error)

    def _manage_models(self) -> None:
        if not self.application.closing:
            self._settings.show_models()

    def _runtime_created(self, panel: RuntimeSetupDialog) -> None:
        self._runtime_dialog = panel

    def _manage_runtime(self) -> None:
        if self.application.closing:
            return
        self._settings.show_runtime()

    def _select_target(self, target: WindowTarget) -> None:
        self.windows_combo.select_target(target)
        self._game_window_changed()
        self.start_button.setEnabled(True)

    def _change_source(self, picker: SourceLanguageCombo) -> None:
        code, name = picker.currentData(), picker.currentText()
        active = self.application.values.source_language
        picker.setCurrentIndex(picker.findData(active))
        if code == active or not self.application.can_change_source:
            return
        if self.application.source_requires_restart(code):
            message = tr("Switch the game text language to {name}?", name=name)
            message += tr("\n\nCapture will restart on the same game window.")
            message += tr(" This clears the current dialogue history.")
            if (
                QMessageBox.question(
                    picker,
                    tr("Change game text language"),
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
        if not self.application.change_source(code):
            return
        self.driver.refresh(inspect=False)

    def _request_restart(self, target: Path) -> None:
        self.application.request_close(target)
        self._settings.hide()
        if central := self.centralWidget():
            central.setEnabled(False)
        self.status.setText(tr("Finishing active work before restarting. Capture will be stopped."))
        self.close()

    def _refresh_models(self) -> None:
        self.application.refresh_models()

    def _tick(self) -> None:
        self.application.set_diagnostics(
            boxes=self.raw_boxes_check.isChecked(), debug=self.debug_check.isChecked()
        )
        self.driver.refresh(inspect=True)

    def _refresh_metrics(self, controls: ApplicationControls) -> None:
        if self.advanced.isChecked():
            self.processing.refresh(
                self.application.view.capture,
                self.application.view.ocr,
                self.application.view.translation,
                self.application.view.visual,
                now=time.monotonic(),
                started_at=controls.started_at,
                capture_rate=controls.capture_rate,
                model=self.routing.summary,
            )

    def _open_warning(self, title: str, message: str) -> None:
        dialog = error_dialog(self, title, message)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _show_restart_failure(self, message: str) -> None:
        if self._restart_failed:
            return
        self._restart_failed = True
        self._settings.hotkeys.activate()
        self._show_capture_stopped()
        self.routing.set_selection(self.application.controls.assignment)
        self.backend_combo.setEnabled(True)
        self.routing.setEnabled(True)
        if central := self.centralWidget():
            central.setEnabled(True)
        self.status.setText(message)
        self._open_warning(tr("Restart failed"), message)

    def _render_application(self, inspect: bool) -> None:
        shutdown = self.application.close_status
        if shutdown.phase == ShutdownPhase.FAILED and not self._restart_failed:
            self._show_restart_failure(shutdown.message)
            return
        if shutdown.phase != ShutdownPhase.FAILED:
            self._restart_failed = False
        if self.application.closing:
            self.close()
            return
        if self._updates:
            self.about.show_update(self._updates.snapshot())
            if self.application.closing:
                return
            if (
                not QApplication.activeModalWidget()
                and not (self._setup_assistant and self._setup_assistant.isVisible())
                and (offer := self._updates.take_offer())
            ):
                self.about.offer(offer)
        self.output_language.set_language(self.application.values.output_language)
        with self.inspector.defer_rendering():
            self._render_scene()
        controls = self.application.controls
        self._render_preparation(controls)
        self.routing.set_models(controls.options, controls.assignment)
        self._refresh_history_models(controls.options)
        if self._setup_assistant:
            self._setup_assistant.refresh_view()
        active = controls.capture_active
        can_change_source = self.application.can_change_source
        self.source_picker.setCurrentIndex(
            self.source_picker.findData(self.application.values.source_language)
        )
        self.source_picker.setEnabled(can_change_source)
        self._settings.refresh_view(
            self.application.values,
            self.application.ocr_options,
            active=active,
            can_change_source=can_change_source,
        )
        self.windows_combo.setEnabled(not active and not controls.changing_source)
        self.refresh_button.setEnabled(not active and not controls.changing_source)
        self.start_button.set_capturing(active)
        self.start_button.setEnabled(
            not controls.changing_source
            and (
                controls.can_stop
                if active
                else isinstance(self.windows_combo.currentData(), WindowTarget)
            )
        )
        progress = self.application.view.progress
        self.capture_status.set_message(
            tr(self.application.view.status),
            tone=progress.tone,
            detail=progress.message,
            busy=progress.busy,
        )
        self.ocr_status.setText(display(self.application.view.ocr_status))
        self.translation_status.setText(display(self.application.view.translation_status))
        self._render_history()
        if inspect:
            self.inspector.refresh()
            self._update_notice()
            self._refresh_metrics(controls)
        self._handle_desktop_action()

    def _render_scene(self) -> None:
        value = self.application.view
        if value.reset != self._render_reset:
            self._render_reset = value.reset
            self.inspector.clear()
            self._clear_scene_display()
        if value.english_reset != self._render_english_reset:
            self._render_english_reset = value.english_reset
            self.inspector.clear_translations()
        if value.observation is not None:
            self.inspector.set_result(value.observation, value.assignments)
        elif value.frame is not None:
            self.inspector.set_frame(value.frame)
        self.inspector.set_translations(value.results)
        self.processing.waiting_blocks = value.scene.waiting_blocks
        self._overlay.set_scene(
            value.scene.tracks
            if self.application.values.translation or self.raw_boxes_check.isChecked()
            else (),
            value.blocks,
            value.english,
            english=self.application.values.translation,
        )
        self._overlay.set_debug(
            debug_display(
                self.application.view.visual,
                value.ocr,
                value.translation,
                time.monotonic(),
                waiting_blocks=value.scene.waiting_blocks,
            )
            if self.debug_check.isChecked() and value.ocr
            else None
        )
        if value.overlay_bounds is not None:
            if error := self._overlay.align(value.overlay_bounds, anchor=value.overlay_anchor):
                self._show_overlay_error(error)
        else:
            self._overlay.hide()

    def _show_overlay_error(self, error: str) -> None:
        dialog = error_dialog(
            self,
            tr("Screen capture setting"),
            error,
            button=QMessageBox.StandardButton.Close,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        retry = dialog.addButton(tr("Retry setup"), QMessageBox.ButtonRole.ActionRole)
        retry.clicked.connect(self._overlay.retry_configuration)
        dialog.open()

    def _handle_desktop_action(self) -> None:
        self._settings.models.cloud.handle_desktop_actions()
        action = self.application.take_desktop_action()
        if action == "runtime":
            self._manage_runtime()
        elif action == "models":
            self._manage_models()
        elif action == "guide":
            self._show_setup_assistant()
        elif isinstance(action, ModelConsent):
            self._ask_preparation(action)

    def _open_demo(self) -> None:
        python = gui_executable()
        try:
            process = subprocess.Popen(module_command(python, "dantetl.ui.demo", "--dialogue"))
            self._demo_processes.append(process)
            self.status.setText(tr("Demo opened. Refresh the window list and select DanteTL Demo."))
            QTimer.singleShot(1200, self, lambda: self._list_windows(refresh=False))
        except OSError as error:
            self.status.setText(tr("Could not open the demo: {error}", error=error))

    def _show_capture_stopped(self) -> None:
        self.start_button.set_capturing(False)
        self.start_button.setEnabled(isinstance(self.windows_combo.currentData(), WindowTarget))
        self.windows_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self._settings.set_capture_active(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._updates:
            self._updates.cancel()
        self.about.hide()
        self.history_panel.hide()
        self.application.request_close()
        self._overlay.hide()
        self.inspector.clear()
        if self._progress_dialog:
            self._progress_dialog.hide()
        self.application.advance(time.monotonic())
        result = self.application.close_status
        if result.phase == ShutdownPhase.FAILED:
            self._show_restart_failure(result.message)
            event.ignore()
            return
        if result.phase != ShutdownPhase.READY:
            self.status.setText(result.message)
            event.ignore()
            return
        self.driver.close()
        self._settings.hide()
        if self._setup_assistant:
            self._setup_assistant.hide()
        self._overlay.close()
        super().closeEvent(event)
