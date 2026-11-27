import os
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest
from ui_fixtures import (
    app_owner,
    replace_backends,
    set_ocr_available,
    settings,
    settle_setup,
    setup_owner,
    show_scene,
)

from dantetl.languages.target import ENGLISH
from dantetl.ocr.observation import CandidateReview
from dantetl.runtime.setup_values import RuntimeTarget, SetupLane
from dantetl.ui.setup_assistant import SetupAssistant

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QMessageBox, QProgressDialog  # noqa: E402
from ui_fixtures import create_window as MainWindow  # noqa: E402
from ui_fixtures import find_model

from dantetl.application.translation_coordinator import model_reference
from dantetl.domain import CapturedFrame, Rect, WindowGeometry  # noqa: E402
from dantetl.ocr.observation import TextTrack  # noqa: E402
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion  # noqa: E402
from dantetl.translation.backend import BackendChoice  # noqa: E402
from dantetl.translation.grouping import TextBlock  # noqa: E402
from dantetl.translation.values import (  # noqa: E402
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)
from dantetl.translation.worker import TranslationSession  # noqa: E402
from dantetl.ui.diagnostics import DebugDisplay  # noqa: E402
from dantetl.ui.inspection import OcrInspector, image_target  # noqa: E402
from dantetl.ui.overlay import OverlayWindow  # noqa: E402
from dantetl.ui.text_layout import layout_english  # noqa: E402


def visible_messages(window):
    return [dialog for dialog in window.findChildren(QMessageBox) if dialog.isVisible()]


def answer_dialog(window, answer):
    (dialog,) = visible_messages(window)
    dialog.button(answer).click()


@pytest.mark.parametrize("code", ["pt-PT", "ar", "ja"])
def test_bundled_language_translates_controls_and_preserves_language_ids(app, code):
    from PySide6.QtCore import QLocale, Qt
    from PySide6.QtWidgets import QDialogButtonBox

    from dantetl.i18n import tr
    from dantetl.ui.controls import OutputLanguageCombo, SourceLanguageCombo
    from dantetl.ui.localization import CATALOGS, InterfaceLanguageCombo, install_language

    locale, direction = QLocale(), app.layoutDirection()
    try:
        install_language(app, code)
        interface = InterfaceLanguageCombo(code)
        assert {interface.itemData(i) for i in range(interface.count())} == {
            "en",
            *(path.stem for path in CATALOGS.glob("*.qm")),
        }
        assert interface.currentData() == code
        assert interface.itemText(interface.findData("en")) == "English"
        for index in range(interface.count()):
            name = interface.itemText(index)
            assert name and name[0] == name[0].upper()
        source, output = SourceLanguageCombo("ja"), OutputLanguageCombo("pt-BR")
        assert source.currentData() == "ja" and output.currentData() == "pt-BR"
        assert source.accessibleName() == tr("Game text language") != "Game text language"
        assert output.currentText() == tr("Portuguese (Brazil)") != "Portuguese (Brazil)"
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        assert buttons.button(QDialogButtonBox.StandardButton.Cancel).text() != "Cancel"
        assert (source.layoutDirection() == Qt.LayoutDirection.RightToLeft) == (code == "ar")
    finally:
        install_language(app, "en")
        QLocale.setDefault(locale)
        app.setLayoutDirection(direction)


def test_capture_visibility_applies_before_show_and_live_failure_preserves_state(app):
    windows = Mock()
    overlay = OverlayWindow(windows)
    bounds = Rect(10, 20, 600, 400)
    try:
        overlay.set_capture_included(False)
        windows.set_overlay_capture.assert_not_called()
        overlay.align(bounds)
        hwnd = int(overlay.winId())
        windows.set_overlay_capture.assert_called_once_with(hwnd, False)
        windows.reset_mock()
        overlay.set_capture_included(True)
        windows.set_overlay_capture.assert_called_once_with(hwnd, True)
        assert overlay.isVisible()
        windows.set_overlay_capture.side_effect = RuntimeError("Unavailable")
        with pytest.raises(RuntimeError, match="Unavailable"):
            overlay.set_capture_included(False)
        windows.reset_mock()
        overlay.set_capture_included(True)  # Failed exclusion did not commit a new state.
        windows.set_overlay_capture.assert_not_called()
        overlay.align(bounds)
        windows.configure_overlay.assert_not_called()
        windows.place_overlay.assert_not_called()
        assert overlay.isVisible() and int(overlay.winId()) == hwnd
    finally:
        overlay.close()
        shiboken6.delete(overlay)


@pytest.mark.parametrize("change_preference", [False, True])
def test_first_show_failure_is_hidden_until_retry_or_preference_change(app, change_preference):
    windows = Mock()
    overlay = OverlayWindow(windows)
    bounds = Rect(10, 20, 600, 400)
    try:
        overlay.set_capture_included(False)
        windows.set_overlay_capture.side_effect = OSError("Fixture affinity error")
        assert overlay.align(bounds) == "Fixture affinity error"
        assert overlay.align(bounds) is None and not overlay.isVisible()
        windows.set_overlay_capture.assert_called_once_with(int(overlay.winId()), False)
        windows.place_overlay.assert_not_called()
        windows.set_overlay_capture.side_effect = None
        if change_preference:
            overlay.set_capture_included(True)
        else:
            overlay.retry_configuration()
        assert overlay.align(bounds) is None and overlay.isVisible()
        windows.set_overlay_capture.assert_called_with(int(overlay.winId()), change_preference)
    finally:
        overlay.close()
        shiboken6.delete(overlay)


def test_first_show_failure_offers_one_working_retry_while_capture_continues(app, tmp_path):
    from ownership_fixtures import Replay, observe

    from dantetl.ui.desktop_driver import DesktopDriver
    from dantetl.ui.main_window import MainWindow as View

    replay = Replay(tmp_path)
    replay.windows.list_targets.return_value = []
    replay.windows.set_overlay_capture.side_effect = OSError("Fixture error in <capture>")
    driver = DesktopDriver()
    window = View(replay.app, replay.windows, driver)
    try:
        window.show()
        replay.app.start_capture(replay.capture.target, 0)
        replay.app.set_diagnostics(boxes=False, debug=True)
        observe(replay)
        driver.refresh(inspect=False)
        driver.refresh(inspect=False)
        dialogs = window.findChildren(QMessageBox)
        assert len(dialogs) == 1 and dialogs[0].text() == "Fixture error in &lt;capture&gt;"
        assert 'href="https://github.com/nhmelo/danteTL/issues"' in dialogs[0].informativeText()
        assert not window._overlay.isVisible() and replay.app.controls.capture_active
        replay.windows.set_overlay_capture.assert_called_once()
        replay.windows.set_overlay_capture.side_effect = None
        next(button for button in dialogs[0].buttons() if button.text() == "Retry setup").click()
        driver.refresh(inspect=False)
        assert window._overlay.isVisible()
    finally:
        driver.close()
        replay.close()
        window.close()


def test_capture_status_and_loading_stay_inside_game_panel_without_inspection(app, tmp_path):
    from ownership_fixtures import Replay, observe
    from PySide6.QtWidgets import QLabel

    from dantetl.application.model_inventory import InventoryData
    from dantetl.ocr.values import OcrSnapshot, OcrState
    from dantetl.ui.desktop_driver import DesktopDriver
    from dantetl.ui.main_window import MainWindow as View

    replay = Replay(tmp_path, data=InventoryData((BackendChoice("Local", Mock(), "base"),)))
    replay.windows.list_targets.return_value = []
    driver = DesktopDriver()
    window = View(replay.app, replay.windows, driver)
    window.show()
    try:
        assert window.sections.game.isAncestorOf(window.status)
        assert all(
            "Stay for the story" not in label.text() for label in window.findChildren(QLabel)
        )
        replay.app.start_capture(replay.capture.target, 1)
        observe(replay)
        replay.ocr.current = OcrSnapshot(OcrState.LOADING)
        driver.refresh(inspect=False)
        assert window.capture_status.progress.isVisible()
        assert "MeikiOCR" in window.capture_status.detail.text()
        observe(replay, 2)
        driver.refresh(inspect=False)
        assert not window.capture_status.progress.isVisible()
        assert window.capture_status.isVisible()
        replay.app.stop()
        driver.refresh(inspect=False)
        assert not window.capture_status.progress.isVisible()
    finally:
        driver.close()
        replay.close()
        window.close()


def test_capture_visibility_ui_remembers_choice_and_rolls_back_native_failure(app, monkeypatch):
    windows, preferences = Mock(), Mock()
    from dantetl.application.preference_values import Preferences

    windows.list_targets.return_value = []
    preferences.current = Preferences(include_overlay_in_capture=False)
    preferences.error = ""
    monkeypatch.setattr("dantetl.composition.PreferenceStore", lambda path: preferences)
    window = MainWindow(windows, Mock())
    try:
        assert not window.capture_overlay_check.isChecked()
        window._overlay.align(Rect(0, 0, 600, 400))
        windows.set_overlay_capture.assert_called_once_with(int(window._overlay.winId()), False)
        window.capture_overlay_check.setChecked(True)
        assert preferences.save.call_args.args[0].include_overlay_in_capture
        preferences.save.reset_mock()
        windows.set_overlay_capture.side_effect = RuntimeError("Fixture native failure")
        window.capture_overlay_check.setChecked(False)
        assert window.capture_overlay_check.isChecked()
        preferences.save.assert_not_called()
        warning = window.findChild(QMessageBox)
        assert warning is not None and warning.isVisible()
        assert "Fixture native failure" in warning.text()
        warning.accept()
    finally:
        window.close()
        window.deleteLater()


def test_first_run_is_explicit_skippable_and_does_not_change_manual_cpu_choice(app, tmp_path):
    from dantetl.runtime.hardware_values import Adapter, HardwareStatus
    from dantetl.runtime.model_store import model_catalog
    from dantetl.ui.setup_assistant import SetupAssistant

    hardware = Mock()
    hardware.snapshot.return_value = None
    models = tuple(replace(b, directory=tmp_path / b.key) for b in model_catalog())
    windows = Mock()
    windows.list_targets.return_value = []
    parent = MainWindow(windows, Mock(), models=models)
    guide = SetupAssistant(
        parent.application.setup, hardware, parent, cloud=settings(parent).models.cloud
    )
    guide.show()
    assert guide.model.currentData() == "gemma"
    assert guide.model.findData("ocr") == -1
    guide.next.click()
    guide.next.click()
    assert not guide.busy and not guide.next.isEnabled()
    guide.runtime.activated.emit(0)  # Choosing the already-selected CPU is still a manual choice.
    hardware.snapshot.return_value = HardwareStatus(
        adapters=(Adapter("RTX", 0x10DE, 16 * 2**30),), llama_device="cpu"
    )
    setup_owner(guide).advance()
    guide.refresh_view()
    assert guide.runtime.currentData().key == "cpu"
    guide.next.click()
    assert guide.pages.currentIndex() == 3 and guide._review and not guide._review.install_runtime
    assert not guide.busy and not list(tmp_path.iterdir())
    guide.skip.click()
    assert not guide.isVisible()
    parent.close()
    guide.deleteLater()
    parent.deleteLater()


def test_first_run_close_drains_worker_and_progress_reports_actual_download(
    app, tmp_path, monkeypatch
):
    from dantetl.runtime.hardware_values import HardwareStatus
    from dantetl.runtime.model_store import model_catalog
    from dantetl.runtime.setup_values import Cancelled, DownloadProgress, ModelProgress

    entered = threading.Event()

    def setup(plan, root, report, cancel):
        report(
            ModelProgress(
                "Downloading", stage=1, stages=3, download=DownloadProgress("fixture.whl", 20, 100)
            )
        )
        entered.set()
        cancel.wait(2)
        raise Cancelled()

    monkeypatch.setattr("dantetl.runtime.first_run.run_setup", setup)
    hardware = Mock()
    hardware.snapshot.return_value = HardwareStatus(llama_device="cpu")
    windows = Mock()
    windows.list_targets.return_value = []
    models = tuple(replace(b, directory=tmp_path / b.key) for b in model_catalog())
    parent = MainWindow(windows, Mock(), models=models, hardware=hardware)
    parent.offer_first_run()
    guide = parent._setup_assistant
    assert guide
    setup_owner(guide).advance()
    guide.refresh_view()
    guide.model.setCurrentIndex(guide.model.findData("gemma"))
    for _ in range(4):
        guide.next.click()
    assert entered.wait(2) and guide.busy
    setup_owner(guide).advance()
    guide.refresh_view()
    assert guide.progress_view.download.value() == 200
    assert guide.progress_view.download.maximum() == 1000
    parent.close()
    assert parent.application.closing  # Parent cannot abandon its setup worker.
    deadline = time.monotonic() + 3
    while guide.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not guide.busy
    setup_owner(guide).advance()
    guide.refresh_view()
    assert guide.progress_view.overall.maximum() == 1
    parent.close()
    parent.deleteLater()


def test_first_run_skip_persists_and_settings_can_reopen_it(app, tmp_path):
    from dantetl.application.preferences import PreferenceStore
    from dantetl.runtime.hardware_values import HardwareStatus
    from dantetl.runtime.model_store import model_catalog

    windows, hardware = Mock(), Mock()
    windows.list_targets.return_value = []
    hardware.snapshot.return_value = HardwareStatus(llama_device="cpu")
    path = tmp_path / "preferences.json"
    models = tuple(replace(b, directory=tmp_path / b.key) for b in model_catalog())
    window = MainWindow(windows, Mock(), models=models, hardware=hardware, preferences_path=path)
    application = app_owner(window)
    window.offer_first_run()
    guide = window.findChild(SetupAssistant)
    assert guide is not None
    assert guide.backdrop_blur.isChecked()
    guide.backdrop_blur.click()
    assert not settings(window).backdrop_blur.isChecked()
    guide.interface_language.setCurrentIndex(guide.interface_language.findData("pt-PT"))
    from dantetl.i18n import tr

    assert settings(window).interface_language.currentData() == "pt-PT"
    assert guide.isVisible() and guide.pages.currentIndex() == 0
    assert guide.heading.text() == tr("Welcome to DanteTL") != "Welcome to DanteTL"
    assert guide.next.text() == tr("Next") != "Next"
    assert (
        guide.sources.tabText(guide.sources.indexOf(guide.cloud_tab))
        == tr("Cloud models")
        != "Cloud models"
    )
    refresh = window.refresh_button
    assert not refresh.text() and not refresh.icon().pixmap(16).isNull()
    assert refresh.toolTip() == refresh.accessibleName() == tr("Refresh") != "Refresh"
    assert settings(window).tabs.tabText(0) == tr("Appearance") != "Appearance"
    guide.skip.click()
    drain(window, lambda: application.preferences.idle)
    assert PreferenceStore(path).initial.setup_seen
    saved = PreferenceStore(path).initial
    assert (saved.interface_language, saved.source_language, saved.output_language) == (
        "pt-PT",
        "ja",
        "en",
    )
    assert not saved.backdrop_blur
    window.offer_first_run()
    assert not guide.isVisible()
    settings(window).setup_assistant.click()
    assert window.findChild(SetupAssistant)
    guide = next(g for g in window.findChildren(SetupAssistant) if g.isVisible())
    guide.close()
    picker = settings(window).interface_language
    picker.setCurrentIndex(picker.findData("ar"))
    assert refresh.toolTip() == refresh.accessibleName() == tr("Refresh") != "Refresh"
    picker.setCurrentIndex(picker.findData("en"))
    assert refresh.toolTip() == refresh.accessibleName() == "Refresh"
    assert settings(window).tabs.tabText(0) == "Appearance"
    window.close()
    window.deleteLater()


def test_model_summary_separates_ratings_capabilities_and_unrated_imports(app):
    from dantetl.ui.model_summary import ModelSummary

    summary = ModelSummary()
    summary.set_model("plamo-2b", 1_800_000_000, True)
    assert summary.values[1].text() == "~ Mixed" and summary.values[1].property("tone") == "warning"
    assert "Heart Rate" in summary.values[1].toolTip()
    assert "support" in summary.cpu.text() and "support" in summary.gpu.text()
    summary.set_model("qwen35-4b", 2_700_000_000, True)
    assert summary.values[2].property("tone") == "error"
    summary.set_model("gemma-e4b", 1_100_000_000, False)
    assert "CPU only" in summary.gpu.text() and "No context" in summary.context.text()
    summary.set_model("custom", 2_000_000_000, True, imported=True)
    assert summary.ratings.isHidden() and "Not rated" in summary.size_label.text()
    assert "Local file" in summary.size_label.text()
    summary.deleteLater()


def test_processing_panel_distinguishes_cached_work_and_resets_stale_capture_metrics(app):
    from dantetl.domain import CaptureSnapshot, CaptureState
    from dantetl.ocr.values import OcrSnapshot, OcrState
    from dantetl.ocr.visual_values import VisualState
    from dantetl.ui.processing_panel import ProcessingPanel

    panel = ProcessingPanel()
    panel.show()
    bounds = Rect(0, 0, 100, 100)
    geometry = WindowGeometry(bounds, bounds, bounds, 144)
    frame = CapturedFrame(8, 1, geometry, b"")
    result = OcrResult(frame, (), 20, 1.02, (Rect(0, 0, 50, 50),))
    ocr = OcrSnapshot(OcrState.READY, result, queue_ms=35, provider="CPUExecutionProvider")
    translation = TranslationSnapshot(
        TranslationState.READY,
        cache_hits=1,
        last_batch_ms=0.1,
        last_inference_ms=92,
        cancelled_obsolete=2,
    )
    panel.refresh(
        CaptureSnapshot(CaptureState.RUNNING, frame),
        ocr,
        translation,
        VisualState(avoided=10),
        now=1,
        started_at=0,
        capture_rate=8,
        model="Gemma",
    )
    assert panel.ocr.values["Scan coverage"].text() == "1 area · 25%"
    assert panel.ocr.values["Device"].text() == "CPU"
    assert panel.translation.values["Last batch"].text() == "1 cached / 0 model"
    assert panel.translation.values["Model call"].text() == "92 ms"
    assert panel.translation.values["Cancelled"].text() == "2"
    # Idle cache hits must not be presented as zero-time model inference.
    assert "Most recent inference" in panel.translation.values["Model call"].toolTip()
    panel.hide()
    panel.refresh(
        None, None, None, VisualState(), now=2, started_at=0, capture_rate=8, model="Gemma"
    )
    assert panel.ocr.values["Device"].text() == "CPU"  # Hidden diagnostics do no formatting.
    panel.show()
    panel.refresh(
        None, None, None, VisualState(), now=3, started_at=0, capture_rate=8, model="Gemma"
    )
    assert panel.ocr.values["Device"].text() == "—"
    assert panel.capture.values["Image"].text() == "—"
    assert panel.translation.values["Last batch"].text() == "—"
    panel.close()
    panel.deleteLater()


def test_model_categories_share_one_worker_and_ocr_never_enters_translation_picker(
    app, tmp_path, monkeypatch
):
    from ui_fixtures import model_manager as ModelManager

    from dantetl.runtime.model_store import ModelBundle
    from dantetl.runtime.setup_values import Cancelled

    ocr = ModelBundle("ocr", "MeikiOCR", tmp_path / "ocr", (), "Reads text", category="recognition")
    gemma = ModelBundle(
        "gemma", "Gemma E2B · Recommended", tmp_path / "gemma", (), "Translates text"
    )
    other = replace(gemma, key="gemma-e4b", name="E4B fixture", directory=tmp_path / "gemma-e4b")
    entered, calls = threading.Event(), []

    def install(bundle, report, cancel):
        calls.append(bundle.key)
        entered.set()
        assert cancel.wait(2)
        raise Cancelled()

    monkeypatch.setattr("dantetl.runtime.model_store.install_model", install)
    manager = ModelManager((ocr, other, gemma))
    assert manager.choice.currentData().key == gemma.key
    assert [manager.choice.itemData(i).key for i in range(manager.choice.count())] == [
        "gemma-e4b",
        "gemma",
    ]
    assert manager.ocr.choice.currentData().key == ocr.key
    assert "Reads text" in manager.ocr.summary.text()
    manager.choice.setCurrentIndex(0)
    manager.refresh()
    # Browsing is not reset to the recommendation.
    assert manager.choice.currentData().key == other.key
    translation_status = manager.status.text()
    manager.category.setCurrentIndex(1)
    assert manager.pages.currentWidget() is manager.ocr_page
    assert not manager.busy
    manager.ocr.download.click()
    assert entered.wait(1) and manager.busy
    operation = manager.service.job(SetupLane.MODELS)
    manager.category.setCurrentIndex(0)  # Navigation leaves work and selections intact.
    manager._download()  # A second section cannot start competing work.
    setup_owner(manager).advance()
    manager.refresh_view()
    assert calls == ["ocr"] and manager.service.job(SetupLane.MODELS).token == operation.token
    assert not manager.choice.isEnabled() and not manager.ocr.download.isEnabled()
    assert manager.status.text() == translation_status
    manager.close()  # Closing still drains the same operation owner.
    settle_setup(manager.service)
    setup_owner(manager).advance()
    manager.refresh_view()
    assert not manager.busy and manager.choice.currentData().key == other.key
    assert manager.ocr.download.isEnabled()
    manager.deleteLater()


@pytest.mark.parametrize(
    "saved,prepared,expected",
    [
        ("auto", True, "gemma"),
        ("auto", False, "gemma-e4b"),
        ("gemma-e4b", True, "gemma-e4b"),
        ("none", True, "none"),
    ],
)
def test_recommended_model_preserves_saved_choice_and_preparation_consent(
    app, tmp_path, monkeypatch, saved, prepared, expected
):
    import json

    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key or prepared)
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({"version": 1, "model": saved}), encoding="utf-8")
    windows, factory = Mock(), Mock()
    windows.list_targets.return_value = []
    choices = (
        BackendChoice(
            "Gemma E2B · Recommended",
            factory,
            "gemma-cpu",
            preparation_id="pending",
            selection_id="gemma",
        ),
        BackendChoice("E4B fixture", factory, "gemma-e4b"),
    )
    window = MainWindow(windows, Mock(), choices, preferences_path=path)
    selected = window.backend_combo.currentData()
    assert (selected.preference_key if selected else "none") == expected
    factory.assert_not_called()
    window.close()
    window.deleteLater()


@pytest.mark.parametrize("model", ["gemma", "gemma-e4b", "qwen35-4b", "plamo-2b", "plamo-0.8b"])
def test_included_ocr_has_ready_badge_and_is_not_in_first_run_download_plan(app, tmp_path, model):
    import hashlib

    from dantetl.runtime.assets import ModelAsset
    from dantetl.runtime.hardware_values import HardwareStatus
    from dantetl.runtime.model_store import ModelBundle, model_catalog

    bundled = tmp_path / "app"
    bundled.mkdir()
    (bundled / "ocr").write_bytes(b"good")
    asset = ModelAsset("fixture", "pinned", "ocr", hashlib.sha256(b"good").hexdigest(), 4)
    ocr = ModelBundle(
        "ocr", "MeikiOCR", tmp_path / "AppData", (asset,), "", bundled, category="recognition"
    )
    models = (ocr, *(b for b in model_catalog() if b.category == "translation"))
    windows, hardware = Mock(), Mock()
    windows.list_targets.return_value = []
    hardware.snapshot.return_value = HardwareStatus(llama_device="cpu")
    window = MainWindow(windows, Mock(), models=models, hardware=hardware)
    application = app_owner(window)
    settings(window).setup_assistant.click()
    guide = window.findChild(SetupAssistant)
    assert guide and application.inventory.snapshot().data.ocr_ready
    guide.model.setCurrentIndex(guide.model.findData(model))
    assert "Included with app" in settings(window).models.ocr.badge.text()
    assert settings(window).models.ocr.download.text() == "Verify / repair OCR files"
    setup_owner(guide).advance()
    guide.refresh_view()
    for _ in range(3):
        guide.next.click()
    assert guide._review and guide._review.model_keys == (model,)
    assert guide.model_help.text() == next(b.description for b in models if b.key == model)
    assert "included, no download" in guide.review.text()
    assert not guide.busy and not ocr.directory.exists()
    guide.close()
    window.close()
    window.deleteLater()


@pytest.mark.parametrize(
    ("key", "already_assigned"),
    [
        ("gemma", True),
        ("gemma", False),
        ("gemma-e4b", False),
        ("qwen35-4b", False),
        ("plamo-2b", False),
        ("plamo-0.8b", False),
    ],
)
def test_first_run_resumes_model_intent_after_restart_but_still_requires_consent(
    app, tmp_path, monkeypatch, key, already_assigned
):
    from dantetl.application.preference_values import Preferences
    from dantetl.application.preferences import PreferenceStore

    path = tmp_path / "preferences.json"
    PreferenceStore(path)._write(
        Preferences(
            model="imported-" + "f" * 64,
            setup_seen=True,
            setup_model=key,
            short_model="local:" + key if already_assigned else "legacy",
            long_model="local:" + key if already_assigned else "legacy",
        )
    )
    windows, factory = Mock(), Mock()
    windows.list_targets.return_value = []
    base = BackendChoice("Imported fixture", factory, "imported-" + "f" * 64)
    gemma = BackendChoice("Language model", factory, f"{key}-cuda", "new-device", selection_id=key)
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key)
    window = MainWindow(windows, Mock(), (base, gemma), preferences_path=path)
    application = app_owner(window)
    window.offer_first_run()
    answer_dialog(window, QMessageBox.StandardButton.No)
    expected = gemma if already_assigned else base
    assert window.backend_combo.currentData() == window.routing.resolve(model_reference(expected))
    factory.assert_not_called()
    drain(window, lambda: application.preferences.idle)
    saved = PreferenceStore(path).initial
    assert saved.setup_model == "" and saved.model == "imported-" + "f" * 64
    window.offer_first_run()
    assert not visible_messages(window)  # Decline is not re-prompted on the next startup.
    window.close()
    window.deleteLater()


def test_hotkey_settings_apply_conflict_disable_and_persistence(app, tmp_path):
    from dantetl.application.hotkeys import HotkeyAction, Hotkeys
    from dantetl.application.preferences import PreferenceStore

    windows = Mock()
    windows.list_targets.return_value = []
    hotkeys = Hotkeys(Mock())
    path = tmp_path / "preferences.json"
    path.write_text('{"version": 1, "original_hotkey": "Ctrl+Shift+F7"}')
    window = MainWindow(windows, Mock(), hotkeys=hotkeys, preferences_path=path)
    application = app_owner(window)
    panel = settings(window).hotkeys
    assert set(panel.rows) == {HotkeyAction.OVERLAY, HotkeyAction.CAPTURE}
    overlay, capture = panel.rows[HotkeyAction.OVERLAY], panel.rows[HotkeyAction.CAPTURE]
    overlay.modifiers.setCurrentText("Ctrl+Shift")
    overlay.key.setCurrentIndex(overlay.key.findData("F7"))
    # Drafts do not change the registered or persisted shortcut.
    settings(window).opacity.setValue(61)
    drain(window, lambda: application.preferences.idle)
    assert PreferenceStore(path).initial.overlay_hotkey == "Ctrl+Alt+F8"
    overlay.apply.click()
    assert hotkeys.shortcut(HotkeyAction.OVERLAY) == "Ctrl+Shift+F7"
    capture.modifiers.setCurrentText("Ctrl+Shift")
    capture.key.setCurrentIndex(capture.key.findData("F7"))
    capture.apply.click()
    assert "other action" in capture.status.label.text()
    assert capture.committed == "Ctrl+Alt+F9"
    assert "Still active" in capture.status.label.text()
    capture.key.setCurrentIndex(0)
    capture.apply.click()
    drain(window, lambda: application.preferences.idle)
    saved = PreferenceStore(path).initial
    assert saved.overlay_hotkey == "Ctrl+Shift+F7" and saved.capture_hotkey == ""
    window.close()
    drain(window, lambda: not window.driver.timer.isActive())
    assert not hotkeys.shortcut(HotkeyAction.OVERLAY)
    window.deleteLater()
    restored = MainWindow(windows, Mock(), hotkeys=hotkeys, preferences_path=path)
    assert hotkeys.shortcut(HotkeyAction.OVERLAY) == "Ctrl+Shift+F7"
    assert not hotkeys.shortcut(HotkeyAction.CAPTURE)
    restored.close()
    restored.deleteLater()


def test_startup_shortcut_conflict_is_visible_and_retryable(app, tmp_path):
    from dantetl.application.hotkeys import HotkeyAction, Hotkeys

    native = Mock()
    native.register.side_effect = OSError("Another app uses this shortcut")
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), hotkeys=Hotkeys(native), cache_path=tmp_path / "cache")
    assert "unavailable" in window.hotkey_notice.label.text()
    row = settings(window).hotkeys.rows[HotkeyAction.OVERLAY]
    assert "Another app" in row.status.label.text()
    assert row.committed == "Ctrl+Alt+F8"
    native.register.side_effect = None
    for row in settings(window).hotkeys.rows.values():
        row.apply.click()
    assert not settings(window).hotkeys.problem
    assert window.hotkey_notice.isHidden()
    window.close()
    window.deleteLater()


def test_hotkeys_keep_selected_window_and_respect_stop_close_and_modal_guards(
    app, tmp_path, monkeypatch
):
    from live_fixtures import frame
    from PySide6.QtWidgets import QDialog

    from dantetl.application.hotkeys import HotkeyAction, Hotkeys
    from dantetl.domain import CaptureSnapshot, CaptureState, WindowTarget

    target = WindowTarget(101, 202, "Our fixture")
    windows = Mock()
    windows.list_targets.return_value = [target]
    windows.read_process_details.return_value = None
    capture = Mock()
    capture.return_value.finished = False
    capture.return_value.snapshot.return_value = CaptureSnapshot(CaptureState.RUNNING, frame(1))
    windows.geometry.return_value = frame(1).geometry
    windows.is_foreground.return_value = True
    monkeypatch.setattr("dantetl.composition.CaptureSession", capture)
    hotkeys = Hotkeys(Mock())
    window = MainWindow(windows, Mock(), hotkeys=hotkeys, cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.driver.timer.stop()
    set_ocr_available(window, False)
    settings(window).hotkeys.capture_focused.setChecked(False)
    hotkeys.activated(HotkeyAction.CAPTURE)
    capture.assert_called_once_with(
        target, windows, fps=8, on_change=window.driver.notify, show_border=True
    )
    window.driver.refresh(inspect=True)
    active = application.live.capture
    before = application.view.reset
    hotkeys.activated(HotkeyAction.OVERLAY)
    assert not window.overlay_check.isChecked() and application.live.capture is active
    assert application.view.reset == before
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert not window.start_button.isEnabled()
    capture.return_value.request_stop.assert_called_once()
    hotkeys.activated(HotkeyAction.CAPTURE)
    capture.return_value.request_stop.assert_called_once()
    application.live.capture = None
    window.start_button.setEnabled(True)
    windows.is_target.return_value = False
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert "closed" in window.status.text()
    assert capture.call_count == 1  # Never substitute a foreground or recycled HWND.
    dialog = QDialog(window)
    dialog.setModal(True)
    dialog.show()
    hotkeys.activated(HotkeyAction.OVERLAY)
    assert not window.overlay_check.isChecked()
    dialog.close()
    application.shutdown.request()
    hotkeys.activated(HotkeyAction.OVERLAY)
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert not window.overlay_check.isChecked() and capture.call_count == 1
    window.close()
    window.deleteLater()


def test_native_hotkey_filter_accepts_both_qt_routes_and_rejects_stale_messages(app):
    import ctypes
    from ctypes import wintypes

    from dantetl.application.hotkeys import HotkeyAction, Hotkeys
    from dantetl.ui.hotkeys_windows import WindowsHotkeyEvents

    registrar = Mock()
    hotkeys = Hotkeys(registrar)
    hotkeys.activated = Mock()
    hotkeys.bind(HotkeyAction.OVERLAY, "Ctrl+Alt+F8")
    identifier = registrar.register.call_args.args[0]
    events = WindowsHotkeyEvents(app, hotkeys)
    message = wintypes.MSG(message=0x0312, wParam=identifier, lParam=(0x77 << 16) | 3)
    pointer = ctypes.addressof(message)
    assert events.nativeEventFilter(b"windows_dispatcher_MSG", pointer) == (True, 0)
    assert events.nativeEventFilter(b"windows_generic_MSG", pointer) == (True, 0)
    assert hotkeys.activated.call_count == 2
    hotkeys.release()
    assert events.nativeEventFilter(b"windows_generic_MSG", pointer) == (False, 0)
    assert hotkeys.activated.call_count == 2
    events.close()


def test_game_window_icons_follow_refreshed_rows_while_old_lookup_is_busy(app):
    from PySide6.QtGui import QColor, QImage

    from dantetl.domain import WindowTarget

    original = WindowTarget(101, 202, "Game")
    other_process = replace(original, process_id=303)
    available = WindowTarget(102, 404, "Second game")
    started, release = threading.Event(), threading.Event()

    def read(target, *, copy):
        if target == other_process:
            raise PermissionError("Process details denied")
        if target == original:
            started.set()
            assert release.wait(5)
        image = QImage(32, 32, QImage.Format.Format_ARGB32)
        image.fill(QColor("red" if target == original else "blue"))
        return "a" * 64, image

    windows = Mock()
    windows.list_targets.return_value = [original]
    windows.read_process_details.side_effect = read
    window = MainWindow(windows, Mock())
    combo = window.windows_combo
    try:
        assert started.wait(2)
        windows.list_targets.return_value = [other_process, available]
        window.refresh_button.click()
        window.refresh_button.click()
        combo.setCurrentIndex(0)
        assert window.start_button.isEnabled() and combo.currentData() == other_process
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while combo.itemIcon(1).isNull():
        assert time.monotonic() < deadline
        app.processEvents()
        time.sleep(0.001)
    assert combo.itemIcon(0).isNull() and combo.currentData() == other_process
    assert combo.itemIcon(1).pixmap(16).toImage().pixelColor(8, 8) == QColor("blue")
    assert windows.read_process_details.call_count == 3
    window.close()


@pytest.mark.parametrize("failure", ["worker", "heading"])
def test_plain_window_list_preserves_capture_target_and_recovers(
    app, tmp_path, monkeypatch, failure
):
    import json

    from dantetl.domain import WindowTarget
    from dantetl.ui import game_window_combo

    first = WindowTarget(11, 42, "Game")
    second = replace(first, hwnd=12)
    other = WindowTarget(20, 24, "Another window")
    windows = Mock()
    windows.list_targets.return_value = [first, second, other]
    windows.read_process_details.side_effect = lambda t, **_: (
        ("a" if t.process_id == first.process_id else "b") * 64,
        None,
    )
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"version": 1, "favorite_executables": ["a" * 64]}))
    window = MainWindow(windows, Mock(), preferences_path=path)
    combo = window.windows_combo

    def wait_for(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline
            app.processEvents()
            time.sleep(0.001)

    wait_for(lambda: combo.count() == 5)
    assert combo.currentData() == first and window.favorite_button.isChecked()
    combo.setCurrentIndex(2)
    window.refresh_button.click()
    window.refresh_button.click()  # The previous refresh may still be resolving this favorite.
    wait_for(lambda: combo.count() == 5)
    assert combo.currentData() == second
    heading_failed = threading.Event()
    with monkeypatch.context() as patch:
        if failure == "worker":
            pool = Mock()
            pool.globalInstance.return_value.start.side_effect = RuntimeError("Pool unavailable")
            patch.setattr(game_window_combo, "QThreadPool", pool)
        else:
            add_section = game_window_combo.add_combo_section

            def fail_second_heading(combo, label):
                if label == "Other windows":
                    heading_failed.set()
                    raise RuntimeError("Heading unavailable")
                add_section(combo, label)

            patch.setattr(game_window_combo, "add_combo_section", fail_second_heading)
        window.refresh_button.click()
        if failure == "heading":
            wait_for(heading_failed.is_set)
        assert [combo.itemData(i) for i in range(combo.count())] == [other, first, second]
        assert "HWND" in combo.itemText(1) and "HWND" in combo.itemText(2)
        assert combo.currentData() == second and not window.favorite_button.isEnabled()
        start = Mock(return_value=False)
        patch.setattr(app_owner(window), "start_capture", start)
        window.start_button.click()
        assert start.call_args.args[0] == second
    window.refresh_button.click()
    wait_for(lambda: combo.count() == 5)
    assert combo.currentData() == second and window.favorite_button.isChecked()
    window.close()


@pytest.mark.parametrize("action", ["choose", "start"])
@pytest.mark.parametrize("manual_refresh", [False, True])
def test_late_favorite_keeps_window_chosen_or_used_for_capture(
    app, tmp_path, monkeypatch, action, manual_refresh
):
    import json

    from dantetl.domain import WindowTarget

    fallback = WindowTarget(1, 10, "A")
    favorite = WindowTarget(2, 20, "B")
    manual = WindowTarget(3, 30, "C")
    release = threading.Event()

    def details(target, **_):
        if target == favorite:
            assert release.wait(3)
            return "a" * 64, None
        return None

    windows = Mock()
    windows.list_targets.return_value = (
        [fallback] if manual_refresh else [manual, favorite, fallback]
    )
    windows.read_process_details.side_effect = details
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"version": 1, "favorite_executables": ["a" * 64]}))
    window = MainWindow(windows, Mock(), preferences_path=path)
    combo = window.windows_combo
    try:
        if manual_refresh:
            windows.list_targets.return_value = [manual, favorite, fallback]
            window.refresh_button.click()
        assert combo.currentData() == fallback
        if action == "choose":
            combo.setCurrentIndex(2)
            assert combo.currentData() == manual
        else:
            start = Mock(return_value=True)
            monkeypatch.setattr(app_owner(window), "start_capture", start)
            window.start_button.click()
            assert start.call_args.args[0] == fallback
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while combo.count() != 5:
        assert time.monotonic() < deadline
        app.processEvents()
        time.sleep(0.001)
    assert combo.itemData(1) == favorite
    assert combo.currentData() == (manual if action == "choose" else fallback)
    window.close()


def test_favorite_windows_group_by_executable_and_return_after_restart(app, tmp_path):
    from PySide6.QtCore import Qt

    from dantetl.application.preferences import PreferenceStore
    from dantetl.domain import WindowTarget

    def until(condition):
        # Includes the background preferences writer reaching disk on a loaded machine.
        deadline = time.monotonic() + 10
        while not condition():
            assert time.monotonic() < deadline
            app.processEvents()
            time.sleep(0.001)

    settings_path = tmp_path / "preferences.json"
    other = WindowTarget(1, 10, "Another game")
    game = WindowTarget(2, 20, "Game")
    second = replace(game, hwnd=3)
    windows = Mock()
    windows.list_targets.return_value = [other, game, second]
    windows.read_process_details.side_effect = lambda t, **_: (
        ("b" if t == other else "a") * 64,
        None,
    )
    window = MainWindow(windows, Mock(), preferences_path=settings_path)
    combo, star = window.windows_combo, window.favorite_button
    combo.setCurrentIndex(2)
    until(star.isEnabled)
    assert not star.isChecked() and star.accessibleName() == "Add to favorites"
    star.click()
    assert star.isChecked() and combo.currentData() == second
    assert [combo.itemData(i) for i in range(combo.count())] == [None, game, second, None, other]
    assert "HWND" in combo.itemText(1) and "HWND" in combo.itemText(2)
    assert [combo.itemText(i) for i in (0, 3)] == ["Favorites", "Other windows"]
    assert all(
        combo.model().flags(combo.model().index(i, 0)) == Qt.ItemFlag.NoItemFlags for i in (0, 3)
    )
    combo.setCurrentIndex(4)
    assert not star.isChecked()
    until(lambda: PreferenceStore(settings_path).initial.favorite_executables == ("a" * 64,))
    window.close()

    windows.list_targets.return_value = [other]
    reopened = MainWindow(windows, Mock(), preferences_path=settings_path)
    combo, star = reopened.windows_combo, reopened.favorite_button
    assert combo.count() == 1 and combo.currentData() == other
    relaunched = WindowTarget(4, 30, "New title")
    windows.list_targets.return_value = [other, relaunched]
    combo.showPopup()
    until(lambda: combo.count() == 4)
    combo.hidePopup()
    assert combo.itemData(1) == relaunched and combo.currentData() == other
    reopened.refresh_button.click()
    until(lambda: combo.count() == 4)
    assert combo.currentData() == relaunched
    assert star.isChecked() and star.accessibleName() == "Remove from favorites"
    star.click()
    assert not star.isChecked() and combo.count() == 2 and combo.currentData() == relaunched
    until(lambda: PreferenceStore(settings_path).initial.favorite_executables == ())
    reopened.close()


@pytest.mark.parametrize("reused_handle", [False, True])
def test_window_refresh_preserves_only_the_selected_process_and_handle(app, reused_handle):
    from dantetl.domain import WindowTarget

    selected = WindowTarget(101, 202, "Original")
    other = WindowTarget(102, 202, "Other")
    windows = Mock()
    windows.list_targets.return_value = [selected, other]
    windows.read_process_details.return_value = None
    window = MainWindow(windows, Mock())
    renamed = replace(selected, title="Renamed")
    windows.list_targets.return_value = [other, renamed]
    window.refresh_button.click()
    assert window.windows_combo.currentData() == renamed
    windows.list_targets.return_value = [other]
    if reused_handle:
        windows.list_targets.return_value.append(replace(selected, process_id=303))
    for _ in range(2):
        window.refresh_button.click()
        window.driver.refresh(inspect=True)
        assert window.windows_combo.currentIndex() == -1
        assert not window.start_button.isEnabled()
        window.start_button.click()
        assert not window.application.controls.capture_active
    window.windows_combo.setCurrentIndex(0)
    assert window.start_button.isEnabled()
    window.close()


def test_capture_hotkey_picks_focused_window_but_never_retargets_running_capture(
    app, tmp_path, monkeypatch
):
    from dantetl.application.hotkeys import HotkeyAction, Hotkeys
    from dantetl.capture.windows import WindowUnavailable
    from dantetl.domain import WindowTarget

    target = WindowTarget(101, 202, "New focused game")
    windows = Mock()
    windows.list_targets.return_value = []  # The initial list can be empty or stale.
    windows.read_process_details.return_value = None
    windows.foreground_target.return_value = target
    capture = Mock()
    monkeypatch.setattr("dantetl.composition.CaptureSession", capture)
    hotkeys = Hotkeys(Mock())
    window = MainWindow(windows, Mock(), hotkeys=hotkeys, cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.driver.timer.stop()
    set_ocr_available(window, False)
    assert not window.start_button.isEnabled()
    hotkeys.activated(HotkeyAction.CAPTURE)
    capture.assert_called_once_with(
        target, windows, fps=8, on_change=window.driver.notify, show_border=True
    )
    assert window.windows_combo.currentData() == target
    windows.foreground_target.side_effect = WindowUnavailable("Focus the game window")
    hotkeys.activated(HotkeyAction.CAPTURE)
    capture.return_value.request_stop.assert_called_once()
    windows.foreground_target.assert_called_once()
    application.live.capture = None
    window.start_button.setEnabled(True)
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert "Focus the game" in window.status.text()
    assert capture.call_count == 1 and window.windows_combo.currentData() == target
    windows.foreground_target.side_effect = None
    windows.is_foreground.return_value = False
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert "Focus changed" in window.status.text() and capture.call_count == 1
    windows.is_foreground.return_value = True
    renamed = replace(target, title="Game with a new title")
    windows.foreground_target.return_value = renamed
    hotkeys.activated(HotkeyAction.CAPTURE)
    assert window.windows_combo.count() == 1 and window.windows_combo.currentData() == renamed
    assert capture.call_count == 2
    application.live.capture = None
    window.close()
    window.deleteLater()


def cell(panel: OcrInspector, row: int, column: int):
    item = panel.table.item(row, column)
    assert item is not None
    return item


def drain(window, predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        window.driver.refresh(inspect=True)
        time.sleep(0.005)


def test_no_models_do_not_start_inference_and_refresh_never_prepares_implicitly(
    app, tmp_path, monkeypatch
):
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock()
    choice = BackendChoice("Gemma", factory, "gemma", "unprepared")
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key)
    window = MainWindow(
        windows, factory, cache_path=tmp_path / "cache", backend_provider=lambda: (choice,)
    )
    assert window.backend_combo.currentData() is None
    assert window.translate_check.isEnabled() and window.models_button.isEnabled()
    drain(window, lambda: not settings(window).busy)
    window.driver.refresh(inspect=True)
    assert "No translation model is downloaded" in window.notice.text()
    assert window.onboarding.property("tone") == "warning"
    window.driver.refresh(inspect=True)
    window.application.refresh_models()
    assert window.backend_combo.currentData() is None
    assert window.translate_check.isEnabled()
    factory.assert_not_called()
    window.close()
    window.deleteLater()


def test_downloaded_model_without_runtime_stays_visible_and_explains_setup(
    app, tmp_path, monkeypatch
):
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock()
    choice = BackendChoice(
        "Gemma · runtime setup needed",
        factory,
        "gemma-unavailable",
        unavailable_reason="Open Runtime setup.",
    )
    setup = Mock()
    monkeypatch.setattr("dantetl.ui.main_window.MainWindow._manage_runtime", setup)
    window = MainWindow(windows, factory, (choice,), scene_feed=True, cache_path=tmp_path / "cache")
    application = app_owner(window)
    assert window.backend_combo.isEnabled() and window.translate_check.isEnabled()
    assert window.backend_combo.currentData() is None  # Never implicitly load an unusable model.
    show_scene(window)
    assert "runtime setup" in window.translation_status.text()
    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, choice))
    setup.assert_called_once()
    assert (
        window.backend_combo.currentData() is None
        and application.translation.lifecycle.publication is None
    )
    factory.assert_not_called()
    window.close()
    window.deleteLater()


@pytest.mark.parametrize("key", ["gemma", "gemma-e4b", "qwen35-4b", "plamo-2b", "plamo-0.8b"])
def test_download_completion_reports_missing_runtime_instead_of_false_readiness(
    app, tmp_path, monkeypatch, key
):
    import hashlib

    from download_fixtures import download_bytes
    from ui_fixtures import model_manager as ModelManager

    from dantetl.runtime.assets import ModelAsset
    from dantetl.runtime.model_store import ModelBundle

    asset = ModelAsset("fixture", "revision", "weights", hashlib.sha256(b"good").hexdigest(), 4)
    bundle = ModelBundle(key, key, tmp_path, (asset,), "", language_model=True)
    monkeypatch.setattr("dantetl.runtime.model_store.download_file", download_bytes(b"good"))
    monkeypatch.setattr("dantetl.runtime.llama.installed_device", lambda: None)
    manager = ModelManager((bundle,))
    manager.download.click()
    assert manager.busy
    settle_setup(manager.service)
    setup_owner(manager).advance()
    manager.refresh_view()
    assert not manager.busy and bundle.present()
    assert "not installed" in manager.status.text() and "Acceleration" in manager.status.text()
    assert manager.download.text() == "Verify / repair"
    assert manager.progress.value() == manager.progress.maximum()
    assert manager.download_badge.text() == "✓  Downloaded"
    assert manager.download_badge.property("tone") == "success"
    assert not manager.runtime_badge.isHidden()
    manager.close()
    manager.deleteLater()


@pytest.mark.parametrize("update,detected", [(True, False), (True, True), (False, False)])
def test_cuda_update_choice_starts_install_only_on_request(
    app, tmp_path, monkeypatch, update, detected
):
    from dantetl.runtime.hardware_values import Adapter, CudaDevice

    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters",
        lambda: (Adapter("RTX", 0x10DE, cuda=CudaDevice(13000, 8, 6, "a" * 32)),),
    )
    install = Mock(side_effect=RuntimeError("Fixture install failed"))
    monkeypatch.setattr("dantetl.runtime.setup.install_runtime", install)
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), cache_path=tmp_path / "cache")
    try:
        window.show()
        if detected:
            settings(window).show_runtime()
            panel = settings(window).runtime
            assert panel is not None
            settle_setup(panel.service)
            panel.refresh_view()
            settings(window).hide()
            install.assert_not_called()
        window.offer_cuda_update()
        prompt = window.findChild(QMessageBox)
        assert prompt and prompt.isVisible()
        buttons = {button.text(): button for button in prompt.buttons()}
        assert set(buttons) == {"Update CUDA", "Continue on CPU"}
        buttons["Update CUDA" if update else "Continue on CPU"].click()
        assert not prompt.isVisible()
        assert settings(window).isVisible() == update
        if update:
            panel = settings(window).runtime
            assert panel and panel.isVisible()
            settle_setup(panel.service)
            panel.refresh_view()
            settle_setup(panel.service)
            panel.refresh_view()
            panel.refresh_view()
            install.assert_called_once()
            assert install.call_args.args[1].key == panel.choice.currentData().key == "nvidia"
            assert panel.error_message == "Fixture install failed" and panel.install.isEnabled()
        else:
            assert settings(window).runtime is None
            install.assert_not_called()
    finally:
        window.close()
        window.deleteLater()


def test_runtime_setup_detects_without_installing_and_close_waits_for_cancel(
    app, tmp_path, monkeypatch
):
    from dantetl.runtime.hardware_values import Adapter, CudaDevice
    from dantetl.runtime.setup_values import Cancelled

    calls = []
    entered = threading.Event()
    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters",
        lambda: (Adapter("RTX", 0x10DE, 16 * 2**30, cuda=CudaDevice(13000, 8, 6, "a" * 32)),),
    )

    def install(root, profile, progress, cancel, gpu=""):
        calls.append(threading.get_ident())
        entered.set()
        assert cancel.wait(2)
        raise Cancelled()

    monkeypatch.setattr("dantetl.runtime.setup.install_runtime", install)
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), cache_path=tmp_path / "cache")
    application = app_owner(window)
    settings(window).show_runtime()
    dialog = settings(window).runtime
    assert dialog is not None
    assert dialog.busy
    settle_setup(dialog.service)
    setup_owner(dialog).advance()
    dialog.refresh_view()
    assert (
        "RTX" in dialog.hardware.accessibleDescription()
        and dialog.choice.currentData().key == "nvidia"
    )
    assert not calls and dialog.install.isEnabled() and not dialog.restart.isEnabled()
    dialog.install.click()
    assert entered.wait(1) and calls != [threading.get_ident()]
    assert not dialog.choice.isEnabled() and not dialog.restart.isEnabled()
    window.close()
    assert application.shutdown.active and dialog.busy
    settle_setup(dialog.service)
    setup_owner(dialog).advance()
    dialog.refresh_view()
    window.driver.refresh(inspect=True)
    assert not dialog.busy and not window.driver.timer.isActive()
    window.deleteLater()


def test_model_download_badges_and_missing_selection_warnings_are_distinct(
    app, tmp_path, monkeypatch
):
    from dantetl.runtime.assets import ModelAsset
    from dantetl.runtime.hardware_values import Adapter, HardwareStatus
    from dantetl.runtime.model_store import ModelBundle

    asset = ModelAsset("test", "revision", "weights", "", 4)
    bundle = ModelBundle("gemma", "Gemma", tmp_path, (asset,), "Fixture", language_model=True)
    windows = Mock()
    windows.list_targets.return_value = []
    hardware = Mock(finished=True)
    hardware.snapshot.return_value = HardwareStatus((Adapter("RTX fixture", 0x10DE),))
    monkeypatch.setattr("dantetl.runtime.llama.installed_device", lambda: None)
    window = MainWindow(windows, Mock(), models=(bundle,), hardware=hardware)
    manager = settings(window).models
    assert "Download required" in manager.download_badge.text()
    assert manager.download_badge.property("tone") == "warning"
    assert window.onboarding.property("tone") == "warning"
    assert window.acceleration_notice.isHidden()
    (tmp_path / "weights").write_bytes(b"good")
    settings(window).show_models()
    drain(window, lambda: manager.download_badge.text() == "✓  Downloaded")
    assert manager.download_badge.text() == "✓  Downloaded"
    assert manager.download_badge.property("tone") == "success"
    assert manager.choice.itemText(0).startswith("✓")
    assert not manager.runtime_badge.isHidden() and "Engine setup" in manager.runtime_badge.text()
    choice = BackendChoice("Gemma", Mock(), "gemma", unavailable_reason="Missing runtime")
    replace_backends(window, (choice,))
    window.driver.refresh(inspect=True)
    assert "No translation model is selected" in window.notice.text()
    assert window.models_button.text() == "Choose model"
    assert "selected" in settings(window).model_notice.label.text()
    window.close()
    assert not settings(window).isVisible()
    window.deleteLater()


def test_cpu_model_selection_is_usable_and_notice_does_not_request_setup(app, monkeypatch):
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock()
    choice = BackendChoice(
        "Gemma · CPU", factory, "gemma-cpu", performance_notice="CPU translation may be slower."
    )
    window = MainWindow(windows, Mock(), (choice,))
    application = app_owner(window)
    setup = Mock(side_effect=AssertionError("CPU selection must not require GPU installation"))
    monkeypatch.setattr(window, "_manage_runtime", setup)
    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, choice))
    window.backend_combo.activated.emit(window.backend_combo.currentIndex())
    window.driver.refresh(inspect=True)
    assert (
        application.translation.lifecycle.snapshot().selected == choice
        and window.translate_check.isEnabled()
    )
    assert window.onboarding.isHidden()
    setup.assert_not_called()
    factory.assert_not_called()  # No capture/worker until requested.
    window.close()
    window.deleteLater()


def test_switch_clears_painted_and_paused_english_rejects_late_reply_and_keeps_model_caches(
    app, tmp_path
):
    class Backend:
        batch_size = 1

        def __init__(self, name):
            self.cache_identity = name
            self.calls = []
            self.entered, self.release = threading.Event(), threading.Event()
            self.release.set()

        def translate(self, texts, *, target=ENGLISH):
            self.calls.append(texts)
            self.entered.set()
            assert self.release.wait(3)
            return tuple(f"{self.cache_identity}: {text}" for text in texts)

        def close(self):
            pass

    old, new, old_again = Backend("E4B fixture"), Backend("Gemma"), Backend("E4B fixture")
    windows = Mock()
    windows.list_targets.return_value = []
    choices = (
        BackendChoice("E4B fixture", lambda: old, "gemma-e4b"),
        BackendChoice("Gemma", lambda: new, "gemma"),
    )
    window = MainWindow(windows, Mock(), choices, scene_feed=True, cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.driver.timer.stop()
    settings(window).inspection_enabled.setChecked(True)
    window.advanced.setChecked(True)
    first = TextBlock((TextTrack(1, "猫", Rect(0, 0, 40, 20)),))
    second = TextBlock((TextTrack(2, "犬", Rect(0, 40, 40, 20)),))

    def poll_until(predicate, blocks):
        deadline = time.monotonic() + 3
        while True:
            show_scene(window, blocks)
            if predicate():
                return
            assert time.monotonic() < deadline
            time.sleep(0.005)

    try:
        poll_until(lambda: bool(application.view.english), (first,))
        bounds = Rect(0, 0, 80, 80)
        frame = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * 25600)
        result = OcrResult(frame, (TextRegion("猫", first.bounds),), 1, 1)
        window.inspector.set_result(result, (1,))
        window.inspector.set_translations(application.view.english)
        window.inspector.pause.setChecked(True)
        assert cell(window.inspector, 0, 3).text() == "E4B fixture: 猫"
        old.entered.clear()
        old.release.clear()
        show_scene(window, (first, second))
        assert old.entered.wait(1)
        poll_until(lambda: not (application.view.english or application.view.results), ())
        assert cell(window.inspector, 0, 3).text().startswith("E4B fixture")
        new.release.clear()
        window.backend_combo.setCurrentIndex(find_model(window.backend_combo, choices[1]))
        assert not application.view.english and not application.view.blocks
        assert cell(window.inspector, 0, 3).text() == "—"
        assert window.inspector.result is result and window.inspector.pause.isChecked()
        old.release.set()
        poll_until(new.entered.is_set, (first, second))
        show_scene(window, (first, second))
        assert not application.view.english
        new.release.set()
        poll_until(lambda: len(application.view.english) == 2, (first, second))
        assert all(t.english.startswith("Gemma:") for t in application.view.english)
        window.inspector.set_result(result, (1,))
        window.inspector.pause.setChecked(False)
        assert cell(window.inspector, 0, 3).text() == "Gemma: 猫"
        replace_backends(window, (replace(choices[0], create=lambda: old_again), choices[1]))
        window.routing.set_models(application.translation.options(), application.translation.plan)
        window.backend_combo.setCurrentIndex(
            find_model(window.backend_combo, application.translation.choices[0])
        )
        assert not application.view.english
        poll_until(lambda: len(application.view.english) == 2, (first, second))
        assert all(t.english.startswith("E4B fixture:") for t in application.view.english)
        assert old_again.calls == []  # Its own cache still works; Gemma never read those entries.
        assert new.calls == [("猫",), ("犬",)]
    finally:
        old.release.set()
        new.release.set()
        window.close()
        drain(
            window,
            lambda: (
                not application.shutdown.active
                or application.translation.lifecycle.publication is None
            ),
        )
        window.deleteLater()


def test_app_shutdown_drains_startup_hardware_check(app, tmp_path):
    windows = Mock()
    windows.list_targets.return_value = []
    hardware = Mock(finished=False)
    hardware.snapshot.return_value = None
    window = MainWindow(windows, Mock(), hardware=hardware, cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.close()
    assert application.shutdown.active and window.driver.timer.isActive()
    window.driver.refresh(inspect=True)
    assert window.driver.timer.isActive()
    hardware.finished = True
    drain(window, lambda: not window.driver.timer.isActive())
    assert not window.driver.timer.isActive()
    window.deleteLater()


def test_runtime_restart_waits_for_translation_and_recovers_if_launch_fails(
    app, tmp_path, monkeypatch
):
    from ownership_fixtures import Route

    from dantetl.application.hotkeys import HotkeyAction, Hotkeys

    windows = Mock()
    windows.list_targets.return_value = []
    native = Mock()
    hotkeys = Hotkeys(native)
    worker = Route()
    monkeypatch.setattr("dantetl.composition.RoutedTranslationSession", lambda *a, **k: worker)
    window = MainWindow(
        windows,
        Mock(),
        (BackendChoice("Fixture", Mock(), "fixture"),),
        scene_feed=True,
        cache_path=tmp_path / "cache",
        hotkeys=hotkeys,
    )
    application = app_owner(window)
    show_scene(window)
    assert application.translation.lifecycle.publication is worker

    def fail_restart(*args, **kwargs):
        assert not hotkeys.shortcut(HotkeyAction.OVERLAY)
        assert not hotkeys.shortcut(HotkeyAction.CAPTURE)
        raise OSError("Could not launch")

    restart = Mock(side_effect=fail_restart)
    monkeypatch.setattr("dantetl.composition.RuntimeRestart", restart)
    settings(window).restart_requested.emit(RuntimeTarget(tmp_path))
    assert application.shutdown.active
    assert worker.stops
    restart.assert_not_called()
    worker.finish()
    window.close()
    drain(window, lambda: not application.shutdown.active)
    restart.assert_called_once()
    assert not application.shutdown.active
    assert [d.text() for d in visible_messages(window)] == ["Could not launch"]
    answer_dialog(window, QMessageBox.StandardButton.Ok)
    assert hotkeys.shortcut(HotkeyAction.OVERLAY) == "Ctrl+Alt+F8"
    assert hotkeys.shortcut(HotkeyAction.CAPTURE) == "Ctrl+Alt+F9"
    central = window.centralWidget()
    assert central and central.isEnabled() and window.driver.timer.isActive()
    window.close()
    window.deleteLater()


def test_restart_cleanup_cannot_block_gui_or_report_finished_before_draining(
    app, tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    owners = []

    class Restart:
        def __init__(self, root, target, *, gpu, on_cleanup, cancel):
            owners.append(threading.get_ident())
            self.on_cleanup = on_cleanup

        def poll(self):
            self.on_cleanup()
            entered.set()
            assert release.wait(3)
            raise RuntimeError("fixture cleanup finished")

    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), cache_path=tmp_path / "cache")
    application = app_owner(window)
    monkeypatch.setattr("dantetl.composition.RuntimeRestart", Restart)
    try:
        settings(window).restart_requested.emit(RuntimeTarget(tmp_path))
        drain(window, entered.is_set)
        assert owners and owners[0] != threading.get_ident()
        for _ in range(20):
            window.driver.refresh(inspect=True)
            app.processEvents()
            assert application.shutdown.snapshot().phase == "restarting"
            assert not visible_messages(window) and application.shutdown.active
        assert "Stopping" in window.status.text()
        release.set()
        drain(window, lambda: not application.shutdown.active)
        assert [d.text() for d in visible_messages(window)] == ["fixture cleanup finished"]
        answer_dialog(window, QMessageBox.StandardButton.Ok)
    finally:
        release.set()
        window.close()
        drain(window, lambda: not window.driver.timer.isActive())
        window.deleteLater()


def test_preparation_decline_preserves_selection_and_accept_loads_once_off_ui(
    app, tmp_path, monkeypatch
):
    windows = Mock()
    windows.list_targets.return_value = []
    entered, release = threading.Event(), threading.Event()
    calls, receipts = [], []

    def load():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        backend = Mock(cache_identity="fake")
        backend.batch_size = 1
        return backend

    base = BackendChoice("E4B fixture", Mock(), "gemma-e4b")
    gemma = BackendChoice("Gemma", load, "gemma", "new")
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key)
    monkeypatch.setattr("dantetl.runtime.model_store.mark_prepared", receipts.append)
    window = MainWindow(windows, Mock(), (base, gemma), cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, gemma))
    assert window.backend_combo.currentData() == window.routing.resolve(model_reference(base))
    assert not entered.is_set() and not receipts
    answer_dialog(window, QMessageBox.StandardButton.No)
    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, gemma))
    answer_dialog(window, QMessageBox.StandardButton.Yes)
    # Draining the previous worker may take more than one GUI poll under build/IO load.
    drain(window, entered.is_set)
    progress = window.findChild(QProgressDialog)
    assert progress is not None and progress.maximum() == 0
    assert not receipts and application.translation.lifecycle.snapshot().proposed == gemma
    assert isinstance(application.translation.lifecycle.publication, TranslationSession)
    assert len(calls) == 1 and calls[0] != threading.get_ident()
    release.set()
    drain(window, lambda: not application.translation.lifecycle.snapshot().preparing)
    drain(window, lambda: receipts == ["new"])
    assert len(calls) == 1
    assert (
        window.backend_combo.currentData() == window.routing.resolve(model_reference(gemma))
        and application.translation.lifecycle.publication is not None
    )
    window.close()
    drain(window, lambda: not window.driver.timer.isActive())
    window.deleteLater()


def test_failed_preparation_restores_previous_selection_without_a_receipt(
    app, tmp_path, monkeypatch
):
    windows = Mock()
    windows.list_targets.return_value = []
    base = BackendChoice("E4B fixture", Mock(), "gemma-e4b")
    bad = BackendChoice("Gemma", Mock(side_effect=RuntimeError("GPU unavailable")), "gemma", "new")
    receipts = []
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key)
    monkeypatch.setattr("dantetl.runtime.model_store.mark_prepared", receipts.append)
    window = MainWindow(windows, Mock(), (base, bad), cache_path=tmp_path / "cache")
    application = app_owner(window)
    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, bad))
    answer_dialog(window, QMessageBox.StandardButton.Yes)
    drain(window, lambda: not application.translation.lifecycle.snapshot().preparing)
    assert window.backend_combo.currentData() == window.routing.resolve(model_reference(base))
    assert not receipts and [d.text() for d in visible_messages(window)] == ["GPU unavailable"]
    answer_dialog(window, QMessageBox.StandardButton.Ok)
    window.close()
    window.deleteLater()


def test_open_consent_dialog_allows_updates_and_keeps_previous_model(app, tmp_path, monkeypatch):
    from ownership_fixtures import Route

    windows = Mock()
    windows.list_targets.return_value = []
    base = BackendChoice("E4B fixture", Mock(), "gemma-e4b")
    gemma = BackendChoice("Gemma", Mock(), "gemma", "new")
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: not key)
    session = Route()
    monkeypatch.setattr("dantetl.composition.RoutedTranslationSession", lambda *a, **k: session)
    window = MainWindow(
        windows, Mock(), (base, gemma), scene_feed=True, cache_path=tmp_path / "cache"
    )
    application = app_owner(window)
    show_scene(window)
    assert application.translation.lifecycle.publication is session

    window.backend_combo.setCurrentIndex(find_model(window.backend_combo, gemma))
    show_scene(window)
    assert session.stops == 0
    answer_dialog(window, QMessageBox.StandardButton.No)
    assert (
        window.backend_combo.currentData() == window.routing.resolve(model_reference(base))
        and application.translation.lifecycle.publication == session
    )
    session.finish()
    window.close()
    window.deleteLater()


def test_main_window_destruction_releases_its_independent_overlay(app):
    from PySide6.QtCore import QCoreApplication, QEvent

    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock())
    overlay = window._overlay
    assert overlay.parentWidget() is None
    window.close()
    drain(window, lambda: not window.driver.timer.isActive())
    shiboken6.delete(window)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(overlay)


def test_closing_during_download_waits_for_cancellation_before_destroying_gui(
    app, tmp_path, monkeypatch
):
    from dantetl.runtime.model_store import ModelBundle
    from dantetl.runtime.setup_values import Cancelled, ModelProgress

    windows = Mock()
    windows.list_targets.return_value = []
    entered = threading.Event()
    cancellation_received = threading.Event()
    allow_exit = threading.Event()

    def install(bundle, progress, cancel):
        progress(ModelProgress("Downloading", 1, 10))
        entered.set()
        assert cancel.wait(2)
        cancellation_received.set()
        assert allow_exit.wait(5)
        raise Cancelled()

    monkeypatch.setattr("dantetl.runtime.model_store.install_model", install)
    bundle = ModelBundle("fake", "Fixture", tmp_path, (), "")
    window = MainWindow(windows, Mock(), models=(bundle,), cache_path=tmp_path / "cache")
    settings(window).show_models()
    manager = settings(window).models
    assert manager is not None
    manager.download.click()
    try:
        assert entered.wait(2)
        assert not window.close()
        assert cancellation_received.wait(2)
        window.driver.refresh(inspect=True)
        assert window.driver.timer.isActive() and manager.busy
    finally:
        allow_exit.set()
    drain(window, lambda: not window.driver.timer.isActive())
    assert not manager.busy
    window.deleteLater()


def test_preview_letterboxing_maps_physical_pixels_once() -> None:
    target = image_target((1920, 1080), (640, 480))
    assert target.getRect() == (0, 60, 640, 360)
    # Original point (960, 540) lands in the preview center, including the top margin.
    scale = target.width() / 1920
    assert (target.x() + 960 * scale, target.y() + 540 * scale) == (320, 240)


@pytest.mark.parametrize("from_settings", [False, True])
def test_idle_source_picker_switches_without_confirmation(app, monkeypatch, from_settings):
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock())
    owner = app_owner(window)
    primary = window.source_picker
    secondary = settings(window).source_picker
    window.show()
    panel = settings(window)
    if from_settings:
        panel.show_recognition()
    native_ids = (int(window.winId()), int(panel.winId()))
    selected_tab = panel.tabs.currentWidget()
    prompt = Mock(side_effect=AssertionError("Idle source changes need no confirmation"))
    monkeypatch.setattr(QMessageBox, "question", prompt)
    picker = secondary if from_settings else primary
    window.output_language.setCurrentIndex(window.output_language.findData("fr"))
    index = picker.findData("es")
    picker.setCurrentIndex(index)
    picker.activated.emit(index)
    drain(window, lambda: owner.can_change_source)
    assert window.isVisible() and panel.isVisible() == from_settings
    assert (int(window.winId()), int(panel.winId())) == native_ids
    assert panel.tabs.currentWidget() is selected_tab
    assert primary.currentData() == secondary.currentData() == "es"
    prompt.assert_not_called()
    assert (owner.values.source_language, owner.values.output_language) == ("es", "fr")
    assert not owner.controls.capture_active and not owner.closing
    window.close()
    drain(window, lambda: not window.driver.timer.isActive())
    window.deleteLater()


def test_output_language_controls_warn_before_capture_and_switch_only_the_affected_model(app):
    from dantetl.translation.backends.plamo import OUTPUT_CODES
    from dantetl.translation.plan import TranslationPlan

    factory = Mock()
    plamo = BackendChoice("PLaMo 2B", factory, "plamo-2b", output_codes=OUTPUT_CODES)
    gemma = BackendChoice("Gemma E2B", factory, "gemma")
    other = BackendChoice("Another model", factory, "other")
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), (plamo, gemma, other))
    panel = settings(window)
    owner = app_owner(window)
    try:
        window.show()
        window.backend_combo.setCurrentIndex(find_model(window.backend_combo, plamo))
        window.settings_button.click()
        assert not panel.findChildren(type(window.output_language))
        window.output_language.setCurrentIndex(window.output_language.findData("pt-BR"))
        assert window.output_language.currentData() == owner.values.output_language == "pt-BR"
        for banner in (window.language_notice, panel.language_notice):
            assert banner.isVisible() and "Portuguese (Brazil)" in banner.label.text()
            assert "PLaMo" in banner.label.text() and "Gemma" in banner.action.text()
        window.output_language.setCurrentIndex(window.output_language.findData("ja"))
        assert owner.values.output_language == "ja"
        assert panel.language_notice.isHidden()
        window.routing.advanced.setChecked(True)
        window.routing.short_model.setCurrentIndex(find_model(window.routing.short_model, other))
        window.output_language.setCurrentIndex(window.output_language.findData("pt-PT"))
        assert "long text and dialogue" in panel.language_notice.label.text()
        assert "short labels" not in panel.language_notice.label.text()
        window.routing.long_enabled.setChecked(False)
        assert panel.language_notice.isHidden() and window.language_notice.isHidden()
        window.routing.long_enabled.setChecked(True)
        panel.language_notice.action.click()
        assert owner.controls.assignment == TranslationPlan("local:other", "local:gemma")
        assert window.output_language.currentData() == "pt-PT"
        assert panel.language_notice.isHidden() and window.language_notice.isHidden()
        assert not owner.controls.capture_active
        factory.assert_not_called()
    finally:
        window.close()
        window.deleteLater()


def test_language_warning_opens_missing_gemma_files_without_downloading(app, tmp_path):
    from dantetl.runtime.model_store import ModelBundle

    factory = Mock()
    plamo = BackendChoice("PLaMo", factory, "plamo-2b", output_codes=frozenset({"en"}))
    bundles = tuple(
        ModelBundle(key, key, tmp_path / key, (), "", language_model=True)
        for key in ("gemma", "plamo-2b")
    )
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), (plamo,), models=bundles)
    panel = settings(window)
    try:
        window.backend_combo.setCurrentIndex(find_model(window.backend_combo, plamo))
        window.output_language.setCurrentIndex(window.output_language.findData("pt-BR"))
        panel.show_models()
        panel.models.choice.setCurrentIndex(1)
        panel.models.category.setCurrentIndex(panel.models.category.findData("cloud"))
        assert window.language_notice.action.text() == "Set up Gemma"
        window.language_notice.action.click()
        assert panel.models.isVisible() and panel.models.category.currentData() == "translation"
        assert panel.models.choice.currentData().key == "gemma"
        assert not panel.models.busy and not list(tmp_path.iterdir())
        assert window.routing.selection.active_models == ("local:plamo-2b",)
        factory.assert_not_called()
    finally:
        window.close()
        window.deleteLater()


def test_settings_edits_persist_and_restore_without_loading_a_model(app, tmp_path, monkeypatch):
    from dantetl.ocr.worker import OcrChoice

    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock()
    choices = (OcrChoice("Automatic", factory, key="auto"), OcrChoice("CPU", factory, key="cpu"))
    path = tmp_path / "preferences.json"
    window = MainWindow(windows, factory, ocr_backends=choices, preferences_path=path)
    application = app_owner(window)
    window.settings_button.click()
    original = settings(window)
    window.settings_button.click()
    assert settings(window) is original
    assert window.output_language.currentData() == "en"
    window.output_language.setCurrentIndex(window.output_language.findData("pt-PT"))
    assert application.values.output_language == application.cloud.test_target.code == "pt-PT"
    assert original.backdrop_blur.isChecked()
    assert original.tabs.currentWidget().isAncestorOf(original.backdrop_blur)
    original.backdrop_blur.click()
    original.inspection_enabled.click()
    window.stabilize_check.setChecked(False)
    settings(window).opacity.setValue(61)
    window.automatic_font.setChecked(False)
    window.font_size.setValue(26)
    window.group_dialogue.setChecked(False)
    window.ocr_device.setCurrentIndex(1)
    window.capture_rate.setValue(144)
    assert application.values.capture_rate == 144
    window.capture_rate.setValue(0)
    assert window.capture_rate.text() == "Unlimited"
    window.translate_check.setChecked(False)
    window.overlay_check.setChecked(False)
    deadline = time.monotonic() + 2
    while not application.preferences.idle:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    window.close()
    window.deleteLater()
    restored = MainWindow(windows, factory, ocr_backends=choices, preferences_path=path)
    assert not settings(restored).backdrop_blur.isChecked()
    assert settings(restored).inspection_enabled.isChecked() and not restored.advanced.isHidden()
    assert restored.output_language.currentData() == "pt-PT"
    assert app_owner(restored).cloud.test_target.code == "pt-PT"
    assert not restored.stabilize_check.isChecked() and not restored.group_dialogue.isChecked()
    assert restored._overlay._opacity == 0.61 and restored.font_size.value() == 26
    assert not restored.automatic_font.isChecked() and restored.font_size.isEnabled()
    assert restored.ocr_device.currentData().key == "cpu" and restored.capture_rate.value() == 0
    assert not restored.translate_check.isChecked() and not restored.overlay_check.isChecked()
    assert not restored.advanced.isChecked() and not restored.inspector.pause.isChecked()
    factory.assert_not_called()
    restored.close()
    restored.deleteLater()


def test_hidden_inspector_does_no_frame_copy_or_table_build_and_resume_uses_latest_pair(
    app, monkeypatch
):
    from dantetl.ui import inspection

    panel = OcrInspector()
    panel.set_active(False)
    copied = Mock(wraps=inspection.frame_image)
    monkeypatch.setattr(inspection, "frame_image", copied)
    bounds = Rect(0, 0, 4, 4)
    geometry = WindowGeometry(bounds, bounds, bounds, 96)
    latest = None
    for sequence, text in enumerate(("猫", "犬", "鳥"), 1):
        frame = CapturedFrame(sequence, sequence, geometry, b"\xff" * 64)
        latest = OcrResult(frame, (TextRegion(text, bounds),), 10, sequence)
        panel.set_result(latest, (7,))
        panel.set_translations((TranslatedText(TranslationRequest(7, 1, text), text + " English"),))
    copied.assert_not_called()
    assert panel.result is None and panel.table.rowCount() == 0
    panel.set_active(True)
    assert latest is not None
    assert copied.call_count == 1
    assert panel.result is latest and panel._display_frame is latest.frame
    assert cell(panel, 0, 1).text() == "鳥" and cell(panel, 0, 3).text() == "鳥 English"
    panel.pause.setChecked(True)
    newer = replace(
        latest, frame=replace(latest.frame, sequence=4), regions=(TextRegion("魚", bounds),)
    )
    panel.set_result(newer, (8,))
    panel.set_translations((TranslatedText(TranslationRequest(8, 1, "魚"), "Fish"),))
    assert panel.result is latest and cell(panel, 0, 3).text() == "鳥 English"
    panel.pause.setChecked(False)
    assert panel.result is newer and cell(panel, 0, 3).text() == "Fish"
    panel.clear()
    panel.set_active(False)
    panel.set_active(True)
    assert panel.result is None and panel._latest_frame is None
    panel.deleteLater()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("key", ["gemma", "gemma-e4b", "qwen35-4b", "plamo-2b", "plamo-0.8b"])
def test_saved_model_choice_survives_runtime_change_without_implicit_preparation(
    app, tmp_path, monkeypatch, prepared, key
):
    from dantetl.application.preference_values import Preferences
    from dantetl.application.preferences import PreferenceStore

    path = tmp_path / "preferences.json"
    store = PreferenceStore(path)
    store._write(Preferences(model=key))
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock()
    base = BackendChoice("Imported fixture", factory, "imported-" + "f" * 64)
    gemma = BackendChoice(
        "Language model", factory, f"{key}-vulkan", "vulkan-receipt", selection_id=key
    )
    monkeypatch.setattr("dantetl.runtime.model_store.is_prepared", lambda key: prepared or not key)
    window = MainWindow(windows, factory, (base, gemma), preferences_path=path)
    application = app_owner(window)
    selected = window.routing.resolve(model_reference(gemma))
    assert selected and window.backend_combo.currentData() == selected
    assert not visible_messages(window)
    factory.assert_not_called()
    if not prepared:
        assert window.onboarding.action.text() == "Prepare model"
        window.onboarding.action.click()
        answer_dialog(window, QMessageBox.StandardButton.No)
        assert window.backend_combo.currentData() == selected
    settings(window).opacity.setValue(63)
    drain(window, lambda: application.preferences.idle)
    assert PreferenceStore(path).initial.model == key
    # Disabling translation preserves the saved model and its preparation intent.
    window.translate_check.setChecked(False)
    drain(window, lambda: application.preferences.idle)
    assert PreferenceStore(path).initial.model == key
    assert not PreferenceStore(path).initial.translation
    factory.assert_not_called()
    window.close()
    window.deleteLater()


def test_inspector_disclosure_and_pause_do_not_change_the_injected_live_session(app, tmp_path):
    from dantetl.application.live_session import LiveSession
    from dantetl.ocr.values import OcrSnapshot, OcrState
    from dantetl.ocr.visual_values import VisualState

    windows = Mock()
    windows.list_targets.return_value = []
    ocr, visual, capture = Mock(), Mock(), Mock(finished=False)
    live = LiveSession(ocr_factory=lambda _: ocr, visual_factory=lambda: visual)
    window = MainWindow(windows, Mock(), cache_path=tmp_path / "cache", live_session=live)
    live.start(capture, Mock())
    bounds = Rect(0, 0, 4, 4)
    geometry = WindowGeometry(bounds, bounds, bounds, 96)
    frozen = None
    for sequence in range(1, 4):
        frame = CapturedFrame(sequence, sequence, geometry, b"\xff" * 64)
        result = OcrResult(frame, (TextRegion("猫", bounds),), 10, sequence)
        ocr.snapshot.return_value = OcrSnapshot(OcrState.READY, result)
        from dantetl.ocr.tracking import prepare_matching

        sources = visual.submit.call_args.args[-1] if visual.submit.called else ()
        visual.poll.return_value = VisualState(
            review=CandidateReview(sequence=sequence, matching=prepare_matching(result, sources)),
            frame=frame,
            request_ocr=True,
        )
        from dantetl.domain import CaptureSnapshot, CaptureState

        windows.geometry.return_value = geometry
        windows.is_foreground.return_value = True
        capture.snapshot.return_value = CaptureSnapshot(CaptureState.RUNNING, frame)
        window.driver.refresh(inspect=True)
        assert live.accepted is result
        if sequence == 1:
            assert window.inspector.result is None
            settings(window).inspection_enabled.setChecked(True)
            window.advanced.setChecked(True)
            frozen = window.inspector.result
            window.inspector.pause.setChecked(True)
        else:
            assert window.inspector.result is frozen
            window.advanced.setChecked(False)
    window.inspector.pause.setChecked(False)
    assert window.inspector.result is None  # Collapsed inspection does no hidden conversion.
    window.advanced.setChecked(True)
    assert window.inspector.result is live.accepted
    capture.finished = ocr.finished = visual.finished = True
    window.close()
    window.deleteLater()


def test_translation_off_leaves_game_unobscured_until_ocr_boxes_are_requested(app, tmp_path):
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), scene_feed=True, cache_path=tmp_path / "cache")
    window.translate_check.setChecked(False)
    block = TextBlock((TextTrack(1, "cat", Rect(20, 20, 40, 20)),))
    show_scene(window, (block,))
    before = window.application.view.scene
    assert window.application.view.overlay_bounds is None
    window.raw_boxes_check.setChecked(True)
    assert window.application.view.overlay_bounds is not None
    window.overlay_check.setChecked(False)
    assert window.application.view.overlay_bounds is None
    assert window.application.view.scene == before
    window.close()
    drain(window, lambda: not window.driver.timer.isActive())
    window.deleteLater()


@pytest.mark.parametrize("cancel_operation", [False, True])
def test_settings_close_can_keep_setup_running_or_cancel_without_stopping_app(
    app, tmp_path, monkeypatch, cancel_operation
):
    from dantetl.runtime.model_store import ModelBundle
    from dantetl.runtime.setup_values import Cancelled

    windows = Mock()
    windows.list_targets.return_value = []
    entered = threading.Event()

    def install(bundle, progress, cancel):
        entered.set()
        assert cancel.wait(2)
        raise Cancelled()

    monkeypatch.setattr("dantetl.runtime.model_store.install_model", install)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    selected = "Cancel operation" if cancel_operation else "Continue in background"
    monkeypatch.setattr(
        QMessageBox,
        "clickedButton",
        lambda self: next(b for b in self.buttons() if b.text() == selected),
    )
    bundle = ModelBundle("fixture", "Fixture", tmp_path, (), "")
    window = MainWindow(windows, Mock(), models=(bundle,), cache_path=tmp_path / "cache")
    application = app_owner(window)
    settings(window).show_models()
    manager = settings(window).models
    manager.download.click()
    assert entered.wait(1) and manager.busy
    settings(window).close()
    assert not application.shutdown.active and window.driver.timer.isActive()
    assert manager.service.job(SetupLane.MODELS).cancelling == cancel_operation
    if not cancel_operation:
        assert settings(window).busy and not settings(window).isVisible()
    window.close()  # Closing the whole app always cancels and drains, without another prompt.
    settle_setup(manager.service)
    setup_owner(manager).advance()
    manager.refresh_view()
    drain(window, lambda: not window.driver.timer.isActive())
    assert not settings(window).busy
    window.deleteLater()


def test_text_selection_and_empty_result_replace_previous_frame_and_boxes(app) -> None:
    bounds = Rect(0, 0, 4, 4)
    source = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * 64)
    panel = OcrInspector()
    result = OcrResult(source, (TextRegion("日本語", Rect(0, 0, 3, 3)),), 5, 1)
    panel.set_result(result)
    panel.table.selectRow(0)
    assert panel.preview.selected == 0
    item = panel.table.item(0, 1)
    assert item is not None and item.text() == "日本語"
    assert "x=0" in panel.details.text()
    panel.set_result(OcrResult(source, (), 5, 2))
    assert panel.table.rowCount() == 0
    assert panel.preview.selected == -1
    assert not panel.copy_button.isEnabled()
    panel.clear()
    assert panel.result is None
    panel.close()
    panel.deleteLater()


def test_overlay_does_not_request_repaint_for_equivalent_track_values(app, monkeypatch) -> None:
    overlay = OverlayWindow(Mock())
    repaint = Mock()
    monkeypatch.setattr(overlay, "update", repaint)
    track = TextTrack(17, "日本語", Rect(100, 200, 120, 30))
    overlay.set_scene((track,))
    assert repaint.call_count == 1
    for _ in range(20):
        overlay.set_scene((replace(track),))
    assert repaint.call_count == 1
    overlay.set_scene((replace(track, bounds=Rect(120, 200, 120, 30)),))
    assert repaint.call_count == 2
    overlay.set_scene(())
    assert repaint.call_count == 3
    overlay.close()
    overlay.deleteLater()


def test_debug_draws_without_translations_and_toggle_preserves_publication(app):
    overlay = OverlayWindow(Mock())
    overlay.resize(400, 300)
    overlay._bounds = Rect(0, 0, 400, 300)
    overlay.set_scene((), english=True)
    assert overlay.grab().toImage().pixelColor(210, 210).alpha() == 0
    overlay.set_debug(DebugDisplay((Rect(200, 200, 40, 40),), (), "Debug"))
    assert overlay.grab().toImage().pixelColor(210, 210).alpha() > 0
    assert overlay._translations == () and overlay._tracks == ()
    overlay.set_debug(None)
    assert overlay.grab().toImage().pixelColor(210, 210).alpha() == 0
    overlay.close()
    overlay.deleteLater()


def test_english_scene_repaints_only_the_moved_box_and_keeps_its_neighbor(app, monkeypatch):
    overlay = OverlayWindow(Mock())
    overlay._bounds = Rect(0, 0, 700, 240)
    overlay.resize(700, 240)
    a = TextTrack(1, "攻撃", Rect(30, 40, 85, 28))
    b = TextTrack(2, "設定", Rect(430, 40, 85, 28))
    translations = tuple(
        TranslatedText(TranslationRequest.from_track(t), word)
        for t, word in ((a, "Attack"), (b, "Settings"))
    )
    overlay.set_scene((a, b), translations=translations, english=True)
    before = overlay.grab().toImage()
    repaint = Mock()
    monkeypatch.setattr(overlay, "update", repaint)
    moved = replace(a, bounds=replace(a.bounds, y=150))
    overlay.set_scene((moved, b), translations=translations, english=True)
    repaint.assert_called_once()
    (dirty,) = repaint.call_args.args
    after = overlay.grab().toImage()
    for x, y in ((35, 45), (35, 155), (435, 45)):
        assert dirty.contains(QPoint(x, y)) == (x == 35)
        assert (after.pixelColor(x, y).alpha() > 0) == (y == 155 or x == 435)
    assert before.copy(430, 40, 85, 28) == after.copy(430, 40, 85, 28)
    overlay.close()
    overlay.deleteLater()


def test_inspector_displays_track_ids_without_replacing_raw_ocr_geometry(app) -> None:
    bounds = Rect(0, 0, 4, 4)
    source = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * 64)
    panel = OcrInspector()
    result = OcrResult(source, (TextRegion("日本語", Rect(1, 1, 2, 2)),), 5, 1)
    panel.set_result(result, (47,))
    item = panel.table.item(0, 0)
    assert item is not None and item.text() == "47"
    assert panel.preview._regions == result.regions
    panel.close()
    panel.deleteLater()


def test_english_layout_wraps_and_stays_inside_screen_at_bottom_right(app) -> None:
    layout = layout_english(
        "A new adventure begins here, and there is more to discover.",
        Rect(1150, 690, 130, 30),
        (1280, 720),
    )
    assert Rect(0, 0, 1280, 720).contains(layout.bounds)
    assert len(layout.lines) > 1
    assert layout.pixel_size >= 12


def test_manual_font_and_overflow_are_explicit(app) -> None:
    layout = layout_english(
        "A very long translation " * 100,
        Rect(0, 0, 100, 30),
        (300, 200),
        automatic=False,
        manual_pixels=24,
    )
    assert layout.pixel_size == 24
    assert layout.truncated and layout.lines[-1].endswith("…")
    assert Rect(0, 0, 300, 200).contains(layout.bounds)


def test_english_column_rejects_translation_for_different_raw_text(app) -> None:
    bounds = Rect(0, 0, 4, 4)
    source = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * 64)
    panel = OcrInspector()
    result = OcrResult(source, (TextRegion("猫", Rect(0, 0, 4, 4)),), 5, 1)
    panel.set_result(result, (1,))
    panel.set_translations((TranslatedText(TranslationRequest(1, 1, "猫"), "Cat"),))
    item = panel.table.item(0, 3)
    assert item is not None and item.text() == "Cat"
    panel.set_result(replace(result, regions=(TextRegion("犬", Rect(0, 0, 4, 4)),)), (1,))
    item = panel.table.item(0, 3)
    assert item is not None and item.text() == "—"
    panel.close()
    panel.deleteLater()


def test_english_overlay_keeps_original_visible_until_current_revision_is_ready(
    app, monkeypatch
) -> None:
    overlay = OverlayWindow(Mock())
    overlay.resize(400, 200)
    overlay._bounds = Rect(0, 0, 400, 200)
    track = TextTrack(1, "猫", Rect(20, 20, 100, 40), text_revision=2)
    overlay.set_scene((track,))
    # Even identical text from a superseded revision must not cover the source.
    overlay.set_scene(
        (track,),
        translations=(TranslatedText(TranslationRequest(1, 1, "猫"), "Cat"),),
        english=True,
    )
    assert overlay.grab().toImage().pixelColor(25, 25).alpha() == 0
    current = (TranslatedText(TranslationRequest.from_track(track), "Cat"),)
    overlay.set_scene((track,), translations=current, english=True)
    assert overlay.grab().toImage().pixelColor(25, 25).alpha() > 0
    repaint = Mock()
    monkeypatch.setattr(overlay, "update", repaint)
    overlay.set_scene((track,), translations=current, english=True)
    repaint.assert_not_called()
    overlay.set_scene((track,), english=True)
    assert overlay.grab().toImage().pixelColor(25, 25).alpha() == 0
    overlay.close()
    overlay.deleteLater()


def test_grouped_english_uses_one_background_and_validates_every_member(app) -> None:
    overlay = OverlayWindow(Mock())
    overlay.resize(400, 200)
    overlay._bounds = Rect(0, 0, 400, 200)
    lines = (
        TextTrack(1, "新しい冒険が始まるので、", Rect(20, 20, 280, 32)),
        TextTrack(2, "準備をしよう。", Rect(20, 68, 240, 32)),
    )
    block = TextBlock(lines)
    translations = (
        TranslatedText(TranslationRequest.from_block(block), "Get ready for a new adventure."),
    )
    overlay.set_scene(lines, (block,), translations, english=True)
    assert overlay.grab().toImage().pixelColor(25, 60).alpha() > 0  # The gap is inside one block.
    overlay.set_scene(
        lines,
        (TextBlock((lines[0], replace(lines[1], text_revision=2))),),
        translations,
        english=True,
    )
    assert overlay.grab().toImage().pixelColor(25, 25).alpha() == 0
    overlay.close()
    overlay.deleteLater()


def test_group_auto_font_uses_line_height_not_paragraph_height(app) -> None:
    layout = layout_english(
        "A complete sentence.", Rect(10, 10, 500, 150), (800, 300), source_line_height=32
    )
    assert layout.pixel_size == round(32 * 0.72)


def test_group_inspection_and_copy_map_to_raw_rows_without_duplicating_english(
    app, monkeypatch
) -> None:
    lines = (
        TextTrack(1, "新しい冒険が始まるので、", Rect(20, 20, 280, 32)),
        TextTrack(2, "準備をしよう。", Rect(20, 68, 240, 32)),
    )
    bounds = Rect(0, 0, 400, 200)
    frame = CapturedFrame(
        1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * (400 * 200 * 4)
    )
    regions = tuple(TextRegion(t.text, t.bounds) for t in lines)
    result = OcrResult(frame, regions, 20, 0)
    panel = OcrInspector()
    panel.set_result(result, (1, 2))
    translated = TranslatedText(
        TranslationRequest.from_block(TextBlock(lines)), "Get ready for a new adventure."
    )
    panel.set_translations((translated,))
    assert cell(panel, 0, 3).text() == translated.english
    assert cell(panel, 1, 3).text() == "↳ Group #1"
    assert lines[1].text in cell(panel, 1, 3).toolTip()
    clipboard = Mock()
    monkeypatch.setattr("dantetl.ui.inspection.QGuiApplication.clipboard", lambda: clipboard)
    panel._copy_english()
    clipboard.setText.assert_called_with(translated.english)
    panel.table.selectRow(1)
    panel._copy_english()
    clipboard.setText.assert_called_with(translated.english)
    panel.set_result(
        replace(result, regions=(regions[0], replace(regions[1], text="違う文章。"))), (1, 2)
    )
    assert cell(panel, 0, 3).text() == cell(panel, 1, 3).text() == "—"
    panel.close()
    panel.deleteLater()


def test_raw_icon_evidence_is_visible_beside_cleaned_translation(app) -> None:
    chars = (TextCharacter("@", Rect(20, 10, 25, 30), 0.3),) + tuple(
        TextCharacter(c, Rect(70 + i * 25, 10, 25, 30), 0.95)
        for i, c in enumerate("道具を確認する")
    )
    raw = TextRegion("@道具を確認する", Rect(15, 10, 235, 30), chars)
    bounds = Rect(0, 0, 400, 100)
    frame = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), b"\xff" * 160000)
    panel = OcrInspector()
    panel.set_result(OcrResult(frame, (raw,), 20, 0), (1,))
    panel.set_translations(
        (TranslatedText(TranslationRequest(1, 1, "道具を確認する"), "Check items"),)
    )
    assert cell(panel, 0, 1).text() == raw.text
    assert cell(panel, 0, 3).text() == "Check items"
    assert panel.preview._regions == (raw,)
    panel.close()
    panel.deleteLater()
