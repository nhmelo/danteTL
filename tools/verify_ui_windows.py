"""Native layout/settings check with our own demo; no game capture or model downloads."""

import argparse
import json
import os
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import shiboken6
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
from verification_isolation import app_owner
from verification_isolation import create_window as MainWindow

from dantetl.application.model_discovery import translation_backends
from dantetl.application.model_factories import create_recognizer, ocr_backends
from dantetl.application.preferences import PreferenceStore
from dantetl.application.updates import Updates
from dantetl.capture.windows import Windows
from dantetl.domain import WindowTarget
from dantetl.runtime.hardware import HardwareCheck
from dantetl.runtime.model_store import model_catalog
from dantetl.translation.cache import TranslationCache
from dantetl.ui.controls import InfoButton
from dantetl.ui.demo import DemoWindow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui"))
    parser.add_argument("--missing-models", action="store_true", help="Show first-run warnings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    demo = DemoWindow(dialogue=True)
    preferences_path = args.output / f"preferences-{time.time_ns()}.json"
    cache_path = args.output / f"cache-{time.time_ns()}.sqlite3"
    cache = TranslationCache(cache_path)
    for game in ("DanteTL Demo", "Second demo"):
        cache.lookup_pending(("Settings", game), reset=False, game=game)
        cache.put("Settings", "Settings", scope=cache.scope)
        cache.put(game, "Map", scope=cache.scope)
    cache.put("Inventory", "Inventory")
    cache.close()
    hardware = HardwareCheck()
    hardware.start()
    models = model_catalog()
    if args.missing_models:
        models = tuple(
            replace(b, directory=args.output / "not-installed" / b.key, bundled_directory=None)
            for b in models
        )
    updates = Updates("0.1.0", args.output, fetch=lambda cancel: None, install=None)
    target = WindowTarget(int(demo.winId()), os.getpid(), demo.windowTitle())
    # Exercise the normal picker with only the owned fixture in screenshots.
    with patch.object(windows, "list_targets", return_value=[target]):
        window = MainWindow(
            windows,
            create_recognizer,
            () if args.missing_models else translation_backends(),
            models=models,
            ocr_backends=ocr_backends(),
            preferences_path=preferences_path,
            cache_path=cache_path,
            hardware=hardware,
            updates=updates,
        )
    application = app_owner(window)
    window.show()
    started, stage = time.monotonic(), 0
    failed = False
    evidence = {}

    def tick():
        nonlocal stage, failed
        try:
            assert time.monotonic() - started < 20, "UI verification timed out"
            settings = window._settings
            if stage in (0, 8, 80, 81):  # Both routing modes in compact and maximized layouts.
                advanced = stage in (80, 81)
                routing = window.routing
                models = (
                    (routing.short_model, routing.long_model) if advanced else (routing.simple,)
                )
                pickers = (window.source_picker, window.output_language, *models)
                assert all(picker.isVisible() for picker in pickers)
                assert len({(p.mapTo(window, QPoint()).x(), p.width()) for p in pickers}) == 1
                if not advanced:
                    routing.advanced.setChecked(True)
                    stage = 80 if stage == 0 else 81
                    return
                routing.advanced.setChecked(False)
                stage = 0 if stage == 80 else 8
            if stage == 0:
                if window.windows_combo.itemIcon(0).isNull():
                    return
                assert window.windows_combo.currentData() == target
                evidence["selected_window_process_icon"] = True
                if not hardware.finished:
                    return
                assert not window.advanced.isVisible() and not window.advanced_panel.isVisible()
                assert window.translate_check.text() == "Translation overlay"
                QTest.keyClick(window.overlay_check, Qt.Key.Key_Space)
                assert not window.overlay_check.isChecked()
                QTest.keyClick(window.overlay_check, Qt.Key.Key_Space)
                assert window.overlay_check.isChecked()
                evidence["keyboard_switch_toggle"] = True
                assert window.border_check.isChecked()
                QTest.keyClick(window.border_check, Qt.Key.Key_Space)
                assert not window.border_check.isChecked()
                assert window.capture_overlay_check.isChecked()
                QTest.keyClick(window.capture_overlay_check, Qt.Key.Key_Space)
                assert not window.capture_overlay_check.isChecked()
                evidence["compact_size"] = [window.width(), window.height()]
                assert window.sections.game.isAncestorOf(window.status)
                assert window.start_button.property("tone") == "idle"
                assert not window.start_button.icon().isNull()
                evidence["capture_icon_and_idle_state"] = True
                star, picker = window.favorite_button, window.windows_combo
                assert star.isEnabled() and not star.isChecked()
                star.click()
                assert star.isChecked() and picker.currentData() == target
                assert picker.itemText(0) == "Favorites" and picker.count() == 2
                window.grab().save(str(args.output / "main.png"))
                with patch.object(windows, "list_targets", return_value=[target]):
                    picker.showPopup()
                picker.view().window().grab().save(str(args.output / "favorites.png"))
                picker.hidePopup()
                star.click()
                assert not star.isChecked() and picker.count() == 1
                evidence["favorite_star_and_section"] = True
                if args.missing_models:
                    assert (
                        window.onboarding.isVisible() and "not downloaded" in window.notice.text()
                    )
                    assert window.onboarding.property("tone") == "warning"
                gear = window.settings_button
                accessible = QAccessible.queryAccessibleInterface(gear)
                assert accessible and accessible.text(QAccessible.Text.Name) == "Settings"
                assert not gear.text() and gear.toolTip() and not gear.icon().pixmap(16).isNull()
                QTest.keyClick(gear, Qt.Key.Key_Space)
                stage = 1
            elif stage == 1:
                assert settings.isVisible() and not settings.isModal()
                assert [settings.tabs.tabText(i) for i in range(settings.tabs.count())] == [
                    "Appearance",
                    "Recognition",
                    "Models",
                    "Acceleration",
                    "Hotkeys",
                    "Cache",
                    "Experimental",
                ]
                assert settings.runtime is None  # Merely opening Settings does not start setup.
                assert window.stabilize_check.isChecked() and window.automatic_font.isChecked()
                settings.grab().save(str(args.output / "appearance.png"))
                info = window.about_button
                assert info.accessibleName() == "About" and not info.icon().pixmap(16).isNull()
                QTest.keyClick(info, Qt.Key.Key_Space)
                assert window.about.isVisible()
                version_label = next(
                    label
                    for label in window.about.findChildren(QLabel)
                    if label.text() == f"v{version('dantetl')}"
                )
                button_bounds = window.about.check.geometry()
                assert version_label.isVisible() and version_label.x() > button_bounds.right()
                label_center_y = version_label.geometry().center().y()
                assert button_bounds.top() <= label_center_y <= button_bounds.bottom()
                credits = next(
                    label
                    for label in window.about.findChildren(QLabel)
                    if label.openExternalLinks()
                )
                assert credits.isVisible() and "© 2026 Nuno Melo" in credits.text()
                assert 'href="https://github.com/nhmelo"' in credits.text()
                assert 'href="https://www.apache.org/licenses/LICENSE-2.0"' in credits.text()
                assert window.about.check.isEnabled()
                window.about.check.click()
                stage = 102
            elif stage == 102:
                if window.about.status.text() != "DanteTL is up to date.":
                    return
                window.about.grab().save(str(args.output / "about.png"))
                window.about.close()
                settings.tabs.setCurrentWidget(settings.cache)
                stage = 101
            elif stage == 101:
                if application.cache.snapshot().busy:
                    return
                settings.cache.refresh_view()
                assert settings.cache.games.count() == 3
                assert settings.cache.status.text() == "Cached translations: 4"
                settings.cache.games.setCurrentRow(0)
                assert (
                    settings.cache.clear_game.isEnabled() and settings.cache.clear_all.isEnabled()
                )
                settings.grab().save(str(args.output / "cache.png"))
                evidence["cache_inventory_and_actions"] = True
                settings.tabs.setCurrentIndex(
                    next(
                        i
                        for i in range(settings.tabs.count())
                        if settings.tabs.tabText(i) == "Experimental"
                    )
                )
                assert not settings.history_enabled.isChecked()
                assert not settings.inspection_enabled.isChecked()
                assert not window.history_button.isVisible()
                settings.grab().save(str(args.output / "experimental.png"))
                QTest.keyClick(settings.history_enabled, Qt.Key.Key_Space)
                assert window.history_button.isVisible() and application.live.history_enabled
                dialogue = window.history_button
                accessible = QAccessible.queryAccessibleInterface(dialogue)
                assert accessible and accessible.text(QAccessible.Text.Name) == "Dialogue history"
                assert dialogue.text() == "Dialogue" and not dialogue.icon().pixmap(16).isNull()
                window.grab().save(str(args.output / "main-dialogue.png"))
                QTest.keyClick(dialogue, Qt.Key.Key_Space)
                assert window.history_panel.isVisible()
                QTest.keyClick(settings.history_enabled, Qt.Key.Key_Space)
                assert (
                    not window.history_panel.isVisible() and not window.history_button.isVisible()
                )
                assert not application.live.history_enabled
                evidence["experimental_history_opt_in"] = True
                QTest.keyClick(settings.inspection_enabled, Qt.Key.Key_Space)
                assert window.advanced.isVisible() and not window.advanced_panel.isVisible()
                QTest.keyClick(window.advanced, Qt.Key.Key_Space)
                assert window.advanced_panel.isVisible()
                window.inspector.pause.setChecked(True)
                window.debug_check.setChecked(True)
                QTest.keyClick(settings.inspection_enabled, Qt.Key.Key_Space)
                assert not window.advanced.isVisible() and not window.advanced_panel.isVisible()
                assert not window.inspector.pause.isChecked() and not window.debug_check.isChecked()
                QTest.keyClick(settings.inspection_enabled, Qt.Key.Key_Space)
                assert window.advanced.isVisible() and not window.advanced.isChecked()
                evidence["experimental_inspection_opt_in"] = True
                settings.opacity.setValue(62)
                window.automatic_font.setChecked(False)
                window.font_size.setValue(24)
                settings.tabs.setCurrentIndex(1)
                stage = 2
            elif stage == 2:
                settings.set_capture_active(True)
                assert settings.ocr_device.isEnabled() and settings.capture_rate.isEnabled()
                assert "next time" in settings.capture_note.text()
                settings.grab().save(str(args.output / "recognition.png"))
                window._manage_models()
                stage = 3
            elif stage == 3:
                assert settings.tabs.tabText(settings.tabs.currentIndex()) == "Models"
                assert settings.models.choice.currentData().key == "gemma"
                assert all(
                    settings.models.choice.itemData(i).key != "ocr"
                    for i in range(settings.models.choice.count())
                )
                assert settings.models.ocr.choice.currentData().key == "ocr"
                assert not settings.models.ocr.details.isVisible()
                expected = (
                    "Download required"
                    if args.missing_models
                    else "Included with app"
                    if settings.models.choice.currentData().included
                    else "Downloaded"
                )
                assert expected in settings.models.download_badge.text()
                settings.grab().save(str(args.output / "models.png"))
                gemma = next(
                    (
                        i
                        for i in range(settings.models.choice.count())
                        if settings.models.choice.itemData(i).key == "gemma"
                    ),
                    None,
                )
                if gemma is not None:
                    settings.models.choice.setCurrentIndex(gemma)
                stage = 31
            elif stage == 31:
                description = settings.models.description
                assert description.height() >= description.heightForWidth(description.width()), (
                    "Model description is clipped after changing selection"
                )
                key = settings.models.choice.currentData().key
                summary = settings.models.summary
                assert summary.isVisible() and summary.cpu.text() == "✓ CPU support"
                if key.startswith("plamo-"):
                    assert summary.values[1].text() == "~ Mixed"
                    assert summary.values[1].property("tone") == "warning"
                if key == "qwen35-4b":
                    assert summary.values[2].text() == "✕ Slow"
                settings.grab().save(str(args.output / f"{key}-status.png"))
                following = {
                    "gemma": "plamo-2b",
                    "plamo-2b": "plamo-0.8b",
                    "plamo-0.8b": "qwen35-4b",
                    "qwen35-4b": "gemma-e4b",
                }.get(key)
                index = next(
                    (
                        i
                        for i in range(settings.models.choice.count())
                        if settings.models.choice.itemData(i).key == following
                    ),
                    -1,
                )
                if index >= 0:
                    settings.models.choice.setCurrentIndex(index)
                    return
                settings.resize(550, 470)
                stage = 32
            elif stage == 32:
                description = settings.models.description
                assert description.height() >= description.heightForWidth(description.width()), (
                    "Model description is clipped in compact Settings"
                )
                settings.grab().save(str(args.output / "model-small.png"))
                settings.resize(680, 620)
                settings.models.category.setCurrentIndex(1)
                settings.models.ocr.manage.click()
                stage = 33
            elif stage == 33:
                assert settings.models.ocr.details.isVisible()
                settings.models.ocr_page.ensureWidgetVisible(settings.models.ocr.download)
                stage = 34
            elif stage == 34:
                assert "does not translate" in settings.models.ocr.summary.text()
                assert settings.models.pages.currentWidget() is settings.models.ocr_page
                assert not settings.models.choice.isVisible()
                assert settings.models.category.isVisible()
                settings.grab().save(str(args.output / "ocr-models.png"))
                evidence["separate_model_pages_with_fixed_menu"] = True
                settings.models.category.setCurrentIndex(0)
                settings.models.runtime_button.click()
                stage = 4
            elif stage == 4:
                if settings.busy:
                    return
                assert window._runtime_dialog and window._runtime_dialog.isVisible()
                assert settings.tabs.tabText(settings.tabs.currentIndex()) == "Acceleration"
                assert (
                    not application.translation.lifecycle.publication and not application.live.ocr
                )
                settings.grab().save(str(args.output / "runtime.png"))
                runtime = settings.runtime
                settings.show_hotkeys()
                assert settings.tabs.tabText(settings.tabs.currentIndex()) == "Hotkeys"
                settings.show_models()
                settings.show_runtime()
                assert (
                    settings.runtime is runtime
                )  # Navigating never replaces an active setup view.
                evidence["separate_setup_tabs_and_navigation"] = True
                settings.resize(550, 470)
                stage = 5
            elif stage == 5:
                assert settings.width() <= 550, "Settings minimum width broke the compact layout"
                settings.grab().save(str(args.output / "settings-small.png"))
                assert all(
                    b.accessibleName() and b.toolTip() for b in settings.findChildren(InfoButton)
                )
                settings.close()
                # Background setup notices may have enlarged the main window since startup.
                evidence["before_expansion_height"] = window.height()
                window.advanced.setChecked(True)
                stage = 6
            elif stage == 6:
                assert window.advanced_panel.isVisible()
                assert window.processing.isVisible()
                assert all(
                    card.isVisible()
                    for card in (
                        window.processing.capture,
                        window.processing.ocr,
                        window.processing.translation,
                    )
                )
                window.processing.grab().save(str(args.output / "processing.png"))
                window.grab().save(str(args.output / "advanced.png"))
                window.advanced.setChecked(False)
                if not application.preferences.idle:
                    return
                saved = PreferenceStore(preferences_path).initial
                assert saved.opacity == 62 and saved.text_size == 24 and not saved.automatic_size
                assert not saved.show_border
                assert not saved.include_overlay_in_capture
                evidence["capture_inclusion_preference_saved"] = True
                evidence["border_preference_saved"] = True
                evidence.update(
                    settings_saved=True, info_controls=len(settings.findChildren(InfoButton))
                )
                window.showMaximized()
                stage = 8
            elif stage == 8:
                assert window.isMaximized()
                assert window.control_panel.width() > 760 and window.control_panel.height() < 850
                assert window.sections.translation.x() > window.sections.game.x()
                assert window.control_panel.mapTo(window, QPoint()).y() <= 24
                assert window.translate_check.mapTo(window, QPoint()).x() < 450
                assert window.border_check.y() == window.translate_check.y()
                assert window.border_check.x() > window.translate_check.x()
                assert window.windows_combo.mapTo(window, QPoint()).y() < 200
                window.grab().save(str(args.output / "maximized.png"))
                window.advanced.setChecked(True)
                stage = 9
            elif stage == 9:
                assert window.isMaximized() and window.advanced_panel.isVisible()
                window.grab().save(str(args.output / "maximized-advanced.png"))
                window.advanced.setChecked(False)
                stage = 10
            elif stage == 10:
                assert window.isMaximized() and not window.advanced_panel.isVisible()
                evidence["maximized_controls_stay_compact_and_left"] = True
                evidence["advanced_preserves_maximized_state"] = True
                window.showNormal()
                stage = 11
            elif stage == 11:
                window.close()
                stage = 7
            elif stage == 7:
                if window.isVisible():
                    return
                assert window.height() <= evidence["before_expansion_height"], (
                    "Collapsing Advanced did not restore its previous compact height"
                )
                evidence["shutdown_finished"] = True
                (args.output / "results.json").write_text(json.dumps(evidence, indent=2), "utf-8")
                print(json.dumps(evidence), flush=True)
                timer.stop()
                app.quit()
        except Exception as error:
            failed = True
            print(f"FAIL at stage {stage}: {error!r}", flush=True)
            timer.stop()
            window._settings.cancel_operations()
            window.hide()
            app.quit()

    timer = QTimer()
    timer.setInterval(180)
    timer.timeout.connect(tick)
    timer.start()
    app.exec()
    updates.finish()
    demo.close()
    shiboken6.delete(app)
    return int(failed)


if __name__ == "__main__":
    from verification_isolation import run_isolated

    run_isolated(main)
