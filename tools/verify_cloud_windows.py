"""Native cloud settings/routing layout with isolated fake accounts; no provider calls."""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication, QScrollArea
from verification_isolation import app_owner, cloud_owner, create_inventory, find_model

from dantetl.application.cloud_services import CloudServices
from dantetl.capture.windows import Windows
from dantetl.composition import application_session, setup_coordinator
from dantetl.paths import state_root
from dantetl.translation.cloud.chatgpt_auth import PLAN_SCOPE, ChatGPTConnections
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.cloud.values import CatalogModel, CloudProfile, CloudProvider
from dantetl.ui.desktop_driver import DesktopDriver
from dantetl.ui.main_window import MainWindow


def drain(app: QApplication, predicate) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Native cloud UI did not settle"
        app.processEvents()
        time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cloud/layout"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    evidence = {}
    with tempfile.TemporaryDirectory(dir=args.output) as directory:
        root = Path(directory)
        connections = ChatGPTConnections(ProtectedStore(root / "accounts"))
        account = "account-" + "a" * 32
        connections.store.write(
            account,
            {
                "email": "demo@example.invalid",
                "access_token": "fixture-not-real",
                "scopes": [PLAN_SCOPE],
                "models": [("gpt-6-luna", "GPT-6 Luna")],
            },
        )
        profile = CloudProfile(
            "cloud-" + "b" * 32, "GPT-6 Luna", CloudProvider.CHATGPT, "gpt-6-luna", account=account
        )
        catalog = CloudProfiles(root / "cloud.json")
        api_profile = CloudProfile.new(CloudProvider.DEEPSEEK)
        missing_profile = CloudProfile.new(CloudProvider.OPENAI)
        gemini, claude = (
            CloudProfile.new(CloudProvider.GEMINI),
            CloudProfile.new(CloudProvider.CLAUDE),
        )
        catalog.write((profile, api_profile, missing_profile, gemini, claude))
        secrets = Mock()
        secrets.read.side_effect = lambda key: (
            "fixture-key" if key in (api_profile.key, gemini.key, claude.key) else None
        )
        service = CloudServices(catalog, lambda: secrets, connections)
        service.start()
        driver = DesktopDriver()
        recognizer = Mock()
        inventory, source_models = create_inventory((windows, recognizer), {})
        application = application_session(
            windows,
            recognizer,
            cloud=service,
            notify=driver.notify,
            inventory=inventory,
            source_models=source_models,
            setup=setup_coordinator(inventory, (), state_root()),
            preferences_path=root / "preferences.json",
            cache_path=root / "cache",
        )
        window = MainWindow(application, windows, driver)
        application = app_owner(window)
        try:
            drain(app, lambda: not application.cloud.busy)
            window.windows_combo.clear()
            window.windows_combo.addItem("DanteTL Demo · 日本語のテスト")
            window.routing.simple.setCurrentIndex(
                find_model(window.routing.simple, window.routing.resolve("cloud:" + profile.key))
            )
            window.routing.game_from_window.setChecked(False)
            window.routing.game.setText("Honkai: Star Rail")
            window.show()
            app.processEvents()
            assert not window.routing.advanced.isChecked()
            assert not window.routing.short_model.isVisible()
            window.grab().save(str(args.output / "main-simple.png"))
            assert window.routing.plan_notice.isVisible()
            assert tuple(c.key for c in window.routing.active_cloud) == (profile.key,)
            assert application.translation.lifecycle.publication is None
            evidence["configuration_without_capture_makes_no_requests"] = True
            window.routing.simple.showPopup()
            app.processEvents()
            window.routing.simple.view().window().grab().save(
                str(args.output / "model-sections.png")
            )
            window.routing.simple.hidePopup()
            window.routing.advanced.setChecked(True)
            app.processEvents()
            assert (
                window.routing.short_model.isVisible()
                and not window.routing.simple.isVisible()
                and window.routing.game.isVisible()
            )
            window.grab().save(str(args.output / "main-advanced-routing.png"))
            window.routing.short_model.setCurrentIndex(
                find_model(
                    window.routing.short_model, window.routing.resolve("cloud:" + gemini.key)
                )
            )
            window.routing.advanced.setChecked(False)
            assert (
                window.routing.selection.short == "cloud:" + gemini.key
                and "Different models" in window.routing.simple.placeholderText()
                and window.routing.simple.currentIndex() == -1
            )
            evidence["simple_selector_and_disclosure_preserve_custom_route"] = True
            window._settings.show_models()
            manager = window._settings.models
            manager.category.setCurrentIndex(manager.category.findData("cloud"))
            app.processEvents()
            panel = manager.cloud
            assert panel.table.rowCount() == 5 and panel.editor is None
            saved, missing = panel.table.item(1, 2), panel.table.item(2, 2)
            assert saved and saved.text() == "✓ Key saved"
            assert missing and "Connection needed" in missing.text()
            window._settings.grab().save(str(args.output / "models-list.png"))
            # Simulate reported usage at the application-owned meter, without sending
            # any provider request or changing the configuration revision.
            asyncio.run(service.allowance(profile.key).admit(profile))
            service.allowance(profile.key).record({"input_tokens": 123, "output_tokens": 7})
            drain(app, lambda: "123 input / 7 output" in panel.usage.text())
            assert profile.name in panel.usage.text() and "1/1000" in panel.usage.text()
            panel.usage.grab().save(str(args.output / "usage.png"))
            panel.table.selectRow(1)
            assert api_profile.name in panel.usage.text() and "0/1000" in panel.usage.text()
            panel.table.selectRow(0)
            assert "1/1000" in panel.usage.text()
            evidence["usage_refreshes_without_configuration_change_and_follows_selected_model"] = (
                True
            )
            panel.edit.click()
            editor = panel.editor
            assert editor
            app.processEvents()
            assert editor.chatgpt.sign_in.isVisible()
            assert "Signed in" in editor.badge.text()
            editor.grab().save(str(args.output / "edit-chatgpt.png"))
            editor.reject()
            panel.add.click()
            editor = panel.editor
            assert editor
            editor.provider.setCurrentIndex(editor.provider.findData(CloudProvider.DEEPSEEK))
            app.processEvents()
            assert editor.key.isVisible() and not editor.chatgpt.isVisible()
            assert not editor.limits.isVisible()
            editor.grab().save(str(args.output / "add-api.png"))
            editor.advanced.setChecked(True)
            editor.resize(480, 460)
            app.processEvents()
            assert editor.limits.isVisible() and editor.save.isVisible()
            editor.grab().save(str(args.output / "editor-small-advanced.png"))
            editor.reject()
            assert panel.editor is None and len(panel.profiles) == 5
            for provider, expected_count in ((CloudProvider.GEMINI, 3), (CloudProvider.CLAUDE, 2)):
                panel.add.click()
                editor = panel.editor
                assert editor
                editor.provider.setCurrentIndex(editor.provider.findData(provider))
                app.processEvents()
                assert editor.model.count() == expected_count + 1
                assert editor.key.isVisible() and not editor.chatgpt.isVisible()
                assert "Key saved" in editor.badge.text() and not editor.key.text()
                editor.model.setCurrentIndex(expected_count - 1)
                editor.grab().save(str(args.output / f"add-{provider}.png"))
                editor.reject()
                selected = gemini if provider == CloudProvider.GEMINI else claude
                index = find_model(window.routing.simple, selected)
                window.routing.simple.setCurrentIndex(index)
                assert (
                    tuple(c.key for c in window.routing.active_cloud) == (selected.key,)
                    and application.translation.lifecycle.publication is None
                )
            evidence["gemini_and_claude_saved_connections_and_model_choices"] = True
            panel.open_editor(gemini)
            editor = panel.editor
            assert editor
            previous = editor.model.currentData()
            with patch(
                "dantetl.application.cloud_services.fetch_models",
                return_value=(
                    CatalogModel("future-gemini", "Future Gemini · fixture", "gemini-no-thinking"),
                ),
            ):
                editor.refresh_models.click()
                drain(app, lambda: cloud_owner(panel)._worker is None)
            assert editor.model.currentData() == previous
            assert editor.model.findData("future-gemini") >= 0
            editor.model.setCurrentIndex(editor.model.findData("future-gemini"))
            assert "Test connection" in editor.model_note.text()
            app.processEvents()
            scroll = editor.findChild(QScrollArea)
            assert scroll and scroll.horizontalScrollBar().maximum() == 0
            editor.grab().save(str(args.output / "provider-model-discovery.png"))
            editor.resize(480, 460)
            app.processEvents()
            assert scroll.horizontalScrollBar().maximum() == 0
            editor.grab().save(str(args.output / "discovery-small.png"))
            editor.reject()
            assert tuple(c.key for c in window.routing.active_cloud) == (claude.key,)
            panel.open_editor(gemini)
            assert panel.editor and panel.editor.model.findData("future-gemini") >= 0
            panel.editor.reject()
            evidence["explicit_discovery_cached_without_changing_committed_route"] = True
            evidence["compact_list_and_draft_dialog"] = True
            evidence["plan_and_api_billing_controls_are_separate"] = True
            window._settings.hide()
            window.showMaximized()
            drain(app, lambda: window.sections.translation.x() > window.sections.game.x())
            assert window.sections.translation.x() > window.sections.game.x()
            window.routing.advanced.setChecked(True)
            app.processEvents()
            window.grab().save(str(args.output / "main-maximized.png"))
            window.control_panel.grab().save(str(args.output / "main-controls-wide.png"))
            window._settings.inspection_enabled.setChecked(True)
            window.advanced.setChecked(True)
            app.processEvents()
            assert (
                window.isMaximized() and window.sections.translation.x() > window.sections.game.x()
            )
            evidence["maximized_controls_use_two_columns_with_inspection_independent"] = True
            window.advanced.setChecked(False)
            window.showNormal()
            window.resize(640, 500)
            drain(app, lambda: window.sections.translation.y() > window.sections.game.y())
            scroll = window.centralWidget()
            assert isinstance(scroll, QScrollArea)
            drain(app, lambda: scroll.horizontalScrollBar().maximum() == 0)
            assert scroll.verticalScrollBar().maximum() > 0
            scroll.ensureWidgetVisible(window.routing.long_model)
            app.processEvents()
            window.grab().save(str(args.output / "main-small-scroll.png"))
            evidence["short_windows_scroll_without_horizontal_clipping"] = True
        finally:
            window.close()
            drain(app, lambda: not window.isVisible())
            window.deleteLater()
            app.processEvents()
    print(json.dumps(evidence, indent=2))
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2), "utf-8")


if __name__ == "__main__":
    main()
