import asyncio
import os
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import httpx2
import pytest
from ui_fixtures import app_owner, cloud_owner, show_scene
from ui_fixtures import settings as window_settings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QMessageBox  # noqa: E402
from ui_fixtures import cloud_panel as CloudSettings  # noqa: E402
from ui_fixtures import (
    cloud_service,  # noqa: E402
    find_model,
)
from ui_fixtures import create_window as MainWindow

from dantetl.application.preference_values import Preferences  # noqa: E402
from dantetl.application.preferences import PreferenceStore  # noqa: E402
from dantetl.application.translation_coordinator import model_reference
from dantetl.domain import Rect  # noqa: E402
from dantetl.languages.target import TARGET_LANGUAGES  # noqa: E402
from dantetl.ocr.observation import TextTrack  # noqa: E402
from dantetl.translation.backend import BackendChoice  # noqa: E402
from dantetl.translation.cloud.backend import CloudBackend  # noqa: E402
from dantetl.translation.cloud.chatgpt_auth import PLAN_SCOPE, ChatGPTConnections  # noqa: E402
from dantetl.translation.cloud.profiles import CloudProfiles  # noqa: E402
from dantetl.translation.cloud.protected_store import ProtectedStore  # noqa: E402
from dantetl.translation.cloud.values import CatalogModel, CloudProfile  # noqa: E402
from dantetl.translation.grouping import TextBlock  # noqa: E402
from dantetl.translation.plan import TranslationPlan  # noqa: E402
from dantetl.translation.routing import RoutedTranslationSession  # noqa: E402


class MemorySecrets:
    def __init__(self):
        self.values = {}
        self.writers = []

    def read(self, key):
        return self.values.get(key)

    def write(self, key, secret):
        self.writers.append(threading.get_ident())
        self.values[key] = secret

    def delete(self, key):
        self.values.pop(key, None)


def drain(app, predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline
        app.processEvents()
        time.sleep(0.005)


def test_cloud_welcome_allows_desktop_updates_until_acknowledged(app, tmp_path, monkeypatch):
    from itertools import chain, repeat

    from PySide6.QtCore import QTimer

    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), cache_path=tmp_path / "cache")
    owner = app_owner(window)
    acknowledge = Mock()
    monkeypatch.setattr(owner.cloud, "acknowledge_welcome", acknowledge)
    monkeypatch.setattr(
        owner.cloud, "take_welcome", Mock(side_effect=chain(["fixture"], repeat("")))
    )
    advances = Mock(wraps=owner.advance)
    monkeypatch.setattr(owner, "advance", advances)
    # Release a regressed blocking dialog so the test can report the failure.
    guard = QTimer(window)
    guard.setSingleShot(True)
    guard.timeout.connect(lambda: [d.reject() for d in window.findChildren(QMessageBox)])
    guard.start(2000)
    try:
        window.driver.refresh(inspect=True)
        (dialog,) = [d for d in window.findChildren(QMessageBox) if d.isVisible()]
        assert not acknowledge.called
        drain(app, lambda: advances.call_count >= 3)
        assert dialog.isVisible()
        dialog.button(QMessageBox.StandardButton.Ok).click()
        acknowledge.assert_called_once_with("fixture")
    finally:
        guard.stop()
        window.close()
        window.deleteLater()


def test_usage_label_names_selected_model_and_switches_without_configuration_change(app, tmp_path):
    first = replace(CloudProfile.new("deepseek"), name="Translation")
    second = replace(CloudProfile.new("openai"), name="Classifier")
    catalog = CloudProfiles(tmp_path / "cloud.json")
    catalog.write((first, second))
    panel = CloudSettings(
        catalog=catalog,
        secrets=MemorySecrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        service = cloud_owner(panel)
        revision = service.revision
        asyncio.run(service.allowance(second.key).admit(second, clock=lambda: 1))
        service.allowance(second.key).record({"input_tokens": 25, "output_tokens": 3})
        panel.refresh_view()
        panel.table.selectRow(0)
        assert "Translation" in panel.usage.text() and "0/1000" in panel.usage.text()
        panel.table.selectRow(1)
        assert "Classifier" in panel.usage.text() and "1/1000" in panel.usage.text()
        assert "25 input / 3 output" in panel.usage.text()
        panel.table.selectRow(0)
        assert "Translation" in panel.usage.text() and "0/1000" in panel.usage.text()
        assert service.revision == revision
    finally:
        panel.close()
        panel.deleteLater()


def test_model_refresh_preserves_draft_and_route_caches_offline_and_keeps_old_list_on_error(
    app, tmp_path, monkeypatch
):
    secrets = MemorySecrets()
    catalog = CloudProfiles(tmp_path / "cloud.json")
    panel = CloudSettings(
        catalog=catalog,
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    calls = []

    def fetch(provider, secret, cancel):
        calls.append((provider, secret, threading.get_ident()))
        return (CatalogModel("future-chat", "Future chat"),)

    monkeypatch.setattr("dantetl.application.cloud_services.fetch_models", fetch)
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("openai"))
        editor.key.setText("fixture-new-key")
        previous = editor.model.currentData()
        editor.refresh_models.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert editor.model.currentData() == previous
        assert not panel.profiles and not secrets.values
        assert calls == [("openai", "fixture-new-key", calls[0][2])]
        assert calls[0][2] != threading.get_ident()
        editor.model.setCurrentIndex(editor.model.findData("future-chat"))
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        saved = panel.profiles[0]
        assert saved.model == "future-chat" and saved.decoding == "default"
        assert len(secrets.values) == 1
        assert cloud_owner(panel).catalog_cache.read("openai", saved.credential) == fetch(
            "openai", "", None
        )
        revision = cloud_owner(panel).revision
        panel.open_editor(saved)
        editor = panel.editor
        assert editor
        assert (
            editor.model.currentData() == "future-chat"
            and "Cached catalog" in editor.model_note.text()
        )

        def fail(*args):
            raise ValueError("Provider unavailable")

        monkeypatch.setattr("dantetl.application.cloud_services.fetch_models", fail)
        editor.refresh_models.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert "unavailable" in editor.status.text()
        assert editor.model.currentData() == saved.model
        assert panel.profiles == (saved,) and cloud_owner(panel).revision == revision
        monkeypatch.setattr("dantetl.application.cloud_services.fetch_models", lambda *args: ())
        editor.refresh_models.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert editor.model.currentData() == saved.model
        assert panel.profiles == (saved,) and cloud_owner(panel).revision == revision
        editor.reject()
        assert panel.editor is None
    finally:
        panel.close()
        panel.deleteLater()


def test_models_setup_saves_secret_off_gui_and_does_not_make_a_request(app, tmp_path):
    secrets = MemorySecrets()
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("deepseek"))
        editor.key.setText("fixture-secret")
        editor.model.setCurrentIndex(1)
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert len(panel.profiles) == 1
        profile = panel.profiles[0]
        assert panel.selected_profile() == profile and cloud_owner(panel).available(profile.key)
        assert secrets.writers and threading.get_ident() not in secrets.writers
        assert "fixture-secret" not in (tmp_path / "cloud.json").read_text()
        assert (
            panel.editor is None
            and cloud_owner(panel).allowance(profile.key).snapshot().requests == 0
        )
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("deepseek"))
        editor.key.setText("must-not-cross-providers")
        editor.provider.setCurrentIndex(editor.provider.findData("chatgpt"))
        assert not editor.key.text()
        editor.reject()
    finally:
        panel.close()
        panel.deleteLater()


@pytest.mark.parametrize("local", [False, True])
def test_cloud_tab_saves_on_next_and_reviews_only_selected_downloads(
    app, tmp_path, monkeypatch, local
):
    from PySide6.QtCore import Qt
    from remote_fixtures import MODEL, Server

    from dantetl.application.ollama_services import OllamaServices
    from dantetl.runtime.hardware_values import Adapter, CudaDevice, HardwareStatus
    from dantetl.runtime.model_store import model_catalog
    from dantetl.translation.ollama.values import OllamaConfig

    entered, release = threading.Event(), threading.Event()

    def install(plan, root, report, cancel):
        entered.set()
        assert release.wait(4)
        return tmp_path / "verified-runtime"

    monkeypatch.setattr("dantetl.runtime.first_run.run_setup", install)
    secrets = MemorySecrets()
    catalog = CloudProfiles(tmp_path / "profiles.json")
    service = cloud_service(
        catalog=catalog,
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    windows, hardware = Mock(), Mock()
    windows.list_targets.return_value = []
    hardware.snapshot.return_value = HardwareStatus(
        (Adapter("RTX", 0x10DE, cuda=CudaDevice(13000, 8, 6, "a" * 32)),), llama_device="cpu"
    )
    models = tuple(
        replace(b, directory=tmp_path / b.key, bundled_directory=None) for b in model_catalog()
    )
    window = MainWindow(windows, Mock(), models=models, hardware=hardware, cloud=service)
    server = Server()
    app_owner(window).ollama = OllamaServices(OllamaConfig(), lambda: None, server.transport)
    window.show()
    window.offer_first_run()
    guide = window._setup_assistant
    assert guide
    try:
        guide.next.click()
        assert guide.model.currentData() == "gemma"
        if not local:
            guide.model.setCurrentIndex(guide.model.findData(""))
        ollama_editor = guide.ollama
        assert ollama_editor
        if local:
            guide.sources.setCurrentWidget(ollama_editor)
            ollama_editor.find_button.click()
            drain(app, lambda: ollama_editor.list.count() == 1)
            ollama_editor.list.item(0).setCheckState(Qt.CheckState.Checked)
            server.info = {"capabilities": ["embedding"]}
        guide.sources.setCurrentWidget(guide.cloud_tab)
        drain(app, lambda: guide.cloud_tab.editor is not None)
        editor = guide.cloud_tab.editor
        assert editor and editor.isVisible() and not editor.isWindow()
        editor.provider.setCurrentIndex(editor.provider.findData("deepseek"))
        editor.model.setCurrentIndex(1)
        # An invalid draft stays on the form and cannot start downloads.
        guide.next.click()
        drain(app, lambda: not service.busy and bool(editor.status.text()))
        assert guide.cloud_tab.editor is editor and not service.profiles
        editor.key.setText("fixture-secret")
        if not local:
            editor.save.click()
            drain(
                app, lambda: bool(guide.cloud_tab.editor and guide.cloud_tab.editor is not editor)
            )
            second = guide.cloud_tab.editor
            assert second and not second.key.text() and not second.model.currentData()
            second.provider.setCurrentIndex(second.provider.findData("openai"))
            second.model.setCurrentIndex(1)
            second.key.setText("second-fixture")
        guide.sources.setCurrentIndex(0)
        guide.next.click()
        if local:
            drain(app, lambda: bool(app_owner(window).ollama.snapshot().error))
            assert ollama_editor.isVisible() and len(service.profiles) == 1
            assert not entered.is_set()
            server.info = {"capabilities": ["completion"]}
            guide.next.click()
        drain(app, lambda: guide.pages.currentIndex() in (2, 3))
        assert {p.provider for p in service.profiles} == (
            {"deepseek"} if local else {"deepseek", "openai"}
        )
        assert service.allowance(service.profiles[0].key).snapshot().requests == 0
        if local:
            assert app_owner(window).ollama.snapshot().config.models == (MODEL,)
            assert guide.pages.currentIndex() == 2
            guide.next.click()
        review = guide._review
        assert review and set(review.model_keys) == ({"ocr", "gemma"} if local else {"ocr"})
        assert not entered.is_set()
        if not local:
            assert not review.install_runtime
            guide.back.click()
            guide.sources.setCurrentWidget(guide.cloud_tab)
            assert guide.cloud_tab.choice.currentData() == next(
                p.key for p in service.profiles if p.provider == "openai"
            )
            assert not guide.cloud_tab.editor
        guide.skip.click()
        assert not guide.isVisible() and app_owner(window).values.setup_seen
    finally:
        release.set()
        window.close()
        window.deleteLater()


def test_cloud_only_needs_no_local_model_and_keeps_english_until_real_change(
    app, tmp_path, monkeypatch
):
    profile = CloudProfile.new("openai")
    catalog = CloudProfiles(tmp_path / "cloud.json")
    catalog.write((profile,))
    secrets = MemorySecrets()
    secrets.write(profile.key, "fixture")
    service = cloud_service(
        catalog=catalog,
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(
        windows, Mock(), scene_feed=True, cache_path=tmp_path / "cache", cloud=service
    )
    application = app_owner(window)
    try:
        drain(app, lambda: not application.cloud.busy)
        window.driver.timer.stop()
        window.driver.wakeup.close()
        panel = window_settings(window).models.cloud
        calls = []

        def handle(request):
            calls.append(request)
            return httpx2.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": "Settings"}}],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 7},
                },
            )

        monkeypatch.setattr(
            "dantetl.application.cloud_services.CloudBackend",
            lambda *args, **kwargs: CloudBackend(
                *args, **kwargs, transport=httpx2.MockTransport(handle)
            ),
        )
        window.routing.simple.setCurrentIndex(
            find_model(window.routing.simple, window.routing.resolve("cloud:" + profile.key))
        )
        assert not application.translation.lifecycle.snapshot().selected
        assert application.controls.assignment.active_models == ("cloud:" + profile.key,)
        block = TextBlock((TextTrack(1, "設定", Rect(10, 10, 100, 30)),))
        show_scene(window, (block,))
        assert isinstance(application.translation.lifecycle.publication, RoutedTranslationSession)
        assert tuple(
            lane.reference
            for lane in application.translation.lifecycle.publication.snapshot().lanes
        ) == ("cloud:" + profile.key,)
        drain(
            app,
            lambda: bool(
                application.translation.lifecycle.publication
                and application.translation.lifecycle.publication.snapshot().results
            ),
        )
        show_scene(window, (block,))
        before = application.view.english
        assert before and before[0].english == "Settings"
        assert profile.name in panel.usage.text()
        assert "1/1000 reserved requests" in panel.usage.text()
        assert "123 input / 7 output" in panel.usage.text()
        for _ in range(30):
            show_scene(window, (block,))
            assert application.view.english == before
        assert len(calls) == 1
        assert service.usage(profile.key).requests == 1
        session = application.translation.lifecycle.publication
        for _ in range(4):
            window.routing.advanced.setChecked(not window.routing.advanced.isChecked())
            show_scene(window, (block,))
            assert (
                application.translation.lifecycle.publication is session
                and application.view.english == before
            )
        assert len(calls) == 1
        panel.open_editor(profile)
        assert panel.editor
        panel.editor.name.setText("Unsaved rename")
        panel.editor.reject()
        show_scene(window, (block,))
        assert (
            application.translation.lifecycle.publication is session
            and application.view.english == before
        )
        assert len(calls) == 1
        window.routing.game_from_window.setChecked(False)
        window.routing.game.setText("Different game")
        window.routing.game.editingFinished.emit()
        assert not application.view.english
        assert application.translation.lifecycle.publication.snapshot().results == ()

        def translated_after_switch():
            show_scene(window, (block,))
            return service.usage(profile.key).output_tokens == 14

        drain(app, translated_after_switch)
        show_scene(window, (block,))
        assert "2/1000 reserved requests" in panel.usage.text()
        assert "246 input / 14 output" in panel.usage.text()
    finally:
        if application.translation.lifecycle.publication:
            application.translation.lifecycle.shutdown()
            drain(
                app,
                lambda: application.translation.lifecycle.finish_shutdown(),
            )
        window.close()
        window.deleteLater()


def test_simple_selection_declined_preparation_and_custom_disclosure_preserve_route(
    app, tmp_path, monkeypatch
):
    profile = CloudProfile.new("openai")
    catalog = CloudProfiles(tmp_path / "cloud.json")
    catalog.write((profile,))
    secrets = MemorySecrets()
    secrets.write(profile.key, "fixture")
    service = cloud_service(
        catalog=catalog,
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    preferences = PreferenceStore(tmp_path / "prefs.json")
    preferences._write(
        replace(Preferences(), model="none", translation_route="cloud", cloud_model=profile.key)
    )
    windows = Mock()
    windows.list_targets.return_value = []
    backend = BackendChoice(
        "Unprepared model", Mock(), "unprepared", preparation_id="fixture-requires-preparation"
    )
    window = MainWindow(
        windows,
        Mock(),
        (backend,),
        preferences_path=preferences.path,
        cache_path=tmp_path / "cache",
        cloud=service,
    )
    application = app_owner(window)
    try:
        drain(app, lambda: not application.cloud.busy)
        window.driver.timer.stop()
        assert window.routing.simple.currentData() == window.routing.resolve(
            model_reference(profile)
        )
        assert not window.routing.advanced.isChecked()
        window.routing.simple.setCurrentIndex(find_model(window.routing.simple, backend))
        app.activeModalWidget().button(QMessageBox.StandardButton.No).click()
        assert window.routing.selection == TranslationPlan(
            "cloud:" + profile.key, "cloud:" + profile.key
        )
        assert window.routing.simple.currentData() == window.routing.resolve(
            model_reference(profile)
        )
        assert (
            not application.translation.lifecycle.snapshot().preparing
            and not application.translation.lifecycle.publication
        )
        window.routing.advanced.setChecked(True)
        window.routing.short_enabled.setChecked(False)
        selected = window.routing.selection
        assert not selected.short_enabled and selected.long == "cloud:" + profile.key
        window.routing.advanced.setChecked(False)
        assert window.routing.selection == selected
        assert window.routing.simple.currentData() == window.routing.resolve(
            model_reference(profile)
        )
        window.routing.long_enabled.setChecked(False)
        assert (
            not window.routing.selection.active_models
            and application.translation.lifecycle.snapshot().selected is None
        )
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.parametrize("provider", ["openai", "gemini", "claude"])
def test_new_model_reuses_saved_api_connection_in_settings_after_reopen(app, tmp_path, provider):
    secrets = MemorySecrets()
    catalog = CloudProfiles(tmp_path / "cloud.json")
    accounts = ChatGPTConnections(ProtectedStore(tmp_path / "accounts"))
    panel = CloudSettings(catalog=catalog, secrets=lambda: secrets, connections=accounts)
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData(provider))
        editor.key.setText("remember-me")
        editor.model.setCurrentIndex(1)
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        first = panel.profiles[0]
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData(provider))
        editor.model.setCurrentIndex(1)
        editor.name.setText("Second model")
        assert editor.connection.currentData() == first.credential
        assert "Saved key" in editor.key.placeholderText() and not editor.key.text()
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert len(panel.profiles) == 2 and len(secrets.values) == 1
    finally:
        panel.close()
        panel.deleteLater()
    reopened = CloudSettings(catalog=catalog, secrets=lambda: secrets, connections=accounts)
    try:
        drain(app, lambda: cloud_owner(reopened)._worker is None)
        assert all(cloud_owner(reopened).available(p.key) for p in reopened.profiles)
        assert len(cloud_owner(reopened).api_accounts) == 1
    finally:
        reopened.close()
        reopened.deleteLater()


def test_two_chatgpt_models_share_persisted_account_without_signing_in_again(
    app, tmp_path, monkeypatch
):
    account = "account-" + "e" * 32
    connections = ChatGPTConnections(ProtectedStore(tmp_path / "accounts"))
    connections.store.write(
        account,
        {
            "access_token": "fixture",
            "scopes": [PLAN_SCOPE],
            "models": [["gpt-6-luna", "Luna"], ["gpt-6.1-sol", "Sol"]],
        },
    )

    def no_network(*args, **kwargs):
        raise AssertionError("A saved account must not sign in again to add models")

    monkeypatch.setattr(connections, "sign_in", no_network)
    monkeypatch.setattr(connections, "refresh_models", no_network)
    catalog = CloudProfiles(tmp_path / "cloud.json")
    panel = CloudSettings(catalog=catalog, secrets=MemorySecrets, connections=connections)
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        for index in (0, 1):
            panel.open_editor()
            editor = panel.editor
            assert editor
            editor.model.setCurrentIndex(index + 1)
            editor.save.click()
            drain(app, lambda: not cloud_owner(panel).busy)
        assert len(panel.profiles) == 2
        assert all(
            p.account == account and cloud_owner(panel).available(p.key) for p in panel.profiles
        )
    finally:
        panel.close()
        panel.deleteLater()
    reopened = CloudSettings(catalog=catalog, secrets=MemorySecrets, connections=connections)
    try:
        drain(app, lambda: cloud_owner(reopened)._worker is None)
        assert all(cloud_owner(reopened).available(p.key) for p in reopened.profiles)
    finally:
        reopened.close()
        reopened.deleteLater()


def test_chatgpt_refresh_populates_empty_picker_from_large_catalog_and_allows_save(
    app, tmp_path, monkeypatch
):
    from dantetl.translation.cloud.auth_work import AuthWork

    key = "account-" + "a" * 32
    connections = ChatGPTConnections(ProtectedStore(tmp_path / "accounts"))
    connections.store.write(
        key,
        {"access_token": "fixture", "scopes": [PLAN_SCOPE], "expires_at": time.time() + 600},
    )
    calls = []

    def respond(request):
        calls.append((request.url.path, threading.get_ident()))
        return httpx2.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "fixture-chat",
                        "display_name": "Fixture model",
                        "visibility": "list",
                        "metadata": "extra catalog information " * 32_768,
                    }
                ]
            },
        )

    transport = httpx2.MockTransport(respond)
    monkeypatch.setattr(
        "dantetl.application.cloud_services.AuthWork",
        lambda event: AuthWork(event, transport=transport),
    )
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=MemorySecrets,
        connections=connections,
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor and editor.model.count() == 1 and editor.model.currentData() is None
        editor.chatgpt.refresh_models.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert editor.model.currentData() is None
        editor.model.setCurrentIndex(editor.model.findData("fixture-chat"))
        assert calls == [("/v1/models", calls[0][1])]
        assert calls[0][1] != threading.get_ident()
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert len(panel.profiles) == 1 and panel.profiles[0].model == "fixture-chat"
        assert cloud_owner(panel).available(panel.profiles[0].key)
        assert len(calls) == 1  # Saving neither signs in again nor makes an inference request.
    finally:
        panel.close()
        panel.deleteLater()


def test_simple_local_preparation_commits_only_when_ready_and_reuses_worker(
    app, tmp_path, monkeypatch
):
    class Engine:
        cache_identity = "simple-prepare-fixture"
        batch_size = 1
        closed = False

        def translate(self, texts, *, target):
            return tuple("Local answer" for _ in texts)

        def close(self):
            self.closed = True

    engine = Engine()
    factory = Mock(return_value=engine)
    backend = BackendChoice(
        "Local", factory, "local-fixture", preparation_id="fixture-prepare-once"
    )
    monkeypatch.setattr("dantetl.runtime.model_store.mark_prepared", lambda *args: None)
    windows = Mock()
    windows.list_targets.return_value = []
    window = MainWindow(windows, Mock(), (backend,), cache_path=tmp_path / "cache")
    application = app_owner(window)
    try:
        drain(app, lambda: not application.cloud.busy)
        window.driver.timer.stop()
        previous = window.routing.selection
        window.routing.simple.setCurrentIndex(find_model(window.routing.simple, backend))
        app.activeModalWidget().button(QMessageBox.StandardButton.Yes).click()
        assert (
            application.translation.lifecycle.snapshot().proposed == backend
            and window.routing.selection == previous
        )
        window.driver.refresh(inspect=True)
        prepared = application.translation.lifecycle.publication
        assert prepared
        drain(app, lambda: prepared.snapshot().state.value == "ready")
        window.driver.refresh(inspect=True)
        assert (
            not app_owner(window).translation.lifecycle.snapshot().preparing
            and window.routing.selection
            == TranslationPlan("local:local-fixture", "local:local-fixture")
        )
        assert (
            application.translation.lifecycle.publication is prepared
            and application.translation.lifecycle.snapshot().standby
            and not engine.closed
        )
        assert factory.call_count == 1
    finally:
        if application.translation.lifecycle.publication:
            application.translation.lifecycle.shutdown()
            drain(
                app,
                lambda: application.translation.lifecycle.finish_shutdown(),
            )
        window.close()
        window.deleteLater()


def test_mixed_route_prepares_assigned_gemma_once_and_restores_deepseek(app, tmp_path, monkeypatch):
    from dantetl.runtime import model_store

    monkeypatch.setattr(model_store, "state_root", lambda: tmp_path)
    cloud = CloudProfile.new("deepseek")
    catalog = CloudProfiles(tmp_path / "cloud.json")
    catalog.update(cloud)
    secrets = MemorySecrets()
    secrets.values[cloud.key] = "fixture"
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock(return_value=Mock(cache_identity="fixture", batch_size=1))
    gemma = BackendChoice("Gemma", factory, "gemma", preparation_id="test-runtime")
    plan = TranslationPlan("local:gemma", model_reference(cloud))
    path = tmp_path / "preferences.json"
    PreferenceStore(path)._write(
        replace(Preferences(), advanced_routing=True, short_model=plan.short, long_model=plan.long)
    )
    for first_launch in (True, False):
        service = cloud_service(
            catalog=catalog,
            secrets=lambda: secrets,
            connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
        )
        window = MainWindow(windows, Mock(), (gemma,), cloud=service, preferences_path=path)
        owner = app_owner(window)
        try:
            assert window.routing.selection == plan
            if first_launch:
                assert window.onboarding.action.text() == "Prepare model"
                window.onboarding.action.click()
                app.activeModalWidget().button(QMessageBox.StandardButton.Yes).click()
                drain(app, lambda: model_store.is_prepared(gemma.preparation_id))
            # A recovered setup intent must leave the cloud assignment in place.
            owner.setup_completed("gemma", None)
            window.driver.refresh(inspect=True)
            assert window.routing.selection == plan and not owner.values.setup_model
            assert not owner.notices().model_warning
            assert factory.call_count == 1
        finally:
            window.close()
            drain(
                app,
                lambda owner=owner: (
                    owner.preferences.idle and not owner.cloud.busy and owner.inventory.finished
                ),
            )
            window.deleteLater()


def test_unconfigured_cloud_route_cannot_silently_use_local_or_start_network(
    app, tmp_path, monkeypatch
):
    windows = Mock()
    windows.list_targets.return_value = []
    local_factory = Mock(side_effect=AssertionError("Unexpected local fallback"))
    prefs = PreferenceStore(tmp_path / "prefs.json")
    missing = "cloud:cloud-" + "f" * 32
    prefs._write(replace(Preferences(), short_model=missing, long_model=missing))
    window = MainWindow(
        windows,
        Mock(),
        (BackendChoice("Local", local_factory, "test"),),
        scene_feed=True,
        cache_path=tmp_path / "cache",
        preferences_path=prefs.path,
    )
    application = app_owner(window)
    try:
        drain(app, lambda: not application.cloud.busy)
        window.driver.timer.stop()
        show_scene(window)
        assert not application.translation.lifecycle.publication and not local_factory.called
        assert "unavailable" in window.notice.text()
        window.onboarding.action.click()
        assert window._settings.models.category.currentData() == "cloud"
        assert window.routing.selection.active_models == (missing,)
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.parametrize("previously_enabled", [False, True])
def test_history_experimental_opt_in_persists_and_keeps_translation_independent(
    app, tmp_path, previously_enabled
):
    windows = Mock()
    windows.list_targets.return_value = []
    factory = Mock(side_effect=AssertionError("No capture: must not load classifier"))
    prefs = PreferenceStore(tmp_path / "history-prefs.json")
    prefs._write(
        replace(Preferences(), translation=False, model="none", history_enabled=previously_enabled)
    )
    choices = (
        BackendChoice("Instruction model", factory, "fixture", classifies=True),
        BackendChoice("Translation only", factory, "gemma-e4b"),
    )
    available: list[tuple[BackendChoice, ...]] = [choices]
    window = MainWindow(
        windows,
        Mock(),
        choices,
        cache_path=tmp_path / "cache",
        preferences_path=prefs.path,
        backend_provider=lambda: available[0],
    )
    application = app_owner(window)
    try:
        drain(app, lambda: not application.cloud.busy)
        panel = window.history_panel
        settings = window_settings(window)
        assert "Experimental" in [settings.tabs.tabText(i) for i in range(settings.tabs.count())]
        assert settings.history_enabled.isChecked() is previously_enabled
        assert panel.model.isEnabled() is previously_enabled
        assert window.history_button.isHidden() is not previously_enabled
        assert panel.model.findData("local:gemma-e4b") == -1
        assignment = application.translation.plan
        settings.history_enabled.setChecked(True)
        assert not window.history_button.isHidden()
        window.history_button.click()
        assert panel.isVisible() and "Experimental" in panel.windowTitle()
        panel.model.setCurrentIndex(panel.model.findData("local:fixture"))
        assert application.live.history_enabled
        assert "separate local model" in panel.model_notice.text()
        panel.hide()
        assert application.live.history_enabled  # closing reader does not disable collection
        drain(app, lambda: application.preferences.idle)
        saved = PreferenceStore(prefs.path).initial
        assert saved.history_enabled and saved.history_model == "local:fixture"
        available[0] = ()
        window.application.refresh_models()
        drain(app, lambda: not application.inventory.snapshot().data.backends)
        assert panel.model_key == "local:fixture" and "available classifier" in panel.status.text()
        window.history_button.click()
        assert panel.isVisible()
        settings.history_enabled.setChecked(False)
        assert not panel.isVisible() and window.history_button.isHidden()
        assert not panel.model.isEnabled() and not application.live.history_enabled
        assert panel.model_key == "local:fixture"
        assert application.translation.plan == assignment and not application.values.translation
        drain(app, lambda: application.preferences.idle)
        assert not PreferenceStore(prefs.path).initial.history_enabled
        assert not factory.called
    finally:
        window.close()
        window.deleteLater()


def test_edit_cancel_and_failed_save_preserve_committed_model_and_shared_key(
    app, tmp_path, monkeypatch
):
    secrets = MemorySecrets()
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("openai"))
        editor.key.setText("original-key")
        editor.model.setCurrentIndex(1)
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        original = panel.profiles[0]
        unchanged = Mock()
        panel.changed.connect(unchanged)
        panel.open_editor(original)
        editor = panel.editor
        assert editor
        editor.name.setText("Unsaved")
        editor.key.setText("replacement-key")
        editor.rate.setValue(333)
        editor.advanced.setChecked(True)
        editor.reject()
        assert panel.editor is None and cloud_owner(panel).catalog.read() == (original,)
        assert list(secrets.values.values()) == ["original-key"]
        assert not unchanged.called
        panel.open_editor(original)
        editor = panel.editor
        assert editor and editor.name.text() == original.name
        assert editor.rate.value() == original.requests_per_minute
        assert not editor.advanced.isChecked() and not editor.key.text()
        editor.name.setText("Keep this draft")
        editor.key.setText("replacement-key")
        monkeypatch.setattr(
            cloud_owner(panel).catalog, "write", Mock(side_effect=OSError("Disk full fixture"))
        )
        editor.save.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert panel.editor is editor and editor.name.text() == "Keep this draft"
        assert "Disk full" in editor.status.text()
        assert cloud_owner(panel).catalog.read() == (original,)
        assert list(secrets.values.values()) == ["original-key"]
        assert not unchanged.called
        editor.reject()
    finally:
        panel.close()
        panel.deleteLater()


def test_test_connection_is_explicit_and_does_not_save_draft_or_credentials(
    app, tmp_path, monkeypatch
):
    backend = Mock()
    backend.translate.return_value = ("Critical Rate",)
    factory = Mock(return_value=backend)
    monkeypatch.setattr("dantetl.application.cloud_services.CloudBackend", factory)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    secrets = MemorySecrets()
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("deepseek"))
        editor.key.setText("unsaved-secret")
        target = TARGET_LANGUAGES["pt-PT"]
        cloud_owner(panel).test_target = target
        editor.model.setCurrentIndex(1)
        editor.test.click()
        assert not factory.called
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        editor.test.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert factory.call_args.args[1] == "unsaved-secret"
        backend.translate.assert_called_once_with(("会心率",), target=target)
        backend.close.assert_called_once()
        assert target.name in editor.status.text()
        assert (
            not panel.profiles
            and not cloud_owner(panel).catalog.path.exists()
            and not secrets.values
        )
        assert editor.key.text() == "unsaved-secret"
        editor.reject()
        assert not cloud_owner(panel)._allowances
    finally:
        panel.close()
        panel.deleteLater()


def test_cancel_dialog_drains_test_cleanup_before_releasing_editor(app, tmp_path, monkeypatch):
    entered, cancelled, cleanup, release = (threading.Event() for _ in range(4))

    class SlowTest:
        def translate(self, texts, *, target):
            entered.set()
            assert cancelled.wait(3)
            return ("Cancelled",)

        def cancel_pending(self):
            cancelled.set()

        def close(self):
            cleanup.set()
            assert release.wait(3)

    monkeypatch.setattr(
        "dantetl.application.cloud_services.CloudBackend", lambda *args, **kwargs: SlowTest()
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=MemorySecrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.open_editor()
        editor = panel.editor
        assert editor
        editor.provider.setCurrentIndex(editor.provider.findData("deepseek"))
        editor.key.setText("fake-key")
        editor.model.setCurrentIndex(1)
        editor.test.click()
        drain(app, entered.is_set)
        editor.reject()
        drain(app, cleanup.is_set)
        assert panel.editor is editor and panel.busy and not editor.key.text()
        assert not panel.add.isEnabled()
        release.set()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert panel.editor is None and not panel.profiles
    finally:
        cancelled.set()
        release.set()
        drain(app, lambda: not cloud_owner(panel).busy)
        panel.close()
        panel.deleteLater()


def test_remove_model_keeps_shared_connection_for_other_models(app, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    secrets = MemorySecrets()
    panel = CloudSettings(
        catalog=CloudProfiles(tmp_path / "cloud.json"),
        secrets=lambda: secrets,
        connections=ChatGPTConnections(ProtectedStore(tmp_path / "accounts")),
    )
    try:
        drain(app, lambda: not cloud_owner(panel).busy)
        for index in (0, 1):
            panel.open_editor()
            editor = panel.editor
            assert editor
            editor.provider.setCurrentIndex(editor.provider.findData("openai"))
            editor.model.setCurrentIndex(index + 1)
            if not index:
                editor.key.setText("shared-secret")
            editor.save.click()
            drain(app, lambda: not cloud_owner(panel).busy)
        assert panel.table.rowCount() == 2
        panel.table.selectRow(0)
        panel.remove.click()
        drain(app, lambda: not cloud_owner(panel).busy)
        assert panel.table.rowCount() == 1 and len(secrets.values) == 1
        assert cloud_owner(panel).available(panel.profiles[0].key)
        panel.open_editor(panel.profiles[0])
        editor = panel.editor
        assert editor and "Key saved" in editor.badge.text()
        editor.reject()
    finally:
        panel.close()
        panel.deleteLater()
