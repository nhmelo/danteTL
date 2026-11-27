"""Both setup entry points share one editor and require an explicit route selection."""

from dataclasses import replace

import pytest
from ownership_fixtures import configured
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from remote_fixtures import MODEL, REMOTE, Server, settle
from translation_fixtures import until

from dantetl.application.ollama_services import OllamaServices
from dantetl.application.preference_values import Preferences
from dantetl.translation.ollama.values import DEFAULT_SERVER, OllamaConfig, OllamaListing
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.selection import AssignmentChange
from dantetl.ui.cloud_settings import CloudSettings
from dantetl.ui.ollama_settings import OllamaSettings
from dantetl.ui.routing_controls import RoutingControls
from dantetl.ui.setup_assistant import SetupAssistant


@pytest.mark.parametrize("ocr_present", [True, False])
def test_welcome_configures_picker_and_reviews_only_missing_ocr(app, tmp_path, ocr_present):
    replay = configured(tmp_path)
    data = replay.inventory.snapshot().data
    replay.discover(
        replace(
            data,
            files=tuple(
                replace(f, present=ocr_present, included=ocr_present) if f.key == "ocr" else f
                for f in data.files
            ),
        )
    )
    server = Server()
    server.rows.append({"name": "other-model:3b", "digest": "b" * 64})
    service = OllamaServices(OllamaConfig(), lambda: None, server.transport)
    replay.app.ollama = service
    parent = QWidget()
    settings = OllamaSettings(service)
    guide = SetupAssistant(
        replay.setup, None, parent, cloud=CloudSettings(replay.cloud, parent), ollama=service
    )
    try:
        guide.show()
        guide.next.click()
        assert guide.sources.isVisible() and guide.sources.tabText(0) == "Local models"
        assert guide.model.isVisible() and guide.sources.tabText(1) == "Ollama API"
        editor = guide.ollama
        assert editor
        guide.sources.setCurrentWidget(editor)
        assert (
            editor.isVisible()
            and not guide.model.isVisible()
            and not editor.save_button.isEnabled()
            and guide.next.isVisible()
            and editor.address.text() == DEFAULT_SERVER
        )
        assert not server.calls
        editor.address.setText(REMOTE)
        editor.find_button.click()
        settle(service)
        guide.refresh_view()
        assert editor.list.count() == 2
        editor.list.item(0).setCheckState(Qt.CheckState.Checked)
        guide.sources.setCurrentIndex(0)
        assert guide.model.isVisible() and guide.next.isVisible() and not replay.operations
        guide.sources.setCurrentWidget(editor)
        assert editor.list.item(0).checkState() == Qt.CheckState.Checked
        assert guide.next.text() == "Next"
        guide.sources.setCurrentIndex(0)
        complete = []
        guide.completed.connect(lambda key, *_: complete.append(key))
        guide.next.click()

        def saved():
            replay.app.advance(0)
            return service.snapshot().saved

        until(saved)
        assert replay.app.values.ollama == OllamaConfig(REMOTE, (OllamaListing(MODEL),))
        settings.refresh_view()
        guide.refresh_view()
        assert settings.list.count() == 2
        if ocr_present:
            assert not guide.isVisible() and not replay.operations and complete == [""]
        else:
            assert guide.review.isVisible()
            guide.back.click()
            assert editor.isVisible() and editor.list.item(0).checkState() == Qt.CheckState.Checked
            guide.next.click()
            guide.refresh_view()
            review = replay.setup.reviewed
            assert review and review.model_keys == ("ocr",) and not review.install_runtime
            assert not replay.operations
            guide.next.click()
            assert replay.setup.busy
            replay.operations[-1].finish()
            replay.setup.advance()
            guide.refresh_view()
            guide.next.click()
            assert complete == [""]
        options = replay.app.controls.options
        assert [o.name for o in options if o.provider == "ollama"] == [MODEL.display_name]
        picker = RoutingControls(Preferences())
        picker.set_models(options, TranslationPlan())
        ref = MODEL.reference(REMOTE)
        option = next(o for o in options if o.reference == ref)
        assert not replay.translation.plan.active_models and picker.simple.currentIndex() == -1
        for combo in (picker.simple, picker.short_model, picker.long_model):
            index = combo.findData(option)
            assert index > 0 and combo.itemText(index - 1) == "Ollama API"
        requests = []
        picker.model_requested.connect(requests.append)
        picker.simple.setCurrentIndex(picker.simple.findData(option))
        assert requests == [AssignmentChange(TranslationPlan(ref, ref), ref)]
        assert replay.translation.assign(requests[0]) == "committed"
        replay.translation.update_ollama(
            OllamaConfig(DEFAULT_SERVER, (OllamaListing(MODEL),)), frozenset()
        )
        assert replay.translation.resolve(ref) is None and replay.translation.plan.short == ref
    finally:
        guide.close()
        replay.close()


def test_settings_reopens_saved_tags_and_refreshes_their_display_metadata(app):
    model = replace(MODEL, name="qwen2.5:1.5b")
    old = OllamaConfig.decode(
        {"server": REMOTE, "models": [{"name": model.name, "digest": model.digest}]}
    )
    server = Server()
    server.rows = [
        {
            "name": model.name,
            "digest": model.digest,
            "size": 986000000,
            "details": {"parameter_size": "1.5B", "quantization_level": "Q4_K_M"},
        },
        {"name": "vision:latest", "digest": "b" * 64},
    ]
    service = OllamaServices(old, lambda: None, server.transport)
    editor = OllamaSettings(service)
    assert not server.calls and editor.address.text() == REMOTE
    assert editor.list.count() == 1 and editor.list.item(0).checkState() == Qt.CheckState.Checked
    editor.find_button.click()
    settle(service)
    editor.refresh_view()
    first = editor.list.item(0)
    assert first.text() == "Qwen 2.5 · 1.5B" and first.checkState() == Qt.CheckState.Checked
    assert all(s in first.toolTip() for s in (model.name, "1.5B", "Q4_K_M", "986 MB"))
    assert editor.list.item(1).checkState() == Qt.CheckState.Unchecked
    assert editor.save_button.isEnabled()
    editor.save_button.click()
    settled = settle(service)
    editor.refresh_view()
    assert settled.config.listings == (OllamaListing(model, "1.5B", "Q4_K_M", 986000000),)
    assert editor.list.count() == 2 and not editor.save_button.isEnabled()
    editor.list.item(0).setCheckState(Qt.CheckState.Unchecked)
    editor.save_button.click()
    assert not settle(service).config.models
