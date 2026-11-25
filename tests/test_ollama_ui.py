"""Settings discovers and verifies models on the selected Ollama server."""

from dataclasses import replace

from PySide6.QtCore import Qt
from test_ollama import MODEL, REMOTE, OllamaConfig, OllamaServices, Server, settle

from dantetl.translation.ollama.values import OllamaListing
from dantetl.ui.ollama_settings import OllamaSettings


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
