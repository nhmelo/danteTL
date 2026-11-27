import hashlib

import pytest

from dantetl.application import model_factories as app
from dantetl.application.model_advice import RECOMMENDED_TRANSLATION
from dantetl.application.model_discovery import inspect_models, translation_backends
from dantetl.runtime import llama as llama_runtime
from dantetl.runtime import model_store
from dantetl.runtime import ocr as ocr_runtime
from dantetl.runtime.assets import ModelAsset


def test_recommended_catalog_and_backend_order_do_not_mix_recognition(tmp_path, monkeypatch):
    from dataclasses import replace

    bundles = tuple(
        replace(b, directory=tmp_path / b.key, assets=(), bundled_directory=None)
        for b in model_store.model_catalog()
    )
    assert bundles[0].key == RECOMMENDED_TRANSLATION == "gemma"
    assert [b.key for b in bundles if b.category == "recognition"] == ["ocr"]
    monkeypatch.setattr(model_store, "model_catalog", lambda: bundles)
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: "cpu")
    monkeypatch.setattr(llama_runtime, "supports_plamo", lambda: True)
    choices = translation_backends()
    assert choices[0].preference_key == "gemma" and "Recommended" in choices[0].name
    assert all(c.preference_key != "ocr" for c in choices)
    assert {"gemma", "gemma-e4b", "qwen35-4b", "plamo-0.8b"} <= {c.preference_key for c in choices}
    for choice in choices:
        if choice.preference_key in ("plamo-2b", "plamo-0.8b"):
            assert choice.output_codes and "fr" in choice.output_codes
            assert choice.output_codes.isdisjoint({"pt-PT", "pt-BR"})


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    asset = ModelAsset("fixture", "pinned", "weights", hashlib.sha256(b"good").hexdigest(), 4)
    bundles = tuple(
        model_store.ModelBundle(key, key, tmp_path / key, (asset,), "", language_model=key != "ocr")
        for key in ("ocr", "gemma", "gemma-e4b", "qwen35-4b")
    )
    monkeypatch.setattr(model_store, "model_catalog", lambda: bundles)
    return {bundle.key: bundle for bundle in bundles}


def test_missing_engine_keeps_downloaded_gemma_visible_with_setup_reason(catalog, monkeypatch):
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: None)
    assert not translation_backends()
    for key in ("ocr", "gemma"):
        catalog[key].directory.mkdir()
        (catalog[key].directory / "weights").write_bytes(b"good")
    choices = translation_backends()
    assert len(choices) == 1 and "Gemma" in choices[0].name
    assert "runtime setup needed" in choices[0].name
    assert "Acceleration" in choices[0].unavailable_reason
    assert (
        "not installed"
        in inspect_models((catalog["gemma"],), include_imports=False).files[0].status
    )
    assert not choices[0].preparation_id


@pytest.mark.parametrize("device", ["cpu", "cuda", "vulkan"])
def test_available_runtime_makes_existing_weights_usable_without_download(
    catalog, monkeypatch, device
):
    catalog["gemma"].directory.mkdir()
    (catalog["gemma"].directory / "weights").write_bytes(b"good")
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: device)
    choice = translation_backends()[0]
    assert choice.key == f"gemma-{device}" and choice.preparation_id
    assert not choice.unavailable_reason
    assert bool(choice.performance_notice) == (device == "cpu")
    if device == "cpu":
        assert "CPU" in choice.name and "GPU" not in choice.name
    assert (
        "Runtime available"
        in inspect_models((catalog["gemma"],), include_imports=False).files[0].status
    )


def test_picker_and_manager_share_one_runtime_and_registry_inspection(catalog, monkeypatch):
    from unittest.mock import Mock

    from dantetl.translation.models import gguf

    device = Mock(side_effect=["cpu", None])
    registry = Mock(side_effect=[(), ValueError("second read")])
    monkeypatch.setattr(llama_runtime, "installed_device", device)
    monkeypatch.setattr(gguf, "read_registry", registry)
    catalog["gemma"].directory.mkdir()
    (catalog["gemma"].directory / "weights").write_bytes(b"good")
    data = inspect_models(tuple(catalog.values()))
    assert data.device == "cpu" and data.backends[0].key == "gemma-cpu"
    assert next(f for f in data.files if f.key == "gemma").device == "cpu"
    assert not data.import_problem
    device.assert_called_once()
    registry.assert_called_once()


def test_incomplete_models_stay_absent(catalog, monkeypatch):
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: None)
    catalog["gemma"].directory.mkdir()
    (catalog["gemma"].directory / "weights").write_bytes(b"bad")
    assert not translation_backends()


def test_broken_optional_registry_keeps_healthy_inventory_and_recovers(
    catalog, tmp_path, monkeypatch
):
    from ownership_fixtures import settle

    from dantetl.application.model_discovery import inspect_models
    from dantetl.application.model_inventory import ModelInventory
    from dantetl.translation.models import gguf

    registry = tmp_path / "imported-models.json"
    broken = b'{"version": 1, "models": broken'
    registry.write_bytes(broken)
    monkeypatch.setattr(gguf, "registry_path", lambda: registry)
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: "cpu")
    monkeypatch.setattr(llama_runtime, "supports_plamo", lambda: True)
    for key in ("ocr", "gemma"):
        catalog[key].directory.mkdir()
        (catalog[key].directory / "weights").write_bytes(b"good")
    owner = ModelInventory(lambda: inspect_models(tuple(catalog.values())), lambda _: None)
    try:
        owner.advance(0)
        settle(owner, lambda: owner.finished)
        snapshot = owner.snapshot()
        assert snapshot.loaded and snapshot.data.ocr_ready
        assert [choice.preference_key for choice in snapshot.data.backends] == ["gemma"]
        assert snapshot.data.import_problem and "imported models" in snapshot.problem
        assert registry.read_bytes() == broken
        registry.write_text('{"version":1,"models":[]}', encoding="utf-8")
        owner.invalidate()
        owner.advance(2)
        settle(owner, lambda: owner.finished, now=2)
        assert owner.snapshot().loaded and not owner.snapshot().problem
        assert owner.snapshot().data.backends == snapshot.data.backends
    finally:
        owner.close()
        settle(owner, lambda: owner.finished)


def test_automatic_ocr_depends_on_runtime_not_downloaded_meiki(catalog, monkeypatch):
    catalog["ocr"].directory.mkdir()
    (catalog["ocr"].directory / "weights").write_bytes(b"good")
    monkeypatch.setattr(app, "gpu_runtime_installed", lambda name="onnxruntime-gpu": False)
    monkeypatch.setattr(ocr_runtime, "gpu_runtime_installed", lambda name="onnxruntime-gpu": False)
    assert len(app.ocr_backends()) == 1 and app.ocr_backends()[0].name.startswith("CPU")
    assert (
        "Acceleration" in inspect_models((catalog["ocr"],), include_imports=False).files[0].status
    )
    monkeypatch.setattr(
        app, "gpu_runtime_installed", lambda name="onnxruntime-gpu": name == "onnxruntime-gpu"
    )
    assert app.ocr_backends()[0].name.startswith("Automatic")


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "vulkan"])
def test_curated_choices_keep_identity_across_runtime_changes(catalog, monkeypatch, device):
    from dantetl.application.preferences import decode_preferences

    monkeypatch.setattr(llama_runtime, "installed_device", lambda: device)
    for key in ("gemma", "gemma-e4b", "qwen35-4b"):
        catalog[key].directory.mkdir()
        (catalog[key].directory / "weights").write_bytes(b"good")
    choices = translation_backends()
    assert {b.preference_key for b in choices} == {"gemma", "gemma-e4b", "qwen35-4b"}
    for choice in choices:
        assert bool(choice.unavailable_reason) == (device is None)
        assert bool(choice.preparation_id) == (device is not None)
        saved = decode_preferences(
            {
                "version": 1,
                "model": choice.preference_key,
                "setup_model": choice.preference_key,
            }
        )
        assert saved.model == saved.setup_model == choice.preference_key


@pytest.mark.parametrize("supported", [False, True])
def test_downloaded_plamo_stays_visible_but_old_engine_needs_update(
    tmp_path, monkeypatch, supported
):
    from dataclasses import replace

    from dantetl.translation.models.curated import MODELS

    models = []
    for definition in MODELS:
        if not definition.key.startswith("plamo-"):
            continue
        bundle = replace(
            definition.bundle(),
            directory=tmp_path / definition.key,
            assets=(ModelAsset("fixture", "pin", "weights", "hash", 4),),
        )
        bundle.directory.mkdir()
        (bundle.directory / "weights").write_bytes(b"good")
        models.append(bundle)
    monkeypatch.setattr(model_store, "model_catalog", lambda: tuple(models))
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: "cpu")
    monkeypatch.setattr(llama_runtime, "supports_plamo", lambda: supported)
    choices = translation_backends()
    assert {c.selection_id for c in choices} == {"plamo-2b", "plamo-0.8b"}
    for choice in choices:
        assert bool(choice.preparation_id) is supported
        assert bool(choice.unavailable_reason) is not supported
        from dantetl.application.preferences import decode_preferences

        saved = decode_preferences(
            {"version": 1, "model": choice.preference_key, "setup_model": choice.preference_key}
        )
        assert saved.model == saved.setup_model == choice.preference_key
    for bundle in models:
        assert (
            "predates" in inspect_models((bundle,), include_imports=False).files[0].status
        ) is not supported
