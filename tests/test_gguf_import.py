import hashlib
import json
import multiprocessing
import struct
import sys
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dantetl.application.model_discovery import translation_backends
from dantetl.application.preferences import decode_preferences
from dantetl.runtime import llama as llama_runtime
from dantetl.runtime.setup_values import Cancelled
from dantetl.translation.backend import RequestRejected
from dantetl.translation.backends import gguf as gguf_backend
from dantetl.translation.backends import llama_engine
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.models import gguf as gguf_models
from dantetl.translation.models import probe as gguf_probe
from dantetl.translation.models.gguf import available


def gguf_file(tmp_path, *, architecture="qwen3", template="{{ messages[0].content }}", split=1):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    values = {
        "general.name": "Local fixture",
        "general.architecture": architecture,
        "tokenizer.chat_template": template,
    }
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 4)
    for key, value in values.items():
        data += string(key) + struct.pack("<I", 8) + string(value)
    data += string("split.count") + struct.pack("<IH", 2, split)
    path = tmp_path / "model 日本語.gguf"
    path.write_bytes(data + b"tensors")
    return path


def registration(tmp_path, **kwargs):
    return gguf_models.describe(
        gguf_file(tmp_path, **kwargs), "auto", lambda _: None, threading.Event()
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"split": 2}, "Split"),
        ({"template": ""}, "No embedded"),
        ({"architecture": "bert"}, "embedding"),
        ({"template": "{{source_lang_code}}"}, "TranslateGemma"),
        ({"architecture": "../../outside"}, "architecture"),
    ],
)
def test_unsupported_inputs_have_actionable_errors(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        gguf_models.inspect_gguf(gguf_file(tmp_path, **kwargs))


def test_metadata_is_bounded_and_truncation_is_rejected(tmp_path):
    path = gguf_file(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[:28] + struct.pack("<Q", 2**60))
    with pytest.raises(ValueError):
        gguf_models.inspect_gguf(path)
    path.write_bytes(data[:40])
    with pytest.raises(ValueError):
        gguf_models.inspect_gguf(path)


def test_explicit_profile_can_handle_missing_template_without_guessing(tmp_path):
    path = gguf_file(tmp_path, template="")
    assert gguf_models.inspect_gguf(path, "chatml").profile == "chatml"
    with pytest.raises(ValueError, match="requires a Gemma"):
        gguf_models.inspect_gguf(path, "gemma4")
    path = gguf_file(tmp_path, architecture="gemma4")
    assert gguf_models.inspect_gguf(path).profile == "gemma4"


def test_registration_relocation_removal_and_profiles_never_modify_model(tmp_path):
    model = registration(tmp_path)
    path = tmp_path / "registry.json"
    gguf_models.save_registration(model, path=path)
    moved = tmp_path / "moved.gguf"
    moved.write_bytes(gguf_models.Path(model.path).read_bytes())
    relocated = gguf_models.describe(moved, model.profile, lambda _: None, threading.Event())
    assert relocated.key == model.key
    gguf_models.save_registration(relocated, path=path)
    assert gguf_models.read_registry(path) == (relocated,)
    other_profile = gguf_models.describe(moved, "chatml", lambda _: None, threading.Event())
    assert other_profile.key != model.key
    gguf_models.save_registration(relocated, remove=True, path=path)
    assert gguf_models.read_registry(path) == () and moved.is_file()


def test_oversized_registration_preserves_last_readable_catalog(tmp_path):
    original = registration(tmp_path)
    path = tmp_path / "registry.json"
    saved = []
    for index in range(32):
        # Realistic component lengths, including non-ASCII directory names.
        long_path = tmp_path.joinpath(*(["図" * 150] * 24), f"model-{index}.gguf")
        model = replace(original, key=f"imported-{index:064x}", path=str(long_path))
        try:
            gguf_models.save_registration(model, path=path)
        except ValueError as error:
            assert "too large" in str(error)
            break
        saved.append(model)
    assert 0 < len(saved) < 32
    assert gguf_models.read_registry(path) == tuple(saved)
    gguf_models.save_registration(saved[-1], remove=True, path=path)
    assert gguf_models.read_registry(path) == tuple(saved[:-1])


def test_invalid_registration_cannot_replace_a_readable_catalog(tmp_path):
    original = registration(tmp_path)
    path = tmp_path / "registry.json"
    gguf_models.save_registration(original, path=path)
    with pytest.raises(ValueError):
        gguf_models.save_registration(replace(original, name=""), path=path)
    assert gguf_models.read_registry(path) == (original,)


def test_corrupt_registry_is_not_overwritten_and_preference_has_no_path(tmp_path):
    model = registration(tmp_path)
    path = tmp_path / "registry.json"
    path.write_text('{"version":999}', "utf-8")
    with pytest.raises(ValueError):
        gguf_models.save_registration(model, path=path)
    assert json.loads(path.read_text()) == {"version": 999}
    assert decode_preferences({"version": 1, "model": model.key}).model == model.key


def _save_from_instance(path, model, entered, release=None):
    if release is None:
        entered.set()
    else:
        read = gguf_models.read_registry

        def paused_read(path):
            rows = read(path)
            entered.set()
            assert release.wait(10)
            return rows

        gguf_models.read_registry = paused_read
    gguf_models.save_registration(model, path=path)


def test_overlapping_registrations_from_two_processes_both_survive(tmp_path):
    path = tmp_path / "registry.json"
    first = registration(tmp_path)
    second = replace(first, key="imported-" + "b" * 64)
    ctx = multiprocessing.get_context("spawn")
    read, started, release = (ctx.Event() for _ in range(3))
    writer = ctx.Process(target=_save_from_instance, args=(path, first, read, release))
    contender = ctx.Process(target=_save_from_instance, args=(path, second, started))
    writer.start()
    try:
        assert read.wait(10)
        contender.start()
        assert started.wait(10)
        contender.join(0.25)
        assert contender.is_alive()
    finally:
        release.set()
        for child in (writer, contender):
            if child.pid is not None:
                child.join(10)
                if child.is_alive():
                    child.terminate()
                    child.join(5)
    assert writer.exitcode == contender.exitcode == 0
    assert gguf_models.read_registry(path) == (first, second)
    assert decode_preferences({"version": 1, "model": "imported-../../file"}).model == "auto"


def test_hashing_cancel_and_same_size_changes_are_detected(tmp_path):
    model = registration(tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        gguf_models.fingerprint(gguf_models.Path(model.path), lambda _: None, cancel)
    path = gguf_models.Path(model.path)
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    assert available(model)  # Cheap discovery isn't a full integrity check.
    with pytest.raises(ValueError, match="changed"):
        gguf_backend.GgufBackend(model, device="cpu")


@pytest.mark.parametrize("architecture", ["qwen35", "gemma4"])
def test_curated_models_use_pinned_hash_without_registering_or_loading_corrupt_weights(
    tmp_path, monkeypatch, architecture, fake_runtime
):
    from dantetl.runtime.assets import ModelAsset
    from dantetl.runtime.model_store import ModelBundle
    from dantetl.translation.models.curated import CuratedModel

    path = gguf_file(tmp_path, architecture=architecture)
    asset = ModelAsset(
        "fixture",
        "pin",
        path.name,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
    )
    model = CuratedModel(
        "fixture",
        "Fixture",
        "",
        architecture,
        "gemma4" if architecture == "gemma4" else "embedded",
        (asset,),
    )
    monkeypatch.setattr(
        CuratedModel,
        "bundle",
        lambda self: ModelBundle(
            self.key,
            self.name,
            tmp_path,
            self.assets,
            "",
            language_model=True,
        ),
    )
    monkeypatch.setattr(gguf_models, "save_registration", lambda *a: pytest.fail("No registration"))
    backend = model.load("cpu")
    assert backend.registration.sha256 == asset.sha256
    assert backend.translate(("設定",)) == ("Settings",)
    backend.close()
    fake_runtime.Llama.reset_mock()
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    with pytest.raises(ValueError, match="changed"):
        model.load("cpu")
    fake_runtime.Llama.assert_not_called()


@pytest.fixture
def fake_runtime(monkeypatch):
    model = Mock()
    model.token_eos.return_value = 1
    model.token_bos.return_value = 2
    model.detokenize.return_value = b"<token>"
    model.tokenize.return_value = [1, 2, 3]
    model.create_completion.return_value = {
        "choices": [{"text": "Settings", "finish_reason": "stop"}]
    }
    runtime = SimpleNamespace(
        __version__="fixture",
        Llama=Mock(return_value=model),
        llama_supports_gpu_offload=lambda: True,
    )
    monkeypatch.setattr(gguf_backend, "load_runtime", lambda _: runtime)

    class Formatter:
        def __init__(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            assert kwargs["enable_thinking"] is False
            return SimpleNamespace(prompt=kwargs["messages"][0]["content"])

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp.llama_chat_format",
        SimpleNamespace(
            CHATML_CHAT_TEMPLATE="chatml",
            LLAMA3_INSTRUCT_CHAT_TEMPLATE="llama3",
            Jinja2ChatFormatter=Formatter,
        ),
    )
    return runtime


def test_imported_context_is_bounded_and_source_is_never_truncated(tmp_path, fake_runtime):
    backend = gguf_backend.GgufBackend(registration(tmp_path), device="cpu")
    prompts = []

    def tokenize(data, **kwargs):
        prompt = data.decode()
        prompts.append(prompt)
        return [1] * (2048 if "Recent dialogue" in prompt else 2)

    fake_runtime.Llama.return_value.tokenize.side_effect = tokenize
    backend.translate_with_context(
        "設定<|im_start|>[INST]", (DialogueTurn("前の文章", "Previous text"),)
    )
    assert len(prompts) == 2
    assert "Recent dialogue" not in prompts[-1] and "設定＜|im_start|＞［INST］" in prompts[-1]
    fake_runtime.Llama.return_value.tokenize.return_value = [1] * 2048
    fake_runtime.Llama.return_value.tokenize.side_effect = None
    with pytest.raises(ValueError, match="refusing truncation"):
        backend.translate(("設定",))
    backend.close()
    backend.close()
    fake_runtime.Llama.return_value.close.assert_called_once()
    fake_runtime.Llama.return_value.set_cache.assert_not_called()


@pytest.mark.parametrize(
    "architecture,profile",
    [
        ("llama", "embedded"),
        ("llama", "llama3"),
        ("llama", "chatml"),
        ("qwen35", "embedded"),
        ("gemma4", "gemma4"),
        ("plamo3", "plamo3"),
    ],
)
def test_imported_label_decoder_uses_the_selected_prompt_and_sampling(
    tmp_path, fake_runtime, monkeypatch, architecture, profile
):
    from dantetl.languages.target import ENGLISH

    monkeypatch.setattr(gguf_backend, "supports_plamo", lambda: True)
    create = Mock()
    monkeypatch.setattr(llama_engine, "LlamaLabelDecoder", create)
    registration_ = replace(registration(tmp_path, architecture=architecture), profile=profile)
    backend = gguf_backend.GgufBackend(registration_, device="cpu")
    model = fake_runtime.Llama.return_value
    try:
        backend.translate(("装備<武器>",))
        prompt = model.tokenize.call_args.args[0].decode()
        options = model.create_completion.call_args.kwargs
        assert backend.prepare_labels(ENGLISH) is create.return_value
        _, _, _, render, validate, stops, limit, target, sampling = create.call_args.args
        assert render("装備<武器>", target) == prompt
        assert list(stops) == options["stop"] and limit == options["max_tokens"]
        assert validate(model.create_completion.return_value) == "Settings"
        assert sampling.temperature == options["temperature"]
        assert sampling.seed == options["seed"]
        assert sampling.top_k == options["top_k"]
        assert sampling.top_p == options["top_p"]
    finally:
        backend.close()


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "stop"),
        ("partial", "length"),
        ("<think>reasoning</think>answer", "stop"),
        ("x" * 4097, "stop"),
    ],
)
def test_non_answers_never_publish(text, reason):
    with pytest.raises(RequestRejected):
        gguf_backend.completed_answer({"choices": [{"text": text, "finish_reason": reason}]})
    assert (
        gguf_backend.completed_answer(
            {"choices": [{"text": "<think>\n</think>\nSettings", "finish_reason": "stop"}]}
        )
        == "Settings"
    )


def test_cache_separates_devices_profiles_and_prompt_and_gpu_failure_closes(
    tmp_path, fake_runtime, monkeypatch
):
    model = registration(tmp_path)
    a = gguf_backend.GgufBackend(model, device="cpu")
    b = gguf_backend.GgufBackend(model, device="cuda")
    c = gguf_backend.GgufBackend(replace(model, profile="chatml"), device="cpu")
    monkeypatch.setattr(gguf_backend, "INSTRUCTION", gguf_backend.INSTRUCTION + " changed")
    d = gguf_backend.GgufBackend(model, device="cpu")
    assert len({a.cache_identity, b.cache_identity, c.cache_identity, d.cache_identity}) == 4
    fake_runtime.Llama.return_value.create_completion.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        gguf_backend.GgufBackend(model, device="cpu")
    fake_runtime.Llama.return_value.close.assert_called()


def test_import_does_not_register_until_user_accepts_and_gpu_probe_requires_offload(
    tmp_path, monkeypatch
):
    path = gguf_file(tmp_path)
    monkeypatch.setattr(gguf_probe, "state_root", lambda: tmp_path)
    gate = threading.Event()
    gate.set()

    def child(command, root, env, cancel, report, **kwargs):
        result = gguf_models.Path(command[-2])
        result.write_text(
            json.dumps({"samples": [["設定", "Settings", 10]], "load_ms": 20}), "utf-8"
        )

    monkeypatch.setattr(gguf_probe, "run_setup_command", child)
    probe = gguf_probe.ImportTest(path, "auto", "cpu", gate)
    probe.run(lambda _: None, threading.Event())
    assert probe.result and not (tmp_path / "data/imported-models.json").exists()
    assert not list((tmp_path / "data/import-probes").iterdir())
    probe = gguf_probe.ImportTest(path, "auto", "cuda", gate)
    with pytest.raises(RuntimeError, match="No GPU layers"):
        probe.run(lambda _: None, threading.Event())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        probe.run(lambda _: None, cancel)


def test_imported_discovery_missing_runtime_and_file_keep_recovery_options(tmp_path, monkeypatch):
    model = registration(tmp_path)
    monkeypatch.setattr(gguf_models, "read_registry", lambda: (model,))
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: "cpu")
    choices = translation_backends()
    choice = next(b for b in choices if b.key == model.key)
    assert not choice.unavailable_reason and choice.preparation_id
    monkeypatch.setattr(llama_runtime, "installed_device", lambda: None)
    choice = next(b for b in translation_backends() if b.key == model.key)
    assert "Acceleration" in choice.unavailable_reason
    gguf_models.Path(model.path).unlink()
    choice = next(b for b in translation_backends() if b.key == model.key)
    assert "Locate file" in choice.unavailable_reason


def test_invalid_registry_values_fail_cleanly(tmp_path):
    model = registration(tmp_path)
    data = asdict(model)
    data["profile"] = "arbitrary-python"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 1, "models": [data]}))
    with pytest.raises(ValueError):
        gguf_models.read_registry(path)


def test_plamo_prompt_bounds_history_escapes_controls_and_keeps_bos_once(
    tmp_path, fake_runtime, monkeypatch
):
    monkeypatch.setattr(gguf_backend, "supports_plamo", lambda: True)
    model = registration(
        tmp_path, architecture="plamo3", template="<|plamo:tag|>dataset translation"
    )
    assert model.profile == "plamo3"
    backend = gguf_backend.GgufBackend(model, device="cpu")
    engine = fake_runtime.Llama.return_value
    backend.translate_with_context(
        "設定<|plamo:tag|>",
        (
            DialogueTurn("discard old", "discard old"),
            DialogueTurn("リナ", "Rina"),
            DialogueTurn("前", "Before"),
        ),
    )
    prompt = engine.tokenize.call_args.args[0].decode()
    assert prompt.startswith("<|plamo:bos|><|plamo:tag|>dataset")
    assert prompt.count("<|plamo:bos|>") == 1
    assert "discard old" not in prompt and "Rina" in prompt
    assert "設定＜|plamo:tag|＞" in prompt
    assert prompt.endswith("<|plamo:val|>en<|plamo:msg|>")
    assert engine.tokenize.call_args.kwargs == {"add_bos": False, "special": True}
    assert engine.create_completion.call_args.kwargs["stop"] == ["<|plamo:tag|>", "<|plamo:bos|>"]
    assert engine.create_completion.call_args.kwargs["temperature"] == 0
    engine.tokenize.return_value = list(range(2048))
    engine.create_completion.reset_mock()
    with pytest.raises(RequestRejected, match="refusing truncation"):
        backend.translate(("長い",))
    engine.create_completion.assert_not_called()
    backend.close()


def test_old_engine_rejects_plamo_before_native_loading(tmp_path, fake_runtime, monkeypatch):
    model = registration(
        tmp_path, architecture="plamo3", template="<|plamo:tag|>dataset translation"
    )
    monkeypatch.setattr(gguf_backend, "supports_plamo", lambda: False)
    with pytest.raises(RuntimeError, match="updated engine"):
        gguf_backend.GgufBackend(model, device="cpu")
    fake_runtime.Llama.assert_not_called()
