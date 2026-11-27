from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dantetl.translation.backends import llama_engine
from dantetl.translation.backends.label_decoder import Sampling


def fake_runtime():
    return SimpleNamespace(__version__="fixture", Llama=Mock())


@pytest.mark.parametrize(
    "device, layers, batch, threads", [("cpu", 0, 128, 4), ("cuda", -1, 1, 2), ("vulkan", -1, 1, 2)]
)
def test_engine_loads_one_bounded_context_on_one_adapter(
    monkeypatch, device, layers, batch, threads
):
    monkeypatch.setattr(llama_engine.os, "cpu_count", lambda: 16)
    runtime = fake_runtime()
    engine = llama_engine.LlamaEngine(runtime, Path("model.gguf"), device)
    options = runtime.Llama.call_args.kwargs
    assert options["model_path"] == "model.gguf" and options["n_ctx"] == llama_engine.CONTEXT
    assert options["n_gpu_layers"] == layers and options["split_mode"] == 0
    assert options["n_batch"] == options["n_ubatch"] == engine.batch == batch
    assert options["n_threads"] == options["n_threads_batch"] == threads


def test_imported_models_keep_default_prompt_evaluation_threads(monkeypatch):
    monkeypatch.setattr(llama_engine.os, "cpu_count", lambda: 16)
    runtime = fake_runtime()
    llama_engine.LlamaEngine(runtime, Path("model.gguf"), "cpu", bound_batch_threads=False)
    options = runtime.Llama.call_args.kwargs
    assert options["n_threads"] == 4 and "n_threads_batch" not in options


def test_completion_uses_the_backends_sampling_and_stops_and_close_is_idempotent():
    runtime = fake_runtime()
    engine = llama_engine.LlamaEngine(runtime, Path("model.gguf"), "cuda")
    model = runtime.Llama.return_value
    engine.complete([1, 2], ("<eos>",), Sampling(temperature=0.7, top_k=20, top_p=0.8, seed=42))
    options = model.create_completion.call_args.kwargs
    assert options["prompt"] == [1, 2] and options["max_tokens"] == llama_engine.OUTPUT_LIMIT
    assert options["stop"] == ["<eos>"] and options["repeat_penalty"] == 1.0
    assert (options["temperature"], options["top_k"], options["top_p"], options["seed"]) == (
        0.7,
        20,
        0.8,
        42,
    )
    engine.close()
    engine.close()
    model.close.assert_called_once()
    assert engine.model is None
