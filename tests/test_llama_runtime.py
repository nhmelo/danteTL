import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dantetl.runtime import llama as llama_runtime


@pytest.fixture(autouse=True)
def isolated_loader(monkeypatch):
    monkeypatch.setattr(llama_runtime, "_runtime", None)


def test_overlapping_loads_initialize_once_without_hiding_other_threads_output(
    monkeypatch, tmp_path
):
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    original = sys.stdout
    runtime = SimpleNamespace(__version__="0.4.7", __file__=str(tmp_path / "日本語/__init__.py"))
    initialize = Mock()
    imports = []

    def importing(name):
        imports.append(name)
        assert sys.stdout is original
        if name == "llama_cpp":
            entered.set()
            assert release.wait(2)
            return runtime
        return SimpleNamespace(ggml_backend_load_all_from_path=initialize)

    def overlapping():
        second.set()
        return llama_runtime.load_runtime("cuda")

    monkeypatch.setattr(llama_runtime, "_dll_handles", [object()])
    monkeypatch.setattr(llama_runtime.importlib, "import_module", importing)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(llama_runtime.load_runtime, "cuda")
        try:
            assert entered.wait(2)
            following = pool.submit(overlapping)
            assert second.wait(2)
            assert sys.stdout is original  # Unrelated output remains usable DURING initialization.
        finally:
            release.set()
        assert first.result(2) is following.result(2) is runtime
    assert sys.stdout is original
    assert imports == ["llama_cpp", "llama_cpp._ggml"]
    initialize.assert_called_once_with(str(tmp_path / "日本語/lib").encode("utf-8"))


def test_failed_initialization_is_not_published_and_can_retry(monkeypatch, tmp_path):
    runtime = SimpleNamespace(__version__="0.4.7", __file__=str(tmp_path / "__init__.py"))
    initialize = Mock(side_effect=[RuntimeError("failed"), None])
    monkeypatch.setattr(llama_runtime, "_dll_handles", [object()])
    monkeypatch.setattr(
        llama_runtime.importlib,
        "import_module",
        lambda name: (
            runtime
            if name == "llama_cpp"
            else SimpleNamespace(ggml_backend_load_all_from_path=initialize)
        ),
    )
    original = sys.stdout
    with pytest.raises(RuntimeError, match="failed"):
        llama_runtime.load_runtime("cpu")
    assert sys.stdout is original
    assert llama_runtime.load_runtime("cpu") is runtime
    assert llama_runtime.load_runtime("cpu") is runtime
    assert initialize.call_count == 2


@pytest.mark.parametrize(
    "files, expected",
    [
        ((), None),
        (("llama.dll",), None),
        (("llama.dll", "ggml-cpu.dll"), "cpu"),
        (("libllama.dll", "ggml-cpu.dll"), "cpu"),
        (("llama.dll", "ggml-cpu-x64.dll", "ggml-cpu-zen4.dll"), "cpu"),
        (("llama.dll", "ggml-cpu.dll", "ggml-cuda.dll"), "cuda"),
        (("llama.dll", "ggml-cpu.dll", "ggml-vulkan.dll"), "vulkan"),
    ],
)
def test_discovery_distinguishes_cpu_gpu_and_missing_engine_without_importing(
    monkeypatch, files, expected
):
    monkeypatch.setattr(llama_runtime, "distribution", lambda _: SimpleNamespace(files=files))
    imports = Mock(side_effect=AssertionError("Discovery must not load native libraries"))
    monkeypatch.setattr(llama_runtime.importlib, "import_module", imports)
    assert llama_runtime.installed_device() == expected
    imports.assert_not_called()


def test_missing_distribution_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        llama_runtime, "distribution", Mock(side_effect=PackageNotFoundError("llama-cpp-python"))
    )
    assert llama_runtime.installed_device() is None


def test_cpu_loading_never_requests_gpu_packages(monkeypatch, tmp_path):
    requested = []

    def package(name):
        requested.append(name)
        assert name == "llama-cpp-python"
        return SimpleNamespace(
            files=(Path("llama_cpp/lib/llama.dll"),),
            locate_file=lambda p: tmp_path / p,
        )

    monkeypatch.setattr(llama_runtime, "distribution", package)
    monkeypatch.setattr(llama_runtime, "_dll_handles", [])
    monkeypatch.setattr(llama_runtime.os, "add_dll_directory", lambda p: object())
    monkeypatch.setenv("PATH", "original")
    runtime = SimpleNamespace(__file__=str(tmp_path / "llama_cpp/__init__.py"))
    initialize = Mock()
    monkeypatch.setattr(llama_runtime, "_load_cpu_backend", initialize)
    monkeypatch.setattr(llama_runtime.importlib, "import_module", lambda _: runtime)
    assert llama_runtime.load_runtime("cpu") is runtime
    assert llama_runtime.load_runtime("cpu") is runtime
    assert requested == ["llama-cpp-python"]
    if sys.platform == "win32":
        initialize.assert_called_once_with(tmp_path / "llama_cpp/lib")


def test_matching_runtime_api_keeps_flash_attention_policy():
    old = SimpleNamespace(__version__="0.3.36")
    new = SimpleNamespace(__version__="0.4.7+cu130")
    for device in ("cpu", "cuda", "vulkan"):
        assert llama_runtime.flash_attention_options(old, device) == {
            "flash_attn": device == "cuda"
        }
        assert llama_runtime.flash_attention_options(new, device) == {
            "flash_attn_type": int(device == "cuda")
        }


def test_dynamic_backend_discovery_precedes_use_and_runs_once(monkeypatch, tmp_path):
    runtime = SimpleNamespace(__version__="0.4.7+cu130", __file__=str(tmp_path / "__init__.py"))
    ggml = SimpleNamespace(ggml_backend_load_all_from_path=Mock())
    monkeypatch.setattr(llama_runtime, "_dll_handles", [object()])
    monkeypatch.setattr(
        llama_runtime.importlib, "import_module", lambda n: runtime if n == "llama_cpp" else ggml
    )
    assert llama_runtime.load_runtime("cuda") is runtime
    assert llama_runtime.load_runtime("cuda") is runtime
    ggml.ggml_backend_load_all_from_path.assert_called_once_with(str(tmp_path / "lib").encode())


@pytest.mark.parametrize(
    "version, supported",
    [
        ("0.3.36", False),
        ("0.3.37", True),
        ("invalid", False),
        ("0.4.7+cu130", True),
        ("0.4.7+dantetl.b12345.cpu", True),
    ],
)
def test_plamo_capability_uses_package_metadata_only(monkeypatch, version, supported):
    monkeypatch.setattr(llama_runtime, "distribution", lambda _: SimpleNamespace(version=version))
    monkeypatch.setattr(
        llama_runtime.importlib,
        "import_module",
        Mock(side_effect=AssertionError("No native imports")),
    )
    assert llama_runtime.supports_plamo() is supported
