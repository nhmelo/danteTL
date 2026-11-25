"""The release CPU build must consume the same source as the runtime dependency lock."""

import os
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dantetl.runtime import probe
from dantetl.runtime.downloads import locked_wheels
from dantetl.runtime.values import PROFILES


def test_installed_cli_resolves_the_current_application_entry_point():
    app = distribution("dantetl")
    (entry,) = (e for e in app.entry_points if e.group == "gui_scripts")
    assert entry.name == "dantetl"
    assert entry.load().__module__ == "dantetl.bootstrap"


@pytest.mark.parametrize("missing_module", [False, True])
def test_app_install_probe_loads_startup_dependencies_before_acceptance(
    monkeypatch, missing_module
):
    load = Mock(
        side_effect=ModuleNotFoundError("missing nested package") if missing_module else None
    )
    app = SimpleNamespace(
        version="0.1.0",
        entry_points=[SimpleNamespace(group="gui_scripts", name="dantetl", load=load)],
    )
    monkeypatch.setattr(probe, "distribution", lambda name: app)
    monkeypatch.setattr("sys.argv", ["probe", "cpu", "--application-only"])
    if missing_module:
        with pytest.raises(ModuleNotFoundError, match="missing nested package"):
            probe.main()
    else:
        probe.main()
    load.assert_called_once_with()


def test_cpu_source_requirement_matches_uv_lock():
    root = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((root / "uv.lock").read_text("utf-8"))
    (package,) = (
        p for p in lock["package"] if p["name"] == "llama-cpp-python" and "registry" in p["source"]
    )
    requirement = (root / "packaging/cpu-runtime.txt").read_text("utf-8").replace("\\\n", "")
    (arguments,) = (
        line.split() for line in requirement.splitlines() if line and not line.startswith("#")
    )
    assert arguments[:2] == [
        f"llama-cpp-python=={package['version']}",
        f"--hash={package['sdist']['hash']}",
    ]
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    settings = project["tool"]["uv"]["config-settings-package"]["llama-cpp-python"]
    assert sorted(arguments[2:]) == sorted(
        f"--config-settings={key}={value}" for key, value in settings.items()
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda profile: profile.key)
def test_locked_runtime_exports_keep_their_ocr_and_translation_engines(tmp_path, profile):
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv") or str(root / ".bootstrap/Scripts/uv.exe")
    export = tmp_path / "pylock.toml"
    command = [
        uv,
        "export",
        "--locked",
        "--offline",
        "--python",
        sys.executable,
        "--no-default-groups",
        "--no-emit-project",
        "--format",
        "pylock.toml",
        "--output-file",
        str(export),
        "--quiet",
    ]
    for group in profile.groups:
        command.extend(("--group", group))
    subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
        env=os.environ | {"UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    packages = {p["name"]: p for p in tomllib.loads(export.read_text("utf-8"))["packages"]}
    expected_ocr = "onnxruntime-gpu" if profile.key == "nvidia" else "onnxruntime"
    assert {name for name in packages if name.startswith("onnxruntime")} == {expected_ocr}
    engine = packages["llama-cpp-python"]
    if profile.key == "cpu":
        assert "sdist" in engine
    else:
        (wheel,) = (w for w in locked_wheels(export) if w.name == "llama-cpp-python")
        if profile.key == "nvidia":
            assert "+cu130-cp312-cp312-win_amd64.whl" in wheel.filename
        else:
            assert "-vulkan/" in wheel.url
