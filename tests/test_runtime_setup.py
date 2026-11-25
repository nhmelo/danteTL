import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import venv
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from dantetl.runtime import restart as runtime_restart
from dantetl.runtime import setup as setup
from dantetl.runtime.hardware_values import Adapter, CudaDevice, recommended_profile
from dantetl.runtime.setup_process import run_setup_command
from dantetl.runtime.setup_values import Cancelled


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters",
        lambda *args: (Adapter("RTX", 0x10DE, cuda=CudaDevice(13000, 8, 6), vulkan=True),),
    )
    (tmp_path / "uv.lock").write_text("pinned")
    (tmp_path / "pyproject.toml").write_text("project")
    uv = tmp_path / ".bootstrap/Scripts/uv.exe"
    uv.parent.mkdir(parents=True)
    uv.touch()
    calls = []

    def run(args, root, environment, cancel, report, **kwargs):
        calls.append(args)
        assert not environment.get("VIRTUAL_ENV")
        assert "UV_PROJECT_ENVIRONMENT" in environment
        target = Path(environment["UV_PROJECT_ENVIRONMENT"])
        assert target != Path(sys.prefix)
        if "venv" in args:
            (target / "Scripts").mkdir(parents=True)
            (target / "Scripts/python.exe").touch()
            (target / "Scripts/pythonw.exe").touch()
        report("Fixture process complete")

    monkeypatch.setattr(setup, "run_setup_command", run)
    monkeypatch.setattr(
        setup, "prepare_wheels", lambda export, *args, **kwargs: export.with_suffix(".txt")
    )
    monkeypatch.setenv("VIRTUAL_ENV", "another-project")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "another-project")
    return tmp_path, calls, run


def test_recommends_discrete_gpu_and_preserves_manual_cpu_fallback():
    assert recommended_profile(()) == "cpu"
    assert recommended_profile((Adapter("Intel", 0x8086, 2**30),)) == "cpu"
    integrated = Adapter("AMD integrated", 0x1002, 2**30, vulkan=True)
    nvidia = Adapter("RTX", 0x10DE, 16 * 2**30, cuda=CudaDevice(13000, 7, 5), vulkan=True)
    amd = Adapter("Radeon", 0x1002, 16 * 2**30, vulkan=True)
    assert recommended_profile((integrated, nvidia)) == "nvidia"
    assert recommended_profile((amd,)) == "amd"


@pytest.mark.parametrize("cuda", [CudaDevice(13030, 6, 1), CudaDevice(12090, 8, 6), None])
def test_vulkan_recommendation_requires_driver_features_and_cuda_checks_both_versions(cuda):
    assert recommended_profile((Adapter("GPU", 0x10DE, cuda=cuda),)) == "cpu"
    assert recommended_profile((Adapter("GPU", 0x10DE, cuda=cuda, vulkan=True),)) == "amd"
    assert recommended_profile((Adapter("Intel", 0x8086, vulkan=True),)) == "amd"


def test_incompatible_device_blocks_download_and_saved_runtime_launch(project, monkeypatch):
    root, calls, _ = project
    setup.install_runtime(root, setup.PROFILES[0], lambda _: None, threading.Event())
    target = setup.ready_runtime(root, "nvidia")
    assert target is not None
    setup.select_runtime(root, target)
    calls.clear()
    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters",
        lambda *args: (Adapter("GTX 1050", 0x10DE, cuda=CudaDevice(13000, 6, 1)),),
    )
    with pytest.raises(RuntimeError, match="6.1"):
        setup.install_runtime(root, setup.PROFILES[0], lambda _: None, threading.Event())
    with pytest.raises(RuntimeError, match="6.1"):
        runtime_restart.RuntimeRestart(root, target)
    with pytest.raises(RuntimeError, match="Vulkan"):
        setup.install_runtime(root, setup.PROFILES[1], lambda _: None, threading.Event())
    assert not calls and setup.selected_runtime(root) == target


def test_native_installer_checks_hash_and_ignores_external_pip_target(tmp_path, monkeypatch):
    target = tmp_path / "private runtime"
    venv.EnvBuilder(with_pip=False).create(target)
    foreign = tmp_path / "other site"
    monkeypatch.setenv("PIP_TARGET", str(foreign))
    wheel = tmp_path / "fixture-1-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("fixture.py", "value = 1")
        archive.writestr(
            "fixture-1.dist-info/METADATA", "Metadata-Version: 2.1\nName: fixture\nVersion: 1\n"
        )
        archive.writestr(
            "fixture-1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("fixture-1.dist-info/RECORD", "fixture.py,,\n")
    requirement = tmp_path / "requirements.txt"
    command = [
        str(target / "Scripts/python.exe"),
        "-I",
        "-B",
        str(Path(setup.__file__).with_name("install_wheels.py")),
        str(requirement),
    ]
    installed = target / "Lib/site-packages/fixture.py"
    for digest, valid in (
        ("0" * 64, False),
        (hashlib.sha256(wheel.read_bytes()).hexdigest(), True),
    ):
        requirement.write_text(f"{wheel.as_uri()} --hash=sha256:{digest}\n", encoding="utf-8")
        result = subprocess.run(command, capture_output=True, timeout=30)
        assert (result.returncode == 0) == valid, result.stderr
        assert installed.exists() == valid
        assert not foreign.exists()


@pytest.mark.parametrize("profile", setup.PROFILES, ids=lambda p: p.key)
def test_install_validates_before_publishing_and_selection_is_explicit(project, profile):
    root, calls, _ = project
    setup.install_runtime(root, profile, lambda p: None, threading.Event())
    ready = setup.ready_runtime(root, profile.key)
    assert ready and "venv" in calls[0]
    assert "--locked" in calls[1] and "--no-default-groups" in calls[1]
    install = calls[2]
    assert Path(install[0]) == ready / "Scripts/python.exe"
    assert Path(install[3]).name == "install_wheels.py"
    assert install[1:3] == ["-I", "-B"]
    assert Path(install[-1]) == ready / "pylock.txt"
    assert calls[-1][1:] == ["-m", "dantetl.runtime.probe", profile.key]
    assert not setup.selected_runtime(root)
    setup.select_runtime(root, ready)
    assert setup.selected_runtime(root) == ready
    before = set((root / ".runtimes").iterdir())
    setup.install_runtime(root, profile, lambda p: None, threading.Event())
    assert set((root / ".runtimes").iterdir()) == before  # Checks/reuses, never resyncs a live env.
    assert len([c for c in calls if "venv" in c]) == 1
    if profile.key == "amd":
        assert "cpu" in calls[1] and "gemma-vulkan" in calls[1] and "amd" not in calls[1]
    if profile.key == "cpu":
        assert "llm-cpu" in calls[1] and "gemma" not in calls[1] and "gpu" not in calls[1]
    (root / "uv.lock").write_text("updated dependencies")
    assert (
        setup.selected_runtime(root) is None
    )  # A changed package cannot silently reuse stale deps.


@pytest.mark.parametrize("error", [RuntimeError("No compatible driver"), Cancelled()])
def test_failure_or_cancel_keeps_active_runtime_and_models(project, monkeypatch, error):
    root, _, run = project
    setup.install_runtime(root, setup.PROFILES[2], lambda p: None, threading.Event())
    previous = setup.ready_runtime(root, "cpu")
    assert previous
    setup.select_runtime(root, previous)
    (root / "models").mkdir()
    (root / "models/weights").write_bytes(b"user data")

    def fail(args, *a, **kw):
        if "dantetl.runtime.probe" in args:
            raise error
        run(args, *a, **kw)

    monkeypatch.setattr(setup, "run_setup_command", fail)
    with pytest.raises(type(error)):
        setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    assert setup.selected_runtime(root) == previous
    assert not setup.ready_runtime(root, "nvidia")
    assert list((root / ".runtimes").iterdir()) == [previous]
    assert (root / "models/weights").read_bytes() == b"user data"


def test_repair_creates_replacement_without_modifying_running_environment(project, monkeypatch):
    root, _, run = project
    setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    old = setup.ready_runtime(root, "nvidia")
    assert old
    setup.select_runtime(root, old)
    (old / "keep").write_text("loaded dll")

    def fail_old(args, *a, **kw):
        if args[0] == str(old / "Scripts/python.exe"):
            raise RuntimeError("Broken library")
        run(args, *a, **kw)

    monkeypatch.setattr(setup, "run_setup_command", fail_old)
    setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    assert setup.ready_runtime(root, "nvidia") != old
    assert (old / "keep").read_text() == "loaded dll"
    assert setup.selected_runtime(root) == old
    setup.select_runtime(root, old)  # Rollback accepts the old verified receipt too.
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(runtime_restart, "OwnedProcess", lambda *a, **kw: process)
    launch = runtime_restart.RuntimeRestart(root, old)
    assert not launch.poll()  # Latest-prepared B must not prevent launching selected A.
    launch.cancel()


@pytest.mark.parametrize("name", ["../.venv", "cpu-../../elsewhere", "C:/Python", ".venv"])
def test_receipts_cannot_launch_or_delete_arbitrary_paths(project, name):
    root, _, _ = project
    with pytest.raises(ValueError):
        setup.runtime_path(root, name)
    (root / "data/runtimes").mkdir(parents=True)
    (root / "data/runtimes/cpu.json").write_text(json.dumps({"environment": name}))
    assert setup.ready_runtime(root, "cpu") is None


def test_restart_commits_only_after_new_window_acknowledges(project, monkeypatch):
    root, _, _ = project
    setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    target = setup.ready_runtime(root, "nvidia")
    assert target
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(runtime_restart, "OwnedProcess", lambda *a, **kw: process)
    restart = runtime_restart.RuntimeRestart(root, target)
    assert not restart.poll() and not setup.selected_runtime(root)
    restart.receipt.parent.mkdir(parents=True)
    restart.receipt.write_text("12345")
    assert restart.poll() and setup.selected_runtime(root) == target


@pytest.mark.parametrize("failure", ["start", "handoff"])
def test_repaired_profile_keeps_previous_selection_on_activation_failure(
    project, monkeypatch, failure
):
    root, _, run = project
    profile = setup.PROFILES[0]
    setup.install_runtime(root, profile, lambda p: None, threading.Event())
    previous = setup.ready_runtime(root, profile.key)
    assert previous
    setup.select_runtime(root, previous)

    def replace_old(args, *a, **kw):
        if args[0] == str(previous / "Scripts/python.exe"):
            raise RuntimeError("Repair fixture")
        run(args, *a, **kw)

    monkeypatch.setattr(setup, "run_setup_command", replace_old)
    setup.install_runtime(root, profile, lambda p: None, threading.Event())
    target = setup.ready_runtime(root, profile.key)
    assert target and target != previous
    assert setup.selected_runtime(root) == previous

    process = Mock()
    process.poll.return_value = 1 if failure == "start" else None
    process.release.side_effect = OSError("handoff failed")
    monkeypatch.setattr(runtime_restart, "OwnedProcess", lambda *a, **kw: process)
    restart = runtime_restart.RuntimeRestart(root, target)
    if failure == "handoff":
        restart.receipt.parent.mkdir(parents=True)
        restart.receipt.write_text("12345")
    with pytest.raises((RuntimeError, OSError)):
        restart.poll()
    assert setup.selected_runtime(root) == previous
    # Both normal launchers use this boundary; A remains launchable after B failed activation.
    process.poll.return_value = None
    process.release.side_effect = None
    restored = runtime_restart.RuntimeRestart(root, previous)
    assert not restored.poll()
    restored.cancel()


def test_failed_restart_preserves_previous_selection(project, monkeypatch):
    root, _, _ = project
    setup.install_runtime(root, setup.PROFILES[2], lambda p: None, threading.Event())
    previous = setup.ready_runtime(root, "cpu")
    assert previous
    setup.select_runtime(root, previous)
    setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    target = setup.ready_runtime(root, "nvidia")
    assert target
    process = Mock()
    process.poll.return_value = 1
    monkeypatch.setattr(runtime_restart, "OwnedProcess", lambda *a, **kw: process)
    restart = runtime_restart.RuntimeRestart(root, target)
    with pytest.raises(RuntimeError, match="could not start"):
        restart.poll()
    assert setup.selected_runtime(root) == previous


def test_subprocess_failure_reports_output_and_cancel_stops_owned_child(tmp_path):
    env = os.environ.copy()
    with pytest.raises(RuntimeError, match="native failure"):
        run_setup_command(
            [sys.executable, "-c", "print('native failure'); raise SystemExit(2)"],
            tmp_path,
            env,
            threading.Event(),
            lambda line: None,
        )
    cancel = threading.Event()
    with pytest.raises(Cancelled):
        run_setup_command(
            [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(20)"],
            tmp_path,
            env,
            cancel,
            lambda line: cancel.set(),
            timeout=3,
        )


def test_cancel_also_stops_installer_descendants(tmp_path):
    cancel = threading.Event()
    child_pid = []

    def started(line):
        child_pid.append(int(line))
        cancel.set()

    command = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
        "print(child.pid,flush=True); time.sleep(20)"
    )
    with pytest.raises(Cancelled):
        run_setup_command(
            [sys.executable, "-u", "-c", command],
            tmp_path,
            os.environ.copy(),
            cancel,
            started,
        )
    assert len(child_pid) == 1
    kernel = ctypes.WinDLL("kernel32")
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x100000, False, child_pid[0])  # SYNCHRONIZE
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle, 2000) == 0
        finally:
            kernel.CloseHandle(handle)


def test_restart_timeout_stops_tree_without_activating_runtime(project, monkeypatch):
    root, _, _ = project
    setup.install_runtime(root, setup.PROFILES[0], lambda p: None, threading.Event())
    target = setup.ready_runtime(root, "nvidia")
    assert target
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(runtime_restart, "OwnedProcess", lambda *a, **kw: process)
    restart = runtime_restart.RuntimeRestart(root, target)
    restart.deadline = 0
    with pytest.raises(RuntimeError, match="timed out"):
        restart.poll()
    process.close.assert_called_once()
    assert setup.selected_runtime(root) is None
