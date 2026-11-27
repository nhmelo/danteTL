"""Install pinned runtimes in user state; publish only validated environments."""

import hashlib
import json
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from dantetl.atomic_file import replace_file
from dantetl.file_lock import file_lock
from dantetl.i18n import tr
from dantetl.paths import is_standalone, module_command, runtime_project
from dantetl.runtime.application import prepare_application
from dantetl.runtime.downloads import prepare_wheels
from dantetl.runtime.hardware import require_profile
from dantetl.runtime.requirements import (
    RUNTIME_PYTHON,
    dependencies_match,
    dependency_fingerprint,
)
from dantetl.runtime.setup_process import clean_runtime_environment, run_setup_command
from dantetl.runtime.setup_values import Cancelled, ModelProgress, Progress
from dantetl.runtime.values import PROFILES, RuntimeProfile


def _fingerprint(root: Path, profile: str) -> str:
    if is_standalone():
        return dependency_fingerprint(runtime_project(root), profile)
    digest = hashlib.sha256(b"runtime-setup-v1")
    for name in ("uv.lock", "pyproject.toml"):
        digest.update((runtime_project(root) / name).read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, str]) -> None:
    replace_file(path, json.dumps(value).encode("utf-8"))


def runtime_path(root: Path, name: str) -> Path:
    """Receipts may select only generated environments inside this project."""
    if not re.fullmatch(r"(?:nvidia|amd|cpu)-[a-f0-9]{32}", name):
        raise ValueError(tr("Invalid runtime environment name."))
    base = (root / ".runtimes").resolve()
    candidate = base / name
    target = candidate.resolve()
    if (
        base.parent != root.resolve()
        or target.parent != base
        or candidate.is_symlink()
        or candidate.is_junction()
    ):
        raise ValueError(tr("Runtime environment escapes this project."))
    return target


def ready_runtime(root: Path, key: str) -> Path | None:
    """Latest prepared environment; independent of the committed selection."""
    if key not in {p.key for p in PROFILES}:
        return None
    try:
        receipt = json.loads((root / "data/runtimes" / f"{key}.json").read_text("utf-8"))
        target = verified_runtime(root, key, receipt["environment"])
        if target and json.loads((target / "ready.json").read_text("utf-8")) == receipt:
            return target
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    target: Path | None = None
    cuda_update: bool = False
    gpu: str = ""


def _inspect_runtime(root: Path, key: str, name: str) -> RuntimeSelection:
    """Validate an environment's own receipt, never a mutable profile pointer."""
    try:
        target = runtime_path(root, name)
        receipt = json.loads((target / "ready.json").read_text("utf-8"))
        if (
            key in {p.key for p in PROFILES}
            and target.name.startswith(key + "-")
            and isinstance(receipt, dict)
            and receipt.get("environment") == name
            and re.fullmatch(r"[a-f0-9]{64}", receipt.get("fingerprint", ""))
            and (target / "Scripts/pythonw.exe").is_file()
        ):
            if is_standalone():
                if not dependencies_match(runtime_project(root), key, target):
                    return RuntimeSelection(cuda_update=key == "nvidia")
            elif receipt != {"environment": name, "fingerprint": _fingerprint(root, key)}:
                return RuntimeSelection()
            return RuntimeSelection(target)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return RuntimeSelection()


def verified_runtime(root: Path, key: str, name: str) -> Path | None:
    return _inspect_runtime(root, key, name).target


def runtime_selection(root: Path) -> RuntimeSelection:
    try:
        value = json.loads((root / "data/runtime.json").read_text("utf-8"))
        if not isinstance(value, dict):
            return RuntimeSelection()
        gpu = value.get("gpu", "")
        if not isinstance(gpu, str) or (
            gpu and (value["profile"] == "cpu" or not re.fullmatch("[a-f0-9]{32}", gpu))
        ):
            return RuntimeSelection()
        return replace(_inspect_runtime(root, value["profile"], value["environment"]), gpu=gpu)
    except (OSError, ValueError, KeyError, TypeError):
        return RuntimeSelection()


def selected_runtime(root: Path) -> Path | None:
    return runtime_selection(root).target


def select_runtime(root: Path, target: Path, gpu: str = "") -> None:
    key = target.name.split("-", 1)[0]
    if verified_runtime(root, key, target.name) != target.resolve():
        raise ValueError(tr("Only a verified runtime can be selected."))
    selection = {"profile": key, "environment": target.name}
    if gpu:
        if key == "cpu" or not re.fullmatch("[a-f0-9]{32}", gpu):
            raise ValueError(
                tr("The selected GPU is unavailable. Choose another GPU in Acceleration.")
            )
        selection["gpu"] = gpu
    _atomic_json(root / "data/runtime.json", selection)


def install_runtime(
    root: Path, profile: RuntimeProfile, progress: Progress, cancel: threading.Event, gpu: str = ""
) -> None:
    """Explicit action only. Reuse completed installs; never mutate a loaded environment."""
    if profile not in PROFILES:
        raise ValueError(tr("Unknown runtime profile."))
    device_env = require_profile(profile.key, cancel, gpu)
    root = root.resolve()
    # Exclusive across app copies sharing this project, without waiting.
    with file_lock(
        root / "data/runtime-install.lock",
        timeout=0,
        busy_message=tr("Another DanteTL instance is already installing a runtime."),
    ):
        _install(root, profile, progress, cancel, device_env)


def _install(
    root: Path,
    profile: RuntimeProfile,
    progress: Progress,
    cancel: threading.Event,
    device_env: dict[str, str],
) -> None:
    environment = clean_runtime_environment(os.environ)
    target = runtime_path(root, f"{profile.key}-{uuid4().hex}")
    environment.update(
        UV_PROJECT_ENVIRONMENT=str(target),
        UV_CACHE_DIR=str(root / ".uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / ".python"),
        PYTHONUTF8="1",
        UV_NO_PROGRESS="1",
    )
    project = runtime_project(root)
    fingerprint = _fingerprint(root, profile.key)
    if is_standalone():
        # Never bind an AppData runtime to an app folder which may be moved or replaced.
        environment["UV_PYTHON_PREFERENCE"] = "only-managed"
        environment["DANTETL_DATA_DIR"] = str(root)
        environment["DANTETL_RESOURCE_DIR"] = str(project)

    def command(args: list[str], stage: str, timeout: float = 1800, step: int = 0) -> None:
        progress(ModelProgress(stage, stage=step, stages=4 if step else 0))
        run_setup_command(
            args,
            project,
            environment | device_env,
            cancel,
            lambda detail: progress(
                ModelProgress(f"{stage}\n{detail}", stage=step, stages=4 if step else 0)
            ),
            timeout=timeout,
        )

    if ready := ready_runtime(root, profile.key):
        if is_standalone():
            environment.update(prepare_application(root, ready / "Scripts/python.exe", cancel))
        try:
            command(
                module_command(ready / "Scripts/python.exe", "dantetl.runtime.probe", profile.key),
                tr("Checking the installed runtime…"),
                timeout=180,
            )
        except (OSError, RuntimeError):
            progress(
                ModelProgress(tr("Existing runtime needs repair. Building a separate replacement…"))
            )
        else:
            progress(ModelProgress(tr("Installed runtime checks passed. Restart to use it.")))
            return

    uv = project.parent / "uv.exe" if is_standalone() else root / ".bootstrap/Scripts/uv.exe"
    if is_standalone() and not uv.is_file():
        raise RuntimeError(
            tr("The bundled installer is missing. Extract the complete DanteTL ZIP.")
        )
    if not uv.is_file():
        located = shutil.which("uv")
        if located:
            uv = Path(located)
        else:
            bootstrap = root / ".runtime-bootstrap"
            command(
                [sys.executable, "-m", "venv", str(bootstrap)], tr("1/3 · Preparing the installer…")
            )
            command(
                [str(bootstrap / "Scripts/python.exe"), "-m", "pip", "install", "uv==0.13.8"],
                tr("1/3 · Downloading the installer…"),
            )
            uv = bootstrap / "Scripts/uv.exe"
    export = target / "pylock.toml"
    args = [
        str(uv),
        "export",
        "--locked",
        "--offline",
        "--quiet",
        "--no-default-groups",
        "--no-emit-project",
        "--format",
        "pylock.toml",
        "--output-file",
        str(export),
    ]
    for group in profile.groups:
        args.extend(("--group", group))
    # Keep the final path: Windows venv entry points must not be renamed after installation.
    try:
        command(
            [str(uv), "venv", "--python", RUNTIME_PYTHON, str(target)],
            tr("Preparing private Python…"),
            step=1,
        )
        if is_standalone():
            shutil.copyfile(project / "runtime-locks" / f"pylock.{profile.key}.toml", export)
        else:
            command(args, tr("Reading locked dependencies…"), step=2)
        requirements = prepare_wheels(
            export,
            root / ".runtime-downloads/wheels",
            lambda value: progress(replace(value, stage=2, stages=4)),
            cancel,
            cpu_wheel=(
                project
                / ("cpu-runtime" if is_standalone() else "build/cpu-runtime")
                / "llama_cpp_python-0.3.37-py3-none-win_amd64.whl"
            )
            if profile.key == "cpu"
            else None,
        )
        installer = (
            project / "install_wheels.py"
            if is_standalone()
            else Path(__file__).with_name("install_wheels.py")
        )
        command(
            [
                str(target / "Scripts/python.exe"),
                "-I",
                "-B",
                str(installer),
                str(requirements),
            ],
            tr("Installing verified dependencies…"),
            step=3,
        )
        if is_standalone():
            progress(ModelProgress(tr("Preparing the application wheel…"), stage=3, stages=4))
            environment.update(prepare_application(root, target / "Scripts/python.exe", cancel))
        else:
            command(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--no-deps",
                    "--python",
                    str(target / "Scripts/python.exe"),
                    "--editable",
                    str(project),
                ],
                tr("Connecting the source application…"),
                step=3,
            )
        command(
            module_command(target / "Scripts/python.exe", "dantetl.runtime.probe", profile.key),
            tr("Checking installed libraries and GPU support…"),
            timeout=180,
            step=4,
        )
        if cancel.is_set():
            raise Cancelled()
        if _fingerprint(root, profile.key) != fingerprint:
            raise RuntimeError(
                tr("The dependencies changed during setup. Retry with the current release.")
            )
        receipt = {"environment": target.name, "fingerprint": fingerprint}
        _atomic_json(target / "ready.json", receipt)
        _atomic_json(root / "data/runtimes" / f"{profile.key}.json", receipt)
    except BaseException:
        # Only remove the fresh environment owned by this call, after subprocesses have exited.
        checked = runtime_path(root, target.name)
        if checked.exists() and checked != Path(sys.prefix).resolve():
            shutil.rmtree(checked)
        raise
    progress(
        ModelProgress(
            tr("Runtime installed and checked. Restart to use it. Model files were kept.")
        )
    )
