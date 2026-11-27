"""Offline app-only updates, with immutable installs shared by existing native runtimes."""

import json
import os
import re
import shutil
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import uuid4

from dantetl.file_lock import file_lock
from dantetl.i18n import tr
from dantetl.paths import is_standalone, module_command, resource_root
from dantetl.runtime.package import ApplicationWheel, bundled_application
from dantetl.runtime.setup_process import clean_runtime_environment, run_setup_command
from dantetl.runtime.setup_values import Cancelled


def _application_path(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}-[a-f0-9]{32}", name):
        raise ValueError(tr("Invalid application install name."))
    base = (root / ".applications").resolve()
    candidate = base / name
    target = candidate.resolve()
    if (
        base.parent != root.resolve()
        or target.parent != base
        or candidate.is_symlink()
        or candidate.is_junction()
    ):
        raise ValueError(tr("Application installation escapes user state."))
    return target


def _ready(root: Path, wheel: ApplicationWheel) -> Path | None:
    try:
        value = json.loads((root / "data/applications" / f"{wheel.sha256}.json").read_text("utf-8"))
        target = _application_path(root, value["directory"])
        if target.name.startswith(wheel.sha256 + "-") and wheel.installed_at(target):
            return target
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _install_lock(root: Path, cancel: threading.Event) -> AbstractContextManager[None]:
    def check() -> None:
        if cancel.is_set():
            raise Cancelled()

    return file_lock(
        root / "data/application-install.lock",
        check=check,
        timeout=120,
        busy_message=tr("Another app update is still running. Try launching again."),
    )


def prepare_application(root: Path, python: Path, cancel: threading.Event) -> dict[str, str]:
    """Before GUI startup or on a worker: install only bundled code, never native dependencies."""
    environment = clean_runtime_environment(os.environ)
    if not is_standalone():
        return environment
    project = resource_root()
    environment.update(
        DANTETL_DATA_DIR=str(root),
        DANTETL_RESOURCE_DIR=str(project),
        PYTHONUTF8="1",
        UV_CACHE_DIR=str(root / ".uv-cache"),
        UV_NO_PROGRESS="1",
    )
    wheel = bundled_application(project)
    with _install_lock(root, cancel):
        if ready := _ready(root, wheel):
            return environment | {"PYTHONPATH": str(ready)}
        target = _application_path(root, f"{wheel.sha256}-{uuid4().hex}")
        target.mkdir(parents=True)
        try:
            requirement = wheel.requirements(target)
            run_setup_command(
                [
                    str(project.parent / "uv.exe"),
                    "pip",
                    "install",
                    "--offline",
                    "--no-index",
                    "--no-deps",
                    "--require-hashes",
                    "--python",
                    str(python),
                    "--target",
                    str(target),
                    "--requirements",
                    str(requirement),
                ],
                project,
                environment,
                cancel,
                lambda _: None,
                timeout=120,
            )
            if not wheel.installed_at(target):
                raise RuntimeError(tr("The app update did not install the expected wheel."))
            environment["PYTHONPATH"] = str(target)
            profile = python.parent.parent.name.split("-", 1)[0]
            run_setup_command(
                module_command(python, "dantetl.runtime.probe", profile, "--application-only"),
                project,
                environment,
                cancel,
                lambda _: None,
                timeout=180,
            )
            if cancel.is_set():
                raise Cancelled()
            if bundled_application(project) != wheel:
                raise RuntimeError(tr("The release changed during its app update. Launch again."))
            pointer = root / "data/applications" / f"{wheel.sha256}.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            temporary = target / "published.json"
            temporary.write_text(json.dumps({"directory": target.name}), "utf-8")
            os.replace(temporary, pointer)
        except BaseException:
            # No pointer was published and owned children have drained. Never touch older code.
            checked = _application_path(root, target.name)
            shutil.rmtree(checked)
            raise
    return environment
