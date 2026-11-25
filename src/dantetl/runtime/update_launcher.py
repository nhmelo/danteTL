"""Bind release discovery and signed installation to the portable Windows distribution."""

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from dantetl.runtime.owned_process import OwnedProcess
from dantetl.runtime.release_values import Release, UpdateProgress
from dantetl.runtime.setup_process import clean_runtime_environment
from dantetl.runtime.update_install import stage_release
from dantetl.runtime.verified_file import check_cancel


def install_release(
    release: Release,
    cancel: threading.Event,
    progress: Callable[[UpdateProgress], None],
    *,
    project: Path,
    state: Path,
    key: Ed25519PublicKey,
) -> None:
    bundle = project.parent.parent
    if state.resolve().is_relative_to(bundle):
        raise ValueError("Move application data outside the portable folder before updating.")
    work = stage_release(release, bundle, key, cancel, progress)
    try:
        # The helper lives outside both folders so Windows can rename them.
        directory = state / "updates"
        directory.mkdir(parents=True, exist_ok=True)
        helper = directory / "DanteTL-update.exe"
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            shutil.copyfile(project / "DanteTL-update.exe", temporary)
            temporary.replace(helper)
        finally:
            temporary.unlink(missing_ok=True)
        environment = clean_runtime_environment(os.environ)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        check_cancel(cancel)
        process = OwnedProcess(
            [
                str(helper),
                str(bundle),
                str(work),
                str(state),
                str(os.getpid()),
                release.version,
            ],
            directory,
            environment,
        )
        try:
            deadline = time.monotonic() + 20
            while not (work / "helper.ready").is_file():
                check_cancel(cancel)
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("The update helper could not start.")
                cancel.wait(0.05)
            process.release()
        except BaseException:
            process.close()
            raise
    except BaseException:
        shutil.rmtree(work)
        raise
