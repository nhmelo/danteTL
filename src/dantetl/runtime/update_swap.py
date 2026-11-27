"""Standalone Windows helper. Hold the old release until the new GUI acknowledges startup."""

import ctypes
import hashlib
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from packaging.version import Version

from dantetl.atomic_file import replace_file
from dantetl.file_lock import file_lock
from dantetl.runtime.owned_process import OwnedProcess
from dantetl.runtime.package import bundled_application, check_portable_folder
from dantetl.runtime.release_values import ISSUES_URL, MANIFEST, SIGNATURE


@contextmanager
def bundle_lease(install: Path, state: Path, *, exclusive: bool = False):
    """GUI instances share the lease. Directory replacement requires exclusive ownership."""
    identity = hashlib.sha256(os.path.normcase(str(install.resolve())).encode()).hexdigest()
    path = state / "data" / f"portable-{identity}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 0 if exclusive else 1, None, 4, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError("Another DanteTL instance is using this installation.")
    try:
        yield
    finally:
        kernel.CloseHandle(handle)


def wait_for_exit(pid: int) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # The parent already exited.
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if kernel.WaitForSingleObject(handle, 120_000) != 0:
            raise TimeoutError("DanteTL did not finish closing.")
    finally:
        kernel.CloseHandle(handle)


def start_gui(install: Path, state: Path, expected: str, *, notice: str = "") -> None:
    token = uuid4().hex
    receipt = state / "data/restarts" / token
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("_PYI", "PYTHON", "UV_"))
        and k
        not in {
            "VIRTUAL_ENV",
            "DANTETL_RESOURCE_DIR",
            "DANTETL_RESTART_TOKEN",
            "DANTETL_STARTUP_NOTICE",
        }
    }
    if notice:
        environment["DANTETL_STARTUP_NOTICE"] = notice
    environment.update(
        DANTETL_UPDATE_TOKEN=token, DANTETL_DATA_DIR=str(state), PYINSTALLER_RESET_ENVIRONMENT="1"
    )
    child = OwnedProcess(
        [str(install / "DanteTL.exe")], install.parent, environment, show_window=True
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if receipt.is_file() and receipt.read_text("utf-8") == expected:
                child.release()
                return
            if child.poll() not in (None, 0):
                break
            time.sleep(0.1)
        raise RuntimeError("The replacement GUI did not acknowledge startup.")
    except BaseException:
        child.close()
        raise
    finally:
        receipt.unlink(missing_ok=True)


def check_transaction_paths(install: Path, work: Path, state: Path) -> None:
    if (
        work.parent != install.parent
        or not work.name.startswith(".dantetl-update-")
        or work == install
        or state.resolve().is_relative_to(install)
        or state.resolve().is_relative_to(work)
        or install.resolve() != install
        or work.resolve() != work
    ):
        raise ValueError("Invalid update directories.")


def swap(
    install: Path, work: Path, state: Path, expected: str, *, launch: Callable = start_gui
) -> None:
    check_transaction_paths(install, work, state)
    previous, candidate = work / "previous", work / "new"
    with file_lock(state / "data/update-install.lock"):
        check_portable_folder(candidate)
        check_portable_folder(install)
        old = bundled_application(install / "_internal/app")
        if old.sha256 != (work / "previous.sha256").read_text("ascii"):
            raise ValueError("The installation changed while the update was being prepared.")
        if (
            Version(expected) <= Version(old.version)
            or bundled_application(candidate / "_internal/app").version != expected
        ):
            raise ValueError("Staged release changed.")
        # Moving directories within the same volume preserves the complete old installation.
        with bundle_lease(install, state, exclusive=True):
            install.rename(previous)
            try:
                candidate.rename(install)
            except OSError:
                previous.rename(install)
                raise
        try:
            launch(install, state, expected)
        except BaseException:
            with bundle_lease(install, state, exclusive=True):
                install.rename(candidate)
                previous.rename(install)
            shutil.rmtree(work, ignore_errors=True)
            raise
    with suppress(OSError):
        shutil.rmtree(work)


def main() -> None:
    install, work, state = (Path(value).absolute() for value in sys.argv[1:4])
    check_transaction_paths(install, work, state)
    check_portable_folder(work / "new")
    if {p.name for p in work.iterdir()} != {"new", "previous.sha256", MANIFEST, SIGNATURE}:
        raise ValueError("Unexpected staging contents.")
    (work / "helper.ready").touch(exist_ok=False)
    departed = False
    try:
        wait_for_exit(int(sys.argv[4]))
        departed = True
        swap(install, work, state, sys.argv[5])
    except Exception as error:
        # Never remove a backup after an incomplete rollback.
        log = state / "logs/update-error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        details = traceback.format_exc()
        log.write_text(f"{error}\nStaging directory: {work}\n\n{details}", "utf-8")
        message = f"DanteTL could not finish updating.\n{error}\n\n{log}"
        if departed and install.is_dir() and not (work / "previous").exists():
            if (work / "new").is_dir() and (work / "previous.sha256").is_file():
                shutil.rmtree(work, ignore_errors=True)
            try:
                start_gui(
                    install,
                    state,
                    bundled_application(install / "_internal/app").version,
                    notice=message,
                )
                return
            except (OSError, RuntimeError, ValueError):
                pass
        ctypes.windll.user32.MessageBoxW(None, f"{message}\n\n{ISSUES_URL}", "DanteTL", 0x10)


def acknowledge_update(state: Path, version: str) -> None:
    token = os.environ.pop("DANTETL_UPDATE_TOKEN", "")
    if len(token) == 32 and all(c in "0123456789abcdef" for c in token):
        replace_file(state / "data/restarts" / token, version.encode("utf-8"))
