"""Verify the relocated PyInstaller EXE, offline OCR and optional desktop startup."""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import cv2
import numpy as np

from dantetl.capture.windows import Windows
from dantetl.domain import WindowTarget
from dantetl.runtime.owned_process import OwnedProcess
from dantetl.runtime.setup_process import run_setup_command

ROOT = Path(__file__).resolve().parents[1]


def files(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result


def wait(process: OwnedProcess, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while process.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Packaged process did not finish")
        time.sleep(0.1)
    assert process.returncode == 0, process.returncode


def verify_cpu_variants(bundle: Path) -> None:
    internal = bundle / "_internal"
    (wheel,) = (internal / "app/cpu-runtime").glob("*.whl")
    with ZipFile(wheel) as archive:
        variants = {
            Path(name).name: hashlib.sha256(archive.read(name)).digest()
            for name in archive.namelist()
            if name.startswith("llama_cpp/lib/ggml-cpu-") and name.endswith(".dll")
        }
    frozen = {
        path.name: hashlib.sha256(path.read_bytes()).digest()
        for path in (internal / "llama_cpp/lib").glob("ggml-cpu*.dll")
    }
    assert {"ggml-cpu-x64.dll", "ggml-cpu-haswell.dll", "ggml-cpu-skylakex.dll"} <= variants.keys()
    assert frozen == variants, "Bundled app and CPU installer must ship the same CPU backends"


def download_helper(exe: Path, output: Path, environment: dict[str, str]) -> None:
    """The windowed EXE must stream progress through its inherited pipe without Python on PATH."""
    payload = b"synthetic download fixture" * 20000
    split = len(payload) // 3
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            ranges.append(requested)
            start = split if requested == f"bytes={split}-" else 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:] if start else payload[:split])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    temporary = output / "download.part"
    progress = []
    thread.start()
    try:
        run_setup_command(
            [
                str(exe),
                "--module",
                "dantetl.runtime.download_transport",
                f"http://127.0.0.1:{server.server_port}/fixture",
                str(temporary),
                str(len(payload)),
            ],
            output,
            environment,
            threading.Event(),
            progress.append,
            timeout=15,
        )
        assert temporary.read_bytes() == payload
        assert ranges == [None, f"bytes={split}-"]
        assert json.loads(progress[-1]) == {"received": len(payload), "total": len(payload)}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        temporary.unlink(missing_ok=True)
        assert not thread.is_alive()


def desktop(
    exe: Path, output: Path, environment: dict[str, str], state: Path, welcome: str
) -> None:
    token = uuid4().hex
    process = OwnedProcess(
        [str(exe)], output, environment | {"DANTETL_RESTART_TOKEN": token}, show_window=True
    )
    receipt = state / "data/restarts" / token
    windows = Windows()

    def owned_windows():
        found = []

        def visit(hwnd, _):
            if windows.process_id(hwnd) == process.pid and windows.user.IsWindowVisible(hwnd):
                found.append(WindowTarget(hwnd, process.pid, windows.title(hwnd)))
            return True

        windows.user.EnumWindows(windows._enum_callback(visit), 0)
        return found

    try:
        deadline = time.monotonic() + 45
        while not receipt.is_file():
            assert process.poll() is None, "App exited before Qt startup"
            assert time.monotonic() < deadline, "App did not acknowledge Qt startup"
            time.sleep(0.1)
        pid = int(receipt.read_text("ascii"))
        assert pid == process.pid, "Bundled CPU must run inside the actual PyInstaller EXE"
        while True:
            owned = owned_windows()
            if any(t.title == welcome for t in owned):
                break
            assert time.monotonic() < deadline, "Fresh app did not show its welcome"
            time.sleep(0.1)
        post = ctypes.WinDLL("user32").PostMessageW
        post.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)
        for title in (welcome, "DanteTL"):
            owned = owned_windows()
            while not any(t.title == title for t in owned):
                assert time.monotonic() < deadline, (title, owned)
                time.sleep(0.1)
                owned = owned_windows()
            for target in owned:
                if target.title == title:
                    post(target.hwnd, 0x0010, 0, 0)
            while any(t.title == title for t in owned_windows()):
                assert time.monotonic() < deadline, "App did not close its owned window"
                time.sleep(0.1)
        wait(process, 15)
    finally:
        process.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "dist/DanteTL-windows-x64.zip")
    parser.add_argument("--bundle", type=Path, help="Verify an already built/extracted bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", help="Skip desktop startup on CI")
    parser.add_argument("--invalid-selection", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        raise ValueError("Use a new directory under this project's artifacts folder.")
    output.mkdir(parents=True)
    if args.bundle:
        bundle = args.bundle.resolve()
    else:
        relocated = output / "Moved app 日本語"
        with ZipFile(args.zip) as archive:
            for name in archive.namelist():
                if not (relocated / name).resolve().is_relative_to(relocated):
                    raise ValueError("ZIP path escapes extraction.")
            archive.extractall(relocated)
        bundle = relocated / "DanteTL"
    before = files(bundle)
    verify_cpu_variants(bundle)
    assert not (bundle / "_internal/app/src").exists(), "No duplicate application source tree"
    assert not (bundle / "_internal/app/bootstrap.py").exists(), "No custom runtime bootstrap"
    exe = bundle / "DanteTL.exe"
    image = exe.read_bytes()
    pe = struct.unpack_from("<I", image, 0x3C)[0]
    assert struct.unpack_from("<H", image, pe + 24 + 68)[0] == 2, "Must be a GUI executable"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "DANTETL_", "_PYI"))
        and key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_QPA_PLATFORM")
    }
    environment.update(
        PATH=str(Path(os.environ["SYSTEMROOT"]) / "System32"),
        LOCALAPPDATA=str(output / "AppData"),
        PYINSTALLER_RESET_ENVIRONMENT="1",
        DANTETL_DISABLE_UPDATE_CHECK="1",
    )
    state = output / "AppData/DanteTL"
    saved = {"profile": "nvidia", "environment": "nvidia-" + "0" * 32}
    if args.invalid_selection:
        (state / "data").mkdir(parents=True)
        (state / "data/runtime.json").write_text(json.dumps(saved), encoding="utf-8")
    report = output / "runtime.json"
    latin_fixture = output / "latin.png"
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "Open inventory",
        (120, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    latin_fixture.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    process = OwnedProcess(
        [
            str(exe),
            "--module",
            "dantetl.runtime.probe",
            "cpu",
            "--ocr-image",
            str(ROOT / "tests/fixtures/ocr.png"),
            "--latin-ocr-image",
            str(latin_fixture),
            "--report",
            str(report),
        ],
        output,
        environment | {"QT_QPA_PLATFORM": "offscreen"},
    )
    try:
        wait(process, 180)
    finally:
        process.close()
    result = json.loads(report.read_text("utf-8"))
    assert Path(result["ocr"]["models"]) == bundle / "_internal/app/models/meiki"
    assert Path(result["latin_ocr"]["models"]) == bundle / "_internal/app/models/paddle-latin-v5"
    download_helper(exe, output, environment)
    if not args.headless:
        desktop(exe, output, environment, state, "DanteTL · Welcome")
        assert (state / "logs/app.log").is_file()
        # A saved interface locale must load from the relocated release before setup appears.
        from PySide6.QtCore import QTranslator

        prefs = state / "data/preferences.json"
        for code in ("pt-PT", "ja"):
            catalog = QTranslator()
            assert catalog.load(str(ROOT / f"src/dantetl/ui/locales/{code}.qm"))
            welcome = catalog.translate("DanteTL", "DanteTL · Welcome")
            assert welcome and welcome != "DanteTL · Welcome"
            prefs.write_text(
                json.dumps({"version": 1, "setup_seen": False, "interface_language": code}), "utf-8"
            )
            desktop(exe, output, environment, state, welcome)
    assert not (state / "models").exists(), "No implicit model download/copy"
    assert not (state / ".runtimes").exists(), "No implicit runtime installation"
    if args.invalid_selection:
        assert json.loads((state / "data/runtime.json").read_text("utf-8")) == saved
    assert files(bundle) == before, "Application resources were modified"
    result.update(
        bundle=str(bundle),
        state=str(state),
        no_python_on_path=True,
        relocated_exe_started=not args.headless,
        localized_welcome=not args.headless,
        unicode_path=any(ord(c) > 127 for c in str(bundle)),
        no_implicit_downloads=True,
        no_files_written_in_app_folder=True,
        verified_files=len(before),
        windowed_download_helper=True,
        invalid_selection=args.invalid_selection,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
