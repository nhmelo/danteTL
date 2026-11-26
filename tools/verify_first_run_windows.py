"""Native welcome/retry/progress fixture with synthetic bytes; never downloads real models."""

import argparse
import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import shiboken6
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from verification_isolation import app_owner
from verification_isolation import create_window as MainWindow

from dantetl.application.preferences import PreferenceStore
from dantetl.capture.windows import Windows
from dantetl.runtime import first_run, model_store
from dantetl.runtime.assets import ModelAsset
from dantetl.runtime.hardware_values import Adapter, CudaDevice, HardwareStatus
from dantetl.runtime.model_store import ModelBundle
from dantetl.runtime.setup_values import DownloadProgress, ModelProgress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/first-run/welcome"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output / uuid4().hex
    state.mkdir()
    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    data = b"fixture" * 1000000
    asset = ModelAsset("fixture", "1", "test.bin", hashlib.sha256(data).hexdigest(), len(data))
    models = tuple(
        ModelBundle(
            k,
            name,
            state / k,
            (asset,),
            "Synthetic verification bytes",
            category="recognition" if k == "ocr" else "translation",
        )
        for k, name in (
            ("ocr", "MeikiOCR"),
            ("gemma", "Gemma E2B · Recommended"),
            ("gemma-e4b", "E4B fixture"),
        )
    )
    bundled = state / "bundled"
    bundled.mkdir()
    (bundled / asset.filename).write_bytes(data)
    models = (replace(models[0], bundled_directory=bundled), *models[1:])
    hardware = Mock()
    hardware.snapshot.return_value = HardwareStatus(
        adapters=(
            Adapter(
                "NVIDIA test adapter", 0x10DE, 16 * 2**30, cuda=CudaDevice(13000, 8, 6, "a" * 32)
            ),
        ),
        llama_device="cpu",
    )
    window = MainWindow(
        windows,
        Mock(),
        models=models,
        hardware=hardware,
        preferences_path=state / "preferences.json",
    )
    application = app_owner(window)
    window.windows_combo.clear()
    window.show()
    window.offer_first_run()
    guide = window._setup_assistant
    stage, ticks = 0, 0
    failed = False
    evidence = {}
    started = time.monotonic()
    model_active = threading.Event()
    runtime_cancelled = threading.Event()

    def download(url, destination, limit, progress, cancel):
        start = destination.stat().st_size if destination.exists() else 0
        if start:
            evidence["resumed_model_bytes"] = start
        model_active.set()
        with destination.open("ab") as output:
            for offset in range(start, len(data), 256 * 1024):
                if cancel.wait(0.05):
                    raise model_store.Cancelled()
                chunk = data[offset : offset + 256 * 1024]
                output.write(chunk)
                progress(offset + len(chunk), len(data))

    def runtime_install(root, profile, progress, cancel, gpu):
        assert profile.key == "nvidia" and model_active.wait(5)
        for step in range(1, 31):
            progress(
                ModelProgress("Runtime fixture", download=DownloadProgress("fixture.whl", step, 30))
            )
            if cancel.wait(0.08):
                runtime_cancelled.set()
                raise model_store.Cancelled()

    def screenshot(name: str) -> None:
        assert guide
        app.processEvents()  # Let layouts settle after changing the page/text.
        guide.grab().save(str(args.output / name))

    def tick():
        nonlocal stage, ticks, failed, guide
        ticks += 1
        try:
            assert time.monotonic() - started < 45
            if guide is None:
                guide = window._setup_assistant
                if guide is None:
                    return
            if stage == 0:
                if guide.hardware_status is None:
                    return
                assert not guide.busy and models[0].included
                assert not any(b.present() for b in models[1:])
                screenshot("welcome.png")
                guide.next.click()
                assert guide.model.currentData() == "gemma"
                screenshot("models.png")
                guide.model.setCurrentIndex(guide.model.findData(""))
                guide.sources.setCurrentWidget(guide.cloud_tab)
                screenshot("cloud.png")
                assert guide.ocr_cuda.isVisible() and not guide.ocr_cuda.isChecked()
                guide.ocr_cuda.click()
                guide.next.click()
                review = guide.service.reviewed
                assert review and review.model_keys == () and review.install_runtime
                assert review.runtime == "nvidia" and not guide.busy
                screenshot("ocr-cuda-review.png")
                guide.back.click()
                guide.ocr_cuda.click()
                guide.sources.setCurrentIndex(0)
                guide.model.setCurrentIndex(guide.model.findData("gemma"))
                assert guide.model.currentData() == "gemma" and guide.model.findData("ocr") == -1
                screenshot("local-model.png")
                assert guide.ollama
                guide.sources.setCurrentWidget(guide.ollama)
                assert not guide.model.isVisible() and not guide.ollama.save_button.isEnabled()
                screenshot("ollama.png")
                guide.sources.setCurrentIndex(0)
                guide.next.click()
                assert guide.runtime.currentData().key == "nvidia"
                screenshot("acceleration.png")
                gpu_index = guide.runtime.currentIndex()
                guide.runtime.setCurrentIndex(0)
                guide.next.click()
                assert guide._review and not guide._review.install_runtime
                guide.back.click()
                guide.runtime.setCurrentIndex(gpu_index)
                guide.next.click()
                assert guide._review and guide._review.install_runtime
                assert guide._review.model == "gemma"
                screenshot("review.png")
                guide.next.click()
                stage = 1
            elif stage == 1:
                if not (
                    0 < guide.progress_view.download.value() < 1000
                    and guide.runtime_progress.download.isVisible()
                ):
                    return
                assert guide.busy and guide.progress_view.download.isVisible()
                assert not guide.progress_view.overall.isVisible()
                assert not guide.runtime_progress.overall.isVisible()
                guide.grab().save(str(args.output / "progress.png"))
                evidence["two_bars_with_real_bytes"] = True
                assert 0 < guide.runtime_progress.download.value() < 1000
                evidence["independent_model_and_runtime_progress"] = True
                guide.skip.click()
                stage = 2
            elif stage == 2:
                if guide.busy:
                    return
                assert "Retry" in guide.next.text()
                assert runtime_cancelled.is_set() and next(state.rglob("*.part")).stat().st_size > 0
                assert not list(state.rglob("*.tmp"))
                guide.next.click()
                stage = 3
            elif stage == 3:
                if guide.busy:
                    return
                assert guide._finished and models[0].present() and models[1].present()
                assert not models[2].present()
                stage = 6
            elif stage == 6:
                prompt = app.activeModalWidget()
                if prompt is None or prompt is guide:
                    return
                guide.grab().save(str(args.output / "complete.png"))
                assert isinstance(prompt, QMessageBox) and "NVIDIA" in prompt.text()
                prompt.grab().save(str(args.output / "restart.png"))
                prompt.button(QMessageBox.StandardButton.Cancel).click()
                guide.skip.click()
                evidence["deferred_restart"] = True
                stage = 4
            elif stage == 4:
                if not application.preferences.idle or not any(
                    item.key == "gemma" and item.present for item in application.setup.files
                ):
                    return
                assert PreferenceStore(state / "preferences.json").initial.setup_seen
                assert "Included with app" in window._settings.models.ocr.badge.text()
                assert "Downloaded" in window._settings.models.download_badge.text()
                assert not models[0].directory.exists()
                window.offer_first_run()
                assert not guide.isVisible()
                window._show_setup_assistant()
                guide = window._setup_assistant
                assert guide and guide.isVisible() and guide.pages.currentIndex() == 0
                guide.skip.click()
                window.close()
                stage = 5
            elif stage == 5:
                if window.isVisible():
                    return
                evidence.update(
                    explicit_downloads=True,
                    gpu_recommendation_cpu_override=True,
                    cancel_retry=True,
                    saved_completion=True,
                    reopen_from_settings=True,
                    clean_close=True,
                    responsive_ticks=ticks,
                )
                (args.output / "results.json").write_text(json.dumps(evidence, indent=2), "utf-8")
                print(json.dumps(evidence), flush=True)
                timer.stop()
                app.quit()
        except Exception:
            import traceback

            traceback.print_exc()
            failed = True
            timer.stop()
            window.close()
            app.quit()

    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(tick)
    timer.start()
    with (
        patch.object(model_store, "download_file", download),
        patch.object(first_run, "install_runtime", runtime_install),
        patch.object(first_run, "ready_runtime", return_value=state / "fixture-runtime"),
    ):
        app.exec()
    shiboken6.delete(app)
    return int(failed)


if __name__ == "__main__":
    from verification_isolation import run_isolated

    run_isolated(main)
