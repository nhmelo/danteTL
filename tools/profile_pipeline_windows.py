"""Time our own changing text fixture; no game input, downloads or user preferences.

Reports publication latency (not monitor scan-out), from Qt fixture painting to
English entering the overlay. Observer callbacks do not alter processing policy.
Run serially with a local GPU runtime; initialization is excluded from trials.
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from pathlib import Path

import shiboken6
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent
from PySide6.QtWidgets import QApplication, QWidget
from verification_isolation import app_owner

from dantetl.application.model_discovery import translation_backends
from dantetl.application.model_factories import create_recognizer
from dantetl.application.preference_values import MAX_CAPTURE_RATE
from dantetl.capture.windows import Windows
from dantetl.composition import application_session, setup_coordinator
from dantetl.domain import WindowTarget
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.ocr.worker import OcrChoice
from dantetl.paths import state_root
from dantetl.translation.backend import (
    BatchResult,
    ContextualBackend,
    LabelDecoder,
    ParallelLabelBackend,
    Source,
)
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.values import TranslationState
from dantetl.ui.demo import DIALOGUE_LINES
from dantetl.ui.desktop_driver import DesktopDriver
from dantetl.ui.inspection import frame_image
from dantetl.ui.main_window import MainWindow


class TextFixture(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DanteTL latency fixture")
        self.resize(900, 580)
        self.lines = ("開始",)
        self.dialogue = False
        self.painted_at = 0.0
        self.pending_paint = True
        self.phase = 0
        self.heartbeat = QTimer(self)
        self.heartbeat.setInterval(16)
        self.heartbeat.timeout.connect(self.animate)
        self.heartbeat.start()

    def animate(self) -> None:
        self.phase = (self.phase + 1) % 600
        self.update()

    def present(self, lines: tuple[str, ...], dialogue: bool = False) -> None:
        self.lines, self.dialogue = lines, dialogue
        self.painted_at = 0
        self.pending_paint = True
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 35, 55))
        # Fresh frames like a running game, without motion behind the text.
        painter.fillRect(self.phase, 570, 20, 4, QColor(55, 75, 95))
        font = QFont("Yu Gothic UI")
        font.setPixelSize(26)
        painter.setFont(font)
        painter.setPen(QColor(245, 245, 245))
        for i, line in enumerate(self.lines):
            x, y = (60, 380 + i * 36) if self.dialogue else (60 + i // 3 * 500, 100 + i % 3 * 145)
            painter.drawText(x, y, line)
        painter.end()
        if self.pending_paint:
            self.painted_at = time.monotonic()
            self.pending_paint = False


def normalized(text: str) -> str:
    return "".join(text.split())


class ProfileWindow(MainWindow):
    trial: dict | None = None
    expected: set[str] = set()
    record_directory = Path()

    def _render_application(self, inspect: bool) -> None:
        application = app_owner(self)
        tracks = application.view.scene.requests
        trial = self.trial
        if trial is not None:
            now = time.monotonic()
            ocr = application.live.ocr.snapshot() if application.live.ocr else None
            result = ocr.result if ocr else None
            scans = trial.setdefault("scans", [])
            if (
                result
                and result.frame.captured_at >= trial["painted_at"]
                and len(scans) < 16
                and not any(s["sequence"] == result.frame.sequence for s in scans)
            ):
                scans.append(
                    dict(
                        sequence=result.frame.sequence,
                        source_ms=round((result.frame.captured_at - trial["painted_at"]) * 1000, 1),
                        ocr_ms=result.duration_ms,
                        coverage=None
                        if result.scanned is None
                        else [(b.x, b.y, b.width, b.height) for b in result.scanned],
                        text=[r.text for r in result.regions],
                        confidence=[r.confidence for r in result.regions],
                    )
                )
                if len(scans) == 1 and trial["case"] == "single-cold":
                    frame_image(result.frame).save(str(self.record_directory / "source.png"))
            if result and any(
                normalized(region.text) in source
                for region in result.regions
                for source in self.expected
                if normalized(region.text)
            ):
                if "ocr_completed_at" not in trial:
                    trial.update(
                        source_at=result.frame.captured_at,
                        ocr_completed_at=result.completed_at,
                        ocr_ms=result.duration_ms,
                        ocr_wait_ms=ocr.queue_ms if ocr else 0,
                        provider=ocr.provider if ocr else "",
                        source_sequence=result.frame.sequence,
                    )
                if application.live.accepted is result:
                    trial.setdefault("accepted_at", now)
            if any(normalized(block.text) in self.expected for block in tracks):
                trial.setdefault("eligible_at", now)
            for block in application.view.scene.candidates:
                if normalized(block.text) in self.expected:
                    trial.setdefault("first_read_ms", {}).setdefault(
                        block.text, round((now - trial["painted_at"]) * 1000, 1)
                    )
        super()._render_application(inspect)
        if trial is not None and application.translation.lifecycle.publication:
            snapshot = application.translation.lifecycle.publication.snapshot()
            translated = tuple(
                item
                for item in self._overlay._translations
                if normalized(item.request.source) in self.expected
            )
            trial["translated"] = {t.request.source: t.english for t in translated}
            if translated:
                trial.setdefault("first_english_at", time.monotonic())
                trial.setdefault("first_queue_ms", snapshot.queue_ms)
                trial.setdefault("first_inference_ms", snapshot.last_inference_ms)
                trial.setdefault("first_cache_hits", snapshot.cache_hits)
                trial.setdefault("first_model_sources", snapshot.translated_sources)
            if len(translated) == len(self.expected):
                trial.setdefault("all_english_at", time.monotonic())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemma", "gemma-e4b", "qwen35-4b"), default="gemma")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--capture-rate", type=int, default=8, help="Samples/s, or 0 for unlimited")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument(
        "--confirmed-only",
        action="store_true",
        help="Measure the control policy by withholding first readings from translation",
    )
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 61))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.capture_rate <= MAX_CAPTURE_RATE:
        parser.error(f"--capture-rate must be between 0 and {MAX_CAPTURE_RATE}")
    args.output.mkdir(parents=True, exist_ok=True)
    unique = str(time.time_ns())
    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    fixture = TextFixture()
    fixture.show()
    backend = next(b for b in translation_backends() if b.preference_key == args.model)
    if backend.unavailable_reason:
        raise RuntimeError(backend.unavailable_reason)
    factory = partial(create_recognizer, device=args.device)
    calls: deque[dict] = deque(maxlen=128)
    calls_lock = threading.Lock()

    class TimedDecoder:
        def __init__(self, decoder: LabelDecoder) -> None:
            self.decoder = decoder
            self.width = decoder.width

        def submit(self, source: Source, target: TargetLanguage) -> Future[BatchResult] | None:
            started = time.perf_counter()
            future = self.decoder.submit(source, target)
            if future is not None:

                def record(done: Future[BatchResult]) -> None:
                    if not done.cancelled():
                        with calls_lock:
                            calls.append(
                                dict(source=source[0], ms=(time.perf_counter() - started) * 1000)
                            )

                future.add_done_callback(record)
            return future

        def step(self) -> None:
            self.decoder.step()

        def reset(self) -> None:
            self.decoder.reset()

        def close(self) -> None:
            self.decoder.close()

    class TimedBackend:
        def __init__(self) -> None:
            engine = backend.create()
            assert isinstance(engine, ContextualBackend)
            self.engine = engine
            self.batch_size, self.cache_identity = engine.batch_size, engine.cache_identity

        def translate(self, texts: tuple[str, ...], *, target=ENGLISH) -> tuple[str, ...]:
            return tuple(self.translate_with_context(text, (), target=target) for text in texts)

        def translate_with_context(
            self, text: str, context: tuple[DialogueTurn, ...], *, target=ENGLISH
        ) -> str:
            started = time.perf_counter()
            answer = self.engine.translate_with_context(text, context, target=target)
            with calls_lock:
                calls.append(dict(source=text, ms=(time.perf_counter() - started) * 1000))
            return answer

        def close(self) -> None:
            self.engine.close()

        @property
        def label_width(self) -> int:
            return self.engine.label_width if isinstance(self.engine, ParallelLabelBackend) else 0

        def prepare_labels(self, target: TargetLanguage) -> LabelDecoder | None:
            if isinstance(self.engine, ParallelLabelBackend):
                decoder = self.engine.prepare_labels(target)
                if decoder:
                    return TimedDecoder(decoder)
            return None

    from verification_isolation import create_cloud, create_inventory

    driver = DesktopDriver()
    ocr = (OcrChoice(args.device, factory, args.capture_rate),)
    inventory, source_models = create_inventory(
        (windows, factory, (replace(backend, preparation_id="", create=TimedBackend),)),
        {"ocr_backends": ocr},
    )
    application = application_session(
        windows,
        factory,
        cloud=create_cloud(),
        notify=driver.notify,
        inventory=inventory,
        source_models=source_models,
        setup=setup_coordinator(inventory, (), state_root()),
        cache_path=args.output / f"cache-{unique}.sqlite3",
        preferences_path=args.output / f"preferences-{unique}.json",
        ocr_backends=ocr,
    )
    window = ProfileWindow(application, windows, driver)
    application = app_owner(window)
    if args.confirmed_only:
        scene = application.live.scene

        def confirmed_scene(*args, **kwargs):
            return replace(scene(*args, **kwargs), candidates=())

        application.live.scene = confirmed_scene
    window.record_directory = args.output
    window.show()
    window._settings.inspection_enabled.setChecked(args.advanced)
    window.advanced.setChecked(args.advanced)
    window.capture_rate.setValue(args.capture_rate)
    window.windows_combo.addItem(
        "Our latency fixture",
        WindowTarget(int(fixture.winId()), os.getpid(), fixture.windowTitle()),
    )
    window.windows_combo.setCurrentIndex(window.windows_combo.count() - 1)
    window.start_button.click()
    fixture.raise_()
    fixture.activateWindow()
    cases = [
        ("single-cold", ("設定",), False),
        ("single-cached", ("設定",), False),
        (
            "menu-cold",
            ("キャラクター", "装備を確認", "攻撃力", "防御力", "速度", "詳細を表示"),
            False,
        ),
        (
            "menu-cached",
            ("キャラクター", "装備を確認", "攻撃力", "防御力", "速度", "詳細を表示"),
            False,
        ),
        ("dialogue-cold", DIALOGUE_LINES, True),
    ]
    cases *= args.repeats
    results: list[dict] = []
    phase = "warmup"
    changed_at = started = time.monotonic()
    index = 0
    failed = False

    refresh_ms: deque[float] = deque(maxlen=100_000)
    heartbeat_ms: deque[float] = deque(maxlen=100_000)
    last_heartbeat = time.monotonic()
    owner, method = (DesktopDriver, "refresh")
    original_refresh = getattr(owner, method)

    def observed_refresh(self, *a, **kw):
        before = time.perf_counter()
        try:
            return original_refresh(self, *a, **kw)
        finally:
            if phase not in ("warmup", "closing"):
                refresh_ms.append((time.perf_counter() - before) * 1000)

    setattr(owner, method, observed_refresh)

    def distribution(values):
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }

    def tick() -> None:
        nonlocal phase, changed_at, index, failed, last_heartbeat
        now = time.monotonic()
        if phase not in ("warmup", "closing"):
            heartbeat_ms.append(max(0.0, (now - last_heartbeat) * 1000 - 10))
        last_heartbeat = now
        try:
            if phase == "closing":
                if window.isVisible():
                    return
                fixture.close()
                timer.stop()
                app.quit()
                return
            assert now - started < 120 * args.repeats, "Profile timed out"
            snapshot = (
                application.translation.lifecycle.publication.snapshot()
                if application.translation.lifecycle.publication
                else None
            )
            assert snapshot is None or snapshot.state != TranslationState.FAILED, snapshot
            if phase == "warmup":
                if not snapshot or not snapshot.results:
                    return
                phase, changed_at = "blank", now
                fixture.present(())
            elif phase == "blank":
                if now - changed_at < 1.0 or application.live._tracker._tracks:
                    return
                name, lines, dialogue = cases[index]
                window.expected = (
                    {normalized("".join(lines))} if dialogue else set(map(normalized, lines))
                )
                fixture.present(lines, dialogue)
                with calls_lock:
                    calls.clear()
                phase, changed_at = "painting", now
            elif phase == "painting":
                if not fixture.painted_at:
                    return
                window.trial = {"case": cases[index][0], "painted_at": fixture.painted_at}
                phase, changed_at = "trial", now
            elif phase == "trial":
                assert window.trial is not None
                if "all_english_at" not in window.trial:
                    assert now - changed_at < 15, (window.trial, window.expected)
                    return
                result = window.trial
                with calls_lock:
                    result["model_calls"] = list(calls)
                for key, value in tuple(result.items()):
                    if key.endswith("_at"):
                        result[key.removesuffix("_at") + "_ms"] = round(
                            (value - result["painted_at"]) * 1000, 1
                        )
                result["inference_note"] = (
                    "Last inference stays visible on cache hits; check first_model_sources."
                )
                results.append(result)
                window.trial = None
                if args.advanced:
                    window.processing.grab().save(
                        str(args.output / f"{cases[index][0]}-metrics.png")
                    )
                index += 1
                if index == len(cases):
                    report = dict(
                        model=args.model,
                        device=args.device,
                        capture_rate=args.capture_rate,
                        advanced=args.advanced,
                        trials=results,
                        refresh_ms=distribution(refresh_ms),
                        heartbeat_lateness_ms=distribution(heartbeat_ms),
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
                    )
                    print(
                        json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True
                    )
                    window.close()
                    phase = "closing"
                else:
                    fixture.present(())
                    phase, changed_at = "blank", now
        except Exception:
            import traceback

            traceback.print_exc()
            fixture.grab().save(str(args.output / "failed-fixture.png"))
            if application.live.ocr:
                snapshot = application.live.ocr.snapshot()
                print("OCR:", snapshot.state, snapshot.message, flush=True)
                if snapshot.result:
                    print([r.text for r in snapshot.result.regions], flush=True)
            failed = True
            window.trial = None
            window.close()
            phase = "closing"

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(tick)
    timer.start()
    app.exec()
    shiboken6.delete(app)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
