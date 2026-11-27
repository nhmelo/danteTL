"""One OCR worker, one pending frame, and generation-checked immutable results."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.i18n import tr
from dantetl.ocr.fixed_line import LineReader, reread_line
from dantetl.ocr.icon_grid import split_icon_grids
from dantetl.ocr.regions import intersects, scan_regions
from dantetl.ocr.scan_plan import ScanCosts, ScanLine, ScanPlan, ScanPlanner, preserves_lines
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextRegion

logger = logging.getLogger(__name__)


class Recognizer(Protocol):
    def recognize(
        self, frame: CapturedFrame, /, *, full_size: tuple[int, int] | None = None
    ) -> tuple[TextRegion, ...]: ...
    def close(self) -> None: ...


@runtime_checkable
class CostedRecognizer(Protocol):
    @property
    def scan_costs(self) -> ScanCosts | None: ...


class _ScanWindow:
    """Rate-limit repeated work while admitting independent crops to an idle reader."""

    def __init__(self) -> None:
        self.until = float("-inf")
        self.areas: tuple[Rect, ...] | None = ()

    def delay(self, areas: tuple[Rect, ...] | None, now: float) -> float:
        if now >= self.until:
            return 0.0
        previous = self.areas
        if (
            areas is not None
            and previous is not None
            and len(areas) + len(previous) <= 4
            and any(all(not intersects(area, old) for old in previous) for area in areas)
        ):
            return 0.0
        return self.until - now

    def start(self, areas: tuple[Rect, ...] | None, now: float) -> None:
        if now >= self.until:
            self.until = now + 0.12
            self.areas = ()
        self.areas = None if areas is None or self.areas is None else self.areas + areas


@dataclass(frozen=True, slots=True)
class OcrChoice:
    name: str
    create: Callable[[], Recognizer]
    capture_fps: int = 8
    key: str = ""


class OcrSession:
    """Own models on a worker thread. No Qt, Windows calls, or unbounded job queues.

    A compatible source switch may reuse the worker. Invalidation discards in-flight results after
    resize/minimize; inspection pause is independent. Shutdown drains inference and cleanup.
    """

    def __init__(
        self,
        factory: Callable[[], Recognizer],
        *,
        require_ack: bool = False,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._factory = factory
        self._notify = on_change or (lambda: None)
        self._condition = threading.Condition()
        self._done = threading.Event()
        self._stop = False
        self._generation = 0
        self._scan_window = _ScanWindow()
        self._last_sequence = 0
        self._pending: tuple[int, CapturedFrame, tuple[Rect, ...] | None] | None = None
        self._require_ack = require_ack
        self._acknowledged = 0
        self._snapshot = OcrSnapshot(OcrState.LOADING)
        self._processed = 0
        self._skipped = 0
        self._queue_ms = 0.0
        self._durations: deque[float] = deque(maxlen=8)
        self._processing_since: float | None = None
        self._thread = threading.Thread(target=self._run, name="ocr-owner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: CapturedFrame, areas: tuple[Rect, ...] | None = None) -> None:
        if areas == ():
            return
        with self._condition:
            if (
                self._stop
                or self._snapshot.state == OcrState.FAILED
                or frame.sequence <= self._last_sequence
            ):
                return
            self._last_sequence = frame.sequence
            if self._pending is not None:
                self._skipped += 1
                prior = self._pending[2]
                areas = None if areas is None or prior is None else (*prior, *areas)
            areas = scan_regions(areas, frame.geometry.client.size)
            if areas == ():
                return
            self._pending = self._generation, frame, areas
            self._condition.notify()

    def acknowledge(self, sequence: int) -> None:
        """Release the next job only after the current result reached tracking."""
        with self._condition:
            self._acknowledged = max(self._acknowledged, sequence)
            self._condition.notify()

    def invalidate(self) -> None:
        with self._condition:
            self._generation += 1
            self._scan_window = _ScanWindow()
            self._last_sequence = 0
            self._acknowledged = 0
            self._pending = None
            self._snapshot = replace(self._snapshot, result=None)
            self._condition.notify()

    def request_stop(self) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._snapshot = OcrSnapshot(
                self._snapshot.state
                if self._snapshot.state in (OcrState.FAILED, OcrState.STOPPED)
                else OcrState.STOPPING,
                message=self._snapshot.message,
            )
            self._condition.notify()

    @property
    def finished(self) -> bool:
        return self._done.is_set() and not self._thread.is_alive()

    def snapshot(self) -> OcrSnapshot:
        with self._condition:
            return replace(
                self._snapshot,
                processed_frames=self._processed,
                skipped_frames=self._skipped,
                queue_ms=self._queue_ms,
                mean_duration_ms=sum(self._durations) / len(self._durations)
                if self._durations
                else 0,
                processing_since=self._processing_since,
            )

    def _ready(self, engine: Recognizer, result: OcrResult | None = None) -> OcrSnapshot:
        return OcrSnapshot(
            OcrState.READY,
            result,
            str(getattr(engine, "message", "")),
            provider=str(getattr(engine, "provider", "")),
        )

    def _run(self) -> None:
        engine: Recognizer | None = None
        try:
            with self._condition:
                if self._stop:
                    return
            engine = self._factory()
            reader = engine if isinstance(engine, LineReader) else None
            planner = ScanPlanner()
            used_detector = False
            with self._condition:
                if self._stop:
                    return
                self._snapshot = self._ready(engine)
            self._notify()
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._stop
                            or (
                                self._pending is not None
                                and (
                                    not self._require_ack
                                    or self._snapshot.result is None
                                    or self._snapshot.result.frame.sequence <= self._acknowledged
                                )
                            )
                        )
                    )
                    if self._stop:
                        return
                    job = self._pending
                    assert job is not None
                    now = time.monotonic()
                    delay = self._scan_window.delay(job[2], now)
                    if delay > 0:
                        del job
                        self._condition.wait(timeout=delay)
                        continue
                    self._scan_window.start(job[2], now)
                    previous = self._snapshot.result if used_detector else None
                    self._pending = None
                    self._processing_since = time.monotonic()
                generation, frame, areas = job
                started = time.monotonic()
                costs = engine.scan_costs if isinstance(engine, CostedRecognizer) else None
                plan = planner.plan(areas, frame.geometry.client.size, generation, costs)
                direct = (
                    reread_line(reader, frame, previous, areas[0])
                    if previous and areas and len(areas) == 1 and reader is not None
                    else None
                )
                if direct is None:
                    regions, scanned, rejected = recognize_planned(engine, frame, plan)
                else:
                    line, sample = direct
                    regions, scanned, rejected = (line,), (sample,), False
                used_detector = direct is None
                completed = time.monotonic()
                result = OcrResult(frame, regions, (completed - started) * 1000, completed, scanned)
                with self._condition:
                    self._processing_since = None
                    published = not self._stop and generation == self._generation
                    if published:
                        planner.observe(
                            tuple(
                                ScanLine(r.bounds, r.confidence, r.vertical)
                                for r in regions
                                if r.text.strip()
                            ),
                            scanned,
                            plan,
                            rejected=rejected,
                        )
                        self._processed += 1
                        self._queue_ms = max(0, (started - frame.captured_at) * 1000)
                        self._durations.append(result.duration_ms)
                        self._snapshot = self._ready(engine, result)
                if published:
                    self._notify()
                # Do not keep old frame buffers alive through the next idle wait.
                del job, frame, result, previous
        except Exception as error:
            logger.exception("Local OCR failed")
            with self._condition:
                self._snapshot = OcrSnapshot(OcrState.FAILED, message=str(error))
            self._notify()
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception as error:
                    logger.exception("OCR cleanup failed")
                    with self._condition:
                        self._snapshot = OcrSnapshot(OcrState.FAILED, message=str(error))
                finally:
                    engine = None
            with self._condition:
                self._pending = None
                self._processing_since = None
                if self._snapshot.state != OcrState.FAILED:
                    self._snapshot = OcrSnapshot(OcrState.STOPPED)
            self._done.set()
            self._notify()


def _layout(frame: CapturedFrame, regions: tuple[TextRegion, ...]) -> tuple[TextRegion, ...]:
    """Layout that reaches past a scanned crop (icons above its labels) uses the whole source."""
    if not any(r.characters for r in regions) or len(frame.bgra) != frame.width * frame.height * 4:
        return regions  # Layout needs character boxes and source pixels.
    pixels = np.frombuffer(frame.bgra, np.uint8).reshape(frame.height, frame.width, 4)
    return split_icon_grids(pixels, regions)


def _recognize_full(engine: Recognizer, frame: CapturedFrame) -> tuple[TextRegion, ...]:
    return _layout(frame, engine.recognize(frame))


def recognize_planned(
    engine: Recognizer,
    frame: CapturedFrame,
    plan: ScanPlan,
) -> tuple[tuple[TextRegion, ...], tuple[Rect, ...] | None, bool]:
    """Evaluate a plan without publishing provisional output or false coverage.

    A geometry/confidence loss or unexpected full retry discards the entire provisional
    result and executes the original plan against the same source frame. No old OCR is
    mixed into this observation. Planning remains separate from retention decisions.
    """
    if plan.changed:
        areas = scan_regions(plan.areas, frame.geometry.client.size)
        candidate = None
        if plan.areas is None:
            candidate = _recognize_full(engine, frame)
        elif areas is not None:
            candidate = _recognize_crops(engine, frame, areas)
        if candidate is not None and preserves_lines(
            plan.expected,
            tuple(
                ScanLine(r.bounds, r.confidence, r.vertical) for r in candidate if r.text.strip()
            ),
        ):
            return candidate, areas, False
    regions, scanned = recognize_regions(engine, frame, plan.baseline)
    return regions, scanned, plan.changed


def recognize_regions(
    engine: Recognizer, frame: CapturedFrame, areas: tuple[Rect, ...] | None
) -> tuple[tuple[TextRegion, ...], tuple[Rect, ...] | None]:
    """Crop on the model worker; map every line/character back to the exact source.

    A detector box at an artificial crop edge may be an incomplete word. Discard
    that provisional crop output and perform one full scan instead of publishing it.
    """
    areas = scan_regions(areas, frame.geometry.client.size)
    if areas is not None:
        regions = _recognize_crops(engine, frame, areas)
        if regions is not None:
            return regions, areas
    return _recognize_full(engine, frame), None


def _recognize_crops(
    engine: Recognizer, frame: CapturedFrame, areas: tuple[Rect, ...]
) -> tuple[TextRegion, ...] | None:
    """None signals a clipped line. An empty tuple is a completed negative scan."""
    if len(frame.bgra) != frame.width * frame.height * 4:
        raise ValueError(tr("Invalid source pixels for regional OCR"))
    pixels = memoryview(frame.bgra)
    regions = []
    for area in areas:
        rows = (
            pixels[(y * frame.width + area.x) * 4 : (y * frame.width + area.x + area.width) * 4]
            for y in range(area.y, area.y + area.height)
        )
        bounds = Rect(0, 0, area.width, area.height)
        crop = CapturedFrame(
            frame.sequence,
            frame.captured_at,
            WindowGeometry(bounds, bounds, bounds, frame.geometry.dpi),
            b"".join(rows),
        )
        for region in engine.recognize(crop, full_size=frame.geometry.client.size):
            b = region.bounds
            if (
                (area.x > 0 and b.x <= 3)
                or (area.y > 0 and b.y <= 3)
                or (area.x + area.width < frame.width and b.x + b.width >= area.width - 3)
                or (area.y + area.height < frame.height and b.y + b.height >= area.height - 3)
            ):
                return None

            regions.append(region.translated(area.x, area.y))
    return _layout(frame, tuple(regions))


def overlay_result_ready(result: OcrResult, latest: CapturedFrame, now: float) -> bool:
    """Allow static frames; expire lagging diagnostics on a changing source after 750 ms."""
    return result.frame.geometry.client.size == latest.geometry.client.size and (
        result.frame.sequence == latest.sequence or now - result.frame.captured_at <= 0.75
    )
