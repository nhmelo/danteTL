"""Offline icon/label segmentation benchmark with real CPU or strict CUDA Meiki inference.

Scores the production recognizer plus tracking content filtering against owned scenes, and
optionally against annotated real frames kept in ignored artifacts. It reports label text
errors, labels read separately/merged/fragmented/missed, icon characters attached to labels,
icon-only phantom rows the translation gate would admit, label boxes covering icons, and
first/warm recognition timings. ``--baseline`` fails on any per-label text regression.
"""

import argparse
import json
import platform
import socket
import sys
from dataclasses import asdict
from io import TextIOWrapper
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from unittest.mock import patch

from icon_corpus import SPECS, Scene, SceneScore, Totals, load_holdout, render, score
from PySide6.QtWidgets import QApplication

from dantetl.domain import Rect
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.ocr.meiki import MeikiRecognizer
from dantetl.ocr.models import model_directory
from dantetl.ocr.text_content import content_region
from dantetl.ocr.values import TextRegion
from dantetl.ocr.worker import recognize_regions

_SETTLED = 5  # Consecutive identical scans; native windows read at most one per scan, four in all.

HOLDOUT_FORMAT = """\
Holdout JSON: {"frames": [{"name", "image", "scope": [[x, y, w, h], ...],
"labels": [{"text", "bounds": [x, y, w, h], "role": "text"|"value"}],
"icons": [{"kind", "bounds": [x, y, w, h]}], "areas": [[x, y, w, h], ...]}]}.
Images resolve beside the JSON. Optional areas run the production regional scan instead of a
full scan. Only rows centered in a scope rectangle are scored, so annotate everything there."""


def _box(r: Rect) -> list[int]:
    return [r.x, r.y, r.width, r.height]


def _region(region: TextRegion) -> dict[str, Any]:
    return dict(
        text=region.text,
        bounds=_box(region.bounds),
        characters=[
            dict(text=c.text, bounds=_box(c.bounds), confidence=round(c.confidence, 4))
            for c in region.characters
        ],
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def _recognize(engine: MeikiRecognizer, scene: Scene) -> tuple[TextRegion, ...]:
    """The OCR worker's own scan path, full or regional, including its post-processing."""
    regions, scanned = recognize_regions(engine, scene.frame, scene.areas)
    if scene.areas is not None and scanned is None:
        raise AssertionError(f"{scene.name}: the regional scan fell back to a full scan")
    return regions


def _measure(engine: MeikiRecognizer, scene: Scene, repeats: int) -> dict[str, Any]:
    """First (cold) read, then repeat until the output settles, then timed settled reads.

    Native-scale windows are read at most one new window per scan and then remembered; a read
    that adds nothing leaves the output unchanged while another window is still unread. Settled
    therefore means unchanged for more consecutive scans than there are windows.
    """
    start = perf_counter()
    regions = _recognize(engine, scene)
    first = (perf_counter() - start) * 1000
    scans, unchanged = 1, 0
    while unchanged < _SETTLED:
        if scans == 16:
            raise AssertionError(f"{scene.name}: recognition did not settle in 16 scans")
        settled = _recognize(engine, scene)
        scans += 1
        unchanged = unchanged + 1 if settled == regions else 0
        regions = settled
    settle = scans - _SETTLED
    warm = []
    for _ in range(repeats):
        start = perf_counter()
        repeated = _recognize(engine, scene)
        warm.append((perf_counter() - start) * 1000)
        if repeated != regions:
            raise AssertionError(f"{scene.name}: settled recognition is not repeatable")
    start = perf_counter()
    rows = tuple(content_region(r, DEFAULT_SOURCE) for r in regions)
    content_ms = (perf_counter() - start) * 1000
    result: SceneScore = score(scene, rows, DEFAULT_SOURCE)
    return dict(
        name=scene.name,
        family=scene.family,
        size=[scene.frame.width, scene.frame.height],
        first_ms=round(first, 2),
        settle_scans=settle,
        warm_ms=[round(t, 2) for t in warm],
        content_ms=round(content_ms, 3),
        score=asdict(result),
        raw=[_region(r) for r in regions],
        rows=[_region(r) for r in rows],
        _score=result,
    )


def _regressions(baseline: dict[str, Any], scenes: list[dict[str, Any]]) -> list[str]:
    """Per-label text errors may not increase and a label may not lose its own row.

    A separately read label may not become merged, fragmented or missed; a merged or fragmented
    one may not go missing.
    """
    rank = {"separate": 0, "merged": 1, "fragmented": 1, "missed": 2}
    previous = {s["name"]: s["score"]["labels"] for s in baseline["scenes"]}
    found = []
    for scene in scenes:
        old = previous.get(scene["name"])
        if old is None:
            continue
        for before, after in zip(old, scene["score"]["labels"], strict=False):
            if before["expected"] != after["expected"]:
                found.append(f"{scene['name']}: label order changed; regenerate the baseline")
                break
            if after["errors"] > before["errors"] or rank[after["state"]] > rank[before["state"]]:
                found.append(
                    f"{scene['name']}: {after['expected']} {before['observed']!r}"
                    f" -> {after['observed']!r} ({before['state']} -> {after['state']})"
                )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=HOLDOUT_FORMAT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", help="Restrict to named owned scenes.")
    parser.add_argument("--holdout", type=Path, help="Annotated real frames; see below.")
    parser.add_argument("--baseline", type=Path, help="A previous output to compare against.")
    args = parser.parse_args()
    if not 2 <= args.repeats <= 20:
        parser.error("Use 2..20 measured repetitions.")
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Regression lines quote Japanese text.
    specs = SPECS
    if args.scene:
        unknown = set(args.scene) - {s.name for s in SPECS}
        if unknown:
            parser.error(f"Unknown scenes: {', '.join(sorted(unknown))}")
        specs = tuple(s for s in SPECS if s.name in args.scene)
    app = QApplication([])
    scenes = [render(spec) for spec in specs]
    if args.holdout:
        scenes.extend(load_holdout(args.holdout))
    measured: list[dict[str, Any]] = []
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")):
        start = perf_counter()
        engine = MeikiRecognizer(model_directory(), device=args.device)
        load_ms = (perf_counter() - start) * 1000
        try:
            if args.device == "cuda" and engine.provider != "CUDAExecutionProvider":
                raise AssertionError(f"Strict CUDA requested but running {engine.provider}")
            provider = engine.provider
            for scene in scenes:
                measured.append(_measure(engine, scene, args.repeats))
                print(f"{scene.name}: {Totals.of([measured[-1]['_score']])}", flush=True)
        finally:
            engine.close()
    families: dict[str, list[SceneScore]] = {}
    for scene in measured:
        families.setdefault(scene["family"], []).append(scene.pop("_score"))
    warm = [t for s in measured for t in s["warm_ms"]]
    report: dict[str, Any] = dict(
        device=args.device,
        provider=provider,
        python=sys.version.split()[0],
        machine=platform.processor() or platform.machine(),
        repeats=args.repeats,
        load_ms=round(load_ms, 1),
        warm_ms_median=round(median(warm), 2),
        warm_ms_p95=round(_percentile(warm, 0.95), 2),
        totals=asdict(Totals.of(s for scores in families.values() for s in scores)),
        families={name: asdict(Totals.of(scores)) for name, scores in families.items()},
        scenes=measured,
    )
    regressions = []
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        regressions = _regressions(baseline, measured)
        report["baseline"] = dict(path=str(args.baseline), totals=baseline["totals"])
        report["regressions"] = regressions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(totals=report["totals"], families=report["families"]), indent=2))
    for line in regressions:
        print(f"REGRESSION {line}")
    app.quit()
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
