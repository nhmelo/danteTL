from dataclasses import replace

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState
from dantetl.ocr.visual_values import VisualState
from dantetl.translation.values import TranslationSnapshot, TranslationState
from dantetl.ui.diagnostics import debug_display


def source():
    bounds = Rect(0, 0, 1280, 720)
    return CapturedFrame(1, 10, WindowGeometry(bounds, bounds, bounds, 144), b"")


def test_debug_distinguishes_changes_from_actual_scanned_coverage_and_model_wait():
    frame = source()
    changed, scanned = Rect(800, 400, 40, 40), Rect(700, 300, 200, 100)
    visual = VisualState(frame=frame, changed_areas=(changed,), scan_areas=None)
    ocr = OcrSnapshot(OcrState.READY, OcrResult(frame, (), 25, 10.1, (scanned,)), queue_ms=80)
    translation = TranslationSnapshot(
        TranslationState.READY,
        pending=3,
        last_batch_ms=400,
        queue_ms=250,
        cache_hits=1,
        translated_sources=2,
        processing_since=10.2,
        cancelled_obsolete=2,
    )
    display = debug_display(visual, ocr, translation, 10.4, waiting_blocks=2)
    assert display.changed == (changed,) and display.scanned == (scanned,)
    for text in (
        "25 ms",
        "wait 80 ms",
        "400 ms",
        "wait 250 ms",
        "1 cache / 2 model",
        "settling 2",
        "Oldest active translation: 0.2s",
        "Obsolete cloud requests cancelled: 2",
    ):
        assert text in display.text
    assert debug_display(visual, ocr, None, 11).changed == ()


def test_full_scan_is_explicit_and_resize_discards_old_coverage():
    frame = source()
    ocr = OcrSnapshot(OcrState.READY, OcrResult(frame, (), 100, 10.1))
    display = debug_display(VisualState(frame=frame), ocr, None, 10.2)
    assert display.scanned == (Rect(0, 0, 1280, 720),)
    assert "full, 100%" in display.text
    changed = replace(frame, geometry=replace(frame.geometry, client=Rect(0, 0, 640, 480)))
    assert debug_display(VisualState(frame=changed), ocr, None, 10.2).scanned == ()
