import threading
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.regions import padded, scan_regions
from dantetl.ocr.tracking import TextTracker
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion
from dantetl.ocr.visual import VisualAnalyzer, VisualSession
from dantetl.ocr.visual_values import VisualState
from dantetl.ocr.worker import OcrSession, recognize_regions
from dantetl.translation.grouping import DialogueGrouper


def frame(sequence=1, *, size=(1000, 600), fill=30):
    bounds = Rect(-300, 50, *size)
    return CapturedFrame(
        sequence,
        sequence / 8,
        WindowGeometry(bounds, bounds, bounds, 168),
        bytes((fill, fill, fill, 255)) * (size[0] * size[1]),
    )


def wait_for(predicate):
    end = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < end
        time.sleep(0.005)


def test_scan_closure_includes_whole_intersected_lines_and_stays_disjoint():
    lines = (Rect(50, 100, 500, 30), Rect(60, 143, 420, 30))
    areas = scan_regions(
        (padded(Rect(200, 120, 10, 10), (1000, 600)), Rect(800, 400, 80, 60)), (1000, 600), lines
    )
    assert areas and len(areas) == 2
    assert all(any(a.contains(line) for a in areas) for line in lines)
    assert all(Rect(0, 0, 1000, 600).contains(a) for a in areas)
    assert scan_regions((Rect(0, 0, 800, 600),), (1000, 600)) is None
    assert scan_regions(tuple(Rect(x * 100, 0, 20, 20) for x in range(5)), (1000, 600)) is None
    assert scan_regions((), (1000, 600)) == ()


def test_crops_use_source_pixels_and_translate_line_and_character_coordinates_once():
    source = frame()
    seen = []

    class Engine:
        def recognize(self, image, *, full_size=None):
            assert full_size == source.geometry.client.size
            seen.append(image)
            return (
                TextRegion(
                    "設定",
                    Rect(10, 10, 40, 20),
                    (TextCharacter("設定", Rect(12, 11, 35, 18), 0.95),),
                    True,
                ),
            )

        def close(self) -> None:
            pass

    areas = (Rect(100, 200, 100, 60), Rect(600, 300, 120, 70))
    regions, coverage = recognize_regions(Engine(), source, areas)
    assert coverage == areas
    assert [r.bounds for r in regions] == [Rect(110, 210, 40, 20), Rect(610, 310, 40, 20)]
    assert regions[1].characters[0].bounds == Rect(612, 311, 35, 18)
    assert regions[1].vertical
    assert [i.geometry.client.size for i in seen] == [a.size for a in areas]
    assert all(i.sequence == source.sequence and i.captured_at == source.captured_at for i in seen)
    assert seen[0].bgra == bytes((30, 30, 30, 255)) * 6000


def test_cut_off_detection_retries_full_frame_instead_of_publishing_a_fragment():
    calls = []

    class Engine:
        def recognize(self, image, *, full_size=None):
            assert full_size == (None if image.width == 1000 else (1000, 600))
            calls.append(image.width)
            return (
                TextRegion("完全な文章" if image.width == 1000 else "完全", Rect(0, 10, 80, 20)),
            )

        def close(self) -> None:
            pass

    regions, coverage = recognize_regions(Engine(), frame(), (Rect(100, 100, 200, 100),))
    assert coverage is None and regions[0].text == "完全な文章"
    assert calls == [200, 1000]
    # The actual window edge is not an artificial crop boundary.
    regions, coverage = recognize_regions(Engine(), frame(), (Rect(0, 100, 200, 100),))
    assert coverage is not None and len(calls) == 3


def test_four_small_crops_can_cost_five_calls_but_publish_only_full_frame_output():
    calls = []

    class Engine:
        def recognize(self, image, *, full_size=None):
            calls.append((image.width, full_size))
            if full_size is None:
                return (TextRegion("完全な文章", Rect(10, 10, 80, 20)),)
            return (TextRegion("部分", Rect(0 if len(calls) == 4 else 10, 10, 60, 20)),)

        def close(self):
            pass

    areas = tuple(Rect(50 + i * 200, 100, 80, 120) for i in range(4))
    regions, coverage = recognize_regions(Engine(), frame(), areas)
    assert sum(a.width * a.height for a in areas) / 600000 == 0.064
    assert len(calls) == 5 and calls[-1] == (1000, None)
    assert coverage is None and tuple(r.text for r in regions) == ("完全な文章",)


def test_partial_misses_cannot_remove_unscanned_tracks_or_split_dialogue():
    tracker, grouper = TextTracker(), DialogueGrouper()
    lines = (
        TextRegion("旅の準備をしてから、", Rect(40, 100, 330, 30)),
        TextRegion("一緒に出かけましょう。", Rect(40, 140, 300, 30)),
    )
    initial = frame()
    tracker.observe(OcrResult(initial, lines, 20, 0.2))
    original = tracker.advance(initial, 1).confirmed
    group = grouper.update(original, original, 1, activity=tracker.snapshot().activity)
    assert len(group.ready) == 1 and len(group.ready[0].lines) == 2
    for i in range(2, 20):
        source = frame(i, fill=i * 3)
        tracker.observe(OcrResult(source, (), 20, source.captured_at, (Rect(700, 400, 100, 100),)))
        assert tracker.advance(source, 100).confirmed == original
        assert (
            grouper.update(original, original, 100, activity=tracker.snapshot().activity).ready
            == group.ready
        )
    # Scanning only the continuation changes the complete group's request, not its first line.
    source = frame(20)
    revised = replace(lines[1], text="ここで待ってください。")
    tracker.observe(
        OcrResult(source, (revised,), 20, source.captured_at, (Rect(20, 135, 400, 45),))
    )
    current = tracker.advance(source, source.captured_at + 0.6).confirmed
    assert current[0] == original[0]
    grouped = grouper.update(
        current, current, source.captured_at + 1, activity=tracker.snapshot().activity
    )
    assert len(grouped.ready) == 1 and grouped.ready[0].text.endswith(revised.text)


def test_later_unrelated_scan_does_not_settle_an_earlier_negative_observation():
    tracker = TextTracker()
    source = frame(1)
    box = Rect(20, 30, 90, 25)
    tracker.observe(OcrResult(source, (TextRegion("設定", box),), 20, 0.2))
    tracker.advance(source, 1)
    source = frame(2, fill=60)
    tracker.observe(OcrResult(source, (), 20, 0.3, (Rect(0, 0, 200, 100),)))
    unrelated = frame(3, fill=60)
    tracker.observe(OcrResult(unrelated, (), 20, 0.4, (Rect(600, 400, 100, 100),)))
    assert tracker.advance(unrelated, 10).visible


def test_worker_unions_waiting_coverage_on_latest_frame_and_waits_for_acknowledgement():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Engine:
        def recognize(self, source, *, full_size=None):
            calls.append((source.sequence, source.width))
            if source.sequence == 1:
                entered.set()
                assert release.wait(4)
            return ()

        def close(self) -> None:
            pass

    session = OcrSession(Engine, require_ack=True)
    session.start()
    try:
        session.submit(frame(1))
        assert entered.wait(4)
        a, b = Rect(10, 20, 100, 60), Rect(600, 400, 120, 70)
        session.submit(frame(2), (a,))
        session.submit(frame(3), (b,))
        release.set()
        wait_for(lambda: session.snapshot().result is not None)
        assert calls == [(1, 1000)]
        session.acknowledge(1)
        wait_for(lambda: (r := session.snapshot().result) is not None and r.frame.sequence == 3)
        assert calls == [(1, 1000), (3, 100), (3, 120)]
        result = session.snapshot().result
        assert result is not None and result.scanned == (a, b)
        session.submit(frame(4), (a,))
        session.invalidate()  # An unacknowledged old result cannot strand resume or stop.
        session.submit(frame(4), (b,))
        wait_for(lambda: session.snapshot().result is not None)
        result = session.snapshot().result
        assert result is not None and result.frame.sequence == 4
    finally:
        release.set()
        session.request_stop()
        wait_for(lambda: session.finished)


def test_idle_worker_starts_disjoint_demand_and_wakes_for_the_last_overlapping_capture(monkeypatch):
    from types import SimpleNamespace

    from dantetl.ocr import worker

    clock = [5.0]
    seen = []
    waiting = threading.Event()

    def monotonic():
        if clock[0] == 5.09:
            waiting.set()
        return clock[0]

    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=monotonic))

    class Engine:
        def recognize(self, source, *, full_size=None):
            seen.append((source.sequence, clock[0]))
            return ()

        def close(self):
            pass

    session = OcrSession(Engine, require_ack=True)
    session.start()
    a, b = Rect(100, 100, 80, 60), Rect(600, 400, 80, 60)
    try:
        c = Rect(300, 100, 80, 60)
        for sequence, areas, now in ((1, (a,), 5.0), (2, (b,), 5.05), (3, (b, c), 5.06)):
            clock[0] = now
            session.submit(frame(sequence), areas)
            wait_for(
                lambda expected=sequence: (
                    (r := session.snapshot().result) is not None and r.frame.sequence == expected
                )
            )
            session.acknowledge(sequence)
        clock[0] = 5.09
        session.submit(frame(4), (b,))
        assert waiting.wait(1)
        assert seen == [(1, 5.0), (2, 5.05), (3, 5.06), (3, 5.06)]
        clock[0] = 5.121
        wait_for(lambda: (r := session.snapshot().result) is not None and r.frame.sequence == 4)
        assert seen[-1] == (4, 5.121)
    finally:
        session.request_stop()
        wait_for(lambda: session.finished)


def ink_frame(sequence, *, label=False, continuation=False):
    source = frame(sequence)
    pixels = np.frombuffer(source.bgra, np.uint8).reshape(600, 1000, 4).copy()
    for x in range(50, 350, 18):
        cv2.rectangle(pixels, (x, 100), (x + 10, 120), (230, 230, 230, 255), 2)
    if continuation:
        for x in range(50, 140, 18):
            cv2.rectangle(pixels, (x, 140), (x + 10, 160), (230, 230, 230, 255), 2)
    if label:
        cv2.putText(
            pixels, "MENU", (770, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230, 255), 2
        )
    return replace(source, bgra=pixels.tobytes())


@pytest.mark.parametrize(
    "mode",
    [
        "substitution",
        "glyph_boxes",
        "weak",
        "wrong_cell",
        "outside",
        "cleared",
        "reset",
        "reader_error",
    ],
)
def test_worker_rereads_only_verified_fixed_line_edits(mode):
    box, area = Rect(110, 60, 64, 20), Rect(75, 35, 140, 75)
    direct_calls, detector_calls = [], []

    def row(text, bounds, strength=0.98):
        chars = ()
        if mode == "glyph_boxes":
            chars = tuple(
                TextCharacter(c, Rect(bounds.x + 16 * i, bounds.y, 16, bounds.height), strength)
                for i, c in enumerate(text)
            )
        return TextRegion(text, bounds, chars, line_confidence=strength)

    class Reader:
        def recognize(self, source, *, full_size=None):
            detector_calls.append(source.sequence)
            if source.sequence == 1:
                return (row("1234", box),)
            b = replace(box, x=box.x - area.x, y=box.y - area.y) if full_size else box
            return (row("124" if mode == "cleared" else "1284", b),)

        def read_line(self, bgr):
            direct_calls.append(True)
            if mode == "reader_error":
                raise ValueError("synthetic failed read")
            return row(
                "7234" if mode == "wrong_cell" else "1284",
                Rect(0, 0, bgr.shape[1], bgr.shape[0]),
                0.45 if mode == "weak" else 0.98,
            )

        def close(self):
            pass

    session = OcrSession(Reader, require_ack=True)
    session.start()
    try:
        source = frame(1, size=(400, 240))
        original = np.frombuffer(source.bgra, np.uint8).reshape(240, 400, 4).copy()
        original[65:76, 145:149, :3] = 205
        session.submit(replace(source, bgra=original.tobytes()))
        wait_for(lambda: session.snapshot().result is not None)
        session.acknowledge(1)
        if mode == "reset":
            session.invalidate()
        original[65:76, 145:149, :3] = 30
        if mode != "cleared":
            original[65:76, 148:152, :3] = 205
        if mode == "outside":
            original[40, 80] = 255
        session.submit(replace(source, sequence=2, bgra=original.tobytes()), (area,))
        wait_for(lambda: (r := session.snapshot().result) and r.frame.sequence == 2)
        result = session.snapshot().result
        assert result and result.regions[0].text == ("124" if mode == "cleared" else "1284")
        if mode in ("substitution", "glyph_boxes"):
            assert direct_calls and detector_calls == [1]
            assert result.covers(box) and not result.covers(area)
            assert result.regions[0].bounds.contains(Rect(145, 65, 7, 11))
            session.acknowledge(2)
            session.submit(replace(source, sequence=3, bgra=original.tobytes()), (area,))
            wait_for(lambda: (r := session.snapshot().result) and r.frame.sequence == 3)
            assert detector_calls == [1, 3] and len(direct_calls) == 1
        else:
            assert detector_calls == [1, 2] and result.scanned == (area,)
            assert bool(direct_calls) == (mode in ("weak", "wrong_cell", "reader_error"))
    finally:
        session.request_stop()
        wait_for(lambda: session.finished)


def test_new_menu_is_scanned_locally_while_unrelated_dialogue_keeps_revealing():
    visual = VisualAnalyzer()
    source = ink_frame(1)
    track = TextTrack(1, "これから長い文章が表示されます", Rect(48, 98, 310, 25))
    result = OcrResult(source, (TextRegion(track.text, track.bounds),), 30, 0.2)
    visual.observe(source, result, (track,), (1,))
    visual.observe(ink_frame(4), result, (track,), (1,))
    state = visual.observe(ink_frame(5, label=True, continuation=True), result, (track,), (1,))
    assert state.request_ocr and state.scan_areas is not None
    assert any(a.contains(Rect(770, 448, 85, 28)) for a in state.scan_areas)
    assert all(not a.contains(track.bounds) for a in state.scan_areas)
    assert state.blocked == {1}


def test_periodic_full_scan_is_not_starved_by_frequent_small_changes():
    visual = VisualAnalyzer()
    full = []
    regional = []
    for i in range(1, 66):
        state = visual.observe(ink_frame(i, label=i % 4 < 2))
        if state.request_ocr:
            (full if state.scan_areas is None else regional).append(i)
    assert len(full) >= 3 and regional
    assert all(b - a <= 25 for a, b in zip(full, full[1:], strict=False))


@pytest.mark.parametrize("outcome", ["static", "mixed_static", "mixed_moving", "invalidate"])
def test_last_deferred_capture_is_scanned_once_without_creating_new_evidence(outcome):
    published = threading.Event()
    worker = VisualSession(on_change=published.set)
    track = TextTrack(1, "これから長い文章が表示されます", Rect(48, 98, 310, 25))
    result = OcrResult(ink_frame(1), (TextRegion(track.text, track.bounds),), 30, 0.2)

    def submit(frame):
        published.clear()
        worker.submit(frame, result, (track,), (1,), ())
        assert published.wait(3)
        return worker.poll(frame.sequence, frame.captured_at)

    worker.start()
    try:
        assert submit(result.frame).request_ocr
        assert submit(ink_frame(4)).request_ocr
        mixed = outcome.startswith("mixed")
        last = ink_frame(5, continuation=True, label=mixed)
        waiting = submit(last)
        assert waiting.blocked == {1} and not waiting.request_ocr
        if outcome == "invalidate":
            worker.invalidate()
            assert not worker.poll(5, 10).request_ocr
            return
        newer = outcome == "mixed_moving"
        state = worker.poll(6 if newer else 5, last.captured_at + (0.05 if newer else 0.17))
        assert state.request_ocr and state.frame is last
        assert state.blocked == waiting.blocked and state.stable_tracks == waiting.stable_tracks

        def covers(box):
            return state.scan_areas is None or any(a.contains(box) for a in state.scan_areas)

        assert covers(track.bounds) != newer
        if mixed:
            assert covers(Rect(770, 448, 85, 28))
        assert not worker.poll(5, 10).request_ocr
    finally:
        worker.request_stop()
        wait_for(lambda: worker.finished)


def test_menu_request_does_not_consume_the_dialogues_later_settling_request():
    visual = VisualAnalyzer()
    first = ink_frame(1)
    track = TextTrack(1, "これから長い文章が表示されます", Rect(48, 98, 310, 25))
    result = OcrResult(first, (TextRegion(track.text, track.bounds),), 30, 0.2)
    visual.observe(first, result, (track,), (1,))
    visual.observe(ink_frame(4), result, (track,), (1,))
    visual.observe(ink_frame(5, continuation=True), result, (track,), (1,))
    menu = visual.observe(ink_frame(6, continuation=True, label=True), result, (track,), (1,))
    assert menu.request_ocr and menu.scan_areas
    assert all(not area.contains(track.bounds) for area in menu.scan_areas)
    settled = visual.observe(ink_frame(7, continuation=True, label=True), result, (track,), (1,))
    assert settled.request_ocr
    assert settled.scan_areas is None or any(
        area.contains(track.bounds) for area in settled.scan_areas
    )


def test_unconsumed_visual_requests_union_coverage_on_the_newest_frame(monkeypatch):
    a, b = Rect(10, 20, 100, 60), Rect(600, 400, 120, 70)

    class Analyzer:
        def observe(self, source, *args):
            return VisualState(
                frame=source, request_ocr=True, scan_areas=(a if source.sequence == 1 else b,)
            )

    monkeypatch.setattr("dantetl.ocr.visual.VisualAnalyzer", lambda language: Analyzer())
    session = VisualSession()
    session.start()
    try:
        session.submit(frame(1), None, (), (), ())
        wait_for(lambda: session._snapshot.frame is not None)
        session.submit(frame(2), None, (), (), ())
        wait_for(lambda: (s := session._snapshot.frame) is not None and s.sequence == 2)
        state = session.poll()
        assert state.request_ocr and state.scan_areas == (a, b)
        assert state.frame and state.frame.sequence == 2
        assert not session.poll().request_ocr
    finally:
        session.request_stop()
        wait_for(lambda: session.finished)


def test_unrelated_partial_scan_cannot_delay_confirmation_of_a_finished_dialogue_edit():
    tracker, visual = TextTracker(), VisualAnalyzer()
    box = Rect(48, 98, 310, 25)
    first = OcrResult(ink_frame(1), (TextRegion("これから長い文章が表示されます", box),), 30, 0.2)
    ids = tracker.observe(first)
    old = tracker.advance(first.frame, 1).confirmed
    visual.observe(first.frame, first, old, ids)
    edit = OcrResult(
        ink_frame(2),
        (TextRegion("新しい冒険を始めましょう。", box),),
        30,
        0.3,
        (Rect(0, 80, 500, 100),),
    )
    ids = tracker.observe(edit)
    visual.observe(edit.frame, edit, old, ids, tracker.snapshot().activity)
    unrelated = OcrResult(ink_frame(4), (), 20, 0.55, (Rect(700, 400, 200, 100),))
    ids = tracker.observe(unrelated)
    state = visual.observe(unrelated.frame, unrelated, old, ids, tracker.snapshot().activity)
    assert state.stable_tracks == ((old[0].track_id, 2, edit.regions[0].text),)
    tracker.confirm_observations(state.stable_tracks, unrelated.frame.captured_at)
    assert tracker.advance(unrelated.frame, 0.55).visible[0].text == edit.regions[0].text
    newer = replace(
        edit, frame=ink_frame(5), regions=(TextRegion("次の文章に変わりました。", box),)
    )
    tracker.observe(newer)
    tracker.confirm_observations(state.stable_tracks, 0.7)
    assert tracker.advance(newer.frame, 0.7).visible[0].text == edit.regions[0].text
