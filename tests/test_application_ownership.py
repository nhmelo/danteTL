"""Application command and completion replays. No QApplication or native models."""

import os
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from live_fixtures import frame
from ownership_fixtures import Replay

from dantetl.application.model_inventory import InventoryData
from dantetl.application.preference_values import Preferences
from dantetl.application.values import ModelConsent, ShutdownPhase
from dantetl.domain import CapturedFrame, CaptureSnapshot, CaptureState, Rect
from dantetl.ocr.observation import CandidateReview
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextRegion
from dantetl.ocr.visual_values import VisualState
from dantetl.runtime.restart import RestartSnapshot
from dantetl.runtime.setup_values import RuntimeTarget
from dantetl.translation.backend import BackendChoice
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.selection import AssignmentChange
from dantetl.translation.values import TranslationState


@pytest.fixture
def replay(tmp_path):
    value = Replay(tmp_path)
    yield value
    value.close()


def test_first_run_offer_survives_pending_account_discovery(replay, monkeypatch):
    release = threading.Event()

    def accounts(_report):
        assert release.wait(3)
        return ()

    monkeypatch.setattr(replay.cloud.connections, "accounts", accounts)
    replay.cloud.start()
    try:
        replay.app.offer_first_run()
        replay.app.advance(0)
        assert not replay.app.can_setup
        assert replay.app.take_desktop_action() is None
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while replay.cloud.busy:
        assert time.monotonic() < deadline
        replay.app.advance(1)
        time.sleep(0.001)
    assert replay.app.can_setup
    assert replay.app.take_desktop_action() == "guide"
    replay.app.advance(2)
    assert replay.app.take_desktop_action() is None


def test_startup_setting_edit_preserves_unresolved_saved_assignment(tmp_path):
    preferences = Preferences(
        model="gemma",
        short_model="local:gemma",
        long_model="cloud:missing",
        translate_short=False,
        translate_long=False,
    )
    replay = Replay(tmp_path, preferences=preferences, loaded=False)
    try:
        replay.app.configure(opacity=60)
        assert replay.app.values.translate_short is False
        assert replay.app.values.translate_long is False
        assert replay.app.values.short_model == "local:gemma"
        # Unknown inventory cannot start inference, even if capture is available.
        assert replay.app.start_capture(replay.capture.target, 1)
        assert replay.live.ocr is None
        replay.discover(InventoryData())
        replay.app.advance(2)
        assert replay.live.ocr is replay.ocr
        assert replay.translation.plan == TranslationPlan(
            "local:gemma", "cloud:missing", False, False
        )
    finally:
        replay.close()


@pytest.mark.parametrize("stopping", ["stop", "close"])
def test_late_inventory_cannot_start_ocr_during_stop_or_close(tmp_path, stopping):
    replay = Replay(tmp_path, loaded=False)
    try:
        replay.app.start_capture(replay.capture.target, 1)
        if stopping == "close":
            replay.app.request_close()
        else:
            replay.app.stop()
        replay.discover(InventoryData())
        replay.app.advance(2)
        assert replay.live.ocr is None
    finally:
        replay.close()


def test_finishing_setup_retries_selection_after_delayed_discovery(replay):
    entered, release = threading.Event(), threading.Event()
    choice = BackendChoice("Downloaded model", Mock(), "downloaded", "weights")

    def inspect():
        entered.set()
        assert release.wait(2)
        return InventoryData((choice,))

    replay.source.inspect = inspect
    try:
        replay.app.setup_completed("downloaded", None)
        replay.app.advance(1)
        assert entered.wait(1)
        for now in range(2, 12):
            replay.app.advance(now)
            assert replay.app.take_desktop_action() is None
            assert replay.app.values.setup_model == "downloaded"
        release.set()
        deadline = time.monotonic() + 2
        while replay.translation.resolve("local:downloaded") is None:
            assert time.monotonic() < deadline
            replay.app.advance(12)
            time.sleep(0.001)
        prompt = replay.app.take_desktop_action()
        assert isinstance(prompt, ModelConsent) and prompt.option.reference == "local:downloaded"
        assert replay.app.values.setup_model == "downloaded"
        replay.app.advance(13)
        assert replay.app.take_desktop_action() is None
        assert not replay.workers
        replay.app.consent(prompt.token, True)
        replay.app.advance(14)
        assert len(replay.workers) == 1 and replay.app.values.setup_model == ""
    finally:
        release.set()


def test_consent_is_one_shot_and_stale_responses_cannot_commit(replay):
    choice = BackendChoice("Candidate", Mock(), "new", "weights-v1")
    replay.discover(InventoryData((choice,)))
    replay.app.advance(1)
    plan = TranslationPlan("local:new", "local:new")
    first = replay.app.assign(AssignmentChange(plan, "local:new"))
    assert first and not replay.workers
    replay.app.consent(first.token, False)
    second = replay.app.assign(AssignmentChange(plan, "local:new"))
    assert second and second.token != first.token
    replay.app.consent(first.token, True)
    assert replay.translation.plan == TranslationPlan()
    replay.app.consent(second.token, True)
    replay.app.consent(second.token, True)
    replay.app.advance(2)
    assert len(replay.workers) == 1
    ready = replay.workers[0]
    ready.state = TranslationState.READY
    replay.app.advance(3)
    assert replay.translation.plan == plan
    assert replay.translation.lifecycle.publication is ready
    replay.app.start_capture(replay.capture.target, 4)
    replay.app.advance(4)
    assert len(replay.routes) == 1
    assert replay.routes[0][1]["prepared"] == ("local:new", ready)
    replay.app.advance(5)
    assert len(replay.workers) == len(replay.routes) == 1


def test_close_while_prompt_open_revokes_its_response(replay):
    replay.discover(InventoryData((BackendChoice("New", Mock(), "new", "weights"),)))
    replay.app.advance(1)
    prompt = replay.app.assign(AssignmentChange(TranslationPlan("local:new"), "local:new"))
    assert prompt
    replay.app.request_close()
    replay.app.consent(prompt.token, True)
    replay.app.advance(2)
    assert not replay.workers and replay.translation.plan == TranslationPlan()


def test_application_preserves_source_acknowledgement_and_partial_coverage(replay):
    replay.app.start_capture(replay.capture.target, 0)
    first = frame(1)
    label = TextRegion("設定", Rect(50, 40, 100, 30))
    replay.capture.current = CaptureSnapshot(CaptureState.RUNNING, frame=first)
    replay.ocr.current = OcrSnapshot(OcrState.READY, OcrResult(first, (label,), 25, 1))
    replay.visual.current = VisualState(review=CandidateReview(sequence=0), frame=first)
    replay.app.advance(1)
    assert replay.app.view.observation is None and not replay.ocr.acknowledged
    replay.visual.current = VisualState(
        review=CandidateReview(sequence=1), frame=first, ocr_sequence=1
    )
    replay.app.advance(1.6)
    original = replay.app.view.scene.tracks
    assert len(original) == 1 and replay.ocr.acknowledged == [1]
    second = frame(2, 2)
    crop = Rect(580, 20, 160, 80)
    region = TextRegion("戻る", Rect(600, 40, 100, 30))
    replay.capture.current = replace(replay.capture.current, frame=second)
    replay.ocr.current = OcrSnapshot(OcrState.READY, OcrResult(second, (region,), 25, 2, (crop,)))
    replay.visual.current = VisualState(
        review=CandidateReview(sequence=2), frame=second, ocr_sequence=2
    )
    replay.app.advance(2.6)
    assert original[0] in replay.app.view.scene.tracks
    assert len(replay.app.view.scene.tracks) == 2
    assert replay.app.view.observation is replay.ocr.current.result
    assert replay.app.view.observation is not None
    assert replay.app.view.observation.regions == (region,)
    for i in range(30):
        replay.app.advance(2.6 + i * 0.03)
    assert set(replay.ocr.acknowledged) == {1, 2}
    assert original[0] in replay.app.view.scene.tracks


def test_unchanged_10000_ticks_have_no_gui_io_catalog_or_route_rebuilds(replay, monkeypatch):
    choice = BackendChoice("Model", Mock(), "same", "weights")
    replay.discover(InventoryData((choice,)))
    replay.app.advance(1)
    options, notice = replay.app.controls.options, replay.app.notices()
    route = Mock(wraps=replay.translation.refresh_route)
    monkeypatch.setattr(replay.translation, "refresh_route", route)
    monkeypatch.setattr(Path, "stat", Mock(side_effect=AssertionError("GUI filesystem read")))
    for i in range(10000):
        replay.app.advance(1 + i * 0.033)
        assert replay.app.controls.options is options
        assert replay.app.notices() is notice
    route.assert_not_called()
    assert not replay.workers and not replay.routes
    assert replay.app.view.english_reset == 0


def test_completed_observations_share_pixels_and_release_superseded_frames(replay):
    class ObservableFrame(CapturedFrame):
        """Weak-referenceable capture value for observing resource lifetime."""

    replay.app.start_capture(replay.capture.target, 0)
    released = []
    for sequence in range(1, 121):
        source = ObservableFrame(sequence, sequence, frame(1).geometry, bytes(1200 * 900 * 4))
        released.append(weakref.ref(source))
        result = OcrResult(source, (), 25, sequence)
        replay.capture.current = CaptureSnapshot(CaptureState.RUNNING, frame=source)
        replay.ocr.current = OcrSnapshot(OcrState.READY, result)
        replay.visual.current = VisualState(
            review=CandidateReview(sequence=sequence), frame=source, ocr_sequence=sequence
        )
        replay.app.advance(sequence + 0.1)
        view = replay.app.view
        assert view.frame is source and view.observation is result
        assert view.capture is replay.capture.current and view.visual is replay.visual.current
        assert view.ocr is replay.ocr.current
    assert sum(reference() is not None for reference in released) == 1


def test_new_lazy_factories_do_not_change_catalog_identity():
    a = BackendChoice("Model", Mock(), "same", "weights")
    b = replace(a, create=Mock())
    assert InventoryData((a,)) == InventoryData((b,))
    assert InventoryData((a,)) != InventoryData((replace(b, preparation_id="new-weights"),))


def test_failed_restart_reopens_setup_and_inventory(replay):
    replay.app.request_close(RuntimeTarget(Path("runtime")))
    replay.app.advance(1)
    assert replay.app.close_status.phase == ShutdownPhase.RESTARTING
    replay.restart.finished = True
    replay.restart.snapshot.return_value = RestartSnapshot("failed", "fixture failure")
    replay.app.advance(2)
    assert replay.app.close_status.phase == ShutdownPhase.FAILED
    assert replay.setup.detect_runtime()
    model = BackendChoice("Discovered after restart failure", Mock(), "recovered")
    replay.discover(InventoryData((model,)))
    replay.app.advance(3)
    assert replay.app.controls.options[0].reference == "local:recovered"


def test_all_owners_construct_and_advance_with_qt_imports_forbidden(tmp_path):
    script = """
import importlib.abc, sys
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'shiboken6'}:
            raise AssertionError('Qt dependency: ' + fullname)
sys.meta_path.insert(0, NoQt())
from pathlib import Path
from ownership_fixtures import Replay
replay = Replay(Path(sys.argv[1]))
replay.app.advance(1)
assert not replay.app.controls.capture_active
replay.close()
replay.app.advance(2)
assert not any(m.startswith(('PySide6', 'shiboken6')) for m in sys.modules)
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(("tests", "src"))}
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
