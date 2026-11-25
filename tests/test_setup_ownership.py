"""Fixed setup lanes, review identity and draining without constructing a dialog."""

import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
from ownership_fixtures import Replay

from dantetl.application.model_inventory import InventoryData
from dantetl.application.model_values import ModelFile
from dantetl.application.setup_coordinator import SetupCoordinator
from dantetl.runtime.hardware_values import HardwareStatus
from dantetl.runtime.model_store import ModelBundle
from dantetl.runtime.setup_values import SetupLane
from dantetl.runtime.values import PROFILES


def configured(tmp_path):
    replay = Replay(tmp_path)
    bundles = tuple(
        ModelBundle(
            key,
            key,
            tmp_path / key,
            (),
            "",
            category="recognition" if key == "ocr" else "translation",
        )
        for key in ("ocr", "model")
    )
    files = tuple(
        ModelFile(
            b.key,
            b.name,
            "",
            b.category,
            0,
            False,
            False,
            False,
            False,
            str(b.directory),
            "",
            str(b.directory),
            "",
            "cpu",
            False,
        )
        for b in bundles
    )
    replay.discover(InventoryData(files=files, device="cpu"))
    replay.setup = SetupCoordinator(replay.inventory, bundles, PROFILES, replay.dependencies)
    return replay


def test_setup_lanes_are_bounded_and_cancel_waits_for_owner_cleanup(tmp_path):
    replay = configured(tmp_path)
    try:
        setup = replay.setup
        assert setup.download("model") and not setup.download("ocr")
        assert setup.detect_runtime() and not setup.install_runtime("cpu")
        assert not setup.test_import("model.gguf", "auto")
        assert len(replay.operations) == 2
        setup.close()
        assert all(op.cancelled for op in replay.operations)
        assert setup.busy and not setup.download("model")
        replay.operations[0].finish()  # A committed download survives the cancellation request.
        setup.advance()
        assert setup.busy
        replay.operations[1].finish(cancelled=True)
        setup.advance()
        assert not setup.busy
        setup.resume()
        assert setup.download("model")
    finally:
        replay.close()


def test_cancel_after_commit_reports_success_and_refreshes_inventory(tmp_path):
    replay = configured(tmp_path)
    try:
        setup = replay.setup
        assert setup.download("model")
        # The worker has committed; discovery can now see the installed file.
        data = replay.source.data
        replay.source.data = replace(
            data,
            files=tuple(replace(f, present=True) if f.key == "model" else f for f in data.files),
        )
        replay.operations[-1].finish()
        setup.cancel(SetupLane.MODELS)
        setup.advance()
        job = setup.job(SetupLane.MODELS)
        assert job.progress.done and not (job.cancelling or job.progress.cancelled or job.busy)
        replay.settle_inventory()
        assert any(f.key == "model" and f.present for f in setup.files)
    finally:
        replay.close()


@pytest.mark.parametrize("outcome", [{"cancelled": True}, {"error": "next download failed"}])
def test_partial_guide_refreshes_verified_files_without_accepting_the_review(tmp_path, outcome):
    replay = configured(tmp_path)
    try:
        setup = replay.setup
        review = setup.review("model", "cpu", HardwareStatus(llama_device="cpu"))
        assert setup.start_review(review.token)
        data = replay.source.data
        replay.source.data = replace(
            data, files=tuple(replace(f, present=True) if f.key == "ocr" else f for f in data.files)
        )
        replay.operations[-1].finish(**outcome)
        setup.advance()
        replay.settle_inventory()
        assert setup.acknowledge_review(review.token) is None
        assert any(f.key == "ocr" and f.present for f in setup.files)
    finally:
        replay.close()


def test_stale_review_repeated_success_and_failed_retry(tmp_path):
    replay = configured(tmp_path)
    try:
        setup = replay.setup
        hardware = HardwareStatus(llama_device="cpu")
        old = setup.review("model", "cpu", hardware)
        current = setup.review("model", "cpu", hardware)
        assert not setup.start_review(old.token)
        assert setup.start_review(current.token)
        assert not setup.start_review(current.token)
        with pytest.raises(ValueError):
            setup.review("model", "cpu", hardware)
        replay.operations[-1].finish(error="fixture failure")
        setup.advance()
        assert setup.start_review(current.token)  # explicit retry uses the same reviewed intent
        replay.operations[-1].finish()
        setup.advance()
        assert not setup.start_review(current.token)  # completed intent is consumed
        assert setup.reviewed == current
    finally:
        replay.close()


@pytest.mark.parametrize(
    ("profile", "cuda_ocr", "llama_device", "upgrade", "install"),
    [
        ("cpu", False, "cpu", False, False),
        ("cpu", True, "cpu", False, True),
        ("amd", False, "vulkan", False, False),
        ("amd", False, "cpu", False, True),
        ("nvidia", False, "cuda", False, True),
        ("nvidia", True, "cpu", False, True),
        ("nvidia", True, "cuda", False, False),
        ("nvidia", True, "cuda", True, True),
    ],
)
def test_review_requests_the_selected_runtime_only_when_needed(
    tmp_path, profile, cuda_ocr, llama_device, upgrade, install
):
    replay = configured(tmp_path)
    try:
        data = replay.source.data
        files = tuple(replace(f, upgrade=upgrade) if f.key == "model" else f for f in data.files)
        replay.discover(replace(data, files=files))
        from dantetl.runtime.hardware_values import Adapter, CudaDevice

        hardware = HardwareStatus(
            (Adapter("Fixture", 0x10DE, cuda=CudaDevice(13000, 8, 6), vulkan=True),),
            cuda_ocr=cuda_ocr,
            llama_device=llama_device,
        )
        assert replay.setup.review("model", profile, hardware).install_runtime is install
        assert not replay.operations
    finally:
        replay.close()


def test_import_capability_does_not_depend_on_downloaded_model_rows(tmp_path):
    replay = Replay(tmp_path, data=InventoryData(device="vulkan"))
    try:
        assert replay.setup.test_import("user.gguf", "auto")
        assert isinstance(replay.dependencies.probe, Mock)
        args = replay.dependencies.probe.call_args.args
        assert args[2] == "vulkan"
        gate = args[3]
        assert isinstance(gate, threading.Event) and not gate.is_set()
        replay.setup.permit_probe()
        assert gate.is_set()
        replay.setup.cancel(SetupLane.MODELS)
        assert replay.setup.probing  # cancellation is not finished teardown
        replay.operations[-1].finish(cancelled=True)
        replay.setup.advance()
        assert not replay.setup.probing and replay.setup.probe_result is None
    finally:
        replay.close()


def test_probe_permit_requires_both_translation_and_classifier_cleanup(tmp_path):
    from dantetl.translation.backend import BackendChoice

    replay = Replay(
        tmp_path, data=InventoryData((BackendChoice("Fixture", Mock(), "fixture"),), device="cpu")
    )
    try:
        replay.translation.refresh_route()
        replay.translation.advance_activity(running=True, enabled=True)
        worker = replay.route_workers[-1]
        assert replay.setup.test_import("user.gguf", "auto")
        assert isinstance(replay.dependencies.probe, Mock)
        gate = replay.dependencies.probe.call_args.args[3]
        replay.live.suspend_history = Mock(return_value=False)
        replay.app.advance(1)
        assert not gate.is_set()
        assert worker.stops and not worker.finished
        worker.finish()
        replay.app.advance(2)
        assert not gate.is_set()
        replay.live.suspend_history.return_value = True
        replay.app.advance(3)
        assert gate.is_set()
    finally:
        replay.close()


def test_cancelled_runtime_completion_cannot_replace_current_snapshot(tmp_path):
    replay = Replay(tmp_path)
    try:
        assert replay.setup.detect_runtime()
        before = replay.setup.runtime
        replay.setup._runtime_result = replace(before, recommendation="amd")
        replay.setup.cancel(SetupLane.RUNTIME)
        replay.operations[0].finish(cancelled=True)
        replay.setup.advance()
        assert replay.setup.runtime == before
    finally:
        replay.close()
