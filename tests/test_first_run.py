import threading
from unittest.mock import Mock

import pytest

from dantetl.runtime import first_run
from dantetl.runtime.model_store import ModelBundle, ModelOperation
from dantetl.runtime.setup_values import Cancelled, DownloadProgress, ModelProgress
from dantetl.runtime.values import PROFILES


def test_review_starts_models_and_runtime_together_and_reports_both(monkeypatch, tmp_path):
    model_order = []
    overlap = threading.Barrier(2, timeout=3)
    download = DownloadProgress("weights", 4, 10)
    models = tuple(ModelBundle(key, key, tmp_path / key, (), "") for key in ("ocr", "translator"))
    plan = first_run.SetupPlan(models, PROFILES[0])

    def model_install(model, report, cancel):
        model_order.append(model)
        report(ModelProgress(download=download))
        if model == models[0]:
            overlap.wait()

    def runtime_install(root, runtime, report, cancel):
        report(ModelProgress("wheel", stage=2, stages=4))
        overlap.wait()

    monkeypatch.setattr(first_run, "install_model", model_install)
    monkeypatch.setattr(first_run, "install_runtime", runtime_install)
    monkeypatch.setattr(first_run, "ready_runtime", lambda *args: tmp_path)
    assert not model_order
    updates = []
    assert first_run.run_setup(plan, tmp_path, updates.append, threading.Event()) == tmp_path
    assert model_order == list(models)
    assert any(
        v.parallel
        and v.parallel[0].download == download
        and v.parallel[0].stage == 1
        and v.parallel[0].stages == 2
        and v.parallel[1].stage == 2
        for v in updates
    )
    assert all(lane.done for lane in updates[-2].parallel)


def test_cancel_before_start_never_installs_models_or_runtime(monkeypatch, tmp_path):
    cancel = threading.Event()
    cancel.set()
    runtime = Mock()
    models = Mock()
    monkeypatch.setattr(first_run, "install_model", models)
    monkeypatch.setattr(first_run, "install_runtime", runtime)
    with pytest.raises(Cancelled):
        first_run.run_setup(
            first_run.SetupPlan((Mock(), Mock()), PROFILES[0]), tmp_path, lambda p: None, cancel
        )
    runtime.assert_not_called()
    models.assert_not_called()


def test_model_failure_still_allows_runtime_to_finish(monkeypatch, tmp_path):
    failed = threading.Event()
    ready = threading.Event()

    def fail(*args):
        failed.set()
        raise OSError("model interrupted")

    def install(*args):
        assert failed.wait(3)
        ready.set()

    monkeypatch.setattr(first_run, "install_model", fail)
    monkeypatch.setattr(first_run, "install_runtime", install)
    monkeypatch.setattr(first_run, "ready_runtime", lambda *args: tmp_path)
    with pytest.raises(RuntimeError, match="model interrupted"):
        first_run.run_setup(
            first_run.SetupPlan((Mock(),), PROFILES[0]), tmp_path, lambda _: None, threading.Event()
        )
    assert ready.is_set()


def test_operation_keeps_download_snapshot_on_completion():
    value = ModelProgress("ready", download=DownloadProgress("file", 10, 10), stage=2, stages=2)
    operation = ModelOperation(lambda progress, cancel: progress(value))
    operation.start()
    operation._thread.join(2)
    result = operation.snapshot()
    assert result.done and result.download == value.download and result.stage == 2


def test_runtime_only_plan_runs_one_installer_and_returns_verified_restart(monkeypatch, tmp_path):
    profile = next(p for p in PROFILES if p.key == "nvidia")
    installer = Mock()
    model_install = Mock()
    monkeypatch.setattr(first_run, "install_model", model_install)
    monkeypatch.setattr(first_run, "install_runtime", installer)
    monkeypatch.setattr(first_run, "ready_runtime", lambda *_: tmp_path)
    updates = []
    assert (
        first_run.run_setup(
            first_run.SetupPlan((), profile), tmp_path, updates.append, threading.Event()
        )
        == tmp_path
    )
    installer.assert_called_once()
    model_install.assert_not_called()
    assert all(not update.parallel for update in updates)
