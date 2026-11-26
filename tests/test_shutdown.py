"""Application shutdown policy without Qt, processes or time-based sleeps."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from dantetl.application.shutdown import ShutdownCoordinator
from dantetl.application.values import ShutdownPhase
from dantetl.runtime.restart import RestartSnapshot
from dantetl.runtime.setup_values import RuntimeTarget


def fixture():
    live, models = Mock(), Mock()
    live.drain.return_value = models.finish_shutdown.return_value = False
    gates = {"setup": True, "preferences": False, "hardware": False}
    cancel, release = Mock(), Mock()
    task = Mock(finished=False)
    task.snapshot.return_value = RestartSnapshot()
    restart = Mock(return_value=task)
    owner = ShutdownCoordinator(
        live,
        models,
        setup_busy=lambda: gates["setup"],
        cancel_setup=cancel,
        preferences_idle=lambda: gates["preferences"],
        hardware_finished=lambda: gates["hardware"],
        release_hotkeys=release,
        restart=restart,
    )
    return owner, live, models, gates, cancel, release, restart, task


def test_shutdown_drains_every_owner_before_releasing_shortcuts_and_launching():
    owner, live, models, gates, cancel, release, restart, task = fixture()
    target = RuntimeTarget(Path("replacement"))
    owner.request(target)
    owner.request(RuntimeTarget(Path("must-not-replace-pending-target")))
    assert owner.active
    live.stop.assert_called_once()
    models.shutdown.assert_called_once()
    for pending in ("setup", "models", "live", "preferences", "hardware"):
        assert owner.advance().phase == ShutdownPhase.DRAINING
        release.assert_not_called()
        restart.assert_not_called()
        if pending == "setup":
            gates["setup"] = False
        elif pending == "models":
            models.finish_shutdown.return_value = True
        elif pending == "live":
            live.drain.return_value = True
        else:
            gates[pending] = True
    assert cancel.call_count >= 5
    assert owner.advance().phase == ShutdownPhase.RESTARTING
    release.assert_called_once()
    restart.assert_called_once_with(target)
    task.snapshot.return_value = RestartSnapshot("ready")
    assert owner.advance().phase == ShutdownPhase.RESTARTING  # Owner still alive.
    task.finished = True
    assert owner.advance().phase == ShutdownPhase.READY
    owner.advance()
    release.assert_called_once()


def test_capture_completion_alone_is_not_application_completion():
    owner, live, models, gates, cancel, release, restart, task = fixture()
    gates.update(setup=False, preferences=True, hardware=True)
    live.drain.return_value = True
    owner.request()
    assert owner.advance().phase == ShutdownPhase.DRAINING
    assert "translation" in owner.snapshot().message
    release.assert_not_called()
    models.finish_shutdown.return_value = True
    assert owner.advance().phase == ShutdownPhase.READY
    restart.assert_not_called()


@pytest.mark.parametrize("launch_failure", [False, True])
def test_failed_restart_resumes_committed_models_only_after_cleanup_then_allows_normal_close(
    launch_failure,
):
    owner, live, models, gates, cancel, release, restart, task = fixture()
    gates.update(setup=False, preferences=True, hardware=True)
    live.drain.return_value = models.finish_shutdown.return_value = True
    if launch_failure:
        restart.side_effect = OSError("launch failed")
    owner.request(RuntimeTarget(Path("candidate")))
    result = owner.advance()
    if not launch_failure:
        assert result.phase == ShutdownPhase.RESTARTING
        task.snapshot.return_value = RestartSnapshot("failed", "cleanup failed")
        assert owner.advance().phase == ShutdownPhase.RESTARTING
        models.resume_after_failed_restart.assert_not_called()
        task.finished = True
        result = owner.advance()
    assert result.phase == ShutdownPhase.FAILED and not owner.active
    models.resume_after_failed_restart.assert_called_once()
    owner.request()
    assert owner.advance().phase == ShutdownPhase.READY
    restart.assert_called_once()  # A later ordinary close never retries the failed restart.
