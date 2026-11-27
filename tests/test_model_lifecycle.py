from unittest.mock import Mock

from dantetl.application.model_lifecycle import ModelPhase
from dantetl.translation.backend import BackendChoice
from dantetl.translation.values import TranslationState


def setup(tmp_path):
    from ownership_fixtures import Replay

    from dantetl.application.model_inventory import InventoryData

    base = BackendChoice("Base", Mock(), "base")
    candidate = BackendChoice("New", Mock(), "new", "new")
    return Replay(tmp_path, data=InventoryData((base, candidate))), base, candidate


def propose(replay, choice):
    from dantetl.translation.plan import TranslationPlan

    return replay.translation.lifecycle.request_assignment(
        TranslationPlan("local:" + choice.key, "local:" + choice.key), choice
    )


def activate(replay):
    replay.translation.refresh_route()
    return replay.translation.advance_activity(running=True, enabled=True).worker


def test_consent_preserves_worker_then_preparation_waits_for_cleanup_and_is_adopted(tmp_path):
    replay, base, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        original = activate(replay)
        assert original is replay.route_workers[0]
        token = propose(replay, candidate)
        assert token is not None
        assert lifecycle.snapshot().phase == ModelPhase.CONSENT
        assert lifecycle.snapshot().selected == base
        assert activate(replay) is original and replay.route_workers[0].stops == 0
        lifecycle.consent(token, False)
        assert lifecycle.snapshot().selected == base and not lifecycle.snapshot().preparing
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.consent(token, True)
        assert lifecycle.advance_preparation() is None and not replay.workers
        replay.route_workers[0].finish()
        lifecycle.advance_preparation()
        prepared = replay.workers[0]
        prepared.state = TranslationState.READY
        outcome = lifecycle.advance_preparation()
        assert outcome and outcome.committed and lifecycle.snapshot().selected == candidate
        assert replay.translation.plan.active_models == ("local:new",)
        assert lifecycle.advance_activity(running=False, enabled=True).worker is None
        assert lifecycle.publication is prepared
        replay.translation.refresh_route()
        assert replay.translation.advance_activity(running=True, enabled=True).worker
        assert replay.routes[-1][1]["prepared"] == ("local:new", prepared)
        assert len(replay.workers) == 1 and prepared.stops == 0
    finally:
        replay.close()


def test_failed_preparation_waits_for_cleanup_and_preserves_committed_assignment(tmp_path):
    replay, base, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        original = replay.translation.plan
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.consent(token, True)
        lifecycle.advance_preparation()
        prepared = replay.workers[0]
        prepared.state = TranslationState.FAILED
        assert lifecycle.advance_preparation() is None
        assert replay.translation.plan == original
        prepared.ended = True
        outcome = lifecycle.advance_preparation()
        assert outcome and not outcome.committed and outcome.error
        assert replay.translation.plan == original and lifecycle.snapshot().selected == base
        assert lifecycle.publication is None and not replay.receipts
    finally:
        replay.close()


def test_route_refresh_during_preparation_cannot_stop_prepared_owner(tmp_path):
    replay, _, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.consent(token, True)
        lifecycle.advance_preparation()
        prepared = replay.workers[0]
        replay.translation.refresh_route()
        assert prepared.stops == 0
        prepared.state = TranslationState.READY
        outcome = lifecycle.advance_preparation()
        assert outcome and outcome.committed
        replay.translation.refresh_route()
        assert replay.translation.advance_activity(running=True, enabled=True).worker
        assert replay.routes[-1][1]["prepared"] == ("local:new", prepared)
    finally:
        replay.close()


def test_import_probe_waits_for_unload_then_resumes_committed_assignment(tmp_path):
    replay, base, _ = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        original = activate(replay)
        plan = replay.translation.plan
        assert not lifecycle.advance_activity(running=True, enabled=True, probing=True).probe_ready
        assert replay.route_workers[0].stops
        replay.route_workers[0].finish()
        assert lifecycle.suspend_for_probe(True)
        assert lifecycle.publication is None and lifecycle.snapshot().selected == base
        assert lifecycle.advance_activity(running=True, enabled=True, probing=True).probe_ready
        lifecycle.suspend_for_probe(False)
        assert activate(replay) is not original and replay.translation.plan == plan
        assert len(replay.route_workers) == 2
    finally:
        replay.close()


def test_phase_follows_consent_drain_preparation_and_commit(tmp_path):
    replay, _, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        activate(replay)
        token = propose(replay, candidate)
        assert token is not None and lifecycle.assignment_busy
        assert lifecycle.snapshot().phase == ModelPhase.CONSENT
        assert lifecycle.snapshot().proposed == candidate and not lifecycle.snapshot().preparing
        lifecycle.consent(token, True)
        # The accepted model waits for the previous owner before it gets a worker.
        assert lifecycle.advance_preparation() is None
        assert lifecycle.snapshot().phase == ModelPhase.DRAINING and lifecycle.snapshot().preparing
        replay.route_workers[0].finish()
        assert lifecycle.advance_preparation() is None
        assert lifecycle.snapshot().phase == ModelPhase.PREPARING
        assert lifecycle.preparation_message
        replay.workers[0].state = TranslationState.READY
        outcome = lifecycle.advance_preparation()
        assert outcome and outcome.committed and not lifecycle.assignment_busy
        assert lifecycle.snapshot().phase == ModelPhase.READY
        assert lifecycle.preparation_message is None and lifecycle.snapshot().proposed is None
        # A used token cannot start a second preparation.
        assert lifecycle.consent(token, True) is None and not lifecycle.assignment_busy
    finally:
        replay.close()


def test_closing_withdraws_consent_and_outlasts_later_preparation_steps(tmp_path):
    replay, base, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.shutdown()
        assert lifecycle.consent(token, True) is None and not lifecycle.assignment_busy
        assert lifecycle.snapshot().phase == ModelPhase.CLOSING
        lifecycle.resume_after_failed_restart()
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.consent(token, True)
        lifecycle.advance_preparation()
        assert lifecycle.snapshot().phase == ModelPhase.PREPARING
        lifecycle.shutdown()
        # An ordinary advance while the prepared owner drains keeps the app closing.
        assert lifecycle.advance_preparation() is None
        assert lifecycle.snapshot().phase == ModelPhase.CLOSING
        replay.workers[0].ended = True
        outcome = lifecycle.advance_preparation()
        assert outcome and not outcome.committed
        assert lifecycle.snapshot().phase == ModelPhase.CLOSING
        assert lifecycle.snapshot().selected == base and not replay.receipts
    finally:
        replay.close()


def test_shutdown_during_preparation_never_commits_or_starts_another_owner(tmp_path):
    replay, base, candidate = setup(tmp_path)
    lifecycle = replay.translation.lifecycle
    try:
        token = propose(replay, candidate)
        assert token is not None
        lifecycle.consent(token, True)
        lifecycle.advance_preparation()
        assert not lifecycle.finish_shutdown()
        assert lifecycle.advance_activity(running=True, enabled=True).worker is None
        replay.workers[0].ended = True
        assert lifecycle.finish_shutdown()
        assert lifecycle.snapshot().selected == base and not replay.receipts
        assert len(replay.workers) == 1 and not replay.routes
    finally:
        replay.close()
