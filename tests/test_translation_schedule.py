"""Scheduling policy with explicit time and manually completed futures, without workers."""

from concurrent.futures import Future
from dataclasses import replace

from dantetl.languages.target import ENGLISH
from dantetl.translation.backend import BatchResult
from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.schedule import Demand, TranslationSchedule
from dantetl.translation.values import TranslationRequest


class Backend:
    cache_identity = "scheduler-fixture"
    batch_size = 1
    max_in_flight = 2

    def translate(self, texts, *, target=ENGLISH):
        raise AssertionError("Planning must not infer")

    def translate_with_context(self, text, context, *, target=ENGLISH):
        raise AssertionError("Planning must not infer")

    def submit_translation(self, sources, *, target=ENGLISH):
        raise AssertionError("The worker owns dispatch")

    def cancel_pending(self):
        raise AssertionError("The worker owns cancellation")

    def close(self):
        raise AssertionError("The worker owns teardown")


def demand(*jobs, generation=1, contexts=None, dialogue=()):
    return Demand(generation, jobs, contexts or dict.fromkeys(jobs, ()), frozenset(dialogue))


def test_cold_sources_are_checked_once_until_withdrawn(tmp_path, monkeypatch):
    cache = TranslationCache(tmp_path / "cache")
    probes = []
    get = cache.get

    def count_get(key):
        probes.append(key)
        return get(key)

    monkeypatch.setattr(cache, "get", count_get)
    schedule, backend = TranslationSchedule(), Backend()
    jobs = tuple(TranslationRequest(i, 1, f"項目{i}") for i in range(128))
    try:
        for i, job in enumerate(jobs):
            batch = schedule.plan(demand(*jobs[i:]), backend, cache)
            assert batch and batch.missing == ((job.source, ()),)
            cache.put(batch.keys[batch.missing[0]], "English")
        assert len(probes) == len(set(probes)) == 128
        assert schedule.plan(demand(), backend, cache) is None
        hit = schedule.plan(demand(jobs[-1]), backend, cache)
        assert hit and not hit.missing and len(probes) == 129
    finally:
        cache.close()


def test_remote_candidates_are_cache_only_until_confirmation(tmp_path):
    schedule, backend = TranslationSchedule(), Backend()
    cache = TranslationCache(tmp_path / "cache")
    cached, cold = TranslationRequest(1, 1, "設定"), TranslationRequest(2, 1, "装備")
    try:
        cache.put(translation_key(backend.cache_identity, cached.source), "Settings")
        current = replace(demand(cached, cold), candidates=frozenset((cached, cold)))
        hit = schedule.plan(current, backend, cache)
        assert hit and hit.sources == ((cached.source, ()),) and not hit.missing
        assert (
            schedule.plan(replace(demand(cold), candidates=frozenset((cold,))), backend, cache)
            is None
        )
        confirmed = schedule.plan(demand(cold), backend, cache)
        assert confirmed and confirmed.missing == ((cold.source, ()),)
    finally:
        cache.close()


def test_cache_remains_available_at_capacity_and_duplicate_sources_share_a_flight(tmp_path):
    schedule, backend = TranslationSchedule(), Backend()
    cache = TranslationCache(tmp_path / "cache")
    a, b, duplicate, cached = (
        TranslationRequest(1, 1, "攻撃力"),
        TranslationRequest(2, 1, "会心率"),
        TranslationRequest(3, 1, "会心率"),
        TranslationRequest(4, 1, "設定"),
    )
    try:
        current = demand(a, b, duplicate)
        first = schedule.plan(current, backend, cache)
        assert first and first.missing == ((a.source, ()),)
        assert first.activate(current.jobs, current.contexts, {a: 9}, now=10, measured=20)
        slow: Future[BatchResult] = Future()
        schedule.submitted(first, slow)
        second = schedule.plan(current, backend, cache)
        assert second and second.missing == ((b.source, ()),)
        fast: Future[BatchResult] = Future()
        schedule.submitted(second, fast)
        assert schedule.plan(current, backend, cache) is None
        assert not schedule.needs_plan(current.generation, current.contexts.items())
        cache.put(translation_key(backend.cache_identity, cached.source), "Settings")
        hit = schedule.plan(demand(a, b, duplicate, cached), backend, cache)
        assert hit and not hit.missing and hit.sources == ((cached.source, ()),)
        assert first.queue_ms == 1000
        fast.set_result(({(b.source, ()): "Critical Rate"}, {}))
        assert schedule.completed() == ((second, fast),)
        assert not slow.done() and schedule.processing_since == 10
        assert schedule.needs_plan(current.generation, current.contexts.items())
        assert schedule.stale(2)
        schedule.reset()
        assert not schedule.stale(2) and schedule.processing_since is None
        assert not slow.cancelled()  # Cancellation stays with the engine owner.
        assert hit.keys[cached.source, ()] == translation_key(backend.cache_identity, cached.source)
    finally:
        cache.close()


def test_dialogue_serialization_allows_menu_progress_and_rechecks_context_before_dispatch(tmp_path):
    schedule, backend = TranslationSchedule(), Backend()
    cache = TranslationCache(tmp_path / "cache")
    first, next_line, menu = (
        TranslationRequest(1, 1, "最初の会話です。"),
        TranslationRequest(2, 1, "次の会話です。"),
        TranslationRequest(3, 1, "設定"),
    )
    try:
        current = demand(first, next_line, menu, dialogue=(first, next_line))
        batch = schedule.plan(current, backend, cache)
        assert batch and batch.serial
        future: Future[BatchResult] = Future()
        schedule.submitted(batch, future)
        independent = schedule.plan(current, backend, cache)
        assert independent and independent.missing == ((menu.source, ()),)
        assert not independent.serial
        history = (DialogueTurn(first.source, "First line."),)
        changed = {first: history, next_line: (), menu: ()}
        assert not batch.activate(current.jobs, changed, {}, now=11, measured=11)
        future.set_result(({}, {}))
        schedule.completed()
        before = schedule.plan(demand(next_line), backend, cache)
        after = schedule.plan(demand(next_line, contexts={next_line: history}), backend, cache)
        assert before and after and before.keys != after.keys
        # Removing the request while cache lookup was in progress cannot dispatch it.
        assert not after.activate((), {}, {}, now=12, measured=12)
    finally:
        cache.close()


def test_changing_first_box_cannot_starve_a_stable_later_request(tmp_path):
    schedule, backend = TranslationSchedule(), Backend()
    backend.max_in_flight = 1
    cache = TranslationCache(tmp_path / "cache")
    stable = TranslationRequest(2, 1, "ここで待っています。")
    selected = []
    try:
        for revision in range(20):
            changing = TranslationRequest(1, revision, f"現在の値は{revision}です。")
            batch = schedule.plan(demand(changing, stable), backend, cache)
            assert batch
            if batch.missing:
                selected.extend(source for source, _ in batch.missing)
                future: Future[BatchResult] = Future()
                schedule.submitted(batch, future)
                answer = {s: "English " + s[0] for s in batch.missing}
                future.set_result((answer, {}))
                schedule.completed()
                for source, english in answer.items():
                    cache.put(batch.keys[source], english)
        assert selected[1] == stable.source
        assert selected.count(stable.source) == 1
    finally:
        cache.close()


def test_waiting_age_survives_reorder_but_not_withdrawal_revision_context_or_reset(tmp_path):
    backend = Backend()
    cache = TranslationCache(tmp_path / "cache")
    a, b, c = (TranslationRequest(i, 1, f"待機{i}番") for i in range(3))
    try:
        for change in ("reorder", "withdrawal", "revision", "context", "generation", "reset"):
            schedule = TranslationSchedule()
            assert schedule.plan(demand(a, b, c), backend, cache)
            contexts = None
            generation = 1
            if change == "withdrawal":
                assert schedule.plan(demand(a, c), backend, cache)
            elif change == "revision":
                b = replace(b, text_revision=b.text_revision + 1)
            elif change == "context":
                contexts = {b: (DialogueTurn("以前の会話", "Earlier."),), c: ()}
            elif change == "generation":
                generation = 2
            elif change == "reset":
                schedule.reset()
            batch = schedule.plan(
                demand(c, b, generation=generation, contexts=contexts), backend, cache
            )
            assert batch
            expected = b if change == "reorder" else c
            assert batch.missing[0][0] == expected.source, change
    finally:
        cache.close()
