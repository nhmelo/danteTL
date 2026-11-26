import time
from concurrent.futures import Future

from dantetl.domain import Rect
from dantetl.languages.target import ENGLISH
from dantetl.ocr.observation import TextTrack
from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.grouping import TextBlock
from dantetl.translation.worker import TranslationSession


class Deferred:
    cache_identity = "controlled-concurrent-fixture"
    batch_size = 1
    max_in_flight = 2

    def __init__(self):
        self.calls = []
        self.closed = False

    def translate(self, texts, *, target=ENGLISH):
        raise AssertionError("Must use deferred submission")

    def translate_with_context(self, text, context, *, target=ENGLISH):
        raise AssertionError("Must use deferred submission")

    def submit_translation(self, sources, *, target=ENGLISH):
        future = Future()
        self.calls.append((sources[0], future))
        return future

    def cancel_pending(self):
        for _, future in self.calls:
            future.cancel()

    def close(self):
        self.cancel_pending()
        self.closed = True

    def finish(self, index):
        source, future = self.calls[index]
        future.set_result(({source: "English: " + source[0]}, {}))


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def block(key, text, *, dialogue=False):
    return TextBlock((TextTrack(key, text, Rect(20, key * 50, 300, 30)),), dialogue=dialogue)


def stop(session):
    session.request_stop()
    until(lambda: session.finished)


def test_cache_only_miss_can_be_promoted_with_identical_remote_source_and_context(tmp_path):
    backend = Deferred()
    path = tmp_path / "cache"
    cache = TranslationCache(path)
    cache.put(translation_key(backend.cache_identity, "設定"), "Settings")
    cache.close()
    session = TranslationSession(lambda: backend, path)
    cached, missing = block(1, "設定"), block(2, "速度")
    session.start()
    try:
        session.update((), retain=(cached, missing), candidates=(cached, missing))
        until(lambda: bool(session.snapshot().cached_candidates))
        assert not backend.calls and not session.snapshot().results
        session.update((missing,), retain=(cached, missing), candidates=(cached,))
        until(lambda: len(backend.calls) == 1)
        assert backend.calls[0][0] == (missing.text, ())
        backend.finish(0)
        until(lambda: session.snapshot().pending == 0)
        assert [r.request.source for r in session.snapshot().results] == [missing.text]
        session.update((cached, missing), retain=(cached, missing))
        assert {r.request.source for r in session.snapshot().results} == {cached.text, missing.text}
        assert not session.snapshot().cached_candidates and len(backend.calls) == 1
    finally:
        stop(session)


def test_independent_answers_publish_out_of_order_with_two_slots_cache_and_source_dedup(tmp_path):
    backend = Deferred()
    path = tmp_path / "cache"
    cache = TranslationCache(path)
    cache.put(translation_key(backend.cache_identity, "設定"), "Settings")
    cache.close()
    session = TranslationSession(lambda: backend, path)
    session.start()
    try:
        items = (block(1, "攻撃力"), block(2, "会心率"), block(3, "防御力"), block(4, "会心率"))
        session.update(items)
        until(lambda: len(backend.calls) == 2)
        session.update((*items, block(5, "設定")))
        until(lambda: len(session.snapshot().results) == 1)
        assert len(backend.calls) == 2
        backend.finish(1)
        until(lambda: len(session.snapshot().results) == 3)
        until(lambda: len(backend.calls) == 3)
        # The slow first term does not block the second term or its duplicate's cache hit.
        assert {r.request.track_id for r in session.snapshot().results} == {2, 4, 5}
        snapshot = session.snapshot().results
        for _ in range(20):
            session.update((*items, block(5, "設定")))
            assert session.snapshot().results == snapshot
        backend.finish(0)
        backend.finish(2)
        until(lambda: len(session.snapshot().results) == 5)
        assert len(backend.calls) == 3
    finally:
        stop(session)
    assert backend.closed


def test_dialogue_is_serial_but_does_not_block_an_independent_menu(tmp_path):
    backend = Deferred()
    publications = []
    session = TranslationSession(
        lambda: backend,
        tmp_path / "cache",
        on_change=lambda: publications.append(session.snapshot()),
    )
    session.start()
    try:
        session.update(
            (
                block(1, "最初の会話です。", dialogue=True),
                block(2, "次の会話です。", dialogue=True),
                block(3, "設定"),
            )
        )
        until(lambda: len(backend.calls) == 2)
        assert [source[0] for source, _ in backend.calls] == ["最初の会話です。", "設定"]
        dialogue_started = session.snapshot().processing_since
        assert dialogue_started is not None
        backend.finish(1)
        until(lambda: any(len(s.results) == 1 for s in publications))
        assert publications[-1].processing_since == dialogue_started
        assert len(backend.calls) == 2
        backend.finish(0)
        until(lambda: len(backend.calls) == 3)
        assert backend.calls[2][0][0] == "次の会話です。"
        backend.finish(2)
        until(lambda: len(session.snapshot().results) == 3)
    finally:
        stop(session)


def test_shared_cloud_call_survives_a_new_owner_and_retained_only_until_withdrawn(tmp_path):
    backend = Deferred()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()
    try:
        label, duplicate = block(1, "設定"), block(2, "設定")
        other = block(3, "速度")
        session.update((label, other))
        until(lambda: len(backend.calls) == 2)
        session.update((duplicate,), retain=(other,))
        backend.finish(0)
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].request.track_id == 2
        assert session.snapshot().cancelled_obsolete == 0
        session.update((duplicate,))
        until(lambda: backend.calls[1][1].cancelled())
        assert session.snapshot().cancelled_obsolete == 1
        assert len(backend.calls) == 2
    finally:
        stop(session)


def test_retained_dialogue_keeps_the_context_of_its_inflight_translation(tmp_path):
    backend = Deferred()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()
    try:
        first = block(1, "最初の会話です。", dialogue=True)
        second = block(1, "次の会話です。", dialogue=True)
        session.update((first,))
        until(lambda: len(backend.calls) == 1)
        backend.finish(0)
        until(lambda: len(session.snapshot().results) == 1)
        session.update((second,))
        until(lambda: len(backend.calls) == 2)
        assert backend.calls[1][0][1]
        label = block(2, "設定")
        session.update((label,), retain=(second, label))
        until(lambda: len(backend.calls) == 3)
        session.update((second, label))
        backend.finish(1)
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].request.source == second.text
        assert len(backend.calls) == 3 and session.snapshot().cancelled_obsolete == 0
    finally:
        stop(session)


def test_new_cache_hit_and_another_owners_fill_publish_with_both_flights_busy(tmp_path):
    backend = Deferred()
    path = tmp_path / "cache"
    writer = TranslationCache(path)
    session = TranslationSession(lambda: backend, path)
    session.start()
    try:
        items = (block(1, "攻撃力"), block(2, "防御力"), block(3, "速度"))
        session.update(items)
        until(lambda: len(backend.calls) == 2)
        # This owner fills an already probed miss and a source the session has never seen.
        writer.put(translation_key(backend.cache_identity, "速度"), "Speed")
        writer.put(translation_key(backend.cache_identity, "設定"), "Settings")
        session.update((*items, block(4, "設定")))
        until(lambda: len(session.snapshot().results) == 2)
        assert {r.english for r in session.snapshot().results} == {"Speed", "Settings"}
        assert len(backend.calls) == 2 and all(not future.done() for _, future in backend.calls)
    finally:
        stop(session)
        writer.close()


def test_swap_cancels_every_inflight_request_and_never_reuses_old_english(tmp_path):
    old, new = Deferred(), Deferred()
    new.cache_identity = "replacement"
    session = TranslationSession(lambda: old, tmp_path / "cache")
    session.start()
    try:
        session.update((block(1, "攻撃力"), block(2, "会心率")))
        until(lambda: len(old.calls) == 2)
        session.switch_backend(lambda: new)
        until(lambda: len(new.calls) == 2)
        assert old.closed and all(f.cancelled() for _, f in old.calls)
        assert not session.snapshot().results
        new.finish(1)
        until(lambda: len(session.snapshot().results) == 1)
        session.invalidate()
        until(lambda: new.calls[0][1].cancelled())
        assert not session.snapshot().results
    finally:
        stop(session)
