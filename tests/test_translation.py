import threading
from dataclasses import replace

import pytest
from live_fixtures import Replay, frame
from translation_fixtures import Backend, stop, track, until

from dantetl.application.live_session import LiveSession
from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.selection import source_language
from dantetl.languages.target import ENGLISH, TARGET_LANGUAGES
from dantetl.ocr.observation import TextTrack
from dantetl.ocr.tracking import TextTracker
from dantetl.ocr.values import OcrResult, TextCharacter, TextRegion
from dantetl.translation.backend import RequestRejected
from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.grouping import DialogueGrouper, TextBlock
from dantetl.translation.values import TranslationRequest, TranslationState
from dantetl.translation.worker import TranslationSession


@pytest.mark.parametrize("confirm", [False, True])
def test_first_read_miss_is_private_until_the_exact_request_is_confirmed(tmp_path, confirm):
    path = tmp_path / "cache"
    backend = Backend()
    session = TranslationSession(lambda: backend, path)
    block = TextBlock((track(),))
    expected = TranslationRequest.from_block(block)
    session.start()
    try:
        session.update((), retain=(block,), candidates=(block,))
        until(lambda: session.snapshot().translated_sources == 1)
        for _ in range(3):
            session.update((), retain=(block,), candidates=(block,))
            snapshot = session.snapshot()
            assert not snapshot.results and not snapshot.cached_candidates
        if confirm:
            session.update((block,), retain=(block,))
            until(lambda: tuple(t.request for t in session.snapshot().results) == (expected,))
        else:
            session.update((), retain=(block,))
        assert backend.calls == [(block.text,)]
    finally:
        session.request_stop()
        until(lambda: session.finished)
    cache = TranslationCache(path)
    try:
        assert bool(cache.get(translation_key(backend.cache_identity, block.text))) == confirm
    finally:
        cache.close()


def test_confirmed_label_gets_next_slot_and_withdrawn_first_read_is_discarded(tmp_path):
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    first, later, ready = (
        TextBlock((track(s, key=i),)) for i, s in enumerate(("設定", "装備", "速度"), 1)
    )
    session.start()
    try:
        session.update((), retain=(first, later), candidates=(first, later))
        assert backend.entered.wait(2)
        session.update((ready,), retain=(first, later, ready), candidates=(later,))
        backend.release.set()
        until(lambda: len(backend.calls) == 3 and session.snapshot().processing_since is None)
        assert backend.calls == [(first.text,), (ready.text,), (later.text,)]
        assert [t.request.source for t in session.snapshot().results] == [ready.text]
        assert not session.snapshot().cached_candidates
    finally:
        backend.release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_confirmation_can_paint_a_prepared_answer_while_the_model_is_busy(tmp_path):
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    label, other = TextBlock((track(),)), TextBlock((track("速度", key=2),))
    session.start()
    try:
        session.update((), retain=(label,), candidates=(label,))
        until(lambda: session.snapshot().translated_sources == 1)
        backend.entered.clear()
        backend.release.clear()
        session.update((other,), retain=(label, other), candidates=(label,))
        assert backend.entered.wait(2)
        session.update((label, other), retain=(label, other))
        assert [r.request.source for r in session.snapshot().results] == [label.text]
    finally:
        backend.release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_live_typewriter_sends_only_first_and_final_readings_to_local_model(tmp_path):
    replay, backend = Replay(), Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()

    def read(seq, now, text):
        row = TextRegion(text, Rect(50, 400, len(text) * 25, 30), line_confidence=0.98)
        scene = replay.observe(frame(seq, now), (row,), settle=0)
        return replay.live.publish(
            session, scene.requests, scene.blocks, candidates=scene.candidates
        )

    try:
        first, finished = "装備", "装備を確認する。"
        assert not read(1, 1, first).translations
        assert backend.entered.wait(2)
        for seq, text in enumerate(("装備を", "装備を確認", finished), 2):
            assert not read(seq, 1 + seq * 0.05, text).translations
        backend.release.set()
        until(lambda: session.snapshot().processing_since is None)
        assert backend.calls == [(first,)] and not session.snapshot().results
        read(5, 1.7, finished)
        until(lambda: bool(session.snapshot().results))
        assert backend.calls == [(first,), (finished,)]
        assert [r.request.source for r in session.snapshot().results] == [finished]
    finally:
        backend.release.set()
        stop(session)


@pytest.mark.parametrize("initial,following", [("en", "es"), ("pt-PT", "pt-BR")])
def test_output_change_discards_inflight_answer_and_reuses_each_languages_cache(
    tmp_path, initial, following
):
    first, second = TARGET_LANGUAGES[initial], TARGET_LANGUAGES[following]
    started, release = threading.Event(), threading.Event()

    class Translator(Backend):
        def translate(self, texts, *, target=ENGLISH):
            self.calls.append((target.code, *texts))
            if target == first:
                started.set()
                assert release.wait(3)
            return tuple(target.name for _ in texts)

    backend = Translator()
    session = TranslationSession(lambda: backend, tmp_path / "cache")
    session.start()
    rows = (track(),)
    try:
        session.update(rows, target=first)
        assert started.wait(3)
        session.update(rows, target=second)
        assert not session.snapshot().results
        release.set()
        until(lambda: any(r.target == second for r in session.snapshot().results))
        assert [r.english for r in session.snapshot().results] == [second.name]
        for target in (first, second):
            session.update(rows, target=target)
            until(lambda: session.snapshot().cache_hits == 1 and session.snapshot().results)
            assert [r.english for r in session.snapshot().results] == [target.name]
        assert backend.calls == [(initial, rows[0].text), (following, rows[0].text)]
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)
    reopened = TranslationCache(tmp_path / "cache")
    try:
        for target in (first, second):
            key = translation_key(backend.cache_identity, rows[0].text, target=target)
            assert reopened.get(key) == target.name
    finally:
        reopened.close()


def test_source_switch_keeps_model_and_scopes_inflight_work_and_cache(tmp_path):
    first, second = source_language("es"), source_language("fr")

    class Translator(Backend):
        language = first

        def set_source(self, language):
            self.language = language

        def translate(self, texts, *, target=ENGLISH):
            language = self.language
            super().translate(texts, target=target)
            return (language.name,) * len(texts)

    backend = Translator()
    session = TranslationSession(lambda: backend, tmp_path / "cache", language=first)
    rows = (track("Menu"),)

    def unexpected_reload():
        pytest.fail("Compatible source switch reopened the model")

    backend.release.clear()
    session.start()
    try:
        session.update(rows)
        assert backend.entered.wait(2)
        session.set_source(second, unexpected_reload)
        session.update(rows)
        assert not session.snapshot().results
        backend.release.set()
        until(lambda: session.snapshot().results)
        assert [r.english for r in session.snapshot().results] == [second.name]
        for language in (first, second):
            session.set_source(language, unexpected_reload)
            session.update(rows)
            until(lambda: session.snapshot().results)
            assert session.snapshot().cache_hits == 1
            assert session.snapshot().results[0].english == language.name
        assert not backend.closed and len(backend.calls) == 2
    finally:
        backend.release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_switching_region_starts_fresh_model_dialogue_context(tmp_path):
    class Contextual(Backend):
        def translate_with_context(self, source, context, *, target):
            self.calls.append((source, target.code, context))
            return target.name

    backend = Contextual()
    worker = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    pt, br = TARGET_LANGUAGES["pt-PT"], TARGET_LANGUAGES["pt-BR"]
    first = TextBlock((track("私の名前はリナです。"),), dialogue=True)
    second = TextBlock((track("明日は一緒に行きましょう。", key=2),), dialogue=True)
    third = TextBlock((track("また明日会いましょう。", key=3),), dialogue=True)
    worker.start()
    try:
        for block, target in ((first, pt), (second, pt), (second, br), (third, br)):
            worker.update((block,), target=target)
            until(lambda: worker.snapshot().pending == 0 and worker.snapshot().results)
        assert backend.calls == [
            (first.text, pt.code, ()),
            (second.text, pt.code, (DialogueTurn(first.text, pt.name),)),
            (second.text, br.code, ()),
            (third.text, br.code, (DialogueTurn(second.text, br.name),)),
        ]
    finally:
        stop(worker)


def test_worker_serves_stable_text_while_earlier_box_keeps_changing(tmp_path):
    stable = track("ここで待っています。", key=2)

    class Changing(Backend):
        batch_size = 1
        revision = 0

        def translate(self, texts, *, target=ENGLISH):
            self.calls.append(texts)
            if texts[0] != stable.text and self.revision < 20:
                self.revision += 1
                # A fresh OCR revision arrives during each inference. The stale
                # answer may fill cache, but cannot publish or jump the waiting line.
                session.update((track(f"現在の値は{self.revision}です。", self.revision), stable))
            return tuple("English " + source for source in texts)

    backend = Changing()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track("現在の値は0です。", 0), stable))
        until(lambda: session.snapshot().pending == 0)
        assert backend.calls[1] == (stable.text,)
        assert backend.calls.count((stable.text,)) == 1
        assert {r.request.source for r in session.snapshot().results} == {
            "現在の値は20です。",
            stable.text,
        }
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_hidden_english_cannot_resurface_on_a_new_menu_while_translation_is_running(tmp_path):
    live, backend = LiveSession(), Backend()
    worker = TranslationSession(lambda: backend, tmp_path / "menu.sqlite3")
    old = TextBlock((track("設定"),))
    arriving = TextBlock((track("地図", key=2),))
    reused = TextBlock((track("装備", revision=2),))
    worker.start()
    try:
        live.publish(worker, (old,), (old,))
        until(lambda: bool(worker.snapshot().results))
        assert live.publish(worker, (old,), (old,)).blocks == (old,)
        backend.entered.clear()
        backend.release.clear()
        live.publish(worker, (arriving,), (old, arriving))
        assert backend.entered.wait(2)
        live.visual_state = replace(live.visual_state, cleared=frozenset(old.lines))
        assert not live.publish(worker, (arriving,), (old, arriving)).blocks
        live.visual_state = replace(live.visual_state, cleared=frozenset())
        assert not live.publish(worker, (reused, arriving), (reused, arriving)).blocks
        backend.release.set()
        until(lambda: worker.snapshot().pending == 0)
        assert set(live.publish(worker, (reused, arriving), (reused, arriving)).blocks) == {
            reused,
            arriving,
        }
    finally:
        backend.release.set()
        worker.request_stop()
        until(lambda: worker.finished)


def test_worker_context_preserves_negation_and_forwards_exact_correction_lineage(tmp_path):
    class Contextual(Backend):
        batch_size = 1

        def __init__(self):
            super().__init__()
            self.contexts = []

        def translate_with_context(self, source, context, *, target=ENGLISH):
            self.contexts.append((source, context))
            return "English " + source

    backend = Contextual()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    first = TextBlock((track("私は彼を信じている。"),), dialogue=True)
    second = TextBlock((track("私は彼を信じていない。", 2),), dialogue=True)
    corrected = TextBlock((track("私は彼女を信じていない。", 3),), dialogue=True)
    session.start()
    try:
        session.update((first,))
        until(lambda: session.snapshot().pending == 0)
        expected = (DialogueTurn(first.text, "English " + first.text),)
        session.update((second,))
        until(lambda: session.snapshot().pending == 0)
        assert backend.contexts[-1] == (second.text, expected)
        session.update((corrected,), corrections=((corrected.lines[0], second.lines[0]),))
        until(lambda: session.snapshot().pending == 0)
        assert backend.contexts[-1] == (corrected.text, expected)
        for _ in range(10):
            session.update((corrected,))
        assert len(backend.contexts) == 3
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_request_rejection_preserves_other_results_and_does_not_retry(tmp_path):
    class Selective(Backend):
        def translate(self, texts, *, target=ENGLISH):
            self.calls.append(texts)
            if "拒否" in texts:
                raise RequestRejected("Incomplete answer")
            return tuple("English " + text for text in texts)

    backend = Selective()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        good = track("設定", key=1)
        session.update((good,))
        until(lambda: len(session.snapshot().results) == 1)
        bad, oversized = track("拒否", key=2), track("語" * 513, key=3)
        requests = (good, bad, oversized, track("戻る", key=4))
        session.update(requests)
        until(lambda: session.snapshot().pending == 0)
        snapshot = session.snapshot()
        assert snapshot.state == TranslationState.READY and len(snapshot.results) == 2
        assert {r.request.track_id for r in snapshot.rejected} == {2, 3}
        count = len(backend.calls)
        for _ in range(30):
            session.update(requests)
        assert len(backend.calls) == count
        session.update((good, track("続ける", revision=2, key=2)))
        until(lambda: session.snapshot().pending == 0)
        assert not session.snapshot().rejected and len(session.snapshot().results) == 2
    finally:
        stop(session)


def test_stale_rejection_cannot_block_new_revision_or_backend(tmp_path):
    class Rejecting(Backend):
        def translate(self, texts, *, target=ENGLISH):
            self.entered.set()
            assert self.release.wait(3)
            raise RequestRejected("Fixture rejection")

    backend = Rejecting()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track(),))
        assert backend.entered.wait(3)
        session.switch_backend(Backend)
        backend.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        assert not session.snapshot().rejected
    finally:
        backend.release.set()
        stop(session)


def test_native_backend_failure_remains_terminal(tmp_path):
    class Broken(Backend):
        def translate(self, texts, *, target=ENGLISH):
            raise RuntimeError("Native engine unusable")

    backend = Broken()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    session.update((track(),))
    until(lambda: session.finished)
    assert session.snapshot().state == TranslationState.FAILED and backend.closed


def test_rejection_is_context_scoped_and_context_reset_releases_same_request(tmp_path):
    class Contextual(Backend):
        batch_size = 1

        def translate_with_context(self, source, context, *, target=ENGLISH):
            if context:
                raise RequestRejected("Cannot answer with this context")
            return "Standalone English"

    session = TranslationSession(Contextual, tmp_path / "cache.sqlite3")
    first = TextBlock((track("私の名前はリナです。"),), dialogue=True)
    second = TextBlock((track("明日は一緒に行きましょう。", key=2),), dialogue=True)
    session.start()
    try:
        session.update((first,))
        until(lambda: session.snapshot().pending == 0)
        session.update((second,))
        until(lambda: session.snapshot().pending == 0)
        assert session.snapshot().rejected and not session.snapshot().results
        session.update((second,), reset_context=True)
        until(lambda: session.snapshot().pending == 0)
        assert not session.snapshot().rejected and session.snapshot().results
    finally:
        stop(session)


def test_rejected_history_is_bounded_and_reset_does_not_suppress_new_work(tmp_path):
    class Rejecting(Backend):
        def translate(self, texts, *, target=ENGLISH):
            self.calls.append(texts)
            raise RequestRejected("Unusable answer")

    backend = Rejecting()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        for key in range(280):
            session.update((track(key=key),))
            until(lambda: session.snapshot().pending == 0)
        assert len(session._requests.rejections) == 256
        assert len(backend.calls) == 280
        session.update((track(key=279),))
        assert session.snapshot().pending == 0
        session.invalidate()
        session.update((track(key=279),))
        until(lambda: session.snapshot().pending == 0)
        assert len(backend.calls) == 281
    finally:
        stop(session)


def test_context_is_frozen_per_request_menu_independent_and_cache_scoped(tmp_path):
    class Contextual(Backend):
        batch_size = 1

        def __init__(self):
            super().__init__()
            self.contexts = []

        def translate_with_context(self, source, context, *, target=ENGLISH):
            self.contexts.append((source, context))
            return f"English {source} ({len(context)})"

    backend = Contextual()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    first = TextBlock((track("私の名前はリナです。"),), dialogue=True)
    second = TextBlock((track("明日は一緒に行きましょう。", key=2),), dialogue=True)
    menu = TextBlock((track("設定", key=3),))
    session.start()
    try:
        session.update((first,))
        until(lambda: session.snapshot().pending == 0)
        previous = session.snapshot().results[0].english
        session.update((second, menu))
        until(lambda: session.snapshot().pending == 0)
        assert backend.contexts == [
            (first.text, ()),
            (second.text, (DialogueTurn(first.text, previous),)),
            (menu.text, ()),
        ]
        for _ in range(5):
            session.update((second, menu))
        assert session.snapshot().pending == 0 and len(backend.contexts) == 3
        session.invalidate()
        session.update((second, menu))
        until(lambda: session.snapshot().pending == 0)
        # The dialogue's standalone result is distinct; menu remains a cache hit.
        assert backend.contexts[-1] == (second.text, ())
        assert len(backend.contexts) == 4
    finally:
        stop(session)


@pytest.mark.parametrize("first_read", [False, True])
def test_context_reset_rejects_in_flight_context_and_retries_without_it(tmp_path, first_read):
    class Contextual(Backend):
        batch_size = 1
        contexts = []

        def translate_with_context(self, source, context, *, target=ENGLISH):
            self.contexts.append(context)
            return self.translate((source,))[0] + (" context" if context else " standalone")

    backend = Contextual()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    first = TextBlock((track("私の名前はリナです。"),), dialogue=True)
    second = TextBlock((track("明日は一緒に行きましょう。", key=2),), dialogue=True)
    session.start()
    try:
        session.update((first,))
        until(lambda: session.snapshot().pending == 0)
        backend.release.clear()
        backend.entered.clear()
        if first_read:
            session.update((), retain=(second,), candidates=(second,))
        else:
            session.update((second,))
        assert backend.entered.wait(2)
        session.update((second,), reset_context=True)
        backend.release.set()
        until(lambda: session.snapshot().pending == 0)
        assert backend.contexts[-2] and not backend.contexts[-1]
        assert session.snapshot().results[0].english.endswith("standalone")
    finally:
        backend.release.set()
        stop(session)


def test_serial_llm_publishes_first_label_before_waiting_for_second(tmp_path):
    class Serial(Backend):
        batch_size = 1

        def translate(self, texts, *, target=ENGLISH):
            if self.calls:
                self.release.clear()
            return super().translate(texts)

    backend = Serial()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.update((track("設定"), track("冒険", key=2)))
    session.start()
    try:
        until(lambda: len(backend.calls) == 2)
        snapshot = session.snapshot()
        assert len(snapshot.results) == 1 and snapshot.results[0].request.source == "設定"
        assert snapshot.pending == 1 and snapshot.processing_since is not None
        assert snapshot.translated_sources == 1 and snapshot.cache_hits == 0
        backend.release.set()
        until(lambda: session.snapshot().pending == 0)
        assert session.snapshot().processing_since is None
        inference = session.snapshot().last_inference_ms
        assert inference > 0
        session.invalidate()
        session.update((track("設定", key=3),))
        until(lambda: session.snapshot().pending == 0)
        assert len(backend.calls) == 2
        assert session.snapshot().cache_hits == 1
        assert session.snapshot().translated_sources == 0
        assert session.snapshot().last_inference_ms == inference
    finally:
        backend.release.set()
        stop(session)


@pytest.mark.parametrize("restart", [False, True])
def test_all_cached_labels_publish_before_first_uncached_call(tmp_path, restart):
    backend = Backend()
    backend.batch_size = 1
    path = tmp_path / "cache.sqlite3"
    notices = []
    session = TranslationSession(
        lambda: backend, path, on_change=lambda: notices.append(session.snapshot())
    )
    session.start()
    try:
        labels = tuple(track(f"設定{i}", key=i) for i in range(6))
        session.update(labels)
        until(lambda: session.snapshot().pending == 0)
        if restart:
            stop(session)
            backend = Backend()
            backend.batch_size = 1
            session = TranslationSession(
                lambda: backend, path, on_change=lambda: notices.append(session.snapshot())
            )
            session.start()
        else:
            session.invalidate()
        backend.calls.clear()
        backend.entered.clear()
        backend.release.clear()
        cached = tuple(replace(t, track_id=t.track_id + 10) for t in labels)
        duplicate = replace(cached[0], track_id=20)
        session.update((track("長い新しい説明文", key=30), *cached, duplicate))
        assert backend.entered.wait(3)
        snapshot = session.snapshot()
        assert any(len(n.results) == 7 and n.cache_hits == 6 for n in notices)
        assert {t.request.track_id for t in snapshot.results} == {10, 11, 12, 13, 14, 15, 20}
        assert snapshot.cache_hits == 6 and snapshot.translated_sources == 0
        assert snapshot.pending == 1 and snapshot.processing_since is not None
        stable = snapshot.results
        for _ in range(20):
            session.update((track("長い新しい説明文", key=30), *cached, duplicate))
            assert session.snapshot().results == stable
        assert backend.calls == [("長い新しい説明文",)]
        backend.release.set()
        until(lambda: session.snapshot().pending == 0)
        assert all(t in session.snapshot().results for t in stable)
    finally:
        backend.release.set()
        stop(session)


@pytest.mark.parametrize("change", ["revision", "retain", "stop", "backend"])
def test_cache_sweep_cannot_publish_obsolete_work(tmp_path, monkeypatch, change):
    backend, replacement = Backend("old"), Backend("new")
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    entered, release = threading.Event(), threading.Event()
    session.start()
    try:
        item = track("設定")
        session.update((item,))
        until(lambda: session.snapshot().pending == 0)
        session.invalidate()
        original_get = TranslationCache.get

        def delayed_get(cache, key):
            answer = original_get(cache, key)
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
            return answer

        monkeypatch.setattr(TranslationCache, "get", delayed_get)
        session.update((item,))
        assert entered.wait(3)
        if change == "revision":
            session.update((track("戻る", revision=2),))
        elif change == "retain":
            session.update((), retain=(TextBlock((item,)),))
        elif change == "stop":
            session.request_stop()
        else:
            session.switch_backend(lambda: replacement)
        assert not session.snapshot().results
        release.set()
        until(lambda: session.snapshot().pending == 0)
        if change in ("retain", "stop"):
            stop(session)
            assert not session.snapshot().results
        else:
            result = session.snapshot().results[0]
            assert result.english == ("old: 戻る" if change == "revision" else "new: 設定")
    finally:
        release.set()
        stop(session)


def test_cache_sweep_revalidates_frozen_dialogue_context(tmp_path, monkeypatch):
    class Contextual(Backend):
        batch_size = 1

        def translate_with_context(self, source, context, *, target=ENGLISH):
            return f"{source}: {'context' if context else 'standalone'}"

    session = TranslationSession(Contextual, tmp_path / "cache.sqlite3")
    first = TextBlock((track("私の名前はリナです。"),), dialogue=True)
    second = TextBlock((track("明日は一緒に行きましょう。", key=2),), dialogue=True)
    entered, release = threading.Event(), threading.Event()
    session.start()
    try:
        for block in (first, second):
            session.update((block,))
            until(lambda: session.snapshot().pending == 0)
        assert session.snapshot().results[0].english.endswith("context")
        session.invalidate()
        session.update((first,))
        until(lambda: session.snapshot().pending == 0)
        original_get = TranslationCache.get

        def delayed_get(cache, key):
            answer = original_get(cache, key)
            if not entered.is_set():
                assert answer is not None and answer.endswith("context")
                entered.set()
                assert release.wait(3)
            return answer

        monkeypatch.setattr(TranslationCache, "get", delayed_get)
        session.update((second,))
        assert entered.wait(3)
        session.update((second,), reset_context=True)
        release.set()
        until(lambda: session.snapshot().pending == 0)
        assert session.snapshot().results[0].english.endswith("standalone")
    finally:
        release.set()
        stop(session)


def test_unchanged_text_movement_and_duplicate_regions_translate_once(tmp_path) -> None:
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track(), track(key=2), track("English", key=3)))
        until(lambda: len(session.snapshot().results) == 2)
        assert backend.calls == [("日本語",)]
        moved = replace(track(), bounds=Rect(300, 300, 200, 40))
        session.update((moved, track(key=2)))
        assert session.snapshot().pending == 0
        assert len(backend.calls) == 1
    finally:
        stop(session)
    assert backend.closed


def test_noisy_readings_do_not_clear_english_or_trigger_more_translation(tmp_path) -> None:
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    tracker = TextTracker()
    sequence = 0

    def observe(text: str, confidence: float, bounds: Rect) -> None:
        nonlocal sequence
        sequence += 1
        now = sequence * 0.25
        viewport = Rect(0, 0, 1280, 720)
        frame = CapturedFrame(sequence, now, WindowGeometry(viewport, viewport, viewport, 96), b"")
        characters = tuple(TextCharacter(char, bounds, confidence) for char in text)
        region = TextRegion(text, bounds, characters)
        tracker.observe(OcrResult(frame, (region,), 20, now + 0.02))
        session.update(tracker.advance(frame, now + 0.02).confirmed)

    session.start()
    try:
        observe("ポーションを使う", 0.95, Rect(100, 100, 200, 40))
        observe("ポーションを使う", 0.9, Rect(100, 100, 200, 40))
        until(lambda: len(session.snapshot().results) == 1)
        translated = session.snapshot().results
        for _ in range(10):
            for _ in range(3):
                observe("ボーションを使う", 0.7, Rect(80, 100, 220, 40))
                assert session.snapshot().results == translated
                assert session.snapshot().pending == 0
            observe("ポーションを使う", 0.8, Rect(100, 100, 200, 40))
    finally:
        stop(session)
    assert backend.calls == [("ポーションを使う",)]


def test_busy_screen_keeps_english_and_dialogue_membership_between_slow_passes(tmp_path) -> None:
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    tracker, grouper = TextTracker(), DialogueGrouper()
    viewport = Rect(0, 0, 3840, 2160)
    regions = tuple(
        TextRegion("設定の確認", Rect(40 + i % 8 * 450, 40 + i // 8 * 100, 260, 32))
        for i in range(64)
    ) + (
        TextRegion("新しい冒険が始まりますが、", Rect(100, 1700, 800, 36)),
        TextRegion("その前に準備をしてください。", Rect(100, 1750, 800, 36)),
    )
    session.start()
    try:
        committed = None
        for sequence in range(1, 10):
            captured = sequence * 1.2
            frame = CapturedFrame(
                sequence, captured, WindowGeometry(viewport, viewport, viewport, 96), b""
            )
            # Miss some labels AND a dialogue member twice, then recover.
            detected = regions if sequence <= 2 or sequence % 3 == 0 else regions[8:-1]
            tracker.observe(OcrResult(frame, detected, 1100, captured + 1.1))
            latest = replace(frame, sequence=sequence + 1, captured_at=captured + 1)
            for offset in (1.1, 1.5, 1.9):
                now = captured + offset
                tracks = tracker.advance(latest, now)
                grouped = grouper.update(tracks.visible, tracks.confirmed, now)
                session.update(grouped.ready)
                if committed is not None:
                    assert session.snapshot().results == committed
                    assert session.snapshot().pending == 0
            if sequence == 2:
                until(lambda: len(session.snapshot().results) == 65)
                committed = session.snapshot().results
        assert committed and any(len(t.request.members) == 2 for t in committed)
        assert sum(len(batch) for batch in backend.calls) == 2
    finally:
        stop(session)


def test_reveal_debounce_preserves_published_english_without_queueing_unready_work(
    tmp_path,
) -> None:
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    block = TextBlock((track(),))
    session.start()
    try:
        session.update((block,))
        until(lambda: bool(session.snapshot().results))
        original = session.snapshot().results
        session.update((), retain=(block,))
        assert session.snapshot().results == original
        assert session.snapshot().pending == 0
        assert len(backend.calls) == 1
        # Retention never accepts a changed source or an in-flight unready result.
        changed = TextBlock((track("別の文章", revision=2),))
        session.update((), retain=(changed,))
        assert not session.snapshot().results and session.snapshot().pending == 0
        session.update((changed,), retain=(changed,))
        until(lambda: bool(session.snapshot().results))
        assert session.snapshot().results[0].request.source == "別の文章"
    finally:
        stop(session)


def test_dialogue_translates_as_one_request_and_rejects_a_late_changed_member(tmp_path) -> None:
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    first = TextTrack(1, "新しい冒険が始まりますが、", Rect(100, 100, 400, 32))
    second = TextTrack(2, "その前に準備をしてください。", Rect(100, 144, 400, 32))
    grouper = DialogueGrouper()
    session.start()
    try:
        pending = grouper.update((first, second), (first,), 0)
        session.update(pending.ready)
        assert session.snapshot().pending == 0
        ready = grouper.update((first, second), (first, second), 0.4)
        session.update(ready.ready)
        assert backend.entered.wait(3)
        assert backend.calls == [(first.text + second.text,)]
        changed = replace(second, text="準備ができたら出発しましょう。", text_revision=2)
        new_block = TextBlock((first, changed))
        session.update((new_block,))
        backend.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        result = session.snapshot().results[0]
        assert result.request == TranslationRequest.from_block(new_block)
        assert backend.calls == [(first.text + second.text,), (first.text + changed.text,)]
        moved = TextBlock(
            tuple(replace(t, bounds=replace(t.bounds, x=200)) for t in new_block.lines)
        )
        session.update((moved,))
        assert session.snapshot().results == (result,)
        assert session.snapshot().pending == 0
        # Splitting a group also rejects its reply even though its anchor line still exists.
        session.update((first,))
        assert session.snapshot().results == ()
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].request == TranslationRequest.from_track(first)
    finally:
        backend.release.set()
        stop(session)


def test_revision_change_and_removed_tracks_reject_inflight_translation(tmp_path) -> None:
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track("古い文章"), track("消えた文章", key=2)))
        assert backend.entered.wait(3)
        session.update((track("新しい文章", revision=2),))
        assert session.snapshot().results == ()
        backend.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        result = session.snapshot().results[0]
        assert result.request == TranslationRequest.from_track(track("新しい文章", revision=2))
        assert result.english.endswith("新しい文章")
        session.update(())
        assert session.snapshot().results == ()
    finally:
        backend.release.set()
        stop(session)


def test_invalidation_prevents_same_key_from_accepting_old_generation(tmp_path) -> None:
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track(),))
        assert backend.entered.wait(3)
        session.invalidate()
        backend.release.set()
        # Stop waits for in-flight work; an invalidated result must not survive cleanup.
        stop(session)
        assert session.snapshot().results == ()
    finally:
        backend.release.set()
        stop(session)


def test_backend_swap_closes_old_before_loading_new_and_uses_separate_cache(tmp_path) -> None:
    old = Backend("old")
    old.release.clear()
    new = Backend("new")
    loaded = threading.Event()

    def load_new():
        assert old.closed
        loaded.set()
        return new

    session = TranslationSession(lambda: old, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track(),))
        assert old.entered.wait(3)
        session.switch_backend(load_new)
        assert not loaded.is_set()
        assert session.snapshot().results == ()
        old.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].english.startswith("new:")
        assert new.calls == [("日本語",)]
    finally:
        old.release.set()
        stop(session)
    assert new.closed


def test_backend_switch_keeps_demand_but_reports_only_the_new_backends_work(tmp_path) -> None:
    old, new = Backend("old"), Backend("new")
    new.release.clear()
    session = TranslationSession(lambda: old, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update((track(),))
        until(lambda: len(session.snapshot().results) == 1)
        before = session.snapshot()
        assert before.translated_sources == 1 and before.last_batch_ms > 0
        session.switch_backend(lambda: new)
        switched = session.snapshot()
        # Results, timings and counts belong to the old backend; the request itself survives.
        assert switched.results == () and switched.pending == 1
        assert switched.translated_sources == switched.cache_hits == 0
        assert switched.last_batch_ms == switched.queue_ms == switched.last_inference_ms == 0
        assert switched.load_ms == 0 and switched.cancelled_obsolete == 0
        assert new.entered.wait(3)
        new.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].english == "new: 日本語"
        assert session.snapshot().translated_sources == 1
    finally:
        new.release.set()
        stop(session)


def test_stop_releases_every_request_and_answer(tmp_path) -> None:
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    session.update((track(key=1),), candidates=(TextBlock((track("候補", key=2),)),))
    until(lambda: len(session.snapshot().results) == 1)
    session.request_stop()
    stopped = session.snapshot()
    assert stopped.results == () and stopped.pending == 0
    assert stopped.rejected == () and stopped.cached_candidates == ()
    until(lambda: session.finished)
    assert backend.closed


def test_switching_game_credits_cached_labels_without_another_model_call(tmp_path):
    path = tmp_path / "cache.sqlite3"
    backend = Backend()
    worker = TranslationSession(lambda: backend, path)
    worker.start()
    try:
        for game in ("First", "Second"):
            worker.update((track("設定"),), game=game)
            until(lambda: worker.snapshot().results)
        assert worker.snapshot().cache_hits == 1
        assert len(backend.calls) == 1
    finally:
        stop(worker)
    cache = TranslationCache(path)
    try:
        assert cache.inventory().games == (("First", 1), ("Second", 1))
    finally:
        cache.close()


def test_cache_reuses_text_after_session_restart_and_track_id_changes(tmp_path) -> None:
    path = tmp_path / "cache.sqlite3"
    for key in (1, 99):
        backend = Backend()
        session = TranslationSession(lambda current=backend: current, path)
        session.start()
        try:
            session.update((track(key=key),))
            until(lambda current=session: len(current.snapshot().results) == 1)
            assert session.snapshot().results[0].request.track_id == key
            assert len(backend.calls) == (1 if key == 1 else 0)
        finally:
            stop(session)


def test_shutdown_is_nonblocking_and_rejects_late_output(tmp_path) -> None:
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    session.update((track(),))
    assert backend.entered.wait(3)
    session.request_stop()
    assert not session.finished
    assert session.snapshot().state == TranslationState.STOPPING
    backend.release.set()
    until(lambda: session.finished)
    assert backend.closed
    assert session.snapshot().results == ()


def test_load_failure_is_visible_and_capture_can_continue_independently(tmp_path) -> None:
    def fail():
        raise RuntimeError("Missing model")

    session = TranslationSession(fail, tmp_path / "cache.sqlite3")
    session.start()
    until(lambda: session.finished)
    stop(session)
    assert session.snapshot().state == TranslationState.FAILED
    assert session.snapshot().message == "Missing model"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_label_capability_failure_closes_loaded_backend(tmp_path, cleanup_fails):
    class BrokenLabels(Backend):
        @property
        def label_width(self):
            raise RuntimeError("Cannot inspect menu support")

        def prepare_labels(self, target):
            raise AssertionError("No decoder should be created")

        def close(self):
            super().close()
            if cleanup_fails:
                raise RuntimeError("Cleanup failed")

    backend = BrokenLabels()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        until(lambda: session.finished)
        assert backend.closed
        assert session.snapshot().state == TranslationState.FAILED
        assert session.snapshot().message == "Cannot inspect menu support"
    finally:
        stop(session)


def test_desired_work_and_batch_size_are_bounded(tmp_path) -> None:
    backend = Backend()
    backend.release.clear()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        session.update(tuple(track(f"日本語{i}", key=i) for i in range(200)))
        assert backend.entered.wait(3)
        assert session.snapshot().pending == 128
        assert len(backend.calls[0]) <= 4
        session.update((track("最新の文章", key=500),))
        backend.release.set()
        until(lambda: len(session.snapshot().results) == 1)
        assert backend.calls[-1] == ("最新の文章",)
        assert len(backend.calls) == 2
    finally:
        backend.release.set()
        stop(session)
