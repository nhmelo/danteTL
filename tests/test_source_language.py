import threading

import pytest
from live_fixtures import Replay, frame
from translation_fixtures import Backend, until

from dantetl.domain import Rect
from dantetl.languages.paddle import PaddleOCRLanguage
from dantetl.languages.selection import DEFAULT_SOURCE, source_language
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.ocr.observation import CandidateReview, TextTrack
from dantetl.ocr.tracking import TextTracker, prepare_matching
from dantetl.ocr.values import OcrResult, TextRegion
from dantetl.ocr.visual import VisualSession
from dantetl.translation.backend import RequestRejected
from dantetl.translation.backends import gemma, gguf, plamo
from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.cloud.protocol import request_body
from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.dialogue.layout import PanelLayout
from dantetl.translation.grouping import DialogueGrouper, TextBlock
from dantetl.translation.worker import TranslationSession

PORTUGUESE = source_language("pt")


def grouped_rows(language, *texts):
    rows = tuple(
        TextTrack(i + 1, text, Rect(100, 100 + 45 * i, 380, 35)) for i, text in enumerate(texts)
    )
    grouped = DialogueGrouper(language=language).update(rows, rows, 0)
    return [[row.text for row in block.lines] for block in grouped.blocks]


def test_new_quoted_speech_starts_a_block_without_ocr_dash_recovery():
    first, second = "Chegou, viu a casa.", "Ficou para jantar."
    assert grouped_rows(PORTUGUESE, first, second, "Vai para casa.") == [
        [first, second, "Vai para casa."]
    ]
    assert grouped_rows(PORTUGUESE, first, second, "“Vai para casa.”") == [
        [first, second],
        ["“Vai para casa.”"],
    ]


@pytest.mark.parametrize("middle_height", [41, 49])
def test_inline_quoted_name_continues_an_established_column(middle_height):
    texts = (
        "The energy that can be stored",
        "while travelling through the",
        "valley with the blessing",
        '"Light of the Lake" increases',
        "to two hundred points.",
    )
    boxes = (
        (100, 100, 690, 49),
        (101, 165, 735, 49),
        (100, 232, 670, middle_height),
        (114, 290, 758, 56),
        (100, 357, 520, 49),
    )
    rows = tuple(
        TextTrack(i, t, Rect(*b)) for i, (t, b) in enumerate(zip(texts, boxes, strict=True))
    )
    language = source_language("en")
    grouper = DialogueGrouper(language=language)
    assert not grouper.update(rows, rows[:-1], 0).ready
    (block,) = grouper.update(rows[::-1], rows, 1).ready
    assert block.lines == rows and block.text == " ".join(texts)


@pytest.mark.parametrize("ending", ['"', "'", "”"])
@pytest.mark.parametrize(
    "first_box,tail_box",
    [
        ((3064, 1950, 616, 35), (3331, 1984, 74, 26)),
        ((3060, 1949, 624, 36), (3325, 1978, 90, 38)),
    ],
)
def test_short_centered_tail_completes_an_inline_quoted_name(ending, first_box, tail_box):
    opening = "“" if ending == "”" else '"'
    rows = (
        TextTrack(
            1,
            "Requires activation of " + opening + "The Dreamed Utopian",
            Rect(*first_box),
        ),
        TextTrack(2, "Dawn" + ending, Rect(*tail_box)),
        TextTrack(3, "Return", Rect(3323, 2020, 94, 30)),
    )
    grouper = DialogueGrouper(language=source_language("en"))
    groups = grouper.update(rows, rows, 0)
    assert [b.lines for b in groups.blocks] == [rows[:2], rows[2:]]
    assert grouper.update(rows, rows, 1).ready == groups.blocks
    assert len(grouper.update(rows, rows, 2, enabled=False).blocks) == 3


def test_percentage_label_needs_established_prose_and_text_after_the_value():
    class EffectLanguage(PaddleOCRLanguage):
        percentage_label_terms = ("bônus",)

    language = EffectLanguage("pt", "Portuguese", "Mapa")
    lead, tail = "Efeito: Quando chega", "bônus:25%. Outra frase."
    assert grouped_rows(language, lead, tail) == [[lead, tail]]
    for first, second in [
        ("Quando chega aqui", tail),
        (lead, "bônus:25%."),
        (lead, "•bônus:25%. Outra frase."),
    ]:
        assert grouped_rows(language, first, second) == [[first], [second]]
    assert grouped_rows(PORTUGUESE, lead, tail) == [[lead], [tail]]


def test_paddle_quote_cues_are_opt_in():
    # Use letters accepted by all three policies so script filtering cannot mask a leak.
    for language, expected in (
        (PORTUGUESE, True),
        (DEFAULT_SOURCE, False),
        (SourceLanguage("test", "Test", "設定"), False),
    ):
        assert language.quoted_term_with_suffix('"光" 明るい') is expected
        assert language.closes_wrapped_quote('必要な "光', "星'") is expected


@pytest.mark.parametrize(
    ("marker", "generic", "japanese"),
    [
        ("A)", True, True),
        ("•", True, True),
        ("-", False, False),
        ("・", False, True),
        ("１、", False, True),
    ],
)
def test_panel_uses_the_selected_list_notation(marker, generic, japanese):
    for language, expected in [(PORTUGUESE, generic), (DEFAULT_SOURCE, japanese)]:
        row = TextTrack(1, marker + " Item", Rect(100, 100, 180, 35))
        block = TextBlock((row,), language=language)
        layout = PanelLayout()
        layout.update((block,))
        assert layout.marked_choice(block) is expected


def test_generic_language_groups_and_publishes_through_the_live_pipeline(tmp_path):
    replay = Replay(language=PORTUGUESE)
    rows = (
        TextRegion("Mapa", Rect(50, 40, 100, 30)),
        TextRegion("O caminho leva até", Rect(100, 650, 380, 35)),
        TextRegion("à praça da vila.", Rect(100, 695, 360, 35)),
    )
    replay.observe(frame(1), rows)
    scene = replay.observe(frame(2, 1.5), rows)
    phrase = "O caminho leva até à praça da vila."
    assert [b.text for b in scene.requests] == ["Mapa", phrase]
    assert [line.text for line in scene.requests[1].lines] == [r.text for r in rows[1:]]

    backend = Backend()
    path = tmp_path / "cache.sqlite3"
    cache = TranslationCache(path)
    cache.put(translation_key(backend.cache_identity, "Mapa"), "Wrong language")
    cache.close()
    session = TranslationSession(lambda: backend, path, language=PORTUGUESE)
    session.start()
    try:
        replay.live.publish(session, scene.requests, scene.blocks)
        until(lambda: len(session.snapshot().results) == 2)
        english = replay.live.publish(session, scene.requests, scene.blocks)
        assert english.blocks == scene.blocks
        assert [t.english for t in english.translations] == [
            f"{backend.cache_identity}: Mapa",
            f"{backend.cache_identity}: {phrase}",
        ]
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_japanese_remains_selected_and_rejects_latin_only_ocr(tmp_path):
    assert source_language("ja") is DEFAULT_SOURCE
    with pytest.raises(KeyError):
        source_language("unregistered")
    regions = (
        TextRegion("Mapa", Rect(50, 40, 100, 30)),
        TextRegion("マップ", Rect(50, 300, 100, 30)),
    )
    replay = Replay()
    replay.observe(frame(1), regions)
    scene = replay.observe(frame(2, 1.5), regions)
    assert len(scene.blocks) == 2
    backend = Backend()
    session = TranslationSession(lambda: backend, tmp_path / "cache.sqlite3")
    session.start()
    try:
        replay.live.publish(session, scene.requests, scene.blocks)
        until(lambda: session.snapshot().results)
        english = replay.live.publish(session, scene.requests, scene.blocks)
        assert [b.text for b in english.blocks] == ["マップ"]
        assert backend.calls == [("マップ",)]
    finally:
        session.request_stop()
        until(lambda: session.finished)


@pytest.mark.parametrize("lower_length, expected_blocks", [(255, 1), (256, 2)])
def test_shared_request_budget_counts_the_language_specific_word_space(
    lower_length, expected_blocks
):
    lines = (
        TextTrack(1, "a" * 255 + "é", Rect(100, 650, 380, 35)),
        TextTrack(2, "é" * (lower_length - 1) + ".", Rect(100, 695, 360, 35)),
    )
    grouped = DialogueGrouper(language=PORTUGUESE).update(lines, lines, 0)
    assert len(grouped.blocks) == expected_blocks
    assert all(len(block.text) <= 512 for block in grouped.blocks)


def test_matching_evidence_is_scoped_to_its_source_language():
    source = frame(1)
    result = OcrResult(source, (TextRegion("por que", Rect(10, 10, 100, 25)),), 25, 1.0)
    tracker = TextTracker(language=PORTUGUESE)
    with pytest.raises(ValueError, match="evidence"):
        tracker.observe(result, review=CandidateReview(matching=prepare_matching(result, ())))
    assert tracker.observe(
        result, review=CandidateReview(matching=prepare_matching(result, (), language=PORTUGUESE))
    ) == (1,)


def test_matching_worker_reuses_pending_source_until_language_invalidation():
    result = OcrResult(frame(1), (TextRegion("Mapa", Rect(10, 10, 100, 25)),), 25, 1.0)
    published = threading.Event()
    worker = VisualSession(language=PORTUGUESE, on_change=published.set)

    def review(sequence):
        published.clear()
        worker.submit(frame(sequence), None, (), (), (), candidate=result)
        assert published.wait(3)
        evidence = worker.poll().review.matching
        assert evidence is not None
        return evidence

    worker.start()
    try:
        first = review(1)
        assert first.language == PORTUGUESE
        assert review(2) is first
        worker.invalidate(language=DEFAULT_SOURCE)
        assert review(2).language == DEFAULT_SOURCE
    finally:
        worker.request_stop()
        until(lambda: worker.finished)


def test_backends_use_the_selected_language_in_their_native_prompt():
    language = PORTUGUESE
    assert "Translate Portuguese video game text" in gemma.translation_prompt(
        "Mapa", language=language
    )
    assert "Translate Portuguese video game text" in gguf.user_prompt("Mapa", (), language=language)
    with pytest.raises(RequestRejected, match="Portuguese as a source"):
        plamo.translation_prompt("Mapa", (), language=language)
    prompt = plamo.translation_prompt("Mapa", (), language=source_language("es"))
    assert "<|plamo:val|>es<|plamo:msg|>Mapa" in prompt
    assert prompt.endswith("<|plamo:val|>en<|plamo:msg|>")
    profile = CloudProfile.new(CloudProvider.CLAUDE)
    body = request_body(profile, "", ("Mapa", ()), language=language)
    assert "Translate CURRENT from Portuguese" in str(body["system"])


@pytest.mark.parametrize("target", TARGET_LANGUAGES.values(), ids=TARGET_LANGUAGES.keys())
def test_backends_request_the_same_output_language(target):
    from dantetl.translation.dialogue.context import DialogueTurn

    history = (DialogueTurn("先に行って。", "…"),)
    for prompt in (
        gemma.translation_prompt("設定", history, target=target),
        gguf.user_prompt("設定", history, target=target),
        str(
            request_body(
                CloudProfile.new(CloudProvider.CLAUDE), "", ("設定", history), target=target
            )["system"]
        ),
    ):
        assert target.name in prompt
        if target.code != "en":
            assert "English" not in prompt
    if target.code in ("pt-PT", "pt-BR"):
        with pytest.raises(RequestRejected, match="PLaMo does not support Portuguese"):
            plamo.translation_prompt("設定", history, target=target)
        return
    output = f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target.code}<|plamo:msg|>"
    prompt = plamo.translation_prompt("設定", history, target=target)
    assert prompt.count(output) == 2 and prompt.endswith(output)


def test_every_published_latin_source_shares_one_install_and_ocr_factory(monkeypatch):
    from unittest.mock import Mock

    from dantetl.application.model_factories import ocr_backends
    from dantetl.languages.selection import SOURCE_LANGUAGES
    from dantetl.ocr import models, paddle
    from dantetl.runtime.model_store import model_catalog

    # The publisher's Latin table includes English, Occitan, Māori and Latin itself.
    published_codes = (
        "en fr de af it es bs pt cs cy da et ga hr uz hu sr-Latn id oc is lt mi ms nl no pl "
        "sk sl sq sv sw tl tr la"
    )
    published = set(published_codes.split())
    assert set(SOURCE_LANGUAGES) == published | {"ja"}
    create = Mock()
    monkeypatch.setattr(paddle, "PaddleRecognizer", create)
    monkeypatch.setattr(
        "dantetl.application.model_factories.gpu_runtime_installed",
        lambda name="onnxruntime-gpu": name == "onnxruntime-gpu",
    )
    for code in published:
        source = SOURCE_LANGUAGES[code]
        assert isinstance(source, PaddleOCRLanguage)
        (bundle,) = [b for b in model_catalog(source) if b.category == "recognition"]
        assert (
            bundle.assets == models.PADDLE_ASSETS and bundle.directory == models.paddle_directory()
        )
        assert bundle.bundled_directory is None
        gpu, cpu = ocr_backends(source)
        assert gpu.key == "auto" and cpu.key == "cpu"
        gpu.create()
        create.assert_called_with(bundle.directory, device="auto")
        assert all(a.url.endswith("/inference.onnx") for a in bundle.assets)
    (jp,) = [b for b in model_catalog(SOURCE_LANGUAGES["ja"]) if b.category == "recognition"]
    assert jp.assets == models.ASSETS


def test_source_change_applies_the_catalog_to_setup_ocr_and_discovery(tmp_path):
    from unittest.mock import Mock

    from ownership_fixtures import Replay

    from dantetl.application.model_inventory import InventoryData, SourceModels
    from dantetl.ocr.worker import OcrChoice
    from dantetl.runtime.model_store import ModelBundle

    bundle = ModelBundle("reader", "Local reader", tmp_path / "reader", (), "")
    discover = Mock(return_value=InventoryData(device="fixture"))
    models = SourceModels((bundle,), (OcrChoice("Test reader", Mock(), key="reader"),), discover)
    replay = Replay(tmp_path, source_models=lambda _: models)
    try:
        assert replay.app.change_source("pt")
        assert not replay.inventory.snapshot().loaded
        replay.settle_inventory()
        replay.app.advance(2)
        assert replay.inventory.snapshot().data.device == "fixture"
        assert tuple(option.key for option in replay.app.ocr_options) == ("reader",)
        assert replay.setup.download(bundle.key)
        discover.assert_called_once()
    finally:
        replay.close()


def test_source_change_drains_active_capture_and_keeps_output_language(tmp_path):

    from ownership_fixtures import Replay

    from dantetl.application.preference_values import Preferences
    from dantetl.application.preferences import PreferenceStore

    replay = Replay(tmp_path, preferences=Preferences(output_language="fr"))
    assert replay.app.start_capture(replay.capture.target, 1)
    assert replay.app.change_source("pt")
    assert not replay.app.closing and not replay.app.change_source("de")
    replay.app.configure(opacity=60)
    finish_source_change(replay)
    replay.close()
    saved = PreferenceStore(tmp_path / "preferences.json").initial
    assert (saved.source_language, saved.output_language) == ("pt", "fr")


def finish_source_change(replay):
    replay.capture.finished = replay.visual.finished = True
    replay.ocr.finished = replay.ocr.stop_requested

    def ready():
        replay.app.advance(2)
        assert not replay.app.closing
        return not replay.app.controls.changing_source

    until(ready)


@pytest.mark.parametrize(
    "source,device,reuse", [("en", "auto", True), ("en", "cpu", False), ("ja", "auto", False)]
)
def test_source_switch_resumes_exact_game_and_reuses_only_compatible_ocr(
    tmp_path, source, device, reuse
):
    from unittest.mock import Mock

    from ownership_fixtures import Replay

    from dantetl.application.preference_values import Preferences
    from dantetl.ocr.worker import OcrChoice

    devices = tuple(OcrChoice(key, Mock(), key=key) for key in ("auto", "cpu"))
    replay = Replay(tmp_path, preferences=Preferences(source_language=source), ocr_choices=devices)
    target = replay.capture.target
    assert replay.app.start_capture(target, 1)
    original = replay.ocr
    replay.app.configure(ocr_device=device)
    assert replay.app.source_requires_restart("pt") is (not reuse)
    assert replay.app.change_source("pt")
    assert original.stop_requested != reuse
    assert not replay.app.start_capture(target, 2)
    replay.app.advance(2)
    if reuse:
        replay.settle_inventory()
        replay.app.advance(2)
        assert not replay.app.controls.changing_source
        assert replay.app.controls.started_at == 1 and not replay.capture.stop_requested
    else:
        assert replay.app.controls.changing_source and len(replay.capture_calls) == 1
        finish_source_change(replay)
    assert [call[0] for call in replay.capture_calls] == [target] * (1 if reuse else 2)
    assert replay.live.ocr is (original if reuse else replay.ocr)
    assert len(replay.recognizers) == (1 if reuse else 2)
    assert not replay.app.view.results and not replay.app.history_turns
    replay.app.stop()
    assert (original if reuse else replay.ocr).stop_requested
    original.finished = True
    replay.close()


@pytest.mark.parametrize("action", ["already-stopping", "stop", "closed-target", "quit"])
def test_source_restart_respects_stops_closed_target_and_quit(tmp_path, action):
    from ownership_fixtures import Replay

    from dantetl.application.preference_values import Preferences
    from dantetl.application.values import ShutdownPhase

    replay = Replay(tmp_path, preferences=Preferences(source_language="en"))
    assert replay.app.start_capture(replay.capture.target, 1)
    original = replay.ocr
    if action == "already-stopping":
        replay.app.stop()
    assert replay.app.change_source("ja")
    if action == "quit":
        replay.close()
        replay.app.advance(2)
        assert replay.app.close_status.phase == ShutdownPhase.READY
    else:
        if action == "stop":
            replay.app.stop()
        if action == "closed-target":
            replay.windows.is_target.return_value = False
        finish_source_change(replay)
        if action == "closed-target":
            assert "closed" in replay.app.view.status
            assert replay.live.ocr is None
        replay.close()
    assert len(replay.capture_calls) == 1 and original.stop_requested
