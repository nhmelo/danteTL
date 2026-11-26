"""Exercise capture -> local OCR -> inspector against our own live Windows fixture."""

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import shiboken6
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent
from PySide6.QtWidgets import QApplication, QWidget
from verification_isolation import app_owner
from verification_isolation import create_window as MainWindow
from verify_translation import process_memory

from dantetl.application.model_discovery import translation_backends
from dantetl.application.model_factories import create_recognizer
from dantetl.capture.windows import Windows
from dantetl.domain import CapturedFrame, WindowTarget, sample_box
from dantetl.ocr.values import OcrState, TextRegion
from dantetl.ocr.worker import OcrChoice
from dantetl.translation.values import TranslationState
from dantetl.ui.demo import DIALOGUE_LINES, DemoWindow


class EquipmentMenu(QWidget):
    """A second populated screen with strong edges throughout the old dialogue area."""

    similar_rows = False

    def show_similar_rows(self) -> None:
        self.similar_rows = True
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#354254"))
        painter.setPen(QColor("#81909d"))
        for x in range(-self.height(), self.width(), 7):
            painter.drawLine(x, 0, x + self.height(), self.height())
        painter.fillRect(0, 0, 16, 16, QColor("#624386"))
        font = QFont("Yu Gothic UI")
        font.setPixelSize(24)
        painter.setFont(font)
        for y, text in ((95, "装備一覧"), (165, "設定"), (235, "地図")):
            painter.fillRect(45, y - 32, 250, 42, QColor("#263447"))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(60, y, text)
        if self.similar_rows:
            box = sample_box(self.width(), self.height())
            painter.fillRect(box.x, box.y - 60, box.width, box.height + 60, QColor("#29465e"))
            font.setPixelSize(20)
            painter.setFont(font)
            for i, text in enumerate((DIALOGUE_LINES[0], "その前に装備を確認してください。")):
                painter.drawText(box.x + 24, box.y - 6 + 30 * i, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save a screenshot of only our test app")
    parser.add_argument(
        "--translation", action="store_true", help="Include real Gemma and model swapping"
    )
    parser.add_argument("--dialogue", action="store_true", help="Verify two-line dialogue grouping")
    parser.add_argument("--ocr-delay-ms", type=int, default=0, help="Simulate a slow OCR worker")
    parser.add_argument(
        "--device", choices=("cpu", "auto", "cuda", "directml", "auto-directml"), default="cpu"
    )
    parser.add_argument(
        "--gemma", action="store_true", help="Start Gemma, swap to another Gemma model and back"
    )
    parser.add_argument(
        "--classifier",
        choices=("gemma", "gemma-e4b", "qwen35-4b"),
        help="Enable a separate prepared local history classifier",
    )
    parser.add_argument(
        "--model",
        choices=("gemma-e4b", "qwen35-4b", "plamo-2b", "plamo-0.8b"),
        help="Start a curated model, swap to another Gemma model and back",
    )
    parser.add_argument("--debug", action="store_true", help="Verify diagnostic overlay rendering")
    parser.add_argument("--transitions", action="store_true", help="Close/reopen the text panel")
    parser.add_argument(
        "--populated-transition",
        action="store_true",
        help="Switch to a textured equipment menu and wait for its translation",
    )
    parser.add_argument(
        "--typewriter", action="store_true", help="Check gradual dialogue readiness"
    )
    args = parser.parse_args()
    selected_model = args.model or ("gemma" if args.gemma else "")
    selected_model = selected_model or "gemma"
    alternate_model = "gemma" if selected_model != "gemma" else "gemma-e4b"
    if (args.gemma or args.model) and not args.translation:
        parser.error("--gemma/--model requires --translation")
    if args.dialogue and not args.translation:
        parser.error("--dialogue requires --translation")
    if args.typewriter and not args.dialogue:
        parser.error("--typewriter requires --dialogue")
    if args.transitions and not args.dialogue:
        parser.error("--transitions requires --dialogue")
    if args.populated_transition and not args.transitions:
        parser.error("--populated-transition requires --transitions")
    if not 0 <= args.ocr_delay_ms <= 1500:
        parser.error("--ocr-delay-ms must be between 0 and 1500")

    class DelayedRecognizer:
        def __init__(self) -> None:
            self.engine = create_recognizer(device=args.device)

        @property
        def scan_costs(self):
            return getattr(self.engine, "scan_costs", None)

        def recognize(
            self, frame: CapturedFrame, *, full_size: tuple[int, int] | None = None
        ) -> tuple[TextRegion, ...]:
            time.sleep(args.ocr_delay_ms / 1000)
            return self.engine.recognize(frame, full_size=full_size)

        def close(self) -> None:
            self.engine.close()

    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    demo = DemoWindow(dialogue=args.dialogue, typewriter=args.typewriter)
    next_menu = EquipmentMenu(demo) if args.populated_transition else None
    if next_menu:
        next_menu.hide()
    if args.typewriter:
        demo.reveal_active = False  # Begin after model initialization/first inference.
    expected = DIALOGUE_LINES[0] if args.dialogue else "新しい冒険が、ここから始まる。"
    translation_source = "".join(DIALOGUE_LINES) if args.dialogue else expected
    demo.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    demo.show()
    factory = (
        DelayedRecognizer if args.ocr_delay_ms else partial(create_recognizer, device=args.device)
    )
    choices = (OcrChoice(args.device, factory, 8, "auto"),)
    if args.device != "cpu":
        choices += (OcrChoice("CPU", create_recognizer, 8, "cpu"),)
    cache_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / (
            f"typewriter-{time.time_ns()}.sqlite3"
            if args.typewriter
            else "verification-translation.sqlite3"
        )
    )
    cache_initially_empty = not cache_path.exists()
    window = MainWindow(
        windows,
        factory,
        # Pipeline fixture explicitly requests real model loading. Onboarding consent
        # is tested separately; never write a preparation receipt from this fixture.
        tuple(replace(b, preparation_id="") for b in translation_backends())
        if args.translation
        else (),
        cache_path=cache_path,
        ocr_backends=choices,
    )
    application = app_owner(window)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    if args.dialogue:
        classifier_key = "local:" + args.classifier if args.classifier else "layout"
        classifier_index = window.history_panel.model.findData(classifier_key)
        assert classifier_index >= 0, "Requested classifier is not installed"
        window.history_panel.model.setCurrentIndex(classifier_index)
        window._settings.history_enabled.setChecked(True)
    model_index = next(
        (
            i
            for i in range(window.backend_combo.count())
            if (b := window.backend_combo.itemData(i)) and b.preference_key == selected_model
        ),
        -1,
    )
    if args.translation:
        assert model_index >= 0, "Requested model is not downloaded"
        window.backend_combo.setCurrentIndex(model_index)
    window._settings.inspection_enabled.setChecked(True)
    window.debug_check.setChecked(args.debug)
    window.show()
    window.advanced.setChecked(True)
    if not args.translation:
        window.translate_check.setChecked(False)
        window.raw_boxes_check.setChecked(True)
    target = WindowTarget(int(demo.winId()), os.getpid(), demo.windowTitle())
    window.windows_combo.addItem("OCR verification fixture", target)
    window.windows_combo.setCurrentIndex(window.windows_combo.count() - 1)
    window.start_button.click()
    started = time.monotonic()
    stage = 0
    stage_started = started
    frozen = None
    initial_size = (0, 0)
    phrase_id = 0
    phrase_bounds = None
    swapped = 0
    grouping_stage = 0
    continuity_started = 0.0
    continuity_polls = 0
    continuity_request = None
    border_owners = None
    border_sequence = 0
    evidence: dict[str, object] = {"translation_cache_initially_empty": cache_initially_empty}
    failure = ""
    reveal_finished = 0.0
    transitions_done = False
    locked_pixels = None
    timeout = 180 if args.device != "cpu" or args.translation else 45
    hidden_since = 0.0
    seen_hidden = False
    rebound = False
    original_scene = window._overlay.set_scene

    def audit_scene(*args, **kwargs):
        nonlocal rebound, seen_hidden
        original_scene(*args, **kwargs)
        if stage in (10, 14):
            visible = any(
                t.request.source == translation_source for t in window._overlay._translations
            )
            rebound |= seen_hidden and visible
            seen_hidden |= not visible

    if args.transitions:
        window._overlay.set_scene = audit_scene

    def close_panel() -> None:
        demo.panel_visible = False
        demo.update()
        if next_menu:
            next_menu.setGeometry(demo.rect())
            next_menu.show()
            next_menu.raise_()

    def tick() -> None:
        nonlocal stage, stage_started, frozen, initial_size, failure, phrase_id, phrase_bounds
        nonlocal swapped, grouping_stage
        nonlocal continuity_started, continuity_polls, continuity_request
        nonlocal border_owners, border_sequence
        nonlocal reveal_finished
        nonlocal transitions_done, locked_pixels, hidden_since
        now = time.monotonic()
        try:
            if now - started > timeout:
                raise RuntimeError(f"Live OCR verification timed out at stage {stage}")
            if stage == 99:
                if application.shutdown.snapshot().phase == "ready":
                    window.close()
                    demo.close()
                    timer.stop()
                    app.quit()
                return
            if application.live.ocr and application.live.ocr.snapshot().state == OcrState.FAILED:
                raise RuntimeError(application.live.ocr.snapshot().message)
            if (
                application.translation.lifecycle.publication
                and application.translation.lifecycle.publication.snapshot().state
                == TranslationState.FAILED
            ):
                raise RuntimeError(application.translation.lifecycle.publication.snapshot().message)
            result = window.inspector.result
            if (
                args.typewriter
                and reveal_finished
                and not swapped
                and application.translation.lifecycle.publication
                and any(
                    t.request.source == translation_source for t in window._overlay._translations
                )
            ):
                evidence.setdefault(
                    "typewriter_to_full_english_seconds", round(now - reveal_finished, 3)
                )
            if stage == 0 and args.typewriter and result is not None:
                demo.reveal_active = True
                if demo.revealed < sum(map(len, DIALOGUE_LINES)):
                    if application.translation.lifecycle.publication:
                        for (
                            translated
                        ) in application.translation.lifecycle.publication.snapshot().results:
                            assert translated.request.source == "旅人", (
                                "Premature dialogue translation: " + translated.request.source
                            )
                    return
                if not reveal_finished:
                    reveal_finished = now
                if not all(line in [r.text for r in result.regions] for line in DIALOGUE_LINES):
                    return
            if stage == 8 and phrase_id and application.live.capture:
                latest = application.live.capture.snapshot().frame
                if latest:
                    current = application.live.tracking.visible
                    assert any(t.track_id == phrase_id for t in current), (
                        "Track blinked between OCR passes"
                    )
                if continuity_request is not None:
                    assert application.translation.lifecycle.publication is not None
                    assert any(
                        t.request == continuity_request
                        for t in application.translation.lifecycle.publication.snapshot().results
                    ), "English disappeared between slow OCR passes"
                    assert any(
                        t.request == continuity_request for t in window._overlay._translations
                    ), "Overlay discarded an unchanged translation"
            if stage == 0 and result is not None:
                text = [r.text for r in result.regions]
                assert expected in text, text
                if args.dialogue:
                    assert DIALOGUE_LINES[1] in text, text
                    assert any(t.text == "旅人" for t in application.live.tracking.visible), (
                        "Unscanned speaker disappeared"
                    )
                assert window.inspector.table.rowCount() == len(result.regions)
                evidence["live_japanese_recognized"] = True
                evidence["ocr_ms"] = round(result.duration_ms, 1)
                evidence["dpi"] = result.frame.geometry.dpi
                if args.device in ("cuda", "directml") and not args.ocr_delay_ms:
                    provider = (
                        "CUDAExecutionProvider" if args.device == "cuda" else "DmlExecutionProvider"
                    )
                    assert (
                        application.live.ocr
                        and application.live.ocr.snapshot().provider == provider
                    )
                    evidence["actual_gpu_provider"] = application.live.ocr.snapshot().provider
                initial_size = result.frame.geometry.client.size
                frozen = result
                tracks = application.live.tracking.visible
                phrase = next((t for t in tracks if t.text == expected), None)
                if phrase is None:
                    return  # Raw OCR can complete before its source-matched track commits.
                phrase_id, phrase_bounds = phrase.track_id, phrase.bounds
                stage = 8
            elif (
                stage == 8
                and result
                and frozen
                and result.frame.sequence >= frozen.frame.sequence + 3
            ):
                tracks = application.live.tracking.visible
                phrase = next(t for t in tracks if t.track_id == phrase_id)
                assert phrase.bounds == phrase_bounds, "Stationary dialogue box changed bounds"
                if args.translation:
                    if application.translation.lifecycle.publication is None:
                        return
                    translated = application.translation.lifecycle.publication.snapshot()
                    if not translated.results or translated.pending:
                        return
                    if not swapped:
                        if not any(
                            t.request.source == translation_source
                            for t in window._overlay._translations
                        ):
                            return  # A partial scan can finish before the full group publishes.
                        if args.typewriter:
                            assert "typewriter_to_full_english_seconds" in evidence
                            evidence["no_partial_dialogue_translations"] = True
                        evidence["first_backend_english"] = next(
                            t.english
                            for t in window._overlay._translations
                            if t.request.source == translation_source
                        )
                        window.backend_combo.setCurrentIndex(
                            next(
                                i
                                for i in range(window.backend_combo.count())
                                if (b := window.backend_combo.itemData(i))
                                and b.preference_key == alternate_model
                            )
                        )
                        swapped = 1
                        return
                    if swapped == 1:
                        evidence["alternate_model_between_loads"] = [
                            t.english for t in translated.results
                        ]
                        window.backend_combo.setCurrentIndex(model_index)
                        swapped = 2
                        return
                    if args.dialogue and grouping_stage == 1:
                        sources = {t.request.source for t in translated.results}
                        if not all(line in sources for line in DIALOGUE_LINES):
                            return  # A serial backend may finish the speaker before lines settle.
                        assert not any(t.request.members for t in translated.results), translated
                        window.group_dialogue.setChecked(True)
                        grouping_stage = 2
                        return
                    english = next(
                        (t for t in translated.results if t.request.source == translation_source),
                        None,
                    )
                    if english is None:
                        return  # Other blocks can finish while a newly regrouped passage settles.
                    if not any(t.request == english.request for t in window._overlay._translations):
                        return  # Publication occurs on the next UI poll after worker completion.
                    if args.ocr_delay_ms and (not args.dialogue or grouping_stage == 2):
                        if not continuity_started:
                            if not any(
                                t.request == english.request for t in window._overlay._translations
                            ):
                                return  # The worker can finish between two UI polls.
                            continuity_started = now
                            continuity_request = english.request
                            if window.border_check.isChecked():
                                capture = application.live.capture
                                assert capture and capture.snapshot().frame
                                frame = capture.snapshot().frame
                                assert frame is not None
                                border_sequence = frame.sequence
                                border_owners = (
                                    capture,
                                    application.live.ocr,
                                    application.translation.lifecycle.publication,
                                )
                                window.border_check.setChecked(False)
                        continuity_polls += 1
                        if now - continuity_started < 3:
                            return
                        evidence["slow_ocr_english_continuity_polls"] = continuity_polls
                        if border_owners:
                            assert border_owners == (
                                application.live.capture,
                                application.live.ocr,
                                application.translation.lifecycle.publication,
                            ), "Border change replaced a processing owner"
                            capture = application.live.capture
                            assert capture is not None
                            frame = capture.snapshot().frame
                            assert frame and frame.sequence > border_sequence
                            evidence["border_toggle_preserves_workers_and_english"] = True
                    assert "adventure" in english.english.lower()
                    evidence["english_after_backend_swap"] = english.english
                    if args.dialogue:
                        assert len(english.request.members) == 2
                        if grouping_stage == 0:
                            window.group_dialogue.setChecked(False)
                            grouping_stage = 1
                            return
                        assert any(t.request.source == "旅人" for t in translated.results)
                        blocks = window._overlay._blocks
                        assert blocks and sum(len(b.lines) == 2 for b in blocks) == 1
                        evidence["dialogue_translated_as_one_block_with_separate_speaker"] = True
                        evidence["grouping_off_on_rebuilds_current_requests"] = True
                        history = application.live.history.turns
                        if not history:
                            return  # Reader settling is independent of the earlier overlay result.
                        assert history[-1].source == english.request.source
                        assert history[-1].speaker in (None, "旅人")
                        assert window.start_button.property("tone") == "recording"
                        assert not window.start_button.icon().isNull()
                        evidence["history_records_complete_group_without_speaker_in_body"] = True
                        evidence["history_speaker"] = history[-1].speaker
                        evidence["history_speaker_english"] = history[-1].speaker_english
                        if history[-1].speaker:
                            name_result = next(
                                t for t in translated.results if t.request.source == "旅人"
                            )
                            assert history[-1].speaker_english == name_result.english.strip()
                        if "history_panel_independent" not in evidence:
                            assert len(history) == 1, (
                                "Grouping/model swaps duplicated the same turn"
                            )
                            before = application.translation.lifecycle.publication
                            window.history_button.click()
                            app.processEvents()
                            assert not window.history_panel.isModal()
                            assert english.english in window.history_panel.reader.toPlainText()
                            assert window.history_panel.reader.verticalScrollBar().value() == 0
                            if history[-1].speaker_english:
                                assert (
                                    f"({history[-1].speaker_english}) 旅人"
                                    in window.history_panel.reader.toPlainText()
                                )
                            if args.output:
                                window.history_panel.grab().save(
                                    str(args.output.with_name("dialogue-history.png"))
                                )
                                window.grab().save(
                                    str(args.output.with_name("capture-running.png"))
                                )
                            window.history_panel.hide()
                            assert application.translation.lifecycle.publication is before
                            evidence["history_panel_independent"] = True
                            demo.raise_()
                            demo.activateWindow()
                    evidence["process_memory"] = process_memory()
                    with patch.object(windows, "is_foreground", return_value=True):
                        window._tick()
                    app.processEvents()  # Deliver the native resize before grabbing the widget.
                    if args.debug:
                        diagnostic = window._overlay._debug
                        assert diagnostic and diagnostic.scanned and "OCR" in diagnostic.text
                        evidence["debug_snapshot"] = diagnostic.text
                        before = window._overlay._translations
                        window.debug_check.setChecked(False)
                        assert window._overlay._debug is None
                        assert window._overlay._translations == before
                        window.debug_check.setChecked(True)
                        assert window._overlay._debug is not None
                        evidence["debug_toggle_preserves_english"] = True
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        window._overlay.grab().save(
                            str(args.output.with_name("english-overlay.png"))
                        )
                    window.automatic_font.setChecked(False)
                    window.font_size.setValue(24)
                    assert window._overlay._manual_size == 24
                    window.automatic_font.setChecked(True)
                    evidence["automatic_and_manual_font_controls"] = True
                    if args.transitions and not transitions_done:
                        if args.ocr_delay_ms >= 900 and not args.debug:
                            b = phrase_bounds
                            assert b is not None
                            locked_pixels = (
                                window._overlay.grab().toImage().copy(b.x, b.y, b.width, b.height)
                            )
                            assert locked_pixels.pixelColor(b.width // 2, b.height // 2).alpha() > 0
                        if args.ocr_delay_ms >= 900:
                            demo.reveal_active = False
                            demo.revealed = 0  # Temporary lettering loss on the original panel.
                        else:
                            close_panel()
                        demo.update()
                        continuity_request = None
                        stage = 12 if args.ocr_delay_ms >= 900 else 10
                        stage_started = time.monotonic()
                        return
                evidence["repeated_ocr_keeps_dialogue_id_and_bounds"] = True
                window.stabilize_check.setChecked(False)
                window.inspector.pause.setChecked(True)
                # Events drained for the grab may have advanced OCR.
                frozen = window.inspector.result
                picker = window._settings.interface_language
                owners = (
                    application.live.capture,
                    application.live.ocr,
                    application.translation.lifecycle.publication,
                )
                game = window.windows_combo.currentData()
                for code in ("pt-PT", "ja", "en"):
                    picker.setCurrentIndex(picker.findData(code))
                    assert (
                        application.live.capture,
                        application.live.ocr,
                        application.translation.lifecycle.publication,
                    ) == owners
                    assert window.inspector.pause.isChecked() and window.inspector.result is frozen
                    assert window.windows_combo.currentData() == game
                    assert (window.start_button.text() == "Stop capture") == (code == "en")
                evidence["live_interface_language_switch"] = True
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    window.grab().save(str(args.output))
                stage, stage_started = 1, now
            elif stage in (12, 13):
                assert any(
                    t.request.source == translation_source for t in window._overlay._translations
                ), "A brief loss of the panel pattern blinked the locked translation"
                if stage == 12:
                    assert now - stage_started < 2, "Transient loss never reached the visual worker"
                    if any(
                        t.track_id == phrase_id for t in application.view.visual.presence.absent
                    ):
                        evidence["transient_pattern_loss_kept_english"] = True
                        if locked_pixels is not None:
                            b = phrase_bounds
                            assert b is not None
                            assert (
                                window._overlay.grab().toImage().copy(b.x, b.y, b.width, b.height)
                                == locked_pixels
                            ), "Painted dialogue changed during visual loss"
                            evidence["transient_pattern_loss_kept_painted_pixels"] = True
                        demo.revealed = sum(map(len, DIALOGUE_LINES))
                        demo.update()
                        stage, stage_started = 13, now
                elif (
                    (accepted := application.live.accepted)
                    and accepted.frame.captured_at > stage_started
                    and expected in (r.text for r in accepted.regions)
                ):
                    close_panel()
                    stage, stage_started = 10, time.monotonic()
            elif stage == 10:
                assert application.capture.rate == 8
                old_visible = any(
                    t.request.source == translation_source for t in window._overlay._translations
                )
                visual_frame = application.view.visual.frame
                if next_menu and visual_frame and visual_frame.captured_at >= stage_started:
                    offset = 4 * (4 * visual_frame.width + 4)
                    if visual_frame.bgra[offset : offset + 3] == bytes.fromhex("864362"):
                        evidence.setdefault("old_visible_on_first_menu_analysis", old_visible)
                        evidence.setdefault(
                            "first_menu_capture_seconds",
                            round(visual_frame.captured_at - stage_started, 3),
                        )
                if old_visible:
                    assert now - stage_started < 4, "Closed panel stayed painted"
                    if visual_frame and visual_frame.captured_at >= stage_started:
                        b = phrase_bounds
                        assert b is not None
                        offset = 4 * (
                            (b.y + b.height // 2) * visual_frame.width + b.x + b.width // 2
                        )
                        assert visual_frame.bgra[offset : offset + 3] != bytes.fromhex("422d17"), (
                            "Dialogue stayed painted after visual analysis of the closed panel"
                        )
                    return
                evidence["panel_hide_seconds"] = round(now - stage_started, 3)
                assert now - stage_started < 0.4, "Closed panel missed the fast hide at 8 FPS"
                if next_menu:
                    assert evidence.get("old_visible_on_first_menu_analysis") is False
                accepted = application.live.accepted
                before_ocr = accepted is None or accepted.frame.captured_at < stage_started
                if before_ocr:
                    assert any(t.track_id == phrase_id for t in application.view.visual.cleared)
                evidence["panel_capture_fps"] = application.capture.rate
                evidence["hidden_before_completed_ocr"] = before_ocr
                if locked_pixels is not None:
                    b = phrase_bounds
                    assert b is not None
                    center = (
                        window._overlay.grab()
                        .toImage()
                        .pixelColor(b.x + b.width // 2, b.y + b.height // 2)
                    )
                    assert center.alpha() == 0, "Dialogue still painted over the closed panel"
                stage, hidden_since = 14, now
                if next_menu:
                    QTimer.singleShot(125, next_menu.show_similar_rows)
            elif stage == 14:
                assert now - stage_started < 15, "Replacement menu never translated"
                shown = {t.request.source for t in window._overlay._translations}
                assert not rebound and translation_source not in shown, (
                    "Dismissed dialogue returned"
                )
                if next_menu and "設定" in shown:
                    evidence.setdefault("new_menu_english_seconds", round(now - stage_started, 3))
                if now - hidden_since < 1 or (next_menu and "設定" not in shown):
                    return
                evidence["continuously_hidden_seconds"] = round(now - hidden_since, 3)
                if next_menu:
                    next_menu.hide()
                demo.panel_visible = True
                demo.update()
                stage, stage_started = 11, time.monotonic()
            elif stage == 11:
                if not any(
                    t.request.source == translation_source for t in window._overlay._translations
                ):
                    assert now - stage_started < 6, "Reopened panel never restored English"
                    return
                evidence["panel_restore_seconds"] = round(now - stage_started, 3)
                if locked_pixels is not None:
                    b = phrase_bounds
                    assert b is not None
                    assert (
                        window._overlay.grab().toImage().copy(b.x, b.y, b.width, b.height)
                        == locked_pixels
                    )
                    evidence["reopened_panel_restored_painted_pixels"] = True
                assert result is not None
                restored = next(t for t in application.live.tracking.visible if t.text == expected)
                evidence["closed_panel_identity_retired"] = restored.track_id != phrase_id
                phrase_id, phrase_bounds = restored.track_id, restored.bounds
                transitions_done = True
                stage = 8
            elif stage == 1 and now - stage_started > 0.7:
                assert window.inspector.result is frozen, "Pause replaced the reviewed frame"
                # The fixture deliberately does not steal foreground focus. Exercise
                # pause rendering with foreground eligibility controlled independently.
                with patch.object(windows, "is_foreground", return_value=True) as foreground:
                    window._tick()
                    assert window._overlay.isVisible(), "Inspector pause hid the live overlay"
                    workers = (
                        application.live.capture,
                        application.live.ocr,
                        application.live.visual,
                        application.translation.lifecycle.publication,
                    )
                    english = window._overlay._translations
                    window.unfocused_overlay_check.setChecked(True)
                    for focused in (False, True, False, True):
                        foreground.return_value = focused
                        window._tick()
                        assert window._overlay.isVisible()
                        assert window._overlay._translations == english
                        style, _ = windows.overlay_properties(int(window._overlay.winId()))
                        assert bool(style & 8) == focused, "Wrong placement after focus change"
                    window.unfocused_overlay_check.setChecked(False)
                    evidence["refocusing_restores_topmost_overlay"] = True
                    window.overlay_check.setChecked(False)
                    window._tick()
                    assert not window._overlay.isVisible()
                    window.overlay_check.setChecked(True)
                    window._tick()
                    assert window._overlay.isVisible()
                    assert window._overlay._translations == english
                    assert workers == (
                        application.live.capture,
                        application.live.ocr,
                        application.live.visual,
                        application.translation.lifecycle.publication,
                    ), "Toggling the overlay replaced processing workers"
                    evidence["overlay_toggle_restores_english_without_worker_restart"] = True
                assert application.live.capture is not None
                captured = application.live.capture.snapshot().frame
                assert captured and frozen and captured.sequence > frozen.frame.sequence
                evidence["pause_freezes_result_while_capture_continues"] = True
                evidence["pause_preserves_live_overlay"] = True
                window.inspector.pause.setChecked(False)
                window.stabilize_check.setChecked(True)
                evidence["raw_comparison_and_pause_toggle"] = True
                demo.resize(760, 500)
                stage = 2
            elif stage == 2 and result and result.frame.geometry.client.size != initial_size:
                assert result.frame.geometry.client.size == windows.geometry(target).client.size
                evidence["resize_produces_matching_ocr_frame"] = True
                demo.showMinimized()
                stage, stage_started = 3, now
            elif stage == 3 and now - stage_started > 0.3:
                assert window.inspector.result is None
                assert not window._overlay.isVisible()
                evidence["minimize_clears_ocr"] = True
                demo.showNormal()
                stage = 4
            elif stage == 4 and result:
                evidence["restore_resumes_ocr"] = True
                window.start_button.click()
                stage = 5
            elif stage == 5 and application.live.capture is None:
                assert window.start_button.property("tone") == "idle"
                assert application.live.ocr is None
                assert application.translation.lifecycle.publication is None
                assert application.live.visual is None
                assert window.inspector.result is None
                if args.device != "cpu":
                    window.ocr_device.setCurrentIndex(1)
                window.start_button.click()
                stage = 6
            elif stage == 6 and result:
                if args.device != "cpu":
                    assert (
                        application.live.ocr
                        and application.live.ocr.snapshot().provider == "CPUExecutionProvider"
                    )
                    evidence["cpu_selection_after_gpu_session"] = True
                evidence["restart_receives_new_result"] = True
                window.close()
                stage = 7
            elif stage == 7 and application.shutdown.snapshot().phase == "ready":
                assert application.live.ocr is None
                assert application.translation.lifecycle.publication is None
                assert application.live.visual is None
                assert not window.isVisible()
                evidence["close_finishes_all_workers"] = True
                evidence["elapsed_seconds"] = round(now - started, 2)
                stage = 99
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            evidence["failure"] = failure
            evidence["failure_state"] = {
                "stage": stage,
                "revealed": demo.revealed,
                "visual_source": application.live.visual_state.ocr_sequence,
                "visual_blocked": sorted(application.live.visual_state.blocked),
                "readings": [
                    {"text": a.observation.text, "pending": a.pending, "growing": a.growing}
                    for a in application.live.tracking.activity
                ],
            }
            traceback.print_exc()
            window.close()
            stage = 99
            print(f"FAIL: {failure!a}", flush=True)
            if now - started > timeout + 5:
                timer.stop()
                app.exit(1)

    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(tick)
    timer.start()
    app.exec()
    shiboken6.delete(app)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
    print(json.dumps(evidence, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    from verification_isolation import run_isolated

    run_isolated(main)
