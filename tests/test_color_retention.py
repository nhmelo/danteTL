"""Colored paragraphs over bright particles must survive hostile completed OCR."""

from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from live_fixtures import Replay as LiveReplay
from visual_fixtures import Replay

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState, TextCharacter, TextRegion
from dantetl.ocr.visual import VisualAnalyzer
from dantetl.translation.values import (
    TranslatedText,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)

SOURCE = "装備キャラの会心率が50%以上の場合、"
GREEN = (170, 240, 60, 255)
WHITE = (245, 245, 245, 255)


def paragraph(
    sequence,
    *,
    digit="5",
    digit_color=GREEN,
    score=0.98,
    extension=None,
    gone=False,
    rows=2,
    subpixel=False,
):
    # Known monospaced glyph cells make this a model-free pixel policy test.
    # Synthetic Japanese source metadata uses Latin shapes for nonnumeric glyphs.
    pixels = np.full((rows * 48 + 70, 760, 4), (35, 25, 20, 255), np.uint8)
    regions = []
    for row in range(rows):
        y = 20 + row * 48
        for x in (90 + sequence % 2 * 70, 20 + len(SOURCE) * 30 + 14):
            center = (x, y + 13 + sequence % 2 * 10)
            cv2.line(pixels, (x - 6, center[1]), (x + 6, center[1]), WHITE, 2)
            cv2.line(pixels, (x, center[1] - 5), (x, center[1] + 5), WHITE, 2)
        chars = []
        for index, char in enumerate(SOURCE):
            box = Rect(20 + index * 30, y, 30, 34)
            glyph = digit if char == "5" else char if char.isascii() else "H"
            color = digit_color if char == "5" else GREEN
            if not gone:
                mask = np.zeros((34, 30), np.uint8)
                cv2.putText(
                    mask, glyph, (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 255, 2, cv2.LINE_AA
                )
                for channel in range(3):
                    alpha = mask.astype(np.float32) / 255
                    if subpixel:
                        alpha = cv2.warpAffine(
                            alpha,
                            np.array([[1, 0, (channel - 1) / 3], [0, 1, 0]], dtype=np.float32),
                            (30, 34),
                        )
                    area = pixels[y : y + 34, box.x : box.x + 30, channel]
                    area[:] = np.rint(area * (1 - alpha) + color[channel] * alpha).astype(np.uint8)
            chars.append(TextCharacter(digit if char == "5" else char, box, score))
        if extension is not None:
            cv2.putText(
                pixels,
                "8",
                (20 + len(SOURCE) * 30 + 3, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                extension,
                2,
                cv2.LINE_AA,
            )
        text = SOURCE.replace("5", digit)
        regions.append(TextRegion(text, Rect(20, y, len(SOURCE) * 30, 34), tuple(chars)))
    viewport = Rect(0, 0, pixels.shape[1], pixels.shape[0])
    frame = CapturedFrame(
        sequence, sequence * 0.5, WindowGeometry(viewport, viewport, viewport, 96), pixels.tobytes()
    )
    return OcrResult(frame, tuple(regions), 25, frame.captured_at)


@pytest.mark.parametrize("subpixel", [False, True])
def test_every_publication_retains_colored_paragraph_through_particles_misses_and_partial_ocr(
    subpixel,
):
    replay, visual, backend = LiveReplay(), VisualAnalyzer(), Mock()
    scene, baseline, translated = None, None, ()
    for sequence in range(1, 30):
        incoming = paragraph(sequence, subpixel=subpixel)
        if sequence > 3:
            # Neither a different reading nor detector extent is proof of text change.
            rows = tuple(replace(r, text=r.text + "ュ", characters=()) for r in incoming.regions)
            if sequence % 4 == 0:
                rows = ()
            incoming = replace(
                incoming,
                regions=rows,
                scanned=(Rect(0, 0, 760, 60),) if sequence % 3 == 0 else None,
            )
        raw = incoming.regions
        source = incoming.frame
        activity = replay.visual.submitted[4] if scene else ()
        replay.visual.current = visual.observe(
            source,
            replay.live.accepted,
            scene.tracks if scene else (),
            replay.live.assignments,
            activity,
            candidate=incoming,
        )
        replay.ocr.current = OcrSnapshot(OcrState.READY, incoming)
        replay.live.reconcile(source, source.captured_at)
        scene = replay.live.scene(source, source.captured_at + 0.6, grouping=True)
        replay.visual.current = visual.observe(
            source,
            replay.live.accepted,
            scene.tracks,
            replay.live.assignments,
            replay.visual.submitted[4],
        )
        replay.live.reconcile(source, source.captured_at + 0.6)
        scene = replay.live.scene(source, source.captured_at + 0.6, grouping=True)
        if sequence == 2:
            assert len(scene.requests) == 1 and len(scene.requests[0].lines) == 2
            translated = (
                TranslatedText(TranslationRequest.from_block(scene.requests[0]), "Effect"),
            )
        backend.snapshot.return_value = TranslationSnapshot(TranslationState.READY, translated)
        english = replay.live.publish(backend, scene.requests, scene.blocks)
        if sequence == 2:
            baseline = english.blocks
        if sequence >= 2:
            assert english.blocks == baseline and len(english.blocks) == 1
            assert not replay.visual.current.hidden
        assert replay.live.accepted is not None and replay.live.accepted.regions == raw


@pytest.mark.parametrize("color", [GREEN, WHITE, (60, 120, 240, 255)])
@pytest.mark.parametrize("score", [0.98, 0.4])
@pytest.mark.parametrize("digit", ["0", "1", "3", "6", "8", "9"])
@pytest.mark.parametrize("subpixel", [False, True])
def test_colored_reference_does_not_freeze_real_numeric_edits_or_mixed_color_digits(
    color, score, digit, subpixel
):
    replay = Replay(paragraph(1, digit_color=color, score=score, subpixel=subpixel))
    old = replay.tracks
    for sequence in (2, 3, 4):
        state = replay.step(
            paragraph(sequence, digit=digit, digit_color=color, score=score, subpixel=subpixel)
        )
        if sequence == 2:
            assert not state.review.unchanged
    assert all(t.text == SOURCE.replace("5", digit) for t in replay.tracks)
    assert [t.track_id for t in replay.tracks] == [t.track_id for t in old]


@pytest.mark.parametrize("color", [GREEN, WHITE])
def test_real_extension_of_a_colored_line_still_unlocks_ocr(color):
    replay = Replay(paragraph(1))
    incoming = paragraph(2, extension=color)
    incoming = replace(
        incoming,
        regions=tuple(replace(r, text=r.text + "8", characters=()) for r in incoming.regions),
    )
    state = replay.step(incoming)
    assert not state.review.unchanged


def test_colored_paragraph_removal_still_retires_after_negative_ocr():
    replay = Replay(paragraph(1))
    for sequence in range(2, 7):
        replay.step(replace(paragraph(sequence, gone=True), regions=()))
    assert not replay.tracks


def test_dense_color_candidates_leave_basic_evidence_for_later_lines():
    replay = Replay(paragraph(1, rows=40))
    old = replay.tracks
    # Change OCR only, with exactly the same pixels, at the end of a dense scene.
    incoming = paragraph(1, rows=40)
    incoming = replace(
        incoming,
        frame=replace(incoming.frame, sequence=2, captured_at=1),
        completed_at=1,
        regions=tuple(replace(r, text="違った観測", characters=()) for r in incoming.regions),
    )
    state = replay.step(incoming)
    assert len(old) == 40 and state.review.unchanged == set(old) and replay.tracks == old


def test_same_pixel_correction_keeps_color_evidence_despite_character_box_jitter():
    replay = Replay(paragraph(1, score=0.75))
    original = replay.tracks
    incoming = paragraph(2, score=0.99)
    corrected = tuple(
        replace(
            r,
            text=r.text.replace("装", "像"),
            characters=tuple(
                replace(
                    c,
                    text=c.text.replace("装", "像"),
                    bounds=Rect(
                        c.bounds.x - 1, c.bounds.y - 1, c.bounds.width, c.bounds.height + 2
                    ),
                )
                for c in r.characters
            ),
        )
        for r in incoming.regions
    )
    replay.step(replace(incoming, regions=corrected))
    settled = replay.tracks
    assert all(t.text.startswith("像") for t in settled)
    assert [t.bounds for t in settled] == [t.bounds for t in original]
    for sequence in range(3, 25):
        state = replay.step(paragraph(sequence))
        assert state.review.unchanged == set(settled) and replay.tracks == settled
