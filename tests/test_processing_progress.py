"""Startup feedback is a pure projection of actual capture/OCR/translation stages."""

from dataclasses import replace

import pytest
from live_fixtures import frame

from dantetl.application.presentation import processing_progress
from dantetl.application.translation_activity import TranslationMode
from dantetl.application.values import ApplicationView, Scene
from dantetl.domain import CaptureSnapshot, CaptureState
from dantetl.ocr.values import OcrResult, OcrSnapshot, OcrState
from dantetl.translation.values import (
    TranslatedText,
    TranslationLaneSnapshot,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)


def ready_view():
    source = frame(1)
    return ApplicationView(
        capture=CaptureSnapshot(CaptureState.RUNNING, source),
        frame=source,
        ocr=OcrSnapshot(OcrState.READY, OcrResult(source, (), 20, 1)),
        translation=TranslationSnapshot(TranslationState.READY),
        translation_mode=TranslationMode.ACTIVE,
    )


def test_startup_explains_each_stage_without_an_invented_percentage():
    view = ready_view()
    stages = (
        (replace(view, capture=CaptureSnapshot(CaptureState.STARTING)), "first captured frame"),
        (replace(view, ocr=OcrSnapshot(OcrState.LOADING)), "Loading MeikiOCR"),
        (replace(view, ocr=OcrSnapshot(OcrState.READY)), "Reading the first frame"),
        (replace(view, scene=Scene(waiting_blocks=1)), "text to finish appearing"),
    )
    for current, expected in stages:
        result = processing_progress(current, translation_enabled=True)
        assert result.busy and expected in result.message
    assert not processing_progress(view, translation_enabled=True).busy


def test_ready_cloud_cannot_hide_a_loading_local_model_or_vice_versa():
    for loading, ready in (("Gemma", "Cloud"), ("Cloud", "Gemma")):
        snapshot = TranslationSnapshot(
            TranslationState.READY,
            lanes=(
                TranslationLaneSnapshot("a", loading, TranslationState.LOADING),
                TranslationLaneSnapshot("b", ready, TranslationState.READY),
            ),
        )
        view = replace(ready_view(), translation=snapshot)
        progress = processing_progress(view, translation_enabled=True)
        assert progress.busy and loading in progress.message and ready not in progress.message


@pytest.mark.parametrize(
    "mode", [TranslationMode.PREPARING, TranslationMode.SWITCHING, TranslationMode.PROBING]
)
def test_transition_status_stays_visible_until_owner_resolves_it(mode):
    view = replace(ready_view(), translation_mode=mode, translation_status="Stage in progress")
    progress = processing_progress(view, translation_enabled=True)
    assert progress.busy and progress.message == view.translation_status


def test_loaded_idle_or_published_translation_is_not_reported_as_startup():
    view = ready_view()
    assert processing_progress(view, translation_enabled=True).message == ""
    view = replace(view, translation=TranslationSnapshot(TranslationState.READY, pending=1))
    assert not processing_progress(view, translation_enabled=True).message
    working = replace(
        view, translation=TranslationSnapshot(TranslationState.READY, processing_since=2)
    )
    assert not processing_progress(working, translation_enabled=True).busy
    assert not processing_progress(view, translation_enabled=False).busy
    view = replace(view, english=(TranslatedText(TranslationRequest(1, 0, "設定"), "Settings"),))
    assert not processing_progress(view, translation_enabled=True).busy


def test_failure_clears_spinner_and_shows_reason_without_hiding_other_lane():
    view = replace(ready_view(), ocr=OcrSnapshot(OcrState.FAILED), ocr_status="OCR failed")
    progress = processing_progress(view, translation_enabled=True)
    assert not progress.busy and progress.tone == "error" and "OCR failed" in progress.message
    snapshot = TranslationSnapshot(
        TranslationState.READY,
        lanes=(
            TranslationLaneSnapshot("a", "Gemma", TranslationState.READY),
            TranslationLaneSnapshot("b", "Cloud", TranslationState.FAILED, "Connection failed"),
        ),
    )
    progress = processing_progress(
        replace(ready_view(), translation=snapshot), translation_enabled=True
    )
    assert (
        not progress.busy and progress.tone == "error" and "Connection failed" in progress.message
    )


@pytest.mark.parametrize(
    "state", [CaptureState.STOPPING, CaptureState.STOPPED, CaptureState.FAILED]
)
def test_stop_failure_and_missing_geometry_do_not_leave_a_loading_spinner(state):
    view = replace(ready_view(), ocr=OcrSnapshot(OcrState.LOADING))
    assert not processing_progress(
        replace(view, capture=CaptureSnapshot(state)), translation_enabled=True
    ).busy
    assert not processing_progress(replace(view, frame=None), translation_enabled=True).busy
