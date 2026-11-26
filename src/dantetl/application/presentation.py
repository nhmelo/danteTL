"""Pure status projection. Formatting cannot start, stop or advance a worker."""

from dataclasses import replace

from dantetl.application.translation_activity import (
    InactiveReason,
    TranslationActivity,
    TranslationMode,
)
from dantetl.application.values import ApplicationView, ProcessingProgress, Scene
from dantetl.domain import CaptureState
from dantetl.i18n import tr
from dantetl.ocr.values import OcrSnapshot, OcrState
from dantetl.ocr.visual_values import VisualState
from dantetl.translation.values import TranslationSnapshot, TranslationState


def reset_scene(view: ApplicationView) -> ApplicationView:
    return replace(
        view,
        scene=Scene(),
        blocks=(),
        english=(),
        results=(),
        observation=None,
        ocr=None,
        frame=None,
        capture=None,
        visual=VisualState(),
        assignments=(),
        overlay_bounds=None,
        overlay_anchor=None,
        reset=view.reset + 1,
    )


def ocr_status(snapshot: OcrSnapshot, visual_message: str, now: float, ocr_model: str) -> str:
    if snapshot.state == OcrState.FAILED:
        return tr("OCR unavailable: {message} Stop and restart to retry.", message=snapshot.message)
    if snapshot.state == OcrState.LOADING:
        return tr(
            "Loading local {ocr_model} models… GPU initialization can take longer.",
            ocr_model=ocr_model,
        )
    if snapshot.result is not None:
        busy = (
            tr("Reading · {value:.1f} s", value=now - snapshot.processing_since)
            if snapshot.processing_since is not None
            else tr("Last scan · {count} regions", count=len(snapshot.result.regions))
        )
        return snapshot.message or visual_message or busy
    elapsed = (
        f" {now - snapshot.processing_since:.0f} s."
        if snapshot.processing_since is not None
        else ""
    )
    return tr("Reading the first frame…{elapsed}", elapsed=elapsed) + (
        tr(" First GPU use may compile kernels; later launches reuse the local cache.")
        if snapshot.provider == "CUDAExecutionProvider"
        else ""
    )


def translation_status(
    activity: TranslationActivity,
    snapshot: TranslationSnapshot | None,
    now: float,
) -> str:
    if activity.mode == TranslationMode.PROBING:
        return tr("Testing imported model; previous translation selection resumes afterward.")
    if activity.mode == TranslationMode.PREPARING:
        return tr("Preparing translation model for this PC…")
    if activity.mode == TranslationMode.SWITCHING:
        return tr("Waiting for the previous translation model to unload…")
    if activity.mode == TranslationMode.INACTIVE:
        return {
            InactiveReason.UNLOADING: tr("Unloading translation model…"),
            InactiveReason.NEEDS_RUNTIME: (
                tr("Translation needs runtime setup. Select a model for instructions.")
            ),
            InactiveReason.OFF: tr("Translation is off."),
            InactiveReason.NO_MODEL: (
                tr("No translation model selected. Open Settings to download one.")
            ),
            InactiveReason.STOPPED: tr("Translation stopped."),
        }[activity.reason]
    assert snapshot is not None
    if activity.mode == TranslationMode.STANDBY:
        return (
            tr("Model ready. Start capture to translate.")
            if snapshot.state == TranslationState.READY
            else tr("Loading the local model…")
        )
    if snapshot.state == TranslationState.FAILED:
        return tr(
            "Translation unavailable: {message} Toggle Translation overlay off/on to retry.",
            message=snapshot.message,
        )
    if snapshot.state == TranslationState.LOADING:
        return tr(
            "Loading/preparing {model_name}… First use can take longer.",
            model_name=activity.model_name,
        )
    if snapshot.state == TranslationState.STOPPING:
        return tr("Waiting for translation to finish unloading…")
    return snapshot.message or (
        tr("Translating · oldest active call {value:.1f} s", value=now - snapshot.processing_since)
        if snapshot.processing_since is not None
        else tr("Ready · {count} translated blocks", count=len(snapshot.results))
    )


def processing_progress(
    view: ApplicationView, *, translation_enabled: bool, ocr_model: str = "MeikiOCR"
) -> ProcessingProgress:
    """Visible startup stages, including partially ready routes; no invented percentage."""
    capture = view.capture
    if capture is None or capture.state not in (CaptureState.STARTING, CaptureState.RUNNING):
        return ProcessingProgress()
    if capture.state == CaptureState.STARTING or capture.frame is None:
        return ProcessingProgress(tr("Waiting for the first captured frame…"), busy=True)
    if view.frame is None:  # Minimized or waiting for matching geometry.
        return ProcessingProgress()
    messages: list[str] = []
    busy = False
    tone = "info"
    ocr = view.ocr
    if ocr and ocr.state == OcrState.LOADING:
        messages.append(
            tr(
                "Loading {ocr_model}… First GPU initialization can take longer.",
                ocr_model=ocr_model,
            )
        )
        busy = True
    elif ocr and ocr.state == OcrState.FAILED:
        messages.append(view.ocr_status)
        tone = "error"
    elif ocr and ocr.result is None and ocr.state == OcrState.READY:
        messages.append(tr("Reading the first frame…"))
        busy = True
    if translation_enabled:
        translation = view.translation
        if view.translation_mode in (
            TranslationMode.PREPARING,
            TranslationMode.SWITCHING,
            TranslationMode.PROBING,
        ):
            messages.append(view.translation_status)
            busy = True
        elif translation and view.translation_mode == TranslationMode.ACTIVE:
            for lane in translation.lanes:
                if lane.state in (TranslationState.LOADING, TranslationState.STOPPING):
                    messages.append(
                        tr(
                            "Loading {name}… {message}", name=lane.name, message=lane.message
                        ).strip()
                    )
                    busy = True
                elif lane.state == TranslationState.FAILED:
                    messages.append(
                        tr("{name} unavailable: {message}", name=lane.name, message=lane.message)
                    )
                    tone = "error"
            if not translation.lanes and translation.state in (
                TranslationState.LOADING,
                TranslationState.STOPPING,
                TranslationState.FAILED,
            ):
                messages.append(view.translation_status)
                busy |= translation.state != TranslationState.FAILED
                if translation.state == TranslationState.FAILED:
                    tone = "error"
        if not messages and view.scene.waiting_blocks and not view.english:
            messages.append(tr("Waiting for text to finish appearing…"))
            busy = True
    return ProcessingProgress("\n".join(messages), busy, tone)
