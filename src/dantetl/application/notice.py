"""Pure, cached presentation policy. Inputs are values, never widgets or workers."""

from dataclasses import dataclass, replace
from functools import lru_cache

from dantetl.application.setup_status import SetupAdvice, acceleration_advice
from dantetl.i18n import display, mark, tr
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.runtime.hardware_values import HardwareStatus
from dantetl.runtime.setup_values import RuntimeTarget
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.references import ModelKind, format_reference, is_imported, reference_kind
from dantetl.translation.selection import AssignmentChange, ModelOption


@dataclass(frozen=True)
class NoticeInput:
    options: tuple[ModelOption, ...]
    plan: TranslationPlan
    ocr_ready: bool
    saving_error: str = ""
    setup_error: str = ""
    setup_busy: bool = False
    preparing: bool = False
    ocr_state: str = ""
    ocr_message: str = ""
    translation_state: str = ""
    translation_message: str = ""
    rejected: bool = False
    hardware: HardwareStatus | None = None
    ocr_device: str = "cpu"
    provider: str = ""
    fallback: str = ""
    target: TargetLanguage = ENGLISH
    ocr_model: str = "MeikiOCR"
    interface_language: str = ENGLISH.code
    discovering: bool = False
    installed_runtime: tuple[str, RuntimeTarget] | None = None


@dataclass(frozen=True)
class LanguageAdvice:
    message: str = ""
    action: str = ""
    change: AssignmentChange | None = None


def _language_advice(
    target: TargetLanguage, plan: TranslationPlan, options: dict[str, ModelOption]
) -> LanguageAdvice:
    affected: dict[str, list[str]] = {}
    for reference, enabled, label in (
        (plan.short, plan.short_enabled, tr("short labels")),
        (plan.long, plan.long_enabled, tr("long text and dialogue")),
    ):
        model = options.get(reference)
        if enabled and model and model.excludes_output(target.code):
            affected.setdefault(reference, []).append(label)
    if not affected:
        return LanguageAdvice()
    models = ", ".join(
        options[key].name.partition(" · ")[0] + (f" ({labels[0]})" if len(labels) == 1 else "")
        for key, labels in affected.items()
    )
    message = tr("{name} output is unavailable with {models}.", name=tr(target.name), models=models)
    # Prefer a Gemma already used by another category, then one ready for use.
    candidates = (
        option
        for key in (format_reference(ModelKind.LOCAL, k) for k in ("gemma", "gemma-e4b"))
        if (option := options.get(key)) and not option.excludes_output(target.code)
    )
    gemma = min(
        candidates,
        key=lambda c: (
            bool(c.unavailable_reason),
            c.reference not in plan.active_models,
            not c.prepared,
        ),
        default=None,
    )
    if gemma is None:
        return LanguageAdvice(message, tr("Set up Gemma"))
    updated = replace(
        plan,
        short=gemma.reference if plan.short in affected else plan.short,
        long=gemma.reference if plan.long in affected else plan.long,
    )
    action = (
        tr("Set up Gemma")
        if gemma.unavailable_reason
        else tr("Use {value}", value=gemma.name.partition(" · ")[0])
    )
    return LanguageAdvice(message, action, AssignmentChange(updated, gemma.reference))


@dataclass(frozen=True)
class Notices:
    message: str = ""
    action: str = mark("Open settings")
    tone: str = "warning"
    destination: str = "models"
    model_warning: str = ""
    model_action: str = "Models"
    save_note: str = ""
    acceleration: SetupAdvice = SetupAdvice()
    model_destination: str = "models"
    language: LanguageAdvice = LanguageAdvice()
    prepare_reference: str = ""


@lru_cache(maxsize=1)
def project_notices(value: NoticeInput) -> Notices:
    options = {c.reference: c for c in value.options}
    active = tuple(options[k] for k in value.plan.active_models if k in options)
    cloud = tuple(c for c in active if not c.local)
    missing = []
    model_action, model_destination, prepare_reference = tr("Models"), "models", ""
    if not value.ocr_ready:
        missing.append(tr("{ocr_model} is not downloaded", ocr_model=value.ocr_model))
    for label, key, enabled in (
        (tr("short labels"), value.plan.short, value.plan.short_enabled),
        (tr("long text"), value.plan.long, value.plan.long_enabled),
    ):
        if not enabled or value.discovering or value.preparing:
            continue
        choice = options.get(key)
        first_issue = not missing
        action, destination = tr("Choose model"), "selection"
        if not key:
            missing.append(tr("No translation model is selected for {label}", label=label))
        elif choice is None:
            missing.append(tr("The model for {label} is unavailable", label=label))
            if reference_kind(key) is ModelKind.OLLAMA:
                action, destination = "Ollama API", "ollama"
            elif reference_kind(key) is ModelKind.CLOUD:
                action, destination = tr("Cloud models"), "cloud"
        elif choice.provider == "ollama" and not choice.prepared:
            missing.append(
                tr("The model for {label} is unavailable", label=label) + f" ({choice.name})"
            )
            action, destination = "Ollama API", "ollama"
        elif not choice.local and not choice.prepared:
            missing.append(tr("Connect the cloud model for {label} in Models", label=label))
            action, destination = tr("Cloud models"), "cloud"
        elif choice.unavailable_reason:
            reason = display(choice.unavailable_reason).rstrip(".")
            if reason not in missing:
                missing.append(reason)
            action, destination = (
                (tr("Models"), "models")
                if is_imported(choice.key)
                else (tr("Acceleration options"), "runtime")
            )
        elif choice.local and not choice.prepared:
            message = tr(
                "{name} needs preparation for this runtime",
                name=display(choice.name).partition(" · ")[0],
            )
            if message not in missing:
                missing.append(message)
            action, destination = tr("Prepare model"), "prepare"
        else:
            continue
        if first_issue:
            model_action, model_destination = action, destination
            if destination == "prepare":
                prepare_reference = key
    if not options and not value.discovering and model_destination not in ("cloud", "ollama"):
        missing.append(tr("No translation model is downloaded or configured"))
        model_action, model_destination = tr("Download models"), "models"
    warning = ". ".join(missing)
    if warning and not any(c.prepared for c in active):
        warning += tr(". Capture can continue, but translation needs setup.")
    message, action, tone, destination = "", tr("Open settings"), "warning", "models"
    if value.saving_error:
        message, tone, destination = value.saving_error, "error", "settings"
    elif value.setup_error:
        message, tone = tr("Setup needs attention: ") + value.setup_error, "error"
    elif value.setup_busy:
        message = tr("Setup is running in the background. Open Settings for progress.")
        action, tone = tr("View setup"), "info"
    elif not value.ocr_ready:
        message, action = warning, tr("Download models")
    elif value.preparing:
        message, action, tone = (
            tr("Preparing the translation model. Capture can continue."),
            "",
            "info",
        )
    elif value.ocr_state in ("failed", "loading"):
        message, action = value.ocr_message, ""
        tone = "error" if value.ocr_state == "failed" else "info"
    elif value.translation_state in ("failed", "loading"):
        message, action = value.translation_message, ""
        tone = "error" if value.translation_state == "failed" else "info"
    elif missing:
        message, action, destination = warning, model_action, model_destination
    elif value.rejected:
        message, action = value.translation_message, ""
        if any(p.provider == "chatgpt" for p in cloud):
            action, destination = tr("Manage usage"), "usage"
    advice = acceleration_advice(
        value.hardware, value.ocr_device, actual_provider=value.provider, fallback=value.fallback
    )
    if value.installed_runtime:
        name, target = value.installed_runtime
        advice = SetupAdvice(
            tr("Restart DanteTL to use {name}.", name=tr(name)),
            action=mark("Restart now"),
            restart_target=target,
        )
    return Notices(
        message,
        action,
        tone,
        destination,
        warning,
        model_action,
        tr("Settings could not be saved. Changes still apply to this session. ")
        + value.saving_error
        if value.saving_error
        else Notices.save_note,
        advice,
        model_destination,
        _language_advice(value.target, value.plan, options),
        prepare_reference,
    )
