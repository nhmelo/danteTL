import re
from dataclasses import dataclass
from typing import Protocol, TypedDict, TypeGuard, Unpack

from dantetl.application.hotkeys import DEFAULT_CAPTURE, DEFAULT_OVERLAY
from dantetl.languages.target import ENGLISH
from dantetl.translation.ollama.values import OllamaConfig

# Numeric targets use the full range supported by Qt's integer editor. Zero is unlimited.
MAX_CAPTURE_RATE = 2**31 - 1


def valid_executable_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class Preferences:
    show_overlay: bool = True
    show_border: bool = True
    include_overlay_in_capture: bool = True
    translation: bool = True
    stabilize: bool = True
    opacity: int = 85
    automatic_size: bool = True
    text_size: int = 20
    group_dialogue: bool = True
    ocr_device: str = "auto"
    capture_rate: int = 8
    model: str = "auto"
    overlay_hotkey: str = DEFAULT_OVERLAY
    capture_hotkey: str = DEFAULT_CAPTURE
    capture_focused_window: bool = True
    setup_seen: bool = False
    setup_model: str = ""
    translation_route: str = "local"
    ollama: OllamaConfig = OllamaConfig()
    cloud_model: str = ""
    game_hint: str = ""
    advanced_routing: bool = False
    short_model: str = "legacy"
    long_model: str = "legacy"
    translate_short: bool = True
    translate_long: bool = True
    game_from_window: bool = True
    history_enabled: bool = False
    history_model: str = ""
    backdrop_blur: bool = True
    source_language: str = "ja"
    output_language: str = ENGLISH.code
    interface_language: str = ENGLISH.code
    inspection_enabled: bool = False
    favorite_executables: tuple[str, ...] = ()


class SettingsEdit(TypedDict, total=False):
    """User-editable settings. Assignment, source changes and setup have dedicated commands."""

    show_overlay: bool
    show_border: bool
    include_overlay_in_capture: bool
    translation: bool
    stabilize: bool
    opacity: int
    automatic_size: bool
    text_size: int
    group_dialogue: bool
    ocr_device: str
    capture_rate: int
    overlay_hotkey: str
    capture_hotkey: str
    capture_focused_window: bool
    game_hint: str
    advanced_routing: bool
    game_from_window: bool
    history_enabled: bool
    history_model: str
    backdrop_blur: bool
    output_language: str
    interface_language: str
    inspection_enabled: bool


class EditSettings(Protocol):
    def __call__(self, **changes: Unpack[SettingsEdit]) -> None: ...
