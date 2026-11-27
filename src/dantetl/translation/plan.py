"""Explicit text-category assignments, independent of providers and presentation."""

from dataclasses import dataclass

from dantetl.ocr.observation import TextTrack
from dantetl.translation.grouping import TextBlock
from dantetl.translation.references import ModelKind, format_reference, reference_kind

SHORT_TEXT_LIMIT = 40


def is_long_text(item: TextTrack | TextBlock) -> bool:
    """Layout groups are longer text, not proof that a character is speaking."""
    return (isinstance(item, TextBlock) and item.dialogue) or len(
        item.text.strip()
    ) > SHORT_TEXT_LIMIT


@dataclass(frozen=True, slots=True)
class TranslationPlan:
    short: str = ""
    long: str = ""
    short_enabled: bool = True
    long_enabled: bool = True

    def model_for(self, item: TextTrack | TextBlock) -> str:
        if is_long_text(item):
            return self.long if self.long_enabled else ""
        return self.short if self.short_enabled else ""

    @property
    def active_models(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                key
                for key, enabled in (
                    (self.short, self.short_enabled),
                    (self.long, self.long_enabled),
                )
                if key and enabled
            )
        )

    @property
    def two_local_models(self) -> bool:
        return sum(reference_kind(key) is ModelKind.LOCAL for key in self.active_models) == 2

    @classmethod
    def legacy(cls, mode: str, local: str, cloud: str) -> "TranslationPlan":
        local = format_reference(ModelKind.LOCAL, local) if local and local != "none" else ""
        cloud = format_reference(ModelKind.CLOUD, cloud) if cloud else ""
        return cls(
            cloud if mode in ("cloud", "short") else local,
            cloud if mode in ("cloud", "long") else local,
        )
