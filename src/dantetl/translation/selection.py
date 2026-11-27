"""UI-facing assignment values. Executable backend recipes stay with their owner."""

from dataclasses import dataclass

from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.references import ModelKind, reference_key, reference_kind


@dataclass(frozen=True)
class OcrOption:
    name: str
    key: str
    capture_fps: int = 8


@dataclass(frozen=True)
class ModelOption:
    reference: str
    name: str
    key: str
    provider: str = ""
    prepared: bool = False
    unavailable_reason: str = ""
    performance_notice: str = ""
    classifies: bool = False
    # None means the model declares no known restriction.
    output_codes: frozenset[str] | None = None

    def excludes_output(self, code: str) -> bool:
        return self.output_codes is not None and code not in self.output_codes

    @property
    def local(self) -> bool:
        return not self.provider

    @property
    def preference_key(self) -> str:
        if reference_kind(self.reference) in (ModelKind.LOCAL, ModelKind.CLOUD):
            return reference_key(self.reference)
        return self.reference


@dataclass(frozen=True)
class AssignmentChange:
    plan: TranslationPlan
    reference: str = ""


@dataclass(frozen=True)
class LocalIdentity:
    key: str
    preparation: str
    unavailable: str = ""


@dataclass(frozen=True)
class CloudIdentity:
    provider: CloudProvider
    model: str
    connection: str
    decoding: str
    model_revision: str
    requests_per_minute: int
    request_limit: int
    game: str
    credential_revision: int

    @classmethod
    def from_profile(cls, profile: CloudProfile, game: str, revision: int) -> "CloudIdentity":
        return cls(
            profile.provider,
            profile.model,
            profile.account
            if profile.provider == CloudProvider.CHATGPT
            else profile.credential_key,
            profile.decoding,
            profile.model_revision,
            profile.requests_per_minute,
            profile.request_limit,
            game,
            revision,
        )


@dataclass(frozen=True)
class ClassifierIdentity:
    reference: str
    model: LocalIdentity | CloudIdentity | None
    available: bool = False
