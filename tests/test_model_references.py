import pytest

from dantetl.translation.references import (
    ModelKind,
    format_reference,
    imported_key,
    is_imported,
    is_remote,
    reference_key,
    reference_kind,
    valid_imported_key,
)


@pytest.mark.parametrize(
    "reference, kind, key, remote",
    [
        ("local:gemma", ModelKind.LOCAL, "gemma", False),
        ("cloud:cloud-" + "a" * 32, ModelKind.CLOUD, "cloud-" + "a" * 32, True),
        ("ollama:" + "b" * 64, ModelKind.OLLAMA, "b" * 64, True),
        ("layout", None, "layout", False),
        ("", None, "", False),
        ("remote:gemma", None, "remote:gemma", False),
    ],
)
def test_persisted_reference_format_round_trips_and_rejects_unknown_kinds(
    reference, kind, key, remote
):
    assert reference_kind(reference) is kind
    assert reference_key(reference) == key
    assert is_remote(reference) is remote
    if kind is not None:
        assert format_reference(kind, key) == reference


def test_imported_keys_require_the_full_digest():
    key = imported_key("c" * 64)
    assert key == "imported-" + "c" * 64 and is_imported(key) and valid_imported_key(key)
    assert is_imported("imported-short") and not valid_imported_key("imported-short")
    assert not is_imported("gemma") and not valid_imported_key("gemma")
