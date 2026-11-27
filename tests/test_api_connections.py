import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from uuid import uuid4

import pytest

from dantetl.translation.cloud.api_connections import ApiConnections
from dantetl.translation.cloud.credentials import WindowsSecrets
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.values import (
    MAX_CLOUD_PROFILES,
    MODELS,
    ApiConnection,
    CloudProfile,
    CloudProvider,
)


class SecretsFixture:
    def __init__(self):
        self.values = {}

    def read(self, key):
        return self.values.get(key)

    def write(self, key, secret):
        self.values[key] = secret

    def delete(self, key):
        self.values.pop(key, None)


def test_full_unicode_catalog_reopens_in_current_and_legacy_encoding(tmp_path):
    catalog = CloudProfiles(tmp_path / "models.json")
    connections, secrets = ApiConnections(catalog), SecretsFixture()
    for index in range(MAX_CLOUD_PROFILES):
        profile = replace(
            CloudProfile.new(CloudProvider.GEMINI),
            name="🎮" * 78 + f"{index:02}",
            model_revision="🌐" * 120,
        )
        connections.save_model(profile, "", "fixture-key", secrets)

    def reopen():
        return ApiConnections(CloudProfiles(catalog.path)).discover(lambda: secrets, pytest.fail)

    saved = reopen()
    assert len(saved.profiles) == len(saved.connections) == len(saved.ready) == MAX_CLOUD_PROFILES
    for path, key, rows in (
        (catalog.path, "profiles", saved.profiles),
        (connections.path, "connections", saved.connections),
    ):
        path.write_text(
            json.dumps({"version": 1, key: [asdict(r) for r in rows]}, indent=2), "utf-8"
        )
    assert connections.path.stat().st_size > 16_384 and catalog.path.stat().st_size > 32_768
    assert reopen() == saved


def test_oversized_edit_preserves_saved_models_connections_and_key(tmp_path):
    catalog = CloudProfiles(tmp_path / "models.json")
    connections, secrets = ApiConnections(catalog), SecretsFixture()
    first, _ = connections.save_model(
        CloudProfile.new(CloudProvider.OPENAI), "", "original", secrets
    )
    originals = {path: path.read_bytes() for path in (catalog.path, connections.path)}
    with pytest.raises(ValueError, match="too large"):
        connections.save_model(
            replace(first, account="a" * 70_000), first.credential, "new", secrets
        )
    assert {path: path.read_bytes() for path in originals} == originals
    assert secrets.values == {first.credential: "original"}
    assert connections.discover(lambda: secrets, pytest.fail).profiles == (first,)


def test_mixed_catalog_can_edit_at_full_capacity_without_mutating_keys_on_overflow(tmp_path):
    catalog = CloudProfiles(tmp_path / "models.json")
    connections, secrets = ApiConnections(catalog), SecretsFixture()
    first, _ = connections.save_model(
        CloudProfile.new(CloudProvider.OPENAI), "", "old-key", secrets
    )
    mixed = tuple(
        CloudProfile(
            "cloud-" + uuid4().hex,
            "Plan model",
            CloudProvider.CHATGPT,
            "gpt-6-luna",
            account="account-" + "a" * 32,
        )
        for _ in range(MAX_CLOUD_PROFILES - 1)
    )
    catalog.write((first, *mixed))
    edited, _ = connections.save_model(
        replace(first, name="Edited API model"), first.credential, "new-key", secrets
    )
    assert len(catalog.read()) == MAX_CLOUD_PROFILES
    assert edited in catalog.read() and connections.secret(edited, secrets) == "new-key"
    before = catalog.read()
    with pytest.raises(ValueError):
        connections.save_model(
            CloudProfile.new(CloudProvider.OPENAI), first.credential, "must-not-save", secrets
        )
    assert catalog.read() == before and connections.secret(edited, secrets) == "new-key"


@pytest.mark.parametrize(
    "provider", [CloudProvider.OPENAI, CloudProvider.GEMINI, CloudProvider.CLAUDE]
)
def test_connection_is_reused_after_restart_model_removal_and_new_model(tmp_path, provider):
    catalog = CloudProfiles(tmp_path / "models.json")
    connections = ApiConnections(catalog)
    secrets = SecretsFixture()
    first, _ = connections.save_model(CloudProfile.new(provider), "", "fixture-key", secrets)
    second, _ = connections.save_model(
        replace(CloudProfile.new(provider), model=MODELS[provider][1][0]),
        first.credential,
        "",
        secrets,
    )
    assert first.credential == second.credential
    assert len(secrets.values) == 1
    # New service instances, no model entries: the connection still exists.
    catalog.write(())
    reloaded = ApiConnections(CloudProfiles(catalog.path))
    assert len(reloaded.discover(lambda: secrets, pytest.fail).connections) == 1
    third, _ = reloaded.save_model(CloudProfile.new(provider), first.credential, "", secrets)
    assert reloaded.secret(third, secrets) == "fixture-key"
    assert "fixture-key" not in connections.path.read_text()
    assert "fixture-key" not in catalog.path.read_text()
    reloaded.forget(first.credential, secrets)
    assert reloaded.secret(third, secrets) == "" and not secrets.values


@pytest.mark.parametrize(
    "other", [CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE]
)
def test_connection_never_crosses_provider_and_key_replacement_rolls_back(
    tmp_path, monkeypatch, other
):
    catalog = CloudProfiles(tmp_path / "models.json")
    connections, secrets = ApiConnections(catalog), SecretsFixture()
    first, _ = connections.save_model(
        CloudProfile.new(CloudProvider.OPENAI), "", "original-secret", secrets
    )
    with pytest.raises(ValueError, match="this provider"):
        connections.save_model(CloudProfile.new(other), first.credential, "", secrets)
    forged = replace(CloudProfile.new(other), credential=first.credential)
    assert connections.secret(forged, secrets) == ""

    def disk_error(*args):
        raise OSError("fixture disk full")

    monkeypatch.setattr(catalog, "write", disk_error)
    with pytest.raises(OSError):
        connections.save_model(first, first.credential, "new-secret", secrets)
    assert connections.secret(first, secrets) == "original-secret"
    with pytest.raises(OSError):
        connections.save_model(
            CloudProfile.new(CloudProvider.OPENAI), "", "abandoned-secret", secrets
        )
    assert list(secrets.values.values()) == ["original-secret"]
    assert len(connections.discover(lambda: secrets, pytest.fail).connections) == 1


def test_legacy_keys_are_adopted_without_moving_or_merging_accounts(tmp_path):
    catalog = CloudProfiles(tmp_path / "models.json")
    first, second = CloudProfile.new(CloudProvider.OPENAI), CloudProfile.new(CloudProvider.OPENAI)
    catalog.write((first, second))
    secrets = SecretsFixture()
    secrets.write(first.key, "first-account")
    secrets.write(second.key, "second-account")
    connections = ApiConnections(catalog)
    assert len(connections.discover(lambda: secrets, pytest.fail).connections) == 2
    assert connections.secret(first, secrets) == "first-account"
    assert connections.secret(second, secrets) == "second-account"
    connections.forget(first.key, secrets)
    assert len(connections.discover(lambda: secrets, pytest.fail).connections) == 1
    assert connections.secret(first, secrets) == ""
    assert connections.secret(second, secrets) == "second-account"
    assert json.loads(catalog.path.read_text())["version"] == 1


def test_rollback_cannot_erase_another_instances_committed_connection(tmp_path, monkeypatch):
    first = ApiConnections(CloudProfiles(tmp_path / "models.json"))
    second = ApiConnections(CloudProfiles(first.catalog.path))
    secrets = SecretsFixture()
    staged, started, release = (threading.Event() for _ in range(3))

    def fail_write(_):
        staged.set()
        assert release.wait(5)
        raise OSError("fixture disk full")

    def save_second():
        started.set()
        return second.save_model(CloudProfile.new(CloudProvider.OPENAI), "", "keep", secrets)

    monkeypatch.setattr(first.catalog, "write", fail_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(
            first.save_model, CloudProfile.new(CloudProvider.OPENAI), "", "discard", secrets
        )
        try:
            assert staged.wait(5)
            saved = pool.submit(save_second)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                saved.result(timeout=0.25)
        finally:
            release.set()
        with pytest.raises(OSError, match="fixture disk full"):
            failed.result(timeout=5)
        model, _ = saved.result(timeout=5)
    inventory = second.discover(lambda: secrets, pytest.fail)
    assert inventory.profiles == (model,)
    assert [c.key for c in inventory.connections] == [model.credential_key]
    assert inventory.ready == {model.credential_key}
    assert secrets.values == {model.credential_key: "keep"}


@pytest.mark.windows
def test_native_vault_round_trip_does_not_modify_stored_key_when_wiping_read_buffer():
    secrets = WindowsSecrets("DanteTL/verification/" + uuid4().hex)
    try:
        assert secrets.read("fixture") is None
        secrets.write("fixture", "fake-owned-test-key")
        assert secrets.read("fixture") == secrets.read("fixture") == "fake-owned-test-key"
        secrets.delete("fixture")
        assert secrets.read("fixture") is None
    finally:
        secrets.delete("fixture")


@pytest.mark.parametrize("provider", list(CloudProvider))
def test_stored_provider_names_load_as_typed_providers_and_save_unchanged(provider):
    if provider is CloudProvider.CHATGPT:
        stored = asdict(
            CloudProfile(
                "cloud-" + "1" * 32, "Plan", provider, "gpt-x", account="account-" + "2" * 32
            )
        )
    else:
        stored = asdict(CloudProfile.new(provider))
    record = json.loads(json.dumps(stored))
    assert record["provider"] == provider.value
    loaded = CloudProfile(**record)
    assert loaded.provider is provider and asdict(loaded) == record


def test_unknown_providers_and_plan_api_connections_are_rejected():
    with pytest.raises(ValueError, match="supported provider"):
        CloudProfile("cloud-" + "3" * 32, "Unknown", "other", "model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="API connection"):
        ApiConnection("api-" + "4" * 32, CloudProvider.CHATGPT, "Plan")
    assert ApiConnection("api-" + "5" * 32, "claude", "Key").provider is CloudProvider.CLAUDE  # type: ignore[arg-type]
