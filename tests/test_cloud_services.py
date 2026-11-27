"""Cloud ownership regressions run without constructing any widgets."""

import asyncio
import subprocess
import sys
import threading
from dataclasses import replace

from remote_fixtures import create, finish

from dantetl.application.cloud_services import CloudServices
from dantetl.translation.cloud.chatgpt_auth import PLAN_SCOPE, ChatGPTConnections
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.state import CloudDraft
from dantetl.translation.cloud.values import CatalogModel, CloudProfile, CloudProvider


def test_save_remove_and_usage_are_application_owned_without_a_view(tmp_path):
    service, secrets = create(tmp_path)
    initial = service.snapshot()
    profile = CloudProfile.new(CloudProvider.DEEPSEEK)
    service.save(CloudDraft(profile, "", "fixture-key"))
    saved = finish(service)
    assert saved.saved and saved.profiles[0].key == profile.key
    assert saved.available == frozenset((profile.key,))
    assert not initial.profiles and not initial.available
    assert secrets.threads and threading.get_ident() not in secrets.threads
    allowance = service.allowance(profile.key)
    asyncio.run(allowance.admit(profile, clock=lambda: 10))
    renamed = replace(saved.profiles[0], name="Renamed")
    service.save(CloudDraft(renamed, renamed.credential_key, ""))
    finish(service)
    assert service.allowance(profile.key) is allowance
    assert service.usage(profile.key).requests == 1
    service.remove(renamed)
    assert not service.available(profile.key)  # revoke before disk completion
    removed = finish(service)
    assert not removed.profiles and secrets.values  # removing does not forget a key


def test_model_refresh_publishes_results_and_completion_together(tmp_path, monkeypatch):
    service, _ = create(tmp_path)
    fetching, fetched, discovering, release = (threading.Event() for _ in range(4))
    models = (CatalogModel("fixture", "Fixture model"),)

    def fetch(*_):
        fetching.set()
        assert fetched.wait(3)
        return models

    def accounts(*_):
        discovering.set()
        assert release.wait(3)
        return ()

    monkeypatch.setattr("dantetl.application.cloud_services.fetch_models", fetch)
    monkeypatch.setattr(service.connections, "accounts", accounts)
    try:
        service.refresh_models(
            CloudDraft(CloudProfile.new(CloudProvider.DEEPSEEK), "", "fixture-key")
        )
        assert fetching.wait(3)
        before = service.snapshot()
        fetched.set()
        assert discovering.wait(3)
        assert before.busy and before.discovered is None
        assert service.snapshot() == before
    finally:
        fetched.set()
        release.set()
        after = finish(service)
    assert after.discovered == models and not after.busy
    assert after.completion == before.completion + 1 and after.version > before.version


def test_cancelled_worker_drains_before_new_commands_and_owner_cleanup(tmp_path, monkeypatch):
    service, _ = create(tmp_path)
    entered, cancelled, cleanup, release = (threading.Event() for _ in range(4))

    class Backend:
        def translate(self, texts, *, target):
            entered.set()
            assert cancelled.wait(3)
            return ("Cancelled",)

        def cancel_pending(self):
            cancelled.set()

        def close(self):
            cleanup.set()
            assert release.wait(3)

    monkeypatch.setattr(
        "dantetl.application.cloud_services.CloudBackend", lambda *a, **k: Backend()
    )
    draft = CloudDraft(CloudProfile.new(CloudProvider.DEEPSEEK), "", "fixture")
    service.test(draft)
    assert entered.wait(3)
    service.cancel()
    assert cleanup.wait(3)
    try:
        assert service.busy
        service.save(draft)  # cannot reset cancellation or replace the in-flight result
        assert service.snapshot().busy and not service.snapshot().profiles
    finally:
        release.set()
    completed = finish(service)
    assert not completed.saved and not completed.profiles
    assert "cancelled" in completed.message.lower()


def test_closed_owner_rejects_late_dialog_actions_and_resumes_after_failed_restart(tmp_path):
    service, secrets = create(tmp_path)
    profile = CloudProfile.new(CloudProvider.DEEPSEEK)
    service.save(CloudDraft(profile, "", "fixture-key"))
    before = finish(service)
    service.close()
    service.remove(profile)
    service.forget_key(before.profiles[0].credential_key)
    service.test(CloudDraft(profile, "", "fixture-key"))
    service.save(CloudDraft(profile, "", "replacement"))
    service.start()
    service._open_browser("https://example.invalid/late")
    assert not service.busy and not service.take_browser()
    assert service.snapshot().available == before.available
    assert set(secrets.values.values()) == {"fixture-key"}
    service.resume()
    service.remove(profile)
    assert not finish(service).profiles


def test_cloud_owner_import_and_startup_reject_all_qt_imports(tmp_path):
    script = """
import importlib.abc, sys, time
from pathlib import Path
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("PySide6", "shiboken6")):
            raise AssertionError("Application owner imported Qt: " + fullname)
sys.meta_path.insert(0, NoQt())
from dantetl.application.cloud_services import CloudServices
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.chatgpt_auth import ChatGPTConnections
from dantetl.translation.cloud.protected_store import ProtectedStore
class NoSecrets:
    def read(self, key): raise AssertionError("Unexpected credential read")
root = Path(sys.argv[1])
service = CloudServices(CloudProfiles(root / "cloud.json"), NoSecrets,
    ChatGPTConnections(ProtectedStore(root / "accounts")))
service.start()
deadline = time.monotonic() + 3
while service.busy:
    service.advance()
    assert time.monotonic() < deadline
    time.sleep(.001)
assert not service.snapshot().problem
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=10)


def test_account_refresh_scopes_recovery_even_if_later_discovery_fails(tmp_path, monkeypatch):
    from dantetl.translation.cloud.values import ChatGPTAccount

    service, _ = create(tmp_path)
    accounts = tuple(
        ChatGPTAccount("account-" + c * 32, c, True, (("fixture", "Fixture"),)) for c in ("a", "b")
    )
    profiles = tuple(
        CloudProfile("cloud-" + c * 32, c, CloudProvider.CHATGPT, "fixture", account=a.key)
        for c, a in zip(("a", "b"), accounts, strict=True)
    )
    service.catalog.write(profiles)
    monkeypatch.setattr(service.connections, "accounts", lambda *_: accounts)
    monkeypatch.setattr(service.connections, "refresh_models", lambda *_: ())
    service.start()
    finish(service)
    assert all(service.available(p.key) for p in profiles)
    service.refresh_account(accounts[1].key)
    finish(service)
    assert service.credential_revision(profiles[0]) == 0
    revision = service.credential_revision(profiles[1])
    assert revision > 0

    def failed_discovery(*_):
        raise ValueError("Catalog read failed after credential recovery")

    monkeypatch.setattr(service.connections, "accounts", failed_discovery)
    service.refresh_account(accounts[1].key)
    assert finish(service).problem
    assert service.credential_revision(profiles[1]) > revision
    assert service.credential_revision(profiles[0]) == 0
    service.close()


def test_discovery_isolates_corrupt_accounts_and_unreadable_api_keys(tmp_path, monkeypatch):
    service, secrets = create(tmp_path)
    # Exercise the on-disk record reader using only disposable plaintext fixture records.
    monkeypatch.setattr(
        "dantetl.translation.cloud.protected_store._protect", lambda data, **_: data
    )
    store = service.connections.store
    good, bad = ("account-" + c * 32 for c in "ab")
    store.write(good, {"access_token": "fixture", "scopes": [PLAN_SCOPE], "models": [["m", "M"]]})
    (store.directory / (bad + ".dat")).write_bytes(b"invalid")
    plan = CloudProfile("cloud-" + "e" * 32, "Plan", CloudProvider.CHATGPT, "m", account=good)
    broken_plan = replace(plan, key="cloud-" + "f" * 32, account=bad)
    service.catalog.write((plan, broken_plan))
    profiles = [CloudProfile.new(CloudProvider.DEEPSEEK) for _ in range(2)]
    for p in profiles:
        service.save(CloudDraft(p, "", "fixture"))
        assert finish(service).saved
    saved = service.catalog.read()
    bad_key = saved[-1].credential_key
    legacy = CloudProfile.new(CloudProvider.OPENAI)
    service.catalog.write((*saved, legacy))

    def read(key):
        if key in (bad_key, legacy.key):
            raise ValueError("sensitive fixture diagnostic")
        return secrets.values.get(key)

    monkeypatch.setattr(secrets, "read", read)
    fresh = CloudServices(
        CloudProfiles(service.catalog.path), lambda: secrets, ChatGPTConnections(store)
    )
    fresh.start()
    snapshot = finish(fresh)
    assert snapshot.profiles == (*saved, legacy)
    assert snapshot.available == {profiles[0].key, plan.key}
    assert [a.key for a in snapshot.accounts] == [good]
    assert "ChatGPT" in snapshot.problem and "API key" in snapshot.problem
    assert "sensitive" not in snapshot.message and "sensitive" not in snapshot.problem
    assert (store.directory / (bad + ".dat")).read_bytes() == b"invalid"
