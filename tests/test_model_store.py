import hashlib
import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
from download_fixtures import download_bytes

from dantetl.runtime import model_store as store
from dantetl.runtime.assets import ModelAsset


def bundle(tmp_path):
    asset = ModelAsset("fixture", "revision", "weights", hashlib.sha256(b"good").hexdigest(), 4)
    return store.ModelBundle("fixture", "Test", tmp_path, (asset,), "")


def test_download_is_explicit_atomic_verified_and_reuses_valid_files(tmp_path, monkeypatch):
    model = bundle(tmp_path)
    download = Mock(side_effect=download_bytes(b"good"))
    monkeypatch.setattr(store, "download_file", download)
    assert not model.present()
    download.assert_not_called()
    progress = []
    store.install_model(model, progress.append, threading.Event())
    assert model.present() and (tmp_path / "weights").read_bytes() == b"good"
    assert progress[-1].received == progress[-1].total == 4
    store.install_model(model, progress.append, threading.Event())
    download.assert_called_once()
    assert not tuple(tmp_path.glob("*.part"))


@pytest.mark.parametrize("payload", [b"bad", b"evil", b"too large"])
def test_bad_download_does_not_replace_existing_file(tmp_path, monkeypatch, payload):
    model = bundle(tmp_path)
    (tmp_path / "weights").write_bytes(b"old")
    monkeypatch.setattr(store, "download_file", download_bytes(payload))
    with pytest.raises(RuntimeError):
        store.install_model(model, lambda p: None, threading.Event())
    assert (tmp_path / "weights").read_bytes() == b"old"
    assert [p.read_bytes() for p in tmp_path.glob("*.part")] == (
        [payload] if len(payload) < 4 else []
    )


def test_cancel_after_last_byte_reuses_verified_partial_on_retry(tmp_path, monkeypatch):
    cancel = threading.Event()
    monkeypatch.setattr(store, "download_file", download_bytes(b"good"))

    def progress(value):
        if value.received:
            cancel.set()

    with pytest.raises(store.Cancelled):
        store.install_model(bundle(tmp_path), progress, cancel)
    assert not bundle(tmp_path).present()
    monkeypatch.setattr(store, "download_file", lambda *a: pytest.fail("Already downloaded"))
    store.install_model(bundle(tmp_path), lambda _: None, threading.Event())
    assert bundle(tmp_path).present()


def test_ocr_repair_uses_bundled_files_offline_and_preserves_app_resources(tmp_path, monkeypatch):
    bundled = tmp_path / "app/models"
    bundled.mkdir(parents=True)
    (bundled / "weights").write_bytes(b"good")
    primary = tmp_path / "AppData/models"
    primary.mkdir(parents=True)
    (primary / "weights").write_bytes(b"evil")  # Same size, incorrect checksum.
    model = replace(bundle(primary), key="ocr", bundled_directory=bundled)
    network = Mock(side_effect=AssertionError("Repair should use local bundled bytes"))
    monkeypatch.setattr(store, "download_file", network)
    progress = []
    store.install_model(model, progress.append, threading.Event())
    assert (primary / "weights").read_bytes() == b"good"
    assert (bundled / "weights").read_bytes() == b"good"
    assert any("Restoring" in p.message for p in progress)
    assert model.local_directory == primary
    network.assert_not_called()


def test_damaged_bundle_repairs_to_appdata_and_downloads_only_bad_asset(tmp_path, monkeypatch):
    bundled = tmp_path / "app/models"
    bundled.mkdir(parents=True)
    (bundled / "weights").write_bytes(b"evil")
    (bundled / "second").write_bytes(b"good")
    primary = tmp_path / "AppData/models"
    model = replace(bundle(primary), key="ocr", bundled_directory=bundled)
    model = replace(model, assets=(*model.assets, replace(model.assets[0], filename="second")))
    network = Mock(side_effect=download_bytes(b"good"))
    monkeypatch.setattr(store, "download_file", network)
    store.install_model(model, lambda p: None, threading.Event())
    network.assert_called_once()
    assert (primary / "weights").read_bytes() == (primary / "second").read_bytes() == b"good"
    assert model.local_directory == primary and model.present()
    assert (bundled / "weights").read_bytes() == b"evil"


def test_cancelled_local_repair_never_publishes_a_partial_file(tmp_path, monkeypatch):
    bundled = tmp_path / "app/models"
    bundled.mkdir(parents=True)
    (bundled / "weights").write_bytes(b"good")
    primary = tmp_path / "AppData/models"
    primary.mkdir(parents=True)
    (primary / "weights").write_bytes(b"evil")
    model = replace(bundle(primary), bundled_directory=bundled)
    cancel = threading.Event()
    monkeypatch.setattr(store, "download_file", lambda *a, **kw: pytest.fail("No network allowed"))
    with pytest.raises(store.Cancelled):
        store.install_model(model, lambda p: cancel.set() if p.received else None, cancel)
    assert (primary / "weights").read_bytes() == b"evil"
    assert (bundled / "weights").read_bytes() == b"good"
    store.install_model(model, lambda _: None, threading.Event())
    assert (primary / "weights").read_bytes() == b"good"


def test_worker_reports_failure_and_cancellation():
    def fail(progress, cancel):
        raise OSError("Disk full")

    operation = store.ModelOperation(fail)
    operation.start()
    operation._thread.join(2)
    assert operation.snapshot().done and operation.snapshot().error == "Disk full"


def test_receipt_is_runtime_and_machine_specific_and_only_written_after_success(
    tmp_path, monkeypatch
):
    model = bundle(tmp_path)
    monkeypatch.setattr(store, "preparation_file", lambda key: tmp_path / f"{key}.json")
    cuda = store.preparation_id(model, "cuda")
    vulkan = store.preparation_id(model, "vulkan")
    assert cuda != vulkan and not store.is_prepared(cuda)
    store.mark_prepared(cuda)
    assert store.is_prepared(cuda) and not store.is_prepared(vulkan)
    monkeypatch.setattr(store, "version", lambda _: "different-runtime")
    assert store.preparation_id(model, "cuda") != cuda


def test_delete_removes_only_pinned_downloads_and_keeps_open_files_whole(tmp_path):
    model = bundle(tmp_path)
    asset = model.assets[0]
    (tmp_path / "weights").write_bytes(b"good")
    (tmp_path / f".{asset.sha256}.part").write_bytes(b"go")
    (tmp_path / "notes.txt").write_text("user file")
    with (tmp_path / "weights").open("rb"), pytest.raises(RuntimeError, match="in use"):
        store.delete_model(model, lambda _: None, threading.Event())
    assert (tmp_path / "weights").read_bytes() == b"good"
    assert (tmp_path / f".{asset.sha256}.part").read_bytes() == b"go"
    progress = []
    store.delete_model(model, progress.append, threading.Event())
    assert not model.present() and progress[-1].message == "Model files deleted."
    assert sorted(p.name for p in tmp_path.iterdir()) == [".download.lock", "notes.txt"]
    store.delete_model(model, lambda _: None, threading.Event())  # Nothing left to delete.


def test_delete_waits_for_no_other_writer(tmp_path):
    model = bundle(tmp_path)
    (tmp_path / "weights").write_bytes(b"good")
    with store.file_lock(tmp_path / ".download.lock"), pytest.raises(OSError, match="downloading"):
        store.delete_model(model, lambda _: None, threading.Event())
    assert model.present()
