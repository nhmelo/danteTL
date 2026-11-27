import threading
from unittest.mock import Mock

import pytest
from ownership_fixtures import settle

from dantetl.application.model_inventory import InventoryData, ModelInventory
from dantetl.translation.backend import BackendChoice


def test_refresh_coalesces_and_late_inventory_cannot_replace_newer_request():
    entered, release = threading.Event(), threading.Event()
    calls = []
    old = InventoryData(prepared=frozenset({"old"}))
    new = InventoryData(prepared=frozenset({"new"}))

    def inspect():
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
            return old
        return new

    owner = ModelInventory(inspect, lambda key: None, interval=100)
    owner.advance(1)
    assert entered.wait(1)
    for _ in range(100):
        owner.invalidate()
        owner.advance(1)
        assert not owner.snapshot().loaded
    release.set()
    settle(owner, lambda: owner.snapshot().loaded)
    assert owner.snapshot().data == new
    assert len(calls) == 2 and threading.get_ident() not in calls
    owner.close()
    settle(owner, lambda: owner.finished)


def test_cached_queries_and_ticks_do_no_discovery_or_receipt_io():
    calls = []
    initial = InventoryData(prepared=frozenset({"receipt"}))
    owner = ModelInventory(lambda: calls.append("scan") or initial, calls.append, initial=initial)
    choice = BackendChoice("Model", Mock(), "key", preparation_id="receipt")
    owner.close()
    for _ in range(10_000):
        owner.advance(0)
        assert owner.prepared(choice)
        assert owner.snapshot().data is initial
    assert not calls


def test_source_change_hides_previous_catalog_and_discards_its_pending_scan():
    entered, release = threading.Event(), threading.Event()
    previous = InventoryData(prepared=frozenset({"old"}))
    current = InventoryData(prepared=frozenset({"new"}))

    def inspect_previous():
        entered.set()
        assert release.wait(2)
        return previous

    owner = ModelInventory(inspect_previous, lambda key: None, initial=previous)
    try:
        owner.advance(1)
        assert entered.wait(1)
        owner.set_inspector(lambda: current)
        snapshot = owner.snapshot()
        assert not snapshot.loaded and snapshot.data != previous
        release.set()
        settle(owner, lambda: owner.snapshot().loaded)
        assert owner.snapshot().data == current
    finally:
        release.set()
        owner.close()
        settle(owner, lambda: owner.finished)


@pytest.mark.parametrize("failed", [False, True])
def test_default_fallback_waits_after_slow_scan_then_discovers_external_change(failed):
    entered, release = threading.Event(), threading.Event()
    calls = []
    current = InventoryData(prepared=frozenset({"original"}))

    def inspect():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        if failed and len(calls) == 1:
            raise OSError("temporary discovery failure")
        return current

    owner = ModelInventory(inspect, lambda key: None)
    try:
        owner.advance(1)
        assert entered.wait(1)
        owner.advance(100)  # The scan took longer than the fallback interval.
        assert len(calls) == 1
        release.set()
        settle(owner, lambda: owner.snapshot().loaded or owner.snapshot().problem, now=100)
        assert owner.finished and len(calls) == 1
        for i in range(10_000):
            owner.advance(100 + i * 0.0029)
        assert owner.finished and len(calls) == 1
        current = InventoryData(prepared=frozenset({"external edit"}))
        owner.advance(130)
        settle(owner, lambda: owner.snapshot().data == current, now=130)
        assert owner.finished and len(calls) == 2
        assert not owner.snapshot().problem
        # Explicit refresh still bypasses the fallback deadline.
        current = InventoryData(prepared=frozenset({"download complete"}))
        owner.invalidate()
        owner.advance(131)
        settle(owner, lambda: owner.snapshot().data == current, now=131)
        assert owner.finished and len(calls) == 3
    finally:
        release.set()
        owner.close()
        settle(owner, lambda: owner.finished, now=131)


def test_close_drains_receipts_and_scan_failure_keeps_last_good_snapshot():
    writes = []
    initial = InventoryData(prepared=frozenset({"prior"}))

    def inspect():
        raise OSError("temporarily inaccessible")

    owner = ModelInventory(
        inspect, lambda key: writes.append((key, threading.get_ident())), initial=initial
    )
    owner.record_prepared("new")
    owner.record_prepared("new")
    owner.close()
    settle(owner, lambda: owner.finished)
    assert [key for key, _ in writes] == ["new"]
    assert writes[0][1] != threading.get_ident()
    assert owner.snapshot().data is initial
    assert owner.snapshot().problem == "temporarily inaccessible"


def test_failed_receipt_does_not_discard_inventory_or_block_shutdown():
    def record(key):
        raise OSError("read-only state")

    initial = InventoryData()
    owner = ModelInventory(lambda: initial, record)
    owner.record_prepared("new")
    owner.close()
    settle(owner, lambda: owner.finished)
    assert owner.snapshot().loaded
    assert "read-only state" in owner.snapshot().problem
    owner.resume()
    owner.advance(10)
    settle(owner, lambda: owner._worker is None, now=10)
    assert "read-only state" in owner.snapshot().problem
    owner.close()
    settle(owner, lambda: owner.finished)
