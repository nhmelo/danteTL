import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.cache_values import CacheInventory


def test_legacy_row_count_is_seeded_once_and_eviction_stays_exact(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE translations "
            "(key TEXT PRIMARY KEY, english TEXT NOT NULL, used INTEGER NOT NULL)"
        )
        db.executemany(
            "INSERT INTO translations VALUES (?,?,?)", [("a", "A", 1), ("b", "B", 2), ("c", "C", 3)]
        )
    db.close()
    cache = TranslationCache(path, disk_limit=3)
    try:
        assert cache.inventory().total == 3
        cache.put("d", "D")
        assert cache.get("a") is None and cache.inventory().total == 3
        assert cache.inventory(clear=True).total == 0
    finally:
        cache.close()


def test_cache_key_separates_backend_settings_and_exact_source() -> None:
    assert translation_key("v1", "猫") != translation_key("v2", "猫")
    assert translation_key("v1", "猫") != translation_key("v1", "犬")
    assert translation_key("v1:beam2", "猫") != translation_key("v1:beam5", "猫")


def test_memory_and_sqlite_limits_and_persistence(tmp_path) -> None:
    path = tmp_path / "cache.sqlite3"
    cache = TranslationCache(path, memory_limit=2, disk_limit=3)
    for index in range(10):
        cache.put(str(index), f"Translation {index}")
    assert len(cache._memory) == 2
    cache.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 3
    restored = TranslationCache(path)
    assert restored.get("9") == "Translation 9"
    assert restored.get("0") is None
    restored.close()


def test_corrupt_disk_cache_keeps_translation_usable_without_deleting_user_file(tmp_path) -> None:
    path = tmp_path / "bad.sqlite3"
    path.write_bytes(b"not a database")
    cache = TranslationCache(path)
    assert cache.warning
    cache.put("key", "English")
    assert cache.get("key") == "English"
    cache.close()
    assert path.read_bytes() == b"not a database"


def test_shared_game_entries_survive_clearing_the_other_game_and_reopening(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache = TranslationCache(path)
    try:
        cache.lookup_pending(("shared", "a"), reset=False, game="A")
        cache.put("shared", "Settings", scope=cache.scope)
        cache.put("a", "Map", scope=cache.scope)
        assert cache.lookup_pending(("shared", "b"), reset=False, game="B")["shared"] == "Settings"
        cache.put("b", "Inventory", scope=cache.scope)
        assert cache.inventory() == CacheInventory(3, (("A", 2), ("B", 2)))
        assert cache.inventory(clear=True, game="A") == CacheInventory(2, (("B", 2),))
        assert cache.get("a") is None and cache.get("shared") == "Settings"
    finally:
        cache.close()
    reopened = TranslationCache(path)
    try:
        assert reopened.inventory() == CacheInventory(2, (("B", 2),))
        assert reopened.inventory(clear=True) == CacheInventory(0, ())
    finally:
        reopened.close()


def test_misses_and_eviction_leave_no_unowned_game_titles_on_disk(tmp_path):
    path = tmp_path / "cache"
    cache = TranslationCache(path, disk_limit=1)
    try:
        for i in range(20):
            cache.lookup_pending(("miss",), reset=True, game=f"untranslated-{i}")
        for game in ("old", "current"):
            cache.lookup_pending((game,), reset=True, game=game)
            cache.put(game, game, scope=cache.scope)
    finally:
        cache.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM cache_games").fetchall() == [("current",)]
    db.close()


def test_reappearing_owned_hit_does_not_invalidate_other_connections(tmp_path, monkeypatch):
    # Keep the separate, periodic recency flush outside this replay.
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    path = tmp_path / "cache"
    cache = TranslationCache(path)
    try:
        cache.lookup_pending(("key",), reset=False, game="A")
        cache.put("key", "Settings", scope=cache.scope)
        with closing(sqlite3.connect(path)) as other:
            version = other.execute("PRAGMA data_version").fetchone()
            for _ in range(20):
                cache.lookup_pending((), reset=False, game="A")
                assert cache.lookup_pending(("key",), reset=False, game="A")["key"] == "Settings"
            assert other.execute("PRAGMA data_version").fetchone() == version
            assert cache.lookup_pending(("key",), reset=False, game="B")["key"] == "Settings"
            assert other.execute("PRAGMA data_version").fetchone() != version
        assert cache.inventory() == CacheInventory(1, (("A", 1), ("B", 1)))
    finally:
        cache.close()


def test_unnamed_capture_uses_the_same_ownership_and_clear_rules(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    try:
        cache.lookup_pending(("shared",), reset=False, game="A")
        cache.put("shared", "Settings", scope=cache.scope)
        assert cache.lookup_pending(("shared", "unnamed"), reset=False)["shared"] == "Settings"
        cache.put("unnamed", "Map", scope=cache.scope)
        assert cache.inventory() == CacheInventory(2, (("", 2), ("A", 1)))
        assert cache.inventory(clear=True, game="") == CacheInventory(1, (("A", 1),))
    finally:
        cache.close()


def test_clear_excludes_old_inflight_writes_and_invalidates_memory_and_pending_hits(tmp_path):
    path = tmp_path / "cache"
    worker, other, manager = (TranslationCache(path) for _ in range(3))
    try:
        worker.lookup_pending(("done", "inflight"), reset=False, game="A")
        old_scope = worker.scope
        worker.put("done", "Done", scope=old_scope)
        other.lookup_pending(("other",), reset=False, game="B")
        assert worker.get("done") == "Done"
        manager.inventory(clear=True, game="A")
        worker.put("inflight", "Late", scope=old_scope)
        other.put("other", "Other", scope=other.scope)
        assert worker.get("done") is None
        assert worker.lookup_pending(("done", "inflight"), reset=False, game="A") == {
            "done": None,
            "inflight": None,
        }
        assert worker.scope != old_scope
        assert manager.inventory() == CacheInventory(1, (("B", 1),))
        manager.inventory(clear=True)
        other.put("late-after-full-clear", "Other", scope=other.scope)
        assert other.lookup_pending(("other",), reset=False, game="B") == {"other": None}
        assert manager.inventory() == CacheInventory(0, ())
    finally:
        worker.close()
        other.close()
        manager.close()


def test_migration_removes_old_miss_only_names_and_preserves_linked_entries(tmp_path):
    path = tmp_path / "cache"
    cache = TranslationCache(path)
    cache.lookup_pending(("saved",), reset=True, game="played")
    cache.put("saved", "Settings", scope=cache.scope)
    cache.close()
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO cache_games(name) VALUES ('miss only')")
        db.execute("PRAGMA user_version=1")
    db.close()
    reopened = TranslationCache(path)
    try:
        assert reopened.get("saved") == "Settings"
        assert reopened.inventory() == CacheInventory(1, (("played", 1),))
    finally:
        reopened.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM cache_games").fetchall() == [("played",)]
    db.close()


def test_a_direct_reader_adopts_another_owners_clear_once_and_keeps_its_memory(tmp_path):
    path = tmp_path / "cache"
    reader, manager = TranslationCache(path), TranslationCache(path)
    evicting = TranslationCache(path, disk_limit=1)
    try:
        manager.inventory(clear=True)
        reader.put("label", "Settings")
        assert reader.get("label") == "Settings"
        # Another owner's eviction leaves the reader's memory hit valid. Only a clear,
        # which a direct reader has already adopted, may discard it.
        evicting.put("other", "Other")
        assert reader.get("label") == "Settings"
    finally:
        for cache in (reader, manager, evicting):
            cache.close()


def test_new_work_on_the_clearing_connection_obtains_a_fresh_scope(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    try:
        cache.lookup_pending(("before",), reset=True, game="A")
        old = cache.scope
        cache.inventory(clear=True)
        cache.lookup_pending(("after",), reset=True, game="A")
        cache.put("before", "Late", scope=old)
        cache.put("after", "Current", scope=cache.scope)
        assert cache.inventory() == CacheInventory(1, (("A", 1),))
        assert cache.get("before") is None
    finally:
        cache.close()


def test_eviction_removes_game_membership(tmp_path):
    cache = TranslationCache(tmp_path / "cache", disk_limit=1)
    try:
        for game in ("A", "B"):
            cache.lookup_pending((game,), reset=False, game=game)
            cache.put(game, "Label", scope=cache.scope)
        assert cache.inventory() == CacheInventory(1, (("B", 1),))
    finally:
        cache.close()


def test_memory_hits_update_disk_recency_before_eviction_and_close(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache = TranslationCache(path, disk_limit=2)
    cache.put("hot", "Often read")
    cache.put("cold", "Never read")
    assert cache.get("hot") == "Often read"
    cache.put("new", "New result")
    cache.close()
    reopened = TranslationCache(path, disk_limit=2)
    assert reopened.get("hot") == "Often read"
    assert reopened.get("cold") is None
    reopened.close()
    # The read immediately before close must also survive a later owner's write.
    writer = TranslationCache(path, disk_limit=2)
    writer.put("next", "Next result")
    writer.close()
    reopened = TranslationCache(path, disk_limit=2)
    assert reopened.get("hot") == "Often read" and reopened.get("new") is None
    reopened.close()


def test_two_owners_account_for_updates_and_evictions_in_one_transaction(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first = TranslationCache(path, disk_limit=3)
    second = TranslationCache(path, disk_limit=3)
    try:
        for index in range(30):
            owner = first if index % 2 else second
            owner.put(str(index), f"English {index}")
            owner.put(str(index), f"English {index} updated")
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == min(
                    index + 1, 3
                )
        assert first.get("29") == "English 29 updated"
        assert second.get("28") == "English 28 updated"
    finally:
        first.close()
        second.close()


def test_concurrent_owner_writes_keep_capacity_and_latest_values(tmp_path):
    path = tmp_path / "shared.sqlite3"
    ready = threading.Barrier(2)

    def write(lane):
        cache = TranslationCache(path, disk_limit=12)
        try:
            ready.wait(timeout=5)
            for index in range(50):
                cache.put(f"{lane}/{index}", f"{lane} English {index}")
                cache.put(f"{lane}/{index}", f"{lane} revised {index}")
                with sqlite3.connect(path) as db:
                    assert db.execute("SELECT COUNT(*) FROM translations").fetchone()[0] <= 12
            assert not cache.warning
        finally:
            cache.close()

    with ThreadPoolExecutor(max_workers=2) as owners:
        list(owners.map(write, ("short", "long")))
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT key, english FROM translations").fetchall()
    assert len(rows) == 12
    assert all(
        english == f"{key.split('/')[0]} revised {key.split('/')[1]}" for key, english in rows
    )
