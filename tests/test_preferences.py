import json
import threading
import time
from dataclasses import replace

from dantetl.application.preference_values import Preferences
from dantetl.application.preferences import PreferenceStore, decode_preferences


def wait_idle(store):
    deadline = time.monotonic() + 2
    while not store.idle:
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_new_install_defaults_and_bad_individual_values_are_safe():
    defaults = Preferences()
    assert defaults.stabilize and defaults.automatic_size and defaults.group_dialogue
    assert defaults.capture_rate == 8 and defaults.model == "auto"
    decoded = decode_preferences(
        {
            "version": 1,
            "opacity": True,
            "capture_rate": -1,
            "text_size": "large",
            "show_overlay": "false",
            "include_overlay_in_capture": "false",
            "ocr_device": "arbitrary.exe",
            "output_language": "unknown",
            "interface_language": "unknown",
            "source_language": "unknown",
            "model": "C:/models/unsafe",
            "group_dialogue": False,
            "unused_future_field": "ignored",
        }
    )
    assert decoded == replace(defaults, group_dialogue=False)


def test_preferences_roundtrip_with_no_handles_paths_or_session_debug_state(tmp_path):
    path = tmp_path / "preferences.json"
    store = PreferenceStore(path)
    assert store.initial == Preferences() and not path.exists()
    desired = replace(
        store.initial,
        opacity=45,
        automatic_size=False,
        text_size=27,
        source_language="sr-Latn",
        ocr_device="cpu",
        capture_rate=400,
        model="gemma",
        translation=False,
        include_overlay_in_capture=False,
        overlay_hotkey="Ctrl+Shift+F7",
        capture_hotkey="",
        capture_focused_window=False,
    )
    store.save(desired)
    wait_idle(store)
    assert not store.error and PreferenceStore(path).initial == desired
    written = json.loads(path.read_text())
    assert written["version"] == 1 and written["model"] == "gemma"
    assert not any(key in written for key in ("hwnd", "debug", "paused", "advanced", "model_path"))
    assert not tuple(tmp_path.glob("*.tmp"))


def test_hotkeys_upgrade_old_settings_and_validate_each_field():
    assert decode_preferences({"version": 1}).overlay_hotkey == "Ctrl+Alt+F8"
    assert decode_preferences({"version": 1, "original_hotkey": "Ctrl+Shift+F10"}) == Preferences()
    assert decode_preferences({"version": 1}).capture_focused_window
    value = decode_preferences({"version": 1, "overlay_hotkey": "Win+F12", "capture_hotkey": ""})
    assert value.overlay_hotkey == "Ctrl+Alt+F8" and value.capture_hotkey == ""


def test_unknown_version_and_corrupt_file_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "preferences.json"
    for content in ('{"version": 99}', "{bad json"):
        path.write_text(content)
        store = PreferenceStore(path)
        assert store.error and store.initial == Preferences()
        store.save(replace(store.initial, opacity=60))
        assert store.idle and path.read_text() == content


def test_slow_disk_coalesces_changes_off_caller_and_keeps_final_value(tmp_path, monkeypatch):
    store = PreferenceStore(tmp_path / "preferences.json")
    entered, release = threading.Event(), threading.Event()
    writes, threads = [], []
    original = store._write

    def write(value):
        threads.append(threading.get_ident())
        if not writes:
            entered.set()
            assert release.wait(2)
        writes.append(value)
        original(value)

    monkeypatch.setattr(store, "_write", write)
    store.save(Preferences(opacity=20))
    assert entered.wait(1)
    for opacity in range(21, 100):
        store.save(Preferences(opacity=opacity))
    assert not store.idle
    release.set()
    wait_idle(store)
    assert [p.opacity for p in writes] == [20, 99]
    assert all(thread != threading.get_ident() for thread in threads)
    assert PreferenceStore(store.path).initial.opacity == 99


def test_failed_atomic_write_preserves_good_file_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "preferences.json"
    store = PreferenceStore(path)
    store.save(Preferences(opacity=40))
    wait_idle(store)
    original = store._write

    def fail(value):
        raise OSError("Disk full")

    monkeypatch.setattr(store, "_write", fail)
    store.save(Preferences(opacity=60))
    wait_idle(store)
    assert "Disk full" in store.error and PreferenceStore(path).initial.opacity == 40
    monkeypatch.setattr(store, "_write", original)
    store.save(Preferences(opacity=60))
    wait_idle(store)
    assert not store.error and PreferenceStore(path).initial.opacity == 60


def test_two_instances_merge_changed_fields_and_preserve_future_fields(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text('{"version":1,"future_option":42,"model":"future-model"}')
    first, second = PreferenceStore(path), PreferenceStore(path)
    first.save(replace(first.current, output_language="fr"))
    second.save(replace(second.current, opacity=42))
    wait_idle(first)
    wait_idle(second)
    first.save(replace(first.current, capture_rate=10))
    wait_idle(first)
    assert not first.error and not second.error
    assert PreferenceStore(path).initial == Preferences(
        output_language="fr", opacity=42, capture_rate=10
    )
    record = json.loads(path.read_text())
    assert (record["future_option"], record["model"]) == (42, "future-model")


def test_preferences_validate_current_file_again_before_saving(tmp_path):
    path = tmp_path / "prefs.json"
    store = PreferenceStore(path)
    path.write_text('{"version":2}')
    store.save(replace(store.current, opacity=60))
    wait_idle(store)
    assert store.error and json.loads(path.read_text()) == {"version": 2}


def test_favorite_edits_merge_across_instances_and_validate_saved_ids(tmp_path):
    path = tmp_path / "prefs.json"
    a, b, c = (letter * 64 for letter in "abc")
    path.write_text(json.dumps({"version": 1, "favorite_executables": [a, 42, "title", a]}))
    first, second = PreferenceStore(path), PreferenceStore(path)
    assert first.initial.favorite_executables == (a,)
    first.save(replace(first.initial, favorite_executables=(a, b)))
    second.save(replace(second.initial, favorite_executables=(c,)))
    wait_idle(first)
    wait_idle(second)
    assert PreferenceStore(path).initial.favorite_executables == (b, c)
