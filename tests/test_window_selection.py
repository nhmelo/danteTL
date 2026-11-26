import ctypes
import os
from unittest.mock import Mock

import pytest

from dantetl.capture.windows import Windows, WindowUnavailable
from dantetl.domain import Rect, WindowGeometry, WindowTarget


@pytest.fixture
def windows():
    value = Windows.__new__(Windows)
    value.supports_background_overlay = True
    value.user = Mock()
    value.user.GetDesktopWindow.return_value = 1
    value.user.GetShellWindow.return_value = 2
    value.user.GetForegroundWindow.return_value = 10
    value.user.IsWindowVisible.return_value = True
    value.user.GetWindowLongPtrW.return_value = 0
    value.dwm = Mock()
    value.dwm.DwmGetWindowAttribute.return_value = 0
    value.process_id = Mock(return_value=os.getpid() + 100)
    value.title = Mock(return_value="Our game fixture")
    rect = Rect(0, 0, 800, 600)
    value.geometry = Mock(return_value=WindowGeometry(rect, rect, rect, 96))
    return value


def test_focused_target_has_exact_identity_and_rejects_focus_races(windows):
    target = windows.foreground_target()
    assert target.hwnd == 10 and target.process_id == os.getpid() + 100
    windows.user.GetForegroundWindow.side_effect = [10, 11]
    with pytest.raises(WindowUnavailable, match="changed"):
        windows.foreground_target()
    windows.user.GetForegroundWindow.side_effect = None
    windows.process_id.side_effect = [os.getpid() + 100, os.getpid() + 101]
    with pytest.raises(WindowUnavailable, match="changed"):
        windows.foreground_target()


def test_overlay_skips_inaccessible_hidden_windows_in_visible_stack(windows):
    game, text, blur, cover, ime = 10, 20, 30, 40, 50
    target = WindowTarget(game, windows.process_id.return_value, "Game")
    above = {game: ime, ime: cover}
    windows.user.GetWindow.side_effect = lambda hwnd, _: above[hwnd]
    windows.user.IsWindowVisible.side_effect = lambda hwnd: hwnd != ime
    windows.user.IsIconic.return_value = False
    positions = []

    def place(batch, hwnd, after, *args):
        positions.append((hwnd, after))
        return batch if after != ime else None

    windows.user.DeferWindowPos.side_effect = place
    assert windows.align_to_game(text, target, Rect(0, 0, 800, 600), frozenset({blur}))
    assert positions == [(blur, cover), (text, cover)]
    windows.user.EndDeferWindowPos.assert_called_once()
    positions.clear()
    above.update({game: blur, blur: ime, ime: text})
    assert windows.align_to_game(text, target, None, frozenset({blur}))
    assert not positions


def test_executable_favorite_id_survives_relaunch_and_distinguishes_installations(windows):
    windows._process_api = Mock()
    windows._icon_api = Mock()  # Executable has no icon.
    paths = iter([r"C:\Games\One\Game.exe", "c:/games/one/GAME.EXE", r"C:\Games\Two\Game.exe"])

    def query_path(process, flags, buffer, size):
        buffer.value = next(paths)
        return True

    windows._process_api.QueryFullProcessImageNameW.side_effect = query_path
    ids = []
    for hwnd, pid in ((10, 42), (20, 99), (30, 100)):
        windows.process_id.return_value = pid
        details = windows.read_process_details(WindowTarget(hwnd, pid, "Fixture"), Mock())
        assert details and details[1] is None
        ids.append(details[0])
    assert ids[0] == ids[1] and ids[0] != ids[2]
    assert windows._process_api.CloseHandle.call_count == 3


@pytest.mark.skipif(os.name != "nt", reason="Uses the Win32 enumeration callback type")
def test_optional_process_dlls_are_not_required_to_list_windows(monkeypatch):
    user = Mock()
    dlls = {"user32": user, "dwmapi": Mock()}

    def load(name, **_):
        if name not in dlls:
            raise OSError("Optional DLL unavailable")
        return dlls[name]

    monkeypatch.setattr(ctypes, "WinDLL", load)
    windows = Windows()
    assert windows.list_targets() == []
    user.EnumWindows.assert_called_once()


@pytest.mark.skipif(os.name != "nt", reason="Uses the Win32 enumeration callback type")
@pytest.mark.parametrize(
    "missing", ["GetWindow", "BeginDeferWindowPos", "DeferWindowPos", "EndDeferWindowPos"]
)
def test_missing_stacking_api_preserves_focused_overlay(monkeypatch, missing):
    user = Mock()
    delattr(user, missing)
    libraries = {"user32": user, "dwmapi": Mock()}
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_: libraries[name])
    windows = Windows()
    assert not windows.supports_background_overlay
    bounds = Rect(0, 0, 800, 600)
    assert not windows.align_to_game(20, WindowTarget(10, 2, "Game"), bounds, frozenset())
    user.SetWindowPos.assert_not_called()
    assert windows.place_overlay(20, bounds, frozenset({30}))
    user.SetWindowPos.assert_called_once_with(20, -1, 0, 0, 800, 600, 0x10)


def test_icon_conversion_failure_releases_handle_and_keeps_favorite_identity(windows):
    from ctypes import wintypes as wt

    def path(process, flags, buffer, size):
        buffer.value = r"C:\Games\Game.exe"
        return True

    def extract(path, index, large, small, count):
        ctypes.cast(large, ctypes.POINTER(wt.HICON))[0] = 123
        return 1

    windows._process_api = Mock()
    windows._process_api.QueryFullProcessImageNameW.side_effect = path
    windows._icon_api = Mock()
    windows._icon_api.ExtractIconExW.side_effect = extract
    target = WindowTarget(10, windows.process_id.return_value, "Fixture")
    expected = windows.read_process_details(target, None)
    assert expected and expected[1] is None
    assert (
        windows.read_process_details(target, Mock(side_effect=RuntimeError("Bad icon"))) == expected
    )
    assert windows.user.DestroyIcon.call_args.args[0].value == 123
    windows._icon_api.ExtractIconExW.side_effect = OSError("Shell unavailable")
    assert windows.read_process_details(target, Mock()) == expected
    windows.user.DestroyIcon.assert_called_once()


@pytest.mark.parametrize("hwnd", [0, 1, 2])
def test_no_focus_desktop_and_shell_are_not_capture_targets(windows, hwnd):
    windows.user.GetForegroundWindow.return_value = hwnd
    with pytest.raises(WindowUnavailable, match="Focus the game"):
        windows.foreground_target()


def test_own_windows_tool_windows_and_unreadable_windows_are_rejected(windows):
    windows.process_id.return_value = os.getpid()
    with pytest.raises(WindowUnavailable):
        windows.foreground_target()
    windows.process_id.return_value += 100
    windows.user.GetWindowLongPtrW.return_value = 0x80
    with pytest.raises(WindowUnavailable):
        windows.foreground_target()
    windows.user.GetWindowLongPtrW.return_value = 0
    windows.dwm.DwmGetWindowAttribute.return_value = -1
    with pytest.raises(WindowUnavailable):
        windows.foreground_target()
    windows.dwm.DwmGetWindowAttribute.return_value = 0
    windows.geometry.side_effect = WindowUnavailable("Closed")
    with pytest.raises(WindowUnavailable):
        windows.foreground_target()
