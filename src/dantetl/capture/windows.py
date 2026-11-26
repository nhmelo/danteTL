"""Own Win32 window identity, physical geometry, and overlay window configuration."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import ntpath
import os
from collections.abc import Callable
from ctypes import wintypes as wt
from functools import cached_property

from dantetl.domain import Rect, WindowGeometry, WindowTarget
from dantetl.i18n import tr

logger = logging.getLogger(__name__)


class WindowUnavailable(RuntimeError):
    """The exact selected window no longer exists or cannot be inspected."""


class WindowClosed(WindowUnavailable):
    """The selected HWND no longer identifies the original process."""


def _rect(value: wt.RECT) -> Rect:
    return Rect(value.left, value.top, value.right - value.left, value.bottom - value.top)


class Windows:
    """Small native boundary. All positions are per-monitor-aware physical pixels."""

    def __init__(self) -> None:
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
        self._enum_callback = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        position_args = [
            wt.HWND,
            wt.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wt.UINT,
        ]
        signatures = {
            "EnumWindows": ([self._enum_callback, wt.LPARAM], wt.BOOL),
            "IsWindow": ([wt.HWND], wt.BOOL),
            "IsWindowVisible": ([wt.HWND], wt.BOOL),
            "IsIconic": ([wt.HWND], wt.BOOL),
            "GetWindowTextLengthW": ([wt.HWND], ctypes.c_int),
            "GetWindowTextW": ([wt.HWND, wt.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetWindowThreadProcessId": ([wt.HWND, ctypes.POINTER(wt.DWORD)], wt.DWORD),
            "GetWindowRect": ([wt.HWND, ctypes.POINTER(wt.RECT)], wt.BOOL),
            "GetClientRect": ([wt.HWND, ctypes.POINTER(wt.RECT)], wt.BOOL),
            "ClientToScreen": ([wt.HWND, ctypes.POINTER(wt.POINT)], wt.BOOL),
            "GetForegroundWindow": ([], wt.HWND),
            "GetDesktopWindow": ([], wt.HWND),
            "GetShellWindow": ([], wt.HWND),
            "GetDpiForWindow": ([wt.HWND], wt.UINT),
            "GetWindowLongPtrW": ([wt.HWND, ctypes.c_int], ctypes.c_ssize_t),
            "SetWindowLongPtrW": ([wt.HWND, ctypes.c_int, ctypes.c_ssize_t], ctypes.c_ssize_t),
            "SetWindowDisplayAffinity": ([wt.HWND, wt.DWORD], wt.BOOL),
            "GetWindowDisplayAffinity": ([wt.HWND, ctypes.POINTER(wt.DWORD)], wt.BOOL),
            "SetWindowPos": (position_args, wt.BOOL),
            "SetProcessDpiAwarenessContext": ([ctypes.c_void_p], wt.BOOL),
        }
        overlay_order = {
            "GetWindow": ([wt.HWND, wt.UINT], wt.HWND),
            "BeginDeferWindowPos": ([ctypes.c_int], wt.HANDLE),
            "DeferWindowPos": ([wt.HANDLE, *position_args], wt.HANDLE),
            "EndDeferWindowPos": ([wt.HANDLE], wt.BOOL),
        }
        self.supports_background_overlay = all(hasattr(self.user, name) for name in overlay_order)
        if self.supports_background_overlay:
            signatures.update(overlay_order)
        for name, (args, result) in signatures.items():
            function = getattr(self.user, name)
            function.argtypes = args
            function.restype = result
        self.dwm.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        self.dwm.DwmGetWindowAttribute.restype = ctypes.c_long

    @cached_property
    def _process_api(self) -> ctypes.WinDLL:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        kernel.OpenProcess.restype = wt.HANDLE
        kernel.CloseHandle.argtypes = [wt.HANDLE]
        kernel.CloseHandle.restype = wt.BOOL
        kernel.QueryFullProcessImageNameW.argtypes = [
            wt.HANDLE,
            wt.DWORD,
            wt.LPWSTR,
            ctypes.POINTER(wt.DWORD),
        ]
        kernel.QueryFullProcessImageNameW.restype = wt.BOOL
        return kernel

    @cached_property
    def _icon_api(self) -> ctypes.WinDLL:
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.ExtractIconExW.argtypes = [
            wt.LPCWSTR,
            ctypes.c_int,
            ctypes.POINTER(wt.HICON),
            ctypes.POINTER(wt.HICON),
            wt.UINT,
        ]
        shell.ExtractIconExW.restype = wt.UINT
        self.user.DestroyIcon.argtypes = [wt.HICON]
        self.user.DestroyIcon.restype = wt.BOOL
        return shell

    def enable_dpi_awareness(self) -> None:
        # Must precede QApplication/window creation. Access denied means already set.
        if not self.user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            error = ctypes.get_last_error()
            if error != 5:
                raise ctypes.WinError(error)

    def process_id(self, hwnd: int) -> int:
        process_id = wt.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        return process_id.value

    def title(self, hwnd: int) -> str:
        length = self.user.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user.GetWindowTextW(hwnd, buffer, len(buffer))
        return buffer.value

    def is_target(self, target: WindowTarget) -> bool:
        return bool(self.user.IsWindow(target.hwnd)) and (
            self.process_id(target.hwnd) == target.process_id
        )

    def read_process_details[T](
        self, target: WindowTarget, copy: Callable[[int], T] | None
    ) -> tuple[str, T | None] | None:
        """Read a stable executable ID and optional icon together on the picker worker."""
        kernel = self._process_api
        process = kernel.OpenProcess(0x1000, False, target.process_id)  # Limited query access
        if not process:
            return None
        try:
            if not self.is_target(target):
                return None
            path = ctypes.create_unicode_buffer(32768)
            size = wt.DWORD(len(path))
            if not kernel.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
                return None
        finally:
            kernel.CloseHandle(process)
        normalized = ntpath.normcase(ntpath.normpath(path.value))
        executable_id = hashlib.sha256(normalized.encode("utf-8", "surrogatepass")).hexdigest()
        image = None
        if copy is not None:
            try:
                image = self._copy_icon(path.value, copy)
            except Exception:
                logger.debug("Window icon unavailable", exc_info=True)
        return (executable_id, image) if self.is_target(target) else None

    def _copy_icon[T](self, path: str, copy: Callable[[int], T]) -> T | None:
        shell = self._icon_api
        icon = wt.HICON()
        try:
            count = shell.ExtractIconExW(path, 0, ctypes.byref(icon), None, 1)
            return copy(icon.value) if count == 1 and icon.value else None
        finally:
            if icon.value:
                self.user.DestroyIcon(icon)

    def list_targets(self) -> list[WindowTarget]:
        targets: list[WindowTarget] = []

        def visit(hwnd: int, _: int) -> bool:
            if target := self._capture_target(hwnd):
                targets.append(target)
            return True

        callback = self._enum_callback(visit)
        if not self.user.EnumWindows(callback, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return sorted(targets, key=lambda item: (item.title.casefold(), item.hwnd))

    def _capture_target(self, hwnd: int) -> WindowTarget | None:
        if (
            not hwnd
            or hwnd in (self.user.GetDesktopWindow(), self.user.GetShellWindow())
            or not self.user.IsWindowVisible(hwnd)
        ):
            return None
        pid = self.process_id(hwnd)
        if not pid or pid == os.getpid() or self.user.GetWindowLongPtrW(hwnd, -20) & 0x80:
            return None  # Exclude our windows and floating tool windows.
        cloaked = wt.DWORD()
        if self.dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)):
            return None
        title = self.title(hwnd)
        if not title or cloaked.value:
            return None
        target = WindowTarget(hwnd, pid, title)
        try:
            geometry = self.geometry(target)
            return target if min(geometry.client.size) > 40 else None
        except WindowUnavailable:
            return None  # Enumeration/selection races naturally with windows closing.

    def foreground_target(self) -> WindowTarget:
        """Resolve only the currently focused eligible window, never an alternate selection."""
        target = self._capture_target(self.user.GetForegroundWindow())
        if target is None:
            raise WindowUnavailable(
                tr("Focus the game window, then press the capture hotkey again.")
            )
        if not self.is_foreground(target) or self.geometry(target).minimized:
            raise WindowUnavailable(
                tr("The focused window changed. Press the capture hotkey again.")
            )
        return target

    def _require_target(self, target: WindowTarget) -> None:
        if not self.is_target(target):
            raise WindowClosed(tr("The selected window has closed. Refresh and select a window."))

    def geometry(self, target: WindowTarget) -> WindowGeometry:
        self._require_target(target)
        client = wt.RECT()
        window = wt.RECT()
        origin = wt.POINT(0, 0)
        if not (
            self.user.GetClientRect(target.hwnd, ctypes.byref(client))
            and self.user.ClientToScreen(target.hwnd, ctypes.byref(origin))
            and self.user.GetWindowRect(target.hwnd, ctypes.byref(window))
        ):
            self._require_target(target)
            raise WindowUnavailable(tr("Could not read the selected window's geometry."))
        visible = wt.RECT()
        result = self.dwm.DwmGetWindowAttribute(
            target.hwnd, 9, ctypes.byref(visible), ctypes.sizeof(visible)
        )  # DWMWA_EXTENDED_FRAME_BOUNDS
        return WindowGeometry(
            client=Rect(origin.x, origin.y, client.right - client.left, client.bottom - client.top),
            visible_bounds=_rect(visible if result == 0 else window),
            window_bounds=_rect(window),
            dpi=self.user.GetDpiForWindow(target.hwnd) or 96,
            minimized=bool(self.user.IsIconic(target.hwnd)),
        )

    def is_foreground(self, target: WindowTarget) -> bool:
        return self.user.GetForegroundWindow() == target.hwnd and self.is_target(target)

    def configure_overlay(self, hwnd: int) -> None:
        # Layered + transparent gives cross-process mouse pass-through. Qt's widget
        # mouse transparency alone only affects widgets within our own application.
        style = self.user.GetWindowLongPtrW(hwnd, -20)
        ctypes.set_last_error(0)
        result = self.user.SetWindowLongPtrW(hwnd, -20, style | 0x80000 | 0x20 | 0x08000000 | 0x80)
        if result == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())

    def set_overlay_capture(self, hwnd: int, included: bool) -> None:
        # Capture targets one game HWND, so the independent overlay is never OCR input.
        # WDA_NONE / WDA_EXCLUDEFROMCAPTURE affect desktop capture of this surface only.
        if not self.user.SetWindowDisplayAffinity(hwnd, 0 if included else 0x11):
            raise RuntimeError(
                tr("Windows could not change whether screen capture includes the overlay.")
            )

    def place_overlay(
        self, hwnd: int, bounds: Rect | None, backdrops: frozenset[int] = frozenset()
    ) -> bool:
        return self._position_surfaces(hwnd, -1, bounds, backdrops)  # HWND_TOPMOST

    def _position_surfaces(
        self,
        hwnd: int,
        after: int,
        bounds: Rect | None,
        backdrops: frozenset[int],
        *,
        keep_order: bool = False,
    ) -> bool:
        flags = 0x0010 | (0x0004 if keep_order else 0)  # NOACTIVATE, optionally NOZORDER
        if bounds is None:
            flags |= 0x0003  # NOMOVE | NOSIZE
        area = bounds or Rect(0, 0, 0, 0)
        position = (hwnd, after, area.x, area.y, area.width, area.height, flags)
        if not backdrops or not self.supports_background_overlay:
            return bool(self.user.SetWindowPos(*position))
        # Commit the text and all visible blur windows in one screen update.
        batch = self.user.BeginDeferWindowPos(1 + len(backdrops))
        if not batch:
            return False
        # Each HWND needs the destination band explicitly. Using the text HWND as
        # the insert-after handle here would use its band before the batch commits.
        # Insert the text last so it ends up in front of the backdrops.
        for backdrop in backdrops:
            batch = self.user.DeferWindowPos(batch, backdrop, after, 0, 0, 0, 0, 0x0013)
            if not batch:
                return False
        batch = self.user.DeferWindowPos(batch, *position)
        return bool(batch and self.user.EndDeferWindowPos(batch))

    def align_to_game(
        self, hwnd: int, target: WindowTarget, bounds: Rect | None, backdrops: frozenset[int]
    ) -> bool:
        """Keep our text and visible blur surfaces directly above the exact game window."""
        cloaked = wt.DWORD()
        if (
            not self.supports_background_overlay
            or not self.is_target(target)
            or not self.user.IsWindowVisible(target.hwnd)
            or self.user.IsIconic(target.hwnd)
            or self.dwm.DwmGetWindowAttribute(
                target.hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
            )
            or cloaked.value
        ):
            return False

        game_topmost = bool(self.user.GetWindowLongPtrW(target.hwnd, -20) & 8)
        text_topmost = bool(self.user.GetWindowLongPtrW(hwnd, -20) & 8)
        # Hidden foreign helpers (including IME windows) can be inaccessible as
        # SetWindowPos insertion handles. They have no bearing on visible stacking.
        remaining = set(backdrops)
        above = target.hwnd
        ordered = False
        for _ in range(len(backdrops) + 256):
            above = self.user.GetWindow(above, 3)  # GW_HWNDPREV
            if not above:
                break
            if above == hwnd:
                ordered = not remaining and game_topmost == text_topmost
                if ordered:
                    break
            elif above in remaining:
                remaining.remove(above)
            elif self.user.IsWindowVisible(above):
                break
        else:
            return False  # No reliable placement on this pass through a changing stack.
        if ordered and bounds is None:
            return True
        # At the boundary between the two bands, inserting after a topmost window
        # can itself make the text topmost. Choose the top of the game's band.
        if ordered:
            above = 0  # Ignored with NOZORDER.
        elif not above or (not game_topmost and self.user.GetWindowLongPtrW(above, -20) & 8):
            above = -1 if game_topmost else (-2 if text_topmost else 0)
        return self._position_surfaces(
            hwnd, above, bounds, frozenset() if ordered else backdrops, keep_order=ordered
        )

    def overlay_properties(self, hwnd: int) -> tuple[int, int]:
        """Read back native flags for the live verification harness."""
        affinity = wt.DWORD()
        if not self.user.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)):
            raise ctypes.WinError(ctypes.get_last_error())
        return self.user.GetWindowLongPtrW(hwnd, -20), affinity.value
