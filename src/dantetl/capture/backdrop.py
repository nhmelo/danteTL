"""Bounded native blur surfaces beneath the click-through Qt overlay."""

import ctypes
import time
from ctypes import wintypes as wt

from dantetl.capture.windows import (
    GWL_EXSTYLE,
    OVERLAY_EX_STYLE,
    SWP_NOACTIVATE,
    SWP_SHOWWINDOW,
    WS_EX_TOPMOST,
    Windows,
)
from dantetl.domain import MAX_OVERLAY_LABELS, Rect
from dantetl.i18n import tr

WS_POPUP = 0x80000000
SS_BLACKRECT = 0x00000004
COLOR_3DDKSHADOW = 21
LWA_COLORKEY = 0x1
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUNDSMALL = 3
DWMWA_BORDER_COLOR = 34
DWMWA_COLOR_NONE = 0xFFFFFFFE


class _AccentPolicy(ctypes.Structure):
    _fields_ = [
        ("state", wt.DWORD),
        ("flags", wt.DWORD),
        ("color", wt.DWORD),
        ("animation", wt.DWORD),
    ]


class _AttributeData(ctypes.Structure):
    _fields_ = [("attribute", ctypes.c_int), ("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class LabelBackdrops:
    """One small HWND per visible box. Window-wide accent blur ignores Qt paint masks."""

    def __init__(self, windows: Windows) -> None:
        self._windows = windows
        user = windows.user
        windows.dwm.DwmSetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        windows.dwm.DwmSetWindowAttribute.restype = ctypes.c_long
        self._accent = getattr(user, "SetWindowCompositionAttribute", None)
        if self._accent:
            self._accent.argtypes = [wt.HWND, ctypes.POINTER(_AttributeData)]
            self._accent.restype = wt.BOOL
        user.CreateWindowExW.argtypes = [
            wt.DWORD,
            wt.LPCWSTR,
            wt.LPCWSTR,
            wt.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wt.HWND,
            wt.HMENU,
            wt.HINSTANCE,
            ctypes.c_void_p,
        ]
        user.CreateWindowExW.restype = wt.HWND
        user.GetSysColor.argtypes = [ctypes.c_int]
        user.GetSysColor.restype = wt.DWORD
        user.SetLayeredWindowAttributes.argtypes = [wt.HWND, wt.COLORREF, wt.BYTE, wt.DWORD]
        user.SetLayeredWindowAttributes.restype = wt.BOOL
        user.DestroyWindow.argtypes = [wt.HWND]
        user.DestroyWindow.restype = wt.BOOL
        user.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
        user.ShowWindow.restype = wt.BOOL
        self._handles: list[int] = []
        self._visible: dict[Rect, int] = {}
        self._available = self._accent is not None

    def clear(self) -> None:
        for hwnd in self._handles:
            self._windows.user.DestroyWindow(hwnd)
        self._handles.clear()
        self._visible.clear()

    @property
    def visible_handles(self) -> frozenset[int]:
        return frozenset(self._visible.values())

    def _create(self, included: bool, overlay: int) -> int:
        user = self._windows.user
        # Use the text window's topmost band for new surfaces.
        style = OVERLAY_EX_STYLE | (user.GetWindowLongPtrW(overlay, GWL_EXSTYLE) & WS_EX_TOPMOST)
        # SS_BLACKRECT paints COLOR_3DDKSHADOW, which the color key makes transparent.
        hwnd = user.CreateWindowExW(
            style, "Static", "", WS_POPUP | SS_BLACKRECT, 0, 0, 0, 0, None, None, None, None
        )
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handles.append(hwnd)
        # Request Windows' small rounded corners once per surface. Qt draws the border.
        # Older compositors may decline these cosmetic attributes and still provide blur.
        dwm = self._windows.dwm
        corner, border = wt.DWORD(DWMWCP_ROUNDSMALL), wt.DWORD(DWMWA_COLOR_NONE)
        dwm.DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner), 4)
        dwm.DwmSetWindowAttribute(hwnd, DWMWA_BORDER_COLOR, ctypes.byref(border), 4)
        key = user.GetSysColor(COLOR_3DDKSHADOW)
        if not user.SetLayeredWindowAttributes(hwnd, key, 255, LWA_COLORKEY):
            raise ctypes.WinError(ctypes.get_last_error())
        # WCA_ACCENT_POLICY / ACCENT_ENABLE_BLURBEHIND, with no color tint.
        policy = _AccentPolicy(3, 0, 0, 0)
        attr = _AttributeData(
            19, ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p), ctypes.sizeof(policy)
        )
        if self._accent is None or not self._accent(hwnd, ctypes.byref(attr)):
            raise OSError(tr("Windows declined the native blur effect."))
        self._windows.set_overlay_capture(hwnd, included)
        return hwnd

    def place(
        self, overlay: int, client: Rect, boxes: tuple[Rect, ...], *, included: bool
    ) -> frozenset[Rect] | None:
        """Preserve native surfaces at unchanged rectangles, including across reordering."""
        if not self._available or len(boxes) > MAX_OVERLAY_LABELS:
            self.clear()
            return None
        requested = {Rect(client.x + b.x, client.y + b.y, b.width, b.height): b for b in boxes}
        deadline = time.perf_counter() + 0.005
        try:
            for box in self._visible.keys() - requested.keys():
                self._windows.user.ShowWindow(self._visible.pop(box), 0)
            used = set(self._visible.values())
            spare = [hwnd for hwnd in self._handles if hwnd not in used]
            for box in requested:
                if box in self._visible:
                    continue
                hwnd = spare.pop() if spare else self._create(included, overlay)
                if not self._windows.user.SetWindowPos(
                    hwnd,
                    overlay,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    SWP_NOACTIVATE | SWP_SHOWWINDOW,
                ):  # Immediately below the text.
                    raise ctypes.WinError(ctypes.get_last_error())
                self._visible[box] = hwnd
                if time.perf_counter() >= deadline:
                    break
        except (OSError, RuntimeError):
            self.clear()
            self._available = False
            return None
        return frozenset(requested[box] for box in self._visible)
