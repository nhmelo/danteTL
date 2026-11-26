"""Bounded native blur surfaces beneath the click-through Qt overlay."""

import ctypes
import time
from ctypes import wintypes as wt

from dantetl.capture.windows import Windows
from dantetl.domain import MAX_OVERLAY_LABELS, Rect
from dantetl.i18n import tr


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
        style = 0x080800A0 | (user.GetWindowLongPtrW(overlay, -20) & 8)
        # SS_BLACKRECT paints COLOR_3DDKSHADOW, which the color key makes transparent.
        hwnd = user.CreateWindowExW(
            style, "Static", "", 0x80000004, 0, 0, 0, 0, None, None, None, None
        )
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handles.append(hwnd)
        # Request Windows' small rounded corners once per surface. Qt draws the border.
        # Older compositors may decline these cosmetic attributes and still provide blur.
        dwm = self._windows.dwm
        dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(wt.DWORD(3)), 4)  # DWMWCP_ROUNDSMALL
        dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(wt.DWORD(0xFFFFFFFE)), 4)
        if not user.SetLayeredWindowAttributes(hwnd, user.GetSysColor(21), 255, 1):
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
                    0x50,
                ):  # SWP_NOACTIVATE | SWP_SHOWWINDOW, immediately below the text.
                    raise ctypes.WinError(ctypes.get_last_error())
                self._visible[box] = hwnd
                if time.perf_counter() >= deadline:
                    break
        except (OSError, RuntimeError):
            self.clear()
            self._available = False
            return None
        return frozenset(requested[box] for box in self._visible)
