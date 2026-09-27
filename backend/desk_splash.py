"""Native start splash — shown before torch/FastAPI are imported."""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

EARLY_SPLASH_NAME = "_early_splash.html"
APP_VERSION = "0.1 alpha"


def ensure_stdio() -> None:
    """pythonw and some GUI hosts leave stdout/stderr None. Uvicorn then crashes on isatty()."""
    try:
        log = None

        def _log_stream():
            nonlocal log
            if log is None:
                from backend.paths import package_root

                path = package_root() / "models" / "vtm_noble.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open("a", encoding="utf-8")
            return log

        if getattr(sys, "stdout", None) is None:
            sys.stdout = _log_stream()
        if getattr(sys, "stderr", None) is None:
            sys.stderr = _log_stream()
    except Exception:
        import io

        if getattr(sys, "stdout", None) is None:
            sys.stdout = io.StringIO()
        if getattr(sys, "stderr", None) is None:
            sys.stderr = io.StringIO()


class WindowBridge:
    """pywebview js_api — close/min/max/drag. Keep this tiny.

    The GUI window must live on a private attribute. pywebview walks every
    public js_api field to build window.pywebview.api; a public ``window``
    pointer reaches WinForms ``native.AccessibilityObject.Bounds.Empty`` and
    recurses until the splash dies (cream flash, then gone).
    """

    def __init__(self) -> None:
        self._window = None

    def bind(self, window: object) -> None:
        self._window = window

    def close(self) -> None:
        window = self._window
        if window is None:
            return
        try:
            window.destroy()
        except Exception:
            pass

    def minimize(self) -> None:
        window = self._window
        if window is None:
            return
        try:
            window.minimize()
        except Exception:
            _win32_show(window, 6)  # SW_MINIMIZE

    def toggle_max(self) -> None:
        def _do() -> None:
            window = self._window
            hwnd = _window_hwnd(window)
            if hwnd <= 0:
                try:
                    if window is not None:
                        window.toggle_fullscreen()
                except Exception:
                    pass
                return
            try:
                import ctypes

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                if int(user32.IsZoomed(hwnd)):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                else:
                    user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            except Exception:
                pass
            hide_native_caption(window)

        _run_on_gui(self._window, _do)

    def start_drag(self) -> None:
        """Begin a native window move from a webview mousedown (WebView2 has no app-region)."""
        def _do() -> None:
            hwnd = _window_hwnd(self._window)
            if hwnd <= 0:
                return
            try:
                import ctypes

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                user32.ReleaseCapture()
                # SC_MOVE | HTCAPTION — works without a visible caption bar.
                user32.SendMessageW(hwnd, 0x0112, 0xF012, 0)
            except Exception:
                pass
            _pin_after_frame_change(self._window)

        _run_on_gui(self._window, _do, wait=True)

    def start_resize(self, edge: str = "bottomright") -> None:
        """js_api alias — HTML grips drive begin/move/end_resize instead of SC_SIZE."""
        self.begin_resize(edge)

    def begin_resize(self, edge: str = "bottomright") -> None:
        """Remember the live frame + cursor so later moves stay on the real window."""
        hit = _resize_edge_hit(edge)

        def _do() -> None:
            _begin_live_resize(self._window, hit, str(edge or ""))

        _run_on_gui(self._window, _do, wait=True)

    def move_resize(self) -> None:
        """Follow the cursor. Origin is the frame from begin_resize, not the page box."""
        def _do() -> None:
            _move_live_resize(self._window)

        _run_on_gui(self._window, _do)

    def end_resize(self) -> None:
        """Re-clip the HWND and dock WebView2 so the next grab hits the new edges."""
        def _do() -> None:
            _end_live_resize(self._window)

        _run_on_gui(self._window, _do, wait=True)

    def splash_ready(self) -> None:
        """JS: splash HTML has painted. Reveal only then — loaded fires on a blank WebView2."""
        reveal_splash(self._window)


_CHROME_RESIZABLE = False

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_CAPTION = 0x00C00000
WS_SYSMENU = 0x00080000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_CLIPSIBLINGS = 0x04000000
WS_CLIPCHILDREN = 0x02000000
WS_EX_APPWINDOW = 0x00040000
WS_EX_DLGMODALFRAME = 0x00000001
WS_EX_WINDOWEDGE = 0x00000100
WS_EX_CLIENTEDGE = 0x00000200
WS_EX_STATICEDGE = 0x00020000
WS_EX_EDGE = (
    WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE
)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
HTCLIENT = 1
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17
HTTRANSPARENT = -1
GWLP_WNDPROC = -4
_WMSZ_FROM_HIT = {
    HTLEFT: 1,
    HTRIGHT: 2,
    HTTOP: 3,
    HTTOPLEFT: 4,
    HTTOPRIGHT: 5,
    HTBOTTOM: 6,
    HTBOTTOMLEFT: 7,
    HTBOTTOMRIGHT: 8,
}
_RESIZE_EDGE_HITS = {
    "left": HTLEFT,
    "w": HTLEFT,
    "right": HTRIGHT,
    "e": HTRIGHT,
    "top": HTTOP,
    "n": HTTOP,
    "bottom": HTBOTTOM,
    "s": HTBOTTOM,
    "topleft": HTTOPLEFT,
    "nw": HTTOPLEFT,
    "topright": HTTOPRIGHT,
    "ne": HTTOPRIGHT,
    "bottomleft": HTBOTTOMLEFT,
    "sw": HTBOTTOMLEFT,
    "bottomright": HTBOTTOMRIGHT,
    "se": HTBOTTOMRIGHT,
}
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_WIN10 = 19
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWCP_DONOTROUND = 1
DWMWA_COLOR_NONE = 0xFFFFFFFE
SPLASH_FILL_RGB = (0x2B, 0x2B, 0x2B)
DESK_FILL_RGB = (0x2B, 0x2B, 0x2B)
# ctypes WndProc callbacks must stay alive. Keyed by HWND.
_NCCALC_HOOKS: dict[int, tuple[object, int]] = {}
_CHILD_HOOKS: dict[int, tuple[object, int, int]] = {}
# Live HTML resize: origin frame in physical pixels + starting cursor.
_LIVE_RESIZE: dict[str, int | str] = {}


def _ptr_fns(user32: object):
    import ctypes

    if ctypes.sizeof(ctypes.c_void_p) == 8:
        get_long = user32.GetWindowLongPtrW  # type: ignore[attr-defined]
        set_long = user32.SetWindowLongPtrW  # type: ignore[attr-defined]
        get_long.restype = ctypes.c_int64
        get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        set_long.restype = ctypes.c_int64
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
    else:
        get_long = user32.GetWindowLongW  # type: ignore[attr-defined]
        set_long = user32.SetWindowLongW  # type: ignore[attr-defined]
        get_long.restype = ctypes.c_int32
        get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        set_long.restype = ctypes.c_int32
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32]
    return get_long, set_long


def _run_on_gui(window: object | None, fn, *, wait: bool = False) -> None:
    """Win32 capture/style changes must run on the WinForms thread.

    Resize/move start must ``wait`` (Invoke). BeginInvoke returns before the
    mouse is still down, so a second scale after a big size change never starts.
    """
    native = getattr(window, "native", None) if window is not None else None
    try:
        if native is not None and bool(getattr(native, "InvokeRequired", False)):
            method_name = "Invoke" if wait else "BeginInvoke"
            method = getattr(native, method_name, None)
            if method is not None:
                from System import Action

                method(Action(fn))
                return
    except Exception:
        pass
    fn()


def _window_hwnd(window: object | None) -> int:
    if window is None:
        return 0
    native = None
    try:
        native = getattr(window, "native", None)
        handle = getattr(native, "Handle", None) if native is not None else None
        if handle is not None:
            value = int(handle.ToInt64()) if hasattr(handle, "ToInt64") else int(handle)
            if value > 0:
                return value
    except Exception:
        pass
    # Only search by title when this is a real GUI window. Tests pass bare objects.
    if native is None or sys.platform != "win32":
        return 0
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        found = user32.FindWindowW(None, "VTM Studio")
        return int(found or 0)
    except Exception:
        return 0


def _window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    if hwnd <= 0 or sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):  # type: ignore[attr-defined]
            return (
                int(rect.left),
                int(rect.top),
                int(rect.right - rect.left),
                int(rect.bottom - rect.top),
            )
    except Exception:
        pass
    return None


def _win32_show(window: object, cmd: int) -> None:
    hwnd = _window_hwnd(window)
    if hwnd <= 0:
        return
    try:
        import ctypes

        ctypes.windll.user32.ShowWindow(hwnd, int(cmd))  # type: ignore[attr-defined]
    except Exception:
        pass


def _bring_to_front(window: object) -> None:
    """Put the window in front with focus. Call on the GUI thread.

    open_desk hides the splash while the desk loads; Windows hands focus to
    another app meanwhile and then refuses a plain show() the foreground, so
    the desk only flashed on the taskbar. Joining the foreground window's
    input queue for the call is the sanctioned way to take it back.
    """
    hwnd = _window_hwnd(window)
    if hwnd <= 0 or sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.IsIconic.argtypes = [ctypes.c_void_p]
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.BringWindowToTop.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        front = user32.GetForegroundWindow()
        if front and int(front) == hwnd:
            return
        ours = kernel32.GetCurrentThreadId()
        theirs = user32.GetWindowThreadProcessId(front, None) if front else 0
        joined = bool(theirs and theirs != ours and user32.AttachThreadInput(ours, theirs, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if joined:
                user32.AttachThreadInput(ours, theirs, False)
    except Exception:
        pass


def _colorref(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return int(r) | (int(g) << 8) | (int(b) << 16)


def _frame_fill_rgb() -> tuple[int, int, int]:
    return DESK_FILL_RGB if _CHROME_RESIZABLE else SPLASH_FILL_RGB


def _is_win11() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return int(sys.getwindowsversion().build) >= 22000
    except Exception:
        return False


def _paint_frame_dark(hwnd: int) -> None:
    """DWM strokes the rounded clip. Match the page fill; COLOR_NONE is Win11-only."""
    try:
        import ctypes

        dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined]
        fill = _frame_fill_rgb()
        dark = ctypes.c_int(1)
        for attr in (DWMWA_USE_IMMERSIVE_DARK_MODE_WIN10, DWMWA_USE_IMMERSIVE_DARK_MODE):
            try:
                dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(dark), ctypes.sizeof(dark)
                )
            except Exception:
                pass
        none = ctypes.c_uint32(DWMWA_COLOR_NONE)
        ink = ctypes.c_uint32(_colorref(fill))
        # COLOR_NONE first (Win11). Win10 ignores it, so follow with the page fill
        # so the stroke is not system white around the dark desk.
        for color in (none, ink):
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_CAPTION_COLOR,
                ctypes.byref(color),
                ctypes.sizeof(color),
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                ctypes.byref(color),
                ctypes.sizeof(color),
            )
        if _is_win11():
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                ctypes.byref(none),
                ctypes.sizeof(none),
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_CAPTION_COLOR,
                ctypes.byref(none),
                ctypes.sizeof(none),
            )
        # pywebview shadow=True (default) calls DwmExtendFrameIntoClientArea(1px).
        # That glass inset is cream-on-cream on splash and a white hairline on the desk.
        class _MARGINS(ctypes.Structure):
            _fields_ = [
                ("cxLeftWidth", ctypes.c_int),
                ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int),
                ("cyBottomHeight", ctypes.c_int),
            ]

        zero = _MARGINS(0, 0, 0, 0)
        dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(zero))
        round_pref = ctypes.c_int(DWMWCP_DONOTROUND)
        dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(round_pref),
            ctypes.sizeof(round_pref),
        )
        # Do not SetWindowTheme("", ""). That falls back to classic 3D edges —
        # the gray-white outline around WS_THICKFRAME on Win10.
    except Exception:
        pass


def _unpack_nchittest(lparam: object) -> tuple[int, int]:
    """Screen coordinates from WM_NCHITTEST. Sign-extend for monitors left/above origin."""
    value = lparam.value if hasattr(lparam, "value") else lparam
    packed = int(value or 0) & 0xFFFFFFFF
    x = packed & 0xFFFF
    y = (packed >> 16) & 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    if y >= 0x8000:
        y -= 0x10000
    return x, y


def _resize_border_px() -> int:
    """Invisible edge thickness. Matches the OS frame; draws nothing."""
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        frame = int(user32.GetSystemMetrics(32) or 0)  # SM_CXFRAME
        pad = int(user32.GetSystemMetrics(92) or 0)  # SM_CXPADDEDBORDER
        return max(6, frame + pad)
    except Exception:
        return 8


def _resize_edge_hit(edge: object) -> int:
    key = str(edge or "").strip().lower().replace("-", "").replace("_", "")
    return int(_RESIZE_EDGE_HITS.get(key, HTCLIENT))


def _resize_hit(x: int, y: int, width: int, height: int, border: int) -> int:
    """HT* for an invisible resize rim. HTCLIENT everywhere else (custom caption clicks)."""
    if border <= 0 or width <= 0 or height <= 0:
        return HTCLIENT
    left = x < border
    right = x >= width - border
    top = y < border
    bottom = y >= height - border
    if top and left:
        return HTTOPLEFT
    if top and right:
        return HTTOPRIGHT
    if bottom and left:
        return HTBOTTOMLEFT
    if bottom and right:
        return HTBOTTOMRIGHT
    if left:
        return HTLEFT
    if right:
        return HTRIGHT
    if top:
        return HTTOP
    if bottom:
        return HTBOTTOM
    return HTCLIENT


def _cursor_pos() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        pt = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):  # type: ignore[attr-defined]
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return None


def _resize_rect_from_delta(
    left: int,
    top: int,
    width: int,
    height: int,
    hit: int,
    dx: int,
    dy: int,
    min_w: int,
    min_h: int,
) -> tuple[int, int, int, int]:
    """New window rect from an edge drag. min_* keeps the frame grabbable."""
    x, y, w, h = int(left), int(top), int(width), int(height)
    min_w = max(1, int(min_w))
    min_h = max(1, int(min_h))
    if hit in (HTLEFT, HTTOPLEFT, HTBOTTOMLEFT):
        x += dx
        w -= dx
    elif hit in (HTRIGHT, HTTOPRIGHT, HTBOTTOMRIGHT):
        w += dx
    if hit in (HTTOP, HTTOPLEFT, HTTOPRIGHT):
        y += dy
        h -= dy
    elif hit in (HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT):
        h += dy
    if w < min_w:
        if hit in (HTLEFT, HTTOPLEFT, HTBOTTOMLEFT):
            x -= min_w - w
        w = min_w
    if h < min_h:
        if hit in (HTTOP, HTTOPLEFT, HTTOPRIGHT):
            y -= min_h - h
        h = min_h
    return x, y, w, h


def _place_hwnd_rect(hwnd: int, x: int, y: int, width: int, height: int) -> None:
    if sys.platform != "win32" or hwnd <= 0:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos(
            hwnd,
            0,
            int(x),
            int(y),
            max(1, int(width)),
            max(1, int(height)),
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
    except Exception:
        pass


def _live_min_size(window: object | None) -> tuple[int, int]:
    min_w, min_h = desk_min_size()
    native = getattr(window, "native", None) if window is not None else None
    try:
        size = getattr(native, "MinimumSize", None) if native is not None else None
        if size is not None:
            min_w = max(min_w, int(size.Width or 0))
            min_h = max(min_h, int(size.Height or 0))
    except Exception:
        pass
    return min_w, min_h


def _begin_live_resize(window: object | None, hit: int, edge: str) -> None:
    _LIVE_RESIZE.clear()
    if not _CHROME_RESIZABLE or hit == HTCLIENT:
        return
    hwnd = _window_hwnd(window)
    if hwnd <= 0:
        return
    try:
        import ctypes

        if int(ctypes.windll.user32.IsZoomed(hwnd)):  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    rect = _window_rect(hwnd)
    mouse = _cursor_pos()
    if rect is None or mouse is None:
        return
    _LIVE_RESIZE.update(
        {
            "hwnd": hwnd,
            "hit": hit,
            "edge": edge,
            "left": rect[0],
            "top": rect[1],
            "width": rect[2],
            "height": rect[3],
            "mx": mouse[0],
            "my": mouse[1],
        }
    )


def _move_live_resize(window: object | None) -> None:
    hwnd = int(_LIVE_RESIZE.get("hwnd") or 0)
    if hwnd <= 0:
        return
    mouse = _cursor_pos()
    if mouse is None:
        return
    min_w, min_h = _live_min_size(window)
    x, y, w, h = _resize_rect_from_delta(
        int(_LIVE_RESIZE["left"]),
        int(_LIVE_RESIZE["top"]),
        int(_LIVE_RESIZE["width"]),
        int(_LIVE_RESIZE["height"]),
        int(_LIVE_RESIZE["hit"]),
        mouse[0] - int(_LIVE_RESIZE["mx"]),
        mouse[1] - int(_LIVE_RESIZE["my"]),
        min_w,
        min_h,
    )
    _place_hwnd_rect(hwnd, x, y, w, h)
    _round_hwnd(hwnd)


def _end_live_resize(window: object | None) -> None:
    _LIVE_RESIZE.clear()
    _pin_after_frame_change(window)


def _pin_after_frame_change(window: object | None) -> None:
    """Keep the rounded clip and WebView2 on the live HWND after move/resize."""
    hwnd = _window_hwnd(window)
    if hwnd > 0:
        _round_hwnd(hwnd)
    _sync_webview(window)


def _nc_hit_test(h_wnd: object, lparam: object) -> int:
    """Resize on the edge; never return HTMIN/MAX/CLOSE (DWM would draw native buttons)."""
    if not _CHROME_RESIZABLE:
        return HTCLIENT
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if int(user32.IsZoomed(h_wnd)):
            return HTCLIENT
        screen_x, screen_y = _unpack_nchittest(lparam)
        pt = wintypes.POINT(screen_x, screen_y)
        user32.ScreenToClient(h_wnd, ctypes.byref(pt))
        rect = wintypes.RECT()
        if not user32.GetClientRect(h_wnd, ctypes.byref(rect)):
            return HTCLIENT
        return _resize_hit(
            int(pt.x),
            int(pt.y),
            int(rect.right - rect.left),
            int(rect.bottom - rect.top),
            _resize_border_px(),
        )
    except Exception:
        return HTCLIENT


def _hook_client_fill(hwnd: int) -> None:
    """WM_NCCALCSIZE → client = window (no DWM stroke). WM_NCHITTEST → resize rim.

    Win10 draws a hairline for WS_THICKFRAME. Returning 0 from NCCALCSIZE is the
    Chromium/Electron fix. That also removes the OS resize border, so NCHITTEST
    must restore an invisible edge. Always HTCLIENT in the caption so DWM does
    not draw native min/max/close on top of the HTML buttons. Chain every other
    message so WebView2 keeps its WinForms WndProc.
    """
    if sys.platform != "win32" or hwnd <= 0 or hwnd in _NCCALC_HOOKS:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get_long, set_long = _ptr_fns(user32)
        old = int(get_long(hwnd, GWLP_WNDPROC) or 0)
        if old == 0:
            return
        wndproc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        user32.CallWindowProcW.restype = ctypes.c_ssize_t
        user32.CallWindowProcW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        def _proc(h_wnd, msg, wparam, lparam):
            try:
                code = int(msg)
                if code == WM_NCCALCSIZE:
                    return 0
                if code == WM_NCHITTEST:
                    return _nc_hit_test(h_wnd, lparam)
                return int(
                    user32.CallWindowProcW(old, h_wnd, msg, wparam, lparam) or 0
                )
            except Exception:
                try:
                    return int(
                        user32.CallWindowProcW(old, h_wnd, msg, wparam, lparam) or 0
                    )
                except Exception:
                    return 0

        callback = wndproc(_proc)
        set_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
        _NCCALC_HOOKS[int(hwnd)] = (callback, old)
    except Exception:
        pass


def _enum_child_hwnds(hwnd: int) -> list[int]:
    found: list[int] = []
    if sys.platform != "win32" or hwnd <= 0:
        return found
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        wndenum = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_ssize_t)

        def _enum(child, _lparam):
            value = int(child or 0)
            if value > 0:
                found.append(value)
            return 1

        callback = wndenum(_enum)
        user32.EnumChildWindows(hwnd, callback, 0)
    except Exception:
        pass
    return found


def _hook_child_passthrough(root: int) -> None:
    """Do not subclass WebView2/Chromium HWNDs. That crashes on launch.

    Resize comes from HTML start_resize + the form's own NCHITTEST rim.
    """
    return


def _style_matches(hwnd: int, *, resizable: bool) -> bool:
    if sys.platform != "win32" or hwnd <= 0:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get_long, _set_long = _ptr_fns(user32)
        style = int(get_long(hwnd, GWL_STYLE) or 0)
        if style & WS_CAPTION or not (style & WS_POPUP):
            return False
        has_frame = bool(style & WS_THICKFRAME)
        return has_frame if resizable else not has_frame
    except Exception:
        return False


def _walk_native(root: object):
    if root is None:
        return
    yield root
    try:
        controls = getattr(root, "Controls", None)
        if controls is None:
            return
        n = int(controls.Count)
    except Exception:
        return
    for i in range(n):
        child = None
        try:
            child = controls[i]
        except Exception:
            try:
                child = controls.Item[i]
            except Exception:
                continue
        yield from _walk_native(child)


def _tint_form(window: object) -> None:
    """Form + WebView2 default color. White DefaultBackgroundColor is the desk hairline."""
    rgb = _frame_fill_rgb()
    hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
    try:
        window.background_color = hex_color
    except Exception:
        pass
    color = None
    try:
        from System.Drawing import Color

        try:
            color = Color.FromArgb(255, rgb[0], rgb[1], rgb[2])
        except Exception:
            color = Color.FromArgb(rgb[0], rgb[1], rgb[2])
    except Exception:
        color = None
    roots = [getattr(window, "native", None), getattr(window, "gui", None)]
    for root in roots:
        if root is None:
            continue
        for ctrl in _walk_native(root):
            if color is None:
                continue
            for attr in ("BackColor", "DefaultBackgroundColor"):
                try:
                    setattr(ctrl, attr, color)
                except Exception:
                    pass
        _fill_browser(root)


def _fill_browser(root: object, client_size: object | None = None) -> None:
    """Keep WebView2 flush to the form so a white DefaultBackgroundColor cannot show."""
    try:
        from System.Windows.Forms import DockStyle, Padding
    except Exception:
        return
    for ctrl in _walk_native(root):
        name = type(ctrl).__name__.lower()
        if "webview" not in name and "browser" not in name:
            continue
        try:
            ctrl.Dock = DockStyle.Fill
        except Exception:
            pass
        try:
            ctrl.Margin = Padding(0)
        except Exception:
            pass
        try:
            ctrl.Padding = Padding(0)
        except Exception:
            pass
        if not _CHROME_RESIZABLE:
            continue
        # Dock.Fill can lag after a top/left size change. Do not Set Size —
        # that COM path aborted pythonw during splash. Layout + public notify only.
        parent = getattr(ctrl, "Parent", None)
        try:
            layout = getattr(parent, "PerformLayout", None) if parent is not None else None
            if callable(layout):
                layout()
        except Exception:
            pass
        _notify_webview_moved(ctrl)


def _notify_webview_moved(ctrl: object) -> None:
    """Public notify only. Reflection/GetType on WebView2 during splash aborts pythonw."""
    fn = getattr(ctrl, "NotifyParentWindowPositionChanged", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _sync_webview(window: object) -> None:
    """Dock-fill only. Do not Set Size or walk COM fields while the splash is coming up."""
    native = getattr(window, "native", None)
    for root in (native, getattr(window, "gui", None)):
        if root is None:
            continue
        _fill_browser(root)


def _request_webview_sync(window: object) -> None:
    _sync_webview(window)


CORNER_RADIUS = 24


def _round_hwnd(hwnd: int) -> None:
    """Clip the HWND to a rounded rect so splash/desk edges are not square."""
    if sys.platform != "win32" or hwnd <= 0:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetWindowRgn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        if int(user32.IsZoomed(hwnd)):
            # Maximized: the 1px DWM stroke is off-screen. A leftover rounded
            # clip from the restored size is what makes one edge vanish until drag.
            user32.SetWindowRgn(hwnd, None, 1)
            return
    except Exception:
        pass
    rect = _window_rect(hwnd)
    if rect is None:
        return
    width, height = max(2, rect[2]), max(2, rect[3])
    try:
        import ctypes

        gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        gdi32.CreateRoundRectRgn.restype = ctypes.c_void_p
        gdi32.CreateRoundRectRgn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        user32.SetWindowRgn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        # Exclusive right/bottom: +1 keeps the last content pixel. Do not inset —
        # that punches a black chip out of the rounded corner.
        hrgn = gdi32.CreateRoundRectRgn(0, 0, width + 1, height + 1, CORNER_RADIUS, CORNER_RADIUS)
        if hrgn:
            user32.SetWindowRgn(hwnd, hrgn, 1)
    except Exception:
        pass


def _apply_frame_style(hwnd: int, *, resizable: bool) -> None:
    if sys.platform != "win32" or hwnd <= 0:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get_long, set_long = _ptr_fns(user32)
        style = int(get_long(hwnd, GWL_STYLE) or 0)
        try:
            was_visible = bool(user32.IsWindowVisible(hwnd))
        except Exception:
            was_visible = True
        style &= ~WS_OVERLAPPEDWINDOW
        style |= WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN
        if was_visible:
            style |= WS_VISIBLE
        else:
            style &= ~WS_VISIBLE
        style &= ~WS_MINIMIZEBOX
        style &= ~WS_MAXIMIZEBOX
        if resizable:
            # Thickframe only. Min/max boxes make Win10 DWM draw caption buttons
            # in a white strip — splash never sets those, which is why it stays clean.
            style |= WS_THICKFRAME
        else:
            style &= ~WS_THICKFRAME
        set_long(hwnd, GWL_STYLE, style)
        try:
            exstyle = int(get_long(hwnd, GWL_EXSTYLE) or 0)
            exstyle |= WS_EX_APPWINDOW
            exstyle &= ~WS_EX_EDGE
            set_long(hwnd, GWL_EXSTYLE, exstyle)
        except Exception:
            pass
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        frame_flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, frame_flags)
        _hook_client_fill(hwnd)
        # Second FRAMECHANGED so WM_NCCALCSIZE hits our hook, not the old proc.
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, frame_flags)
        _paint_frame_dark(hwnd)
        _round_hwnd(hwnd)
        _hook_child_passthrough(hwnd)
        from backend.app_icon import apply_window_icon

        apply_window_icon(hwnd)
    except Exception:
        pass


def work_area_rect() -> tuple[int, int, int, int]:
    """Primary monitor work area: left, top, width, height (taskbar excluded)."""
    if sys.platform != "win32":
        return (0, 0, 1920, 1080)
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # type: ignore[attr-defined]
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width >= 640 and height >= 480:
            return (int(rect.left), int(rect.top), width, height)
    except Exception:
        pass
    return (0, 0, 1920, 1080)


def work_area_size() -> tuple[int, int]:
    _left, _top, width, height = work_area_rect()
    return (width, height)


def splash_origin(width: int, height: int) -> tuple[int, int]:
    """Top-left that places a window of this size in the work-area center."""
    left, top, work_w, work_h = work_area_rect()
    w = min(max(1, int(width)), work_w)
    h = min(max(1, int(height)), work_h)
    return (left + max(0, (work_w - w) // 2), top + max(0, (work_h - h) // 2))


def desk_min_size() -> tuple[int, int]:
    """Smallest window that still keeps the rail + stage usable."""
    work_w, work_h = work_area_size()
    want_w, want_h = 1180, 740
    return (
        min(work_w, max(960, min(want_w, work_w - 24))),
        min(work_h, max(640, min(want_h, work_h - 24))),
    )


def maximize_window(window: object) -> None:
    hwnd = _window_hwnd(window)
    if hwnd > 0:
        try:
            import ctypes

            ctypes.windll.user32.ShowWindow(hwnd, 3)  # type: ignore[attr-defined]
            hide_native_caption(window, resizable=True)
            return
        except Exception:
            pass
    try:
        getattr(window, "maximize")()
    except Exception:
        pass
    hide_native_caption(window, resizable=True)


def hide_native_caption(window: object, *, resizable: bool | None = None) -> None:
    """Strip the Windows title bar after show. Opaque window, real bounds.

    Safer than pywebview frameless/transparent, which can walk
    AccessibilityObject.Bounds.Empty until Python stops responding.
    Do not replace WndProc without chaining, or touch WinForms ControlBox —
    that blanks WebView2. _hook_client_fill chains CallWindowProc for everything
    except WM_NCCALCSIZE (client rect = window rect, no 1px DWM stroke) and
    WM_NCHITTEST (invisible resize rim; HTCLIENT so DWM skips native caption).
    """
    global _CHROME_RESIZABLE
    if resizable is not None:
        _CHROME_RESIZABLE = bool(resizable)

    def _do() -> None:
        hwnd = _window_hwnd(window)
        if hwnd <= 0:
            return
        # SWP_FRAMECHANGED on every Resize lets DWM restroke the hairline.
        # Only rebuild the frame when pywebview put the caption back.
        if _style_matches(hwnd, resizable=_CHROME_RESIZABLE) and hwnd in _NCCALC_HOOKS:
            _paint_frame_dark(hwnd)
            _round_hwnd(hwnd)
        else:
            _apply_frame_style(hwnd, resizable=_CHROME_RESIZABLE)
        _hook_child_passthrough(hwnd)
        _tint_form(window)

    _run_on_gui(window, _do)


def _retry_caption(window: object) -> None:
    def _retries() -> None:
        for wait in (0.03, 0.08, 0.16, 0.32, 0.64, 1.2, 2.4):
            time.sleep(wait)
            hide_native_caption(window)

    threading.Thread(target=_retries, name="vtm-caption", daemon=True).start()


def keep_caption_hidden(window: object, *, resizable: bool = False) -> None:
    """Hide the OS title bar from the GUI thread only (shown / loaded)."""
    hide_native_caption(window, resizable=resizable)

    def _hide(*_a, **_k) -> None:
        hide_native_caption(window)

    events = getattr(window, "events", None)
    if events is not None:
        for name in ("shown", "loaded"):
            ev = getattr(events, name, None)
            if ev is None:
                continue
            try:
                ev += _hide
            except Exception:
                pass

    _retry_caption(window)

    native = getattr(window, "native", None)
    try:
        if native is not None:
            if hasattr(native, "Resize"):
                native.Resize += lambda *_a: _round_live(window)
            # Resize does not fire after a drag. DWM repaints the stroke then.
            if hasattr(native, "ResizeEnd"):
                native.ResizeEnd += lambda *_a: _after_user_frame(window)
    except Exception:
        pass


def _round_live(window: object | None) -> None:
    hwnd = _window_hwnd(window)
    if hwnd > 0:
        _round_hwnd(hwnd)


def _after_user_frame(window: object | None) -> None:
    hide_native_caption(window)
    _pin_after_frame_change(window)


_HOLD_REVEAL = False
_SPLASH_WINDOW: list[object | None] = [None]
_SPLASH_SHOWN = [False]
SWP_HIDEWINDOW = 0x0080


def _desk_rect(width: int, height: int) -> tuple[int, int, int, int]:
    """Centered desk bounds: left, top, width, height."""
    work_left, work_top, work_w, work_h = work_area_rect()
    width = min(max(320, int(width)), work_w)
    height = min(max(240, int(height)), work_h)
    x = work_left + max(0, (work_w - width) // 2)
    y = work_top + max(0, (work_h - height) // 2)
    return (x, y, width, height)


def _place_hwnd(
    hwnd: int,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    hidden: bool = False,
    keep_size: bool = False,
) -> None:
    if sys.platform != "win32" or hwnd <= 0:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        flags = SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        if keep_size:
            flags |= SWP_NOSIZE
        if hidden:
            flags |= SWP_HIDEWINDOW
        user32.SetWindowPos(hwnd, 0, int(x), int(y), int(width), int(height), flags)
    except Exception:
        pass


def _native_scale(window: object) -> float:
    """pywebview stores Form.Size in physical pixels. Our sizes are logical."""
    native = getattr(window, "native", None)
    try:
        scale = float(getattr(native, "_scale", 1) or 1)
        if scale > 0:
            return scale
    except Exception:
        pass
    return 1.0


def _apply_desk_size(window: object, x: int, y: int, width: int, height: int, *, hidden: bool = False) -> None:
    """Raise min-size and actually move/resize. Splash locks min_size to itself."""
    try:
        window.resizable = True
    except Exception:
        pass
    try:
        window.min_size = desk_min_size()
    except Exception:
        pass
    native = getattr(window, "native", None)
    scale = _native_scale(window)
    try:
        if native is not None:
            from System.Drawing import Size

            min_w, min_h = desk_min_size()
            native.MinimumSize = Size(int(min_w * scale), int(min_h * scale))
    except Exception:
        pass
    try:
        window.resize(width, height)
    except Exception:
        pass
    # pywebview move() sends SWP_SHOWWINDOW, which would flash the desk mid-load.
    # resize() already set a DPI-aware size — don't overwrite it with logical pixels.
    if hidden:
        _place_hwnd(
            _window_hwnd(window),
            int(x * scale),
            int(y * scale),
            0,
            0,
            hidden=True,
            keep_size=True,
        )
        return
    try:
        window.move(x, y)
    except Exception:
        pass


def desk_handoff_url(url: str) -> str:
    """Mark the operator desk navigation so React does not paint a second splash."""
    base = (url or "").strip()
    if not base or "desk=1" in base:
        return base
    if "#" in base:
        path, frag = base.split("#", 1)
        sep = "&" if "?" in path else "?"
        return f"{path}{sep}desk=1#{frag}"
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}desk=1"


def reveal_splash(window: object | None = None) -> None:
    """Show the splash only after HTML has painted (or the fallback timeout)."""
    if _HOLD_REVEAL:
        return
    target = window if window is not None else _SPLASH_WINDOW[0]
    if target is None or _SPLASH_SHOWN[0]:
        return
    _SPLASH_SHOWN[0] = True

    def _do() -> None:
        hide_native_caption(target)
        try:
            target.show()
        except Exception:
            pass

    _run_on_gui(target, _do)


def reveal_when_painted(window: object, *, timeout: float = 2.0) -> None:
    """Keep the HWND hidden until splash JS says it painted. loaded is a blank WebView2."""
    _SPLASH_WINDOW[0] = window
    _SPLASH_SHOWN[0] = False

    def _timeout() -> None:
        time.sleep(max(0.35, float(timeout)))
        reveal_splash(window)

    threading.Thread(target=_timeout, name="vtm-reveal", daemon=True).start()


def set_desk_frame(window: object, width: int, height: int, *, hidden: bool = False) -> None:
    """Grow into the operator desk: resizable, movable, still no OS title bar."""
    global _CHROME_RESIZABLE
    _CHROME_RESIZABLE = True
    x, y, width, height = _desk_rect(width, height)

    def _do() -> None:
        hide_native_caption(window, resizable=True)
        _apply_desk_size(window, x, y, width, height, hidden=hidden)
        hide_native_caption(window, resizable=True)

    _run_on_gui(window, _do)


def open_desk(
    window: object,
    url: str,
    width: int,
    height: int,
    *,
    timeout: float = 8.0,
) -> None:
    """Keep splash up until boot is done, then swap to a larger desk without a mid-load flash."""
    global _HOLD_REVEAL
    _HOLD_REVEAL = True
    target = desk_handoff_url(url)
    state = {"shown": False, "loading": False}

    def _show(*_a, **_k) -> None:
        def _do() -> None:
            global _HOLD_REVEAL
            if state["shown"] or not state["loading"]:
                return
            state["shown"] = True
            _HOLD_REVEAL = False
            hide_native_caption(window, resizable=True)
            try:
                window.show()
            except Exception:
                pass
            # show() / load_url re-apply Sizable chrome after the splash retries
            # have already finished. Strip it again on this reveal.
            hide_native_caption(window, resizable=True)
            _bring_to_front(window)
            _retry_caption(window)

        _run_on_gui(window, _do)

    events = getattr(window, "events", None)
    if events is not None:
        ev = getattr(events, "loaded", None)
        if ev is not None:
            try:
                ev += _show
            except Exception:
                pass

    def _start() -> None:
        try:
            window.hide()
        except Exception:
            pass
        set_desk_frame(window, width, height, hidden=True)
        state["loading"] = True
        try:
            window.load_url(target)
        except Exception:
            _show()

    _run_on_gui(window, _start)

    def _timeout() -> None:
        time.sleep(max(0.4, float(timeout)))
        _show()

    threading.Thread(target=_timeout, name="vtm-desk-reveal", daemon=True).start()


def create_splash_window(
    webview: object,
    *,
    url: str | None = None,
    html: str | None = None,
    width: int,
    height: int,
    js_api: WindowBridge | None = None,
):
    """Centered splash. Fixed size, draggable, never frameless, transparent, or DWM-shadowed."""
    from backend.app_icon import claim_app_id

    claim_app_id()
    _left, _top, work_w, work_h = work_area_rect()
    w = min(max(320, int(width)), work_w)
    h = min(max(240, int(height)), work_h)
    x, y = splash_origin(w, h)
    kw: dict = {
        "width": w,
        "height": h,
        "x": x,
        "y": y,
        "resizable": False,
        "min_size": (w, h),
        "hidden": True,
        "shadow": False,
        "background_color": "#2B2B2B",
    }
    if js_api is not None:
        kw["js_api"] = js_api
    if url is not None:
        return webview.create_window("VTM Studio", url, **kw)
    return webview.create_window("VTM Studio", html=html, **kw)


WEBVIEW2_ARGS = "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"
_NO_MIC_FLAG = "--disable-audio-input"
_MIC_KIND = 1
_CAM_KIND = 2
_DENY = 2


def merge_webview2_args(raw: object) -> str:
    """Append mic-off without adding a second --disable-features (Chromium keeps the last)."""
    args = str(raw or "").strip()
    if _NO_MIC_FLAG not in args.split():
        args = f"{args} {_NO_MIC_FLAG}".strip()
    return args


def mute_webview_microphone() -> None:
    """Env fallback. pywebview overwrites AdditionalBrowserArguments, so also patch it."""
    os.environ[WEBVIEW2_ARGS] = merge_webview2_args(os.environ.get(WEBVIEW2_ARGS))


def stamp_webview2_creation_props(props: object) -> object:
    """Flags must be on the props object before WebView2 starts the environment.

    A leftover msedgewebview2 from the last close reuses the user-data folder and
    ignores AdditionalBrowserArguments — that is the second-launch mic lock.
    ExclusiveUserDataFolderAccess refuses to join that zombie environment.
    """
    try:
        props.AdditionalBrowserArguments = merge_webview2_args(
            getattr(props, "AdditionalBrowserArguments", "")
        )
    except Exception:
        pass
    try:
        props.ExclusiveUserDataFolderAccess = True
    except Exception:
        pass
    return props


def webview2_user_data_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env_key in ("LOCALAPPDATA", "APPDATA"):
        root = os.environ.get(env_key) or ""
        if root:
            dirs.append(Path(root) / "pywebview")
    try:
        import webview

        settings = getattr(webview, "settings", None) or {}
        raw = None
        if isinstance(settings, dict):
            raw = settings.get("STORAGE_PATH") or settings.get("storage_path")
        else:
            raw = getattr(settings, "STORAGE_PATH", None) or getattr(
                settings, "storage_path", None
            )
        if raw:
            dirs.append(Path(str(raw)))
    except Exception:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for folder in dirs:
        try:
            key = str(folder.resolve()).lower()
        except OSError:
            key = str(folder).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(folder)
    return out


def _webview2_cmdline_is_ours(cmd: str, folders: list[Path]) -> bool:
    text = cmd.lower().replace("/", "\\")
    if "pywebview" in text:
        return True
    for folder in folders:
        needle = str(folder).lower().replace("/", "\\")
        if needle and needle in text:
            return True
    return False


WEBVIEW2_RUNTIME_KEY = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WEBVIEW2_MIN_BUILD = (86, 0, 622, 0)


def webview2_version() -> str | None:
    """Installed WebView2 runtime version, or None when it is missing.

    Mirrors pywebview's check. Without the runtime pywebview silently falls
    back to the IE engine, which cannot run the desk UI (blank window).
    """
    if os.name != "nt":
        return None
    import winreg

    paths = (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_RUNTIME_KEY}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_RUNTIME_KEY}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_RUNTIME_KEY}"),
    )
    for hive, path in paths:
        try:
            with winreg.OpenKey(hive, path) as key:
                build = str(winreg.QueryValueEx(key, "pv")[0])
        except OSError:
            continue
        try:
            parts = tuple(int(p) for p in build.split("."))
        except ValueError:
            continue
        if parts >= WEBVIEW2_MIN_BUILD:
            return build
    return None


def kill_orphan_webview2() -> list[int]:
    """Drop leftover WebView2 runtimes that keep the WASAPI mic after close."""
    if os.name != "nt":
        return []
    folders = webview2_user_data_dirs()
    killed: list[int] = []
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        # Only read cmdline for WebView2 hosts — fetching it for every process
        # cost ~4s before the splash could open.
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if name not in {"msedgewebview2.exe", "msedgewebview2"}:
                    continue
                cmd = " ".join(str(part) for part in (proc.cmdline() or []) if part)
                if not _webview2_cmdline_is_ours(cmd, folders):
                    continue
                pid = int(proc.info["pid"])
                proc.kill()
                killed.append(pid)
            except (psutil.Error, TypeError, ValueError):
                continue
        return killed
    return _kill_orphan_webview2_cim(folders)


def _kill_orphan_webview2_cim(folders: list[Path]) -> list[int]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^msedgewebview2\\.exe$' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
            if os.name == "nt"
            else 0,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    killed: list[int] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cmd = str(item.get("CommandLine") or "")
        if not _webview2_cmdline_is_ours(cmd, folders):
            continue
        try:
            pid = int(item.get("ProcessId") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 4:
            continue
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)),
            )
            killed.append(pid)
        except OSError:
            continue
    return killed


def should_deny_webview_permission(kind: object) -> bool:
    try:
        return int(kind) in {_MIC_KIND, _CAM_KIND}
    except (TypeError, ValueError):
        name = str(kind).lower()
        return "microphone" in name or name.endswith(".camera") or name == "camera"


def deny_webview_media_permission(_sender: object, args: object) -> None:
    """The desk never uses getUserMedia. Deny so Chromium cannot keep the WASAPI capture."""
    try:
        kind = getattr(args, "PermissionKind", None)
        if not should_deny_webview_permission(kind):
            return
        if hasattr(args, "set_State"):
            args.set_State(_DENY)
        else:
            args.State = _DENY
        args.Handled = True
    except Exception:
        return


def webview_media_deny_origins(source: object = None) -> list[str]:
    """Local desk + splash file origins. Chromium can request the mic on file:// too."""
    origins = [
        "http://127.0.0.1:8765/",
        "http://localhost:8765/",
        "http://127.0.0.1:5173/",
        "http://localhost:5173/",
        "file:///",
    ]
    raw = str(source or "").strip()
    if raw:
        origins.append(raw if raw.endswith("/") else f"{raw}/")
        if raw.lower().startswith("file:"):
            origins.append("file:///")
    folder = splash_asset_dir()
    if folder is not None:
        try:
            uri = folder.resolve().as_uri()
            if not uri.endswith("/"):
                uri = f"{uri}/"
            origins.append(uri)
            origins.append(f"{uri}{EARLY_SPLASH_NAME}")
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for origin in origins:
        if origin in seen:
            continue
        seen.add(origin)
        out.append(origin)
    return out


def _attach_webview_media_block(sender: object) -> None:
    try:
        core = sender.CoreWebView2
    except Exception:
        return
    if core is None:
        return
    try:
        core.PermissionRequested += deny_webview_media_permission
    except Exception:
        pass
    try:
        profile = core.Profile
        setter = getattr(profile, "SetPermissionState", None)
        if setter is None:
            return
        source = getattr(core, "Source", None)
        for origin in webview_media_deny_origins(source):
            for kind in (_MIC_KIND, _CAM_KIND):
                try:
                    setter(kind, origin, _DENY)
                except Exception:
                    continue
    except Exception:
        return


def patch_webview2_no_microphone() -> None:
    """Put --disable-audio-input on the real WebView2 args before the environment starts.

    pywebview sets AdditionalBrowserArguments to only ElasticOverscroll, which drops
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS. Chromium then opens the default mic.
    """
    mute_webview_microphone()
    try:
        import webview.platforms.edgechromium as edge
    except Exception:
        return
    _patch_edge_init_args(edge)
    _patch_edge_permission(edge)


class _CreationPropsGuard:
    """Merge --disable-audio-input onto AdditionalBrowserArguments before the environment starts."""

    def __init__(self, real: object) -> None:
        object.__setattr__(self, "_real", real)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_real":
            object.__setattr__(self, name, value)
            return
        if name == "CreationProperties" and value is not None:
            stamp_webview2_creation_props(value)
        setattr(object.__getattribute__(self, "_real"), name, value)

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_real"), name)


def _is_winforms_control(value: object) -> bool:
    """True for CLR WebView2 / Control — those must not be wrapped for Controls.Add."""
    if value is None or isinstance(value, _CreationPropsGuard):
        return False
    name = type(value).__name__
    mod = str(getattr(type(value), "__module__", "") or "")
    if name in {"WebView2", "Control", "Form"}:
        return True
    return "Microsoft.Web.WebView2" in mod or "System.Windows.Forms" in mod


def _hook_ensure_core_webview(wv: object) -> None:
    """Stamp mic-off flags on the real control before WebView2 starts."""
    orig = getattr(wv, "EnsureCoreWebView2Async", None)
    if orig is None or getattr(orig, "_vtm_stamped", False):
        return

    def ensure(*args: object, **kwargs: object) -> object:
        try:
            stamp_webview2_creation_props(getattr(wv, "CreationProperties", None))
        except Exception:
            pass
        return orig(*args, **kwargs)

    ensure._vtm_stamped = True  # type: ignore[attr-defined]
    try:
        wv.EnsureCoreWebView2Async = ensure
    except Exception:
        pass


class _WebviewSlot:
    """Wrap EdgeChrome.webview so CreationProperties assignment is intercepted.

    Python fakes (tests) stay wrapped. The real WinForms WebView2 is stored
    unwrapped so ``form.Controls.Add(self.webview)`` still receives a Control.
    """

    def __init__(self) -> None:
        self.private = "_vtm_webview"

    def __get__(self, obj: object, owner: type | None = None) -> object:
        if obj is None:
            return self
        return obj.__dict__.get(self.private)

    def __set__(self, obj: object, value: object) -> None:
        if value is None:
            obj.__dict__[self.private] = None
            return
        if isinstance(value, _CreationPropsGuard):
            obj.__dict__[self.private] = value
            return
        if _is_winforms_control(value):
            _hook_ensure_core_webview(value)
            try:
                stamp_webview2_creation_props(getattr(value, "CreationProperties", None))
            except Exception:
                pass
            obj.__dict__[self.private] = value
            return
        obj.__dict__[self.private] = _CreationPropsGuard(value)


def _patch_edge_init_args(edge: object) -> None:
    chrome = getattr(edge, "EdgeChrome", None)
    if chrome is None:
        return
    if getattr(chrome, "_vtm_no_mic_args", False):
        return
    try:
        chrome.webview = _WebviewSlot()
        chrome._vtm_no_mic_args = True
    except Exception:
        return


def _patch_edge_permission(edge: object) -> None:
    chrome = getattr(edge, "EdgeChrome", None)
    if chrome is None:
        return
    orig = chrome.on_webview_ready
    if getattr(orig, "_vtm_no_mic", False):
        return

    def on_webview_ready(self: object, sender: object, args: object) -> None:
        orig(self, sender, args)
        _attach_webview_media_block(sender)

    on_webview_ready._vtm_no_mic = True  # type: ignore[attr-defined]
    chrome.on_webview_ready = on_webview_ready


UI_PUBLIC_FILES = ("splash-art.png", "splash-mark.png", "favicon.svg", "favicon.ico", "icons.svg")


def ui_public_files(dist: Path) -> dict[str, Path]:
    """Vite public files copied to ui/dist that FastAPI must serve by name."""
    found: dict[str, Path] = {}
    for name in UI_PUBLIC_FILES:
        path = dist / name
        if path.is_file():
            found[name] = path
    return found


def splash_asset_dir() -> Path | None:
    """Folder that already contains splash-art.png (so relative <img> works)."""
    from .paths import package_root, ui_dist_dir

    for folder in (ui_dist_dir(), package_root() / "ui" / "public"):
        if (folder / "splash-art.png").is_file():
            return folder
    return None


def _public_png(name: str) -> Path | None:
    folder = splash_asset_dir()
    if folder is None:
        return None
    path = folder / name
    return path if path.is_file() else None


def splash_art_path() -> Path | None:
    return _public_png("splash-art.png")


def splash_mark_path() -> Path | None:
    return _public_png("splash-mark.png")


def write_early_splash(
    *,
    api_origin: str = "",
    error: str = "",
    label: str = "Starting…",
) -> Path | None:
    """Write splash HTML next to the PNGs. WebView2 blocks file:// images in html=."""
    folder = splash_asset_dir()
    if folder is None:
        return None
    path = folder / EARLY_SPLASH_NAME
    path.write_text(
        early_splash_html(api_origin=api_origin, error=error, label=label),
        encoding="utf-8",
    )
    return path


def early_splash_html(
    *,
    error: str = "",
    label: str = "Starting…",
    api_origin: str = "",
) -> str:
    """Splash HTML. Load from a file beside splash-art.png so the art appears immediately."""
    art = splash_art_path()
    mark = splash_mark_path()
    art_css = "background-image:url('splash-art.png');" if art is not None else ""
    mark_html = (
        '<img class="mark" src="splash-mark.png" width="72" height="72" alt="" />'
        if mark is not None
        else ""
    )
    notice = html.escape(error.strip()) if error.strip() else ""
    notice_html = f'<p class="err" id="err">{notice}</p>' if notice else '<p class="err" id="err" hidden></p>'
    line = html.escape(label or "Starting…")
    origin = html.escape((api_origin or "").strip().rstrip("/"), quote=True)
    version = html.escape(APP_VERSION)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <link rel="icon" type="image/png" href="splash-mark.png" />
  <title>VTM Studio</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #2b2b2b;
      color: #cfcfcf;
      font-family: "Segoe UI", "Yu Gothic UI", sans-serif;
    }}
    .krita-splash {{
      height: 100%;
      width: 100%;
      min-height: 100%;
      display: flex;
      overflow: hidden;
      background: #2b2b2b;
      user-select: none;
      cursor: grab;
      -webkit-app-region: drag;
    }}
    .krita-splash:active {{
      cursor: grabbing;
    }}
    .krita-splash-art {{
      flex: 1 1 auto;
      min-width: 0;
      background-color: #1a1a1a;
      {art_css}
      background-repeat: no-repeat;
      background-size: cover;
      background-position: 82% 38%;
    }}
    .krita-splash-panel {{
      position: relative;
      z-index: 1;
      width: min(34vw, 268px);
      flex: 0 0 min(34vw, 268px);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 22px;
      background: #333;
      border-left: 1px solid #1c1c1c;
    }}
    .krita-splash-plate {{
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 28px;
    }}
    .mark {{
      display: block;
      width: 72px;
      height: 72px;
      margin: 0 auto 12px;
      object-fit: contain;
      background: transparent;
    }}
    .kicker {{
      margin: 0;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-align: center;
      color: #8d8d8d;
    }}
    h1 {{
      margin: 4px 0 0;
      font-size: 22px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-align: center;
      color: #cfcfcf;
    }}
    .ver {{
      margin: 6px 0 0;
      color: #8d8d8d;
      font-size: 11px;
      text-align: center;
    }}
    .status {{
      width: 100%;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .line {{
      margin: 0;
      font-size: 12px;
      color: #8d8d8d;
      text-align: center;
    }}
    .bar {{
      height: 4px;
      border-radius: 0;
      background: #1f1f1f;
      border: 1px solid #1c1c1c;
      overflow: hidden;
    }}
    .bar b {{
      position: relative;
      display: block;
      width: 0%;
      height: 100%;
      background: #5a9fd6;
      border-radius: 0;
      overflow: hidden;
      transition: width 120ms linear;
    }}
    .bar b i {{
      position: absolute;
      top: 0;
      width: 28%;
      height: 100%;
      border-radius: 0;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.28), transparent);
      animation: bead-flow 1.35s ease-in-out infinite;
    }}
    @keyframes bead-flow {{
      0% {{ left: -28%; opacity: 0.25; }}
      18% {{ opacity: 1; }}
      82% {{ opacity: 1; }}
      100% {{ left: 100%; opacity: 0.25; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .bar b {{ transition: none; }}
      .bar b i {{ animation: none; left: 100%; opacity: 1; }}
    }}
    .err {{
      margin: 8px 0 0;
      color: #d08080;
      font-size: 12px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="krita-splash" role="status">
    <div class="krita-splash-art" aria-hidden="true"></div>
      <aside class="krita-splash-panel">
      <div class="krita-splash-plate">
      <header>
        {mark_html}
        <p class="kicker">VTM</p>
        <h1>Studio</h1>
        <p class="ver">{version}</p>
      </header>
      <div class="status">
        <p class="line" id="line">{line}</p>
        <div class="bar" id="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
          <b id="fill"><i></i></b>
        </div>
        {notice_html}
      </div>
      </div>
    </aside>
  </div>
  <script>
    const API = "{origin}";
    const WEIGHTS = {{ model: 0.46, character: 0.34, lab: 0.2 }};
    const started = Date.now();
    const lineEl = document.getElementById("line");
    const fillEl = document.getElementById("fill");
    const barEl = document.getElementById("bar");
    const errEl = document.getElementById("err");
    let lastPct = 0;
    let sawBoot = false;

    function startDrag() {{
      try {{
        const api = window.pywebview && window.pywebview.api;
        if (!api || !api.start_drag) return;
        const ret = api.start_drag();
        if (ret && typeof ret.then === "function") ret.catch(function () {{}});
      }} catch (e) {{}}
    }}

    document.addEventListener("mousedown", function (e) {{
      if (e.button === 0) startDrag();
    }});

    function clamp(n, lo, hi) {{
      return Math.max(lo, Math.min(hi, n));
    }}

    function paint(pct, label) {{
      const next = clamp(pct, 0, 100);
      lastPct = Math.max(lastPct, next);
      const shown = Math.round(lastPct);
      if (lineEl) lineEl.textContent = label || "Loading resources…";
      if (fillEl) fillEl.style.width = lastPct.toFixed(1) + "%";
      if (barEl) barEl.setAttribute("aria-valuenow", String(shown));
    }}

    function overall(boot) {{
      if (boot && typeof boot.progress === "number") {{
        let label = boot.progress_label || "Loading resources…";
        if (boot.ready) label = "Ready";
        return {{ pct: clamp(Number(boot.progress) * 100, 0, 100), label: label }};
      }}
      const stages = (boot && boot.stages) || {{}};
      let pct = 0;
      for (const key of ["model", "character", "lab"]) {{
        const row = stages[key] || {{}};
        pct += clamp(Number(row.progress) || 0, 0, 1) * WEIGHTS[key];
      }}
      let label = "Loading resources…";
      for (const key of ["lab", "character", "model"]) {{
        if ((stages[key] || {{}}).state === "run") {{
          label = stages[key].label || label;
          break;
        }}
      }}
      if (boot && boot.ready) label = "Ready";
      return {{ pct: clamp(pct, 0, 1) * 100, label: label }};
    }}

    window.__vtmSplash = function (boot) {{
      if (!boot) return;
      sawBoot = true;
      const bar = overall(boot);
      paint(boot.ready ? 100 : bar.pct, bar.label);
      if (errEl) {{
        const msg = String(boot.error || "").trim();
        errEl.hidden = !msg;
        errEl.textContent = msg;
      }}
    }};

    function creep() {{
      if (sawBoot) return;
      const elapsed = Date.now() - started;
      paint(Math.min(6, (elapsed / 10000) * 6), "Loading resources…");
    }}

    async function tick() {{
      if (API) {{
        try {{
          const res = await fetch(API + "/api/boot", {{ cache: "no-store" }});
          if (res.ok) {{
            window.__vtmSplash(await res.json());
            return;
          }}
        }} catch (e) {{}}
      }}
      creep();
    }}

    creep();
    setInterval(tick, 400);

    function pingReady() {{
      try {{
        const api = window.pywebview && window.pywebview.api;
        if (!api || !api.splash_ready) return false;
        const ret = api.splash_ready();
        if (ret && typeof ret.then === "function") ret.catch(function () {{}});
        return true;
      }} catch (e) {{
        return false;
      }}
    }}

    function afterPaint() {{
      requestAnimationFrame(function () {{
        requestAnimationFrame(function () {{
          if (pingReady()) return;
          let n = 0;
          const id = setInterval(function () {{
            n += 1;
            if (pingReady() || n > 40) clearInterval(id);
          }}, 50);
        }});
      }});
    }}

    window.addEventListener("pywebviewready", afterPaint);
    if (document.readyState === "complete") afterPaint();
    else window.addEventListener("load", afterPaint);
  </script>
</body>
</html>
"""
