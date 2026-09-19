from pathlib import Path
import os

from backend.desk_splash import (
    EARLY_SPLASH_NAME,
    early_splash_html,
    splash_art_path,
    ui_public_files,
    write_early_splash,
)


def test_desk_applies_hat_window_icon() -> None:
    text = Path(__file__).resolve().parents[1] / "desk_splash.py"
    src = text.read_text(encoding="utf-8")
    assert "apply_window_icon" in src
    assert "claim_app_id" in src
    assert "splash-mark.png" in src
    assert "rel=\"icon\"" in src


def test_ui_public_files_maps_splash_art(tmp_path: Path) -> None:
    (tmp_path / "splash-art.png").write_bytes(b"png")
    (tmp_path / "favicon.svg").write_text("<svg />", encoding="utf-8")
    found = ui_public_files(tmp_path)
    assert found["splash-art.png"] == tmp_path / "splash-art.png"
    assert found["favicon.svg"] == tmp_path / "favicon.svg"
    assert "icons.svg" not in found


def test_early_splash_html_is_local_and_named() -> None:
    page = early_splash_html(label="Loading model")
    assert "VTM Noble" in page or "Noble" in page
    assert "krita-splash" in page
    assert "Loading model" in page
    assert "splash-art.png" in page
    assert "file://" not in page
    assert "Tracking stays idle" not in page
    assert 'id="bar"' in page
    assert "0.1 alpha" in page
    assert "win-dots" not in page
    assert "win-caption" not in page
    assert "aria-label=\"Minimize\"" not in page
    assert "start_drag" in page
    assert "splash_ready" in page
    assert "pywebviewready" in page
    assert "Operator desk" not in page
    assert "2026" not in page
    assert "background-position: 82% 38%" in page
    assert "__vtmSplash" in page
    assert "bead-flow" in page
    assert "<i></i>" in page
    art = splash_art_path()
    if art is not None:
        assert "url('splash-art.png')" in page


def test_early_splash_html_embeds_api_origin() -> None:
    page = early_splash_html(api_origin="http://127.0.0.1:8765/")
    assert 'const API = "http://127.0.0.1:8765"' in page
    assert "/api/boot" in page


def test_write_early_splash_sits_beside_png(tmp_path: Path, monkeypatch) -> None:
    art = tmp_path / "splash-art.png"
    art.write_bytes(b"png")
    (tmp_path / "splash-mark.png").write_bytes(b"png")
    monkeypatch.setattr("backend.desk_splash.splash_asset_dir", lambda: tmp_path)
    path = write_early_splash(api_origin="http://127.0.0.1:8765")
    assert path == tmp_path / EARLY_SPLASH_NAME
    text = path.read_text(encoding="utf-8")
    assert "splash-art.png" in text
    assert "file://" not in text
    assert "http://127.0.0.1:8765" in text


def test_early_splash_html_escapes_error() -> None:
    page = early_splash_html(error="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_mute_webview_microphone_sets_disable_audio_input(monkeypatch) -> None:
    from backend.desk_splash import WEBVIEW2_ARGS, mute_webview_microphone

    monkeypatch.delenv(WEBVIEW2_ARGS, raising=False)
    mute_webview_microphone()
    assert "--disable-audio-input" in os.environ[WEBVIEW2_ARGS].split()
    mute_webview_microphone()
    assert os.environ[WEBVIEW2_ARGS].count("--disable-audio-input") == 1


def test_merge_webview2_args_appends_once() -> None:
    from backend.desk_splash import merge_webview2_args

    merged = merge_webview2_args("--disable-features=ElasticOverscroll")
    assert "--disable-features=ElasticOverscroll" in merged
    assert merged.count("--disable-audio-input") == 1
    assert merge_webview2_args(merged) == merged
    assert "--disable-gpu" not in merged


def test_should_deny_webview_mic_and_camera() -> None:
    from backend.desk_splash import should_deny_webview_permission

    assert should_deny_webview_permission(1) is True
    assert should_deny_webview_permission(2) is True
    assert should_deny_webview_permission(3) is False
    assert should_deny_webview_permission("Microphone") is True


def test_patch_edge_init_injects_disable_audio_input() -> None:
    from backend import desk_splash

    class Chrome:
        def __init__(self, form, window, cache_dir):
            props = type("P", (), {})()
            props.AdditionalBrowserArguments = "--disable-features=ElasticOverscroll"
            self.webview = type("W", (), {})()
            self.webview.CreationProperties = props

        def on_webview_ready(self, sender, args):
            return None

    class Edge:
        EdgeChrome = Chrome

    desk_splash._patch_edge_init_args(Edge)
    inst = Edge.EdgeChrome(None, None, "")
    args = inst.webview.CreationProperties.AdditionalBrowserArguments
    assert "--disable-features=ElasticOverscroll" in args
    assert "--disable-audio-input" in args
    assert inst.webview.CreationProperties.ExclusiveUserDataFolderAccess is True


def test_patch_edge_init_works_without_python_source(monkeypatch) -> None:
    import inspect

    from backend import desk_splash

    def boom(*_a, **_k):
        raise OSError("could not find source")

    monkeypatch.setattr(inspect, "getsource", boom)

    class Chrome:
        def __init__(self, form, window, cache_dir):
            props = type("P", (), {})()
            props.AdditionalBrowserArguments = "--disable-features=ElasticOverscroll"
            self.webview = type("W", (), {})()
            self.webview.CreationProperties = props

    class Edge:
        EdgeChrome = Chrome

    desk_splash._patch_edge_init_args(Edge)
    inst = Edge.EdgeChrome(None, None, "")
    args = inst.webview.CreationProperties.AdditionalBrowserArguments
    assert "--disable-audio-input" in args
    assert inst.webview.CreationProperties.ExclusiveUserDataFolderAccess is True


def test_webview_media_deny_origins_includes_file(monkeypatch, tmp_path: Path) -> None:
    from backend import desk_splash

    art = tmp_path / "splash-art.png"
    art.write_bytes(b"png")
    monkeypatch.setattr(desk_splash, "splash_asset_dir", lambda: tmp_path)
    origins = desk_splash.webview_media_deny_origins("file:///tmp/_early_splash.html")
    assert "file:///" in origins
    assert "http://127.0.0.1:8765/" in origins
    assert any(item.endswith("_early_splash.html") or item.endswith("_early_splash.html/") for item in origins)


def test_webview2_cmdline_only_matches_pywebview_folder() -> None:
    from pathlib import Path

    from backend.desk_splash import _webview2_cmdline_is_ours

    ours = Path(r"C:\Users\me\AppData\Local\pywebview")
    ours_cmd = (
        r"msedgewebview2.exe --user-data-dir=C:\Users\me\AppData\Local\pywebview "
        r"--embedded-browser-webview=1"
    )
    other = (
        r"msedgewebview2.exe --user-data-dir=C:\Users\me\AppData\Local\Cursor "
        r"--embedded-browser-webview=1"
    )
    assert _webview2_cmdline_is_ours(ours_cmd, [ours]) is True
    assert _webview2_cmdline_is_ours(other, [ours]) is False


def test_kill_orphan_webview2_kills_matching_pid(monkeypatch) -> None:
    import sys

    from backend import desk_splash

    class Proc:
        def __init__(self) -> None:
            self.info = {
                "pid": 4242,
                "name": "msedgewebview2.exe",
                "cmdline": [
                    "msedgewebview2.exe",
                    "--user-data-dir=C:\\Users\\me\\AppData\\Local\\pywebview",
                ],
            }
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    proc = Proc()

    class FakePsutil:
        class Error(Exception):
            pass

        @staticmethod
        def process_iter(_attrs: object) -> list:
            return [proc]

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    killed = desk_splash.kill_orphan_webview2()
    assert killed == [4242]
    assert proc.killed is True


def test_main_patches_webview_after_import() -> None:
    src = Path(__file__).resolve().parents[1] / "__main__.py"
    text = src.read_text(encoding="utf-8")
    assert text.find("import webview") < text.find("patch_webview2_no_microphone")
    assert "kill_orphan_webview2" in text


def test_mute_webview_microphone_keeps_existing_args(monkeypatch) -> None:
    from backend.desk_splash import WEBVIEW2_ARGS, mute_webview_microphone

    monkeypatch.setenv(WEBVIEW2_ARGS, "--disable-features=ElasticOverscroll")
    mute_webview_microphone()
    flags = os.environ[WEBVIEW2_ARGS].split()
    assert "--disable-features=ElasticOverscroll" in flags
    assert "--disable-audio-input" in flags
    assert "--disable-gpu" not in flags
    assert "--single-process" not in flags


def test_create_splash_window_skips_webview2_hang_flags() -> None:
    from backend.desk_splash import WindowBridge, create_splash_window, work_area_size

    class FakeWebview:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def create_window(self, title, url=None, html=None, **kwargs):
            self.kwargs = {"title": title, "url": url, "html": html, **kwargs}
            return self.kwargs

    fake = FakeWebview()
    bridge = WindowBridge()
    window = create_splash_window(
        fake,
        url="file:///tmp/splash.html",
        width=1000,
        height=562,
        js_api=bridge,
    )
    assert window["title"] == "VTM Noble"
    assert "transparent" not in window
    assert "frameless" not in window
    assert window["js_api"] is bridge
    assert window["resizable"] is False
    assert window["hidden"] is True
    assert window["shadow"] is False
    assert window["background_color"] == "#2B2B2B"
    work_w, work_h = work_area_size()
    assert window["width"] == min(1000, work_w)
    assert window["height"] == min(562, work_h)
    assert window["min_size"] == (window["width"], window["height"])
    assert isinstance(window["x"], int)
    assert isinstance(window["y"], int)


def test_window_bridge_close_and_minimize() -> None:
    from backend.desk_splash import WindowBridge

    class FakeWindow:
        def __init__(self) -> None:
            self.destroyed = False
            self.minimized = False

        def destroy(self) -> None:
            self.destroyed = True

        def minimize(self) -> None:
            self.minimized = True

    bridge = WindowBridge()
    bridge.close()
    bridge.minimize()
    fake = FakeWindow()
    bridge.bind(fake)
    bridge.minimize()
    bridge.close()
    assert fake.minimized is True
    assert fake.destroyed is True


def test_window_bridge_toggle_max_falls_back() -> None:
    from backend.desk_splash import WindowBridge

    class FakeWindow:
        def __init__(self) -> None:
            self.toggled = False

        def toggle_fullscreen(self) -> None:
            self.toggled = True

    fake = FakeWindow()
    bridge = WindowBridge()
    bridge.bind(fake)
    bridge.toggle_max()
    assert fake.toggled is True


def test_window_bridge_start_drag_without_hwnd() -> None:
    from backend.desk_splash import WindowBridge

    bridge = WindowBridge()
    bridge.start_drag()
    bridge.bind(object())
    bridge.start_drag()
    bridge.start_resize("left")
    bridge.start_resize("topright")
    bridge.begin_resize("bottom")
    bridge.move_resize()
    bridge.end_resize()


def test_resize_edge_names() -> None:
    from backend.desk_splash import (
        HTCLIENT,
        HTLEFT,
        HTTOPRIGHT,
        HTBOTTOMRIGHT,
        _resize_edge_hit,
    )

    assert _resize_edge_hit("left") == HTLEFT
    assert _resize_edge_hit("ne") == HTTOPRIGHT
    assert _resize_edge_hit("bottom-right") == HTBOTTOMRIGHT
    assert _resize_edge_hit("nope") == HTCLIENT


def test_desk_min_size_fits_work_area() -> None:
    from backend.desk_splash import desk_min_size, work_area_size

    work_w, work_h = work_area_size()
    min_w, min_h = desk_min_size()
    assert min_w <= work_w
    assert min_h <= work_h
    assert min_w >= 800
    assert min_h >= 560


def test_splash_origin_centers_on_work_area(monkeypatch) -> None:
    from backend.desk_splash import splash_origin

    monkeypatch.setattr("backend.desk_splash.work_area_rect", lambda: (0, 0, 1920, 1080))
    x, y = splash_origin(1000, 562)
    assert x == (1920 - 1000) // 2
    assert y == (1080 - 562) // 2

    monkeypatch.setattr("backend.desk_splash.work_area_rect", lambda: (80, 40, 1600, 900))
    x, y = splash_origin(800, 500)
    assert x == 80 + (1600 - 800) // 2
    assert y == 40 + (900 - 500) // 2


def test_set_desk_frame_resizes_and_moves() -> None:
    from backend.desk_splash import set_desk_frame, work_area_size

    class Fake:
        def __init__(self) -> None:
            self.size = None
            self.pos = None

        def resize(self, w, h) -> None:
            self.size = (w, h)

        def move(self, x, y) -> None:
            self.pos = (x, y)

    fake = Fake()
    set_desk_frame(fake, 1380, 920)
    work_w, work_h = work_area_size()
    assert fake.size is not None
    assert fake.size[0] <= work_w
    assert fake.size[1] <= work_h
    assert fake.pos is not None
    assert fake.resizable is True


def test_desk_handoff_url_marks_operator_desk() -> None:
    from backend.desk_splash import desk_handoff_url

    assert desk_handoff_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765?desk=1"
    assert desk_handoff_url("http://127.0.0.1:8765/") == "http://127.0.0.1:8765/?desk=1"
    assert desk_handoff_url("http://127.0.0.1:5173?x=1") == "http://127.0.0.1:5173?x=1&desk=1"
    assert desk_handoff_url("http://127.0.0.1:8765?desk=1") == "http://127.0.0.1:8765?desk=1"


def test_open_desk_hides_until_loaded(monkeypatch) -> None:
    import backend.desk_splash as desk_splash
    from backend.desk_splash import open_desk

    monkeypatch.setattr(desk_splash, "work_area_rect", lambda: (0, 0, 1920, 1080))

    class Event:
        def __init__(self) -> None:
            self.handlers = []

        def __iadd__(self, fn):
            self.handlers.append(fn)
            return self

    class Fake:
        def __init__(self) -> None:
            self.hidden = 0
            self.shown = 0
            self.url = None
            self.size = None
            self.pos = None
            self.resizable = False
            self.events = type("Events", (), {})()
            self.events.loaded = Event()

        def hide(self) -> None:
            self.hidden += 1

        def show(self) -> None:
            self.shown += 1

        def load_url(self, url: str) -> None:
            self.url = url

        def resize(self, w, h) -> None:
            self.size = (w, h)

        def move(self, x, y) -> None:
            self.pos = (x, y)

    fake = Fake()
    desk_splash._HOLD_REVEAL = False
    open_desk(fake, "http://127.0.0.1:8765", 1380, 920, timeout=60)
    assert fake.hidden == 1
    assert fake.shown == 0
    assert fake.url == "http://127.0.0.1:8765?desk=1"
    assert fake.size == (1380, 920)
    assert fake.resizable is True
    fake.events.loaded.handlers[0]()
    assert fake.shown == 1
    fake.events.loaded.handlers[0]()
    assert fake.shown == 1
    desk_splash._HOLD_REVEAL = False


def test_desk_rect_is_larger_than_splash(monkeypatch) -> None:
    from backend.desk_splash import _desk_rect, splash_origin

    monkeypatch.setattr("backend.desk_splash.work_area_rect", lambda: (0, 0, 1920, 1080))
    splash = splash_origin(1000, 562)
    desk = _desk_rect(1380, 920)
    assert desk[2] > 1000
    assert desk[3] > 562
    assert desk[2] == 1380
    assert desk[3] == 920
    assert splash != (desk[0], desk[1])


def test_main_handoff_opens_desk_not_same_window_size() -> None:
    import inspect
    from pathlib import Path

    from backend import __main__ as main

    src = inspect.getsource(main.main)
    assert "open_desk(" in src
    assert "set_desk_frame(window" not in src
    app = Path(__file__).resolve().parents[2] / "ui" / "src" / "App.tsx"
    text = app.read_text(encoding="utf-8")
    assert "DESK_HANDOFF" in text
    assert "has('desk')" in text
    assert "WindowResize" in text
    assert "desk-shell\" aria-busy" not in text


def test_maximize_window_without_hwnd() -> None:
    from backend.desk_splash import maximize_window

    class Fake:
        def __init__(self) -> None:
            self.maximized = False

        def maximize(self) -> None:
            self.maximized = True

    fake = Fake()
    maximize_window(fake)
    assert fake.maximized is True


def test_hide_native_caption_without_hwnd() -> None:
    from backend.desk_splash import _fill_browser, hide_native_caption

    _fill_browser(object())
    hide_native_caption(object())


def test_dwm_border_is_color_none() -> None:
    import inspect

    from backend.desk_splash import (
        DESK_FILL_RGB,
        DWMWA_BORDER_COLOR,
        DWMWA_COLOR_NONE,
        DWMWCP_DONOTROUND,
        SPLASH_FILL_RGB,
        WS_EX_EDGE,
        _apply_frame_style,
        _colorref,
        _frame_fill_rgb,
        _paint_frame_dark,
        _round_hwnd,
        early_splash_html,
    )
    import backend.desk_splash as desk_splash

    assert DWMWA_COLOR_NONE == 0xFFFFFFFE
    assert DWMWA_BORDER_COLOR == 34
    assert DWMWCP_DONOTROUND == 1
    assert WS_EX_EDGE & 0x00000100
    assert _colorref(SPLASH_FILL_RGB) == 0x002B2B2B
    assert _colorref(DESK_FILL_RGB) == 0x002B2B2B
    prev = desk_splash._CHROME_RESIZABLE
    try:
        desk_splash._CHROME_RESIZABLE = False
        assert _frame_fill_rgb() == SPLASH_FILL_RGB
        desk_splash._CHROME_RESIZABLE = True
        assert _frame_fill_rgb() == DESK_FILL_RGB
    finally:
        desk_splash._CHROME_RESIZABLE = prev
    paint = inspect.getsource(_paint_frame_dark)
    assert "DWMWA_COLOR_NONE" in paint
    assert "_colorref(fill)" in paint
    assert "DWMWCP_DONOTROUND" in paint
    assert "DWMWA_USE_IMMERSIVE_DARK_MODE_WIN10" in paint
    assert "DwmExtendFrameIntoClientArea" in paint
    assert "_MARGINS(0, 0, 0, 0)" in paint
    assert 'SetWindowTheme(hwnd, "", "")' not in paint
    tint = inspect.getsource(desk_splash._tint_form)
    assert "DefaultBackgroundColor" in tint
    clip = inspect.getsource(_round_hwnd)
    assert "CreateRoundRectRgn(0, 0, width + 1, height + 1," in clip
    assert "IsZoomed" in clip
    frame = inspect.getsource(_apply_frame_style)
    assert "WS_EX_EDGE" in frame
    assert "WS_THICKFRAME | WS_MINIMIZEBOX" not in frame
    assert "_hook_client_fill" in frame
    assert "_hook_child_passthrough" in frame
    hide = inspect.getsource(desk_splash.hide_native_caption)
    assert "FormBorderStyle" not in hide
    assert "_style_matches" in hide
    assert "_hook_child_passthrough" in hide
    hook = inspect.getsource(desk_splash._hook_client_fill)
    assert "WM_NCCALCSIZE" in hook
    assert "WM_NCHITTEST" in hook
    assert "WM_WINDOWPOSCHANGED" not in hook
    assert "_nc_hit_test" in hook
    assert "CallWindowProcW" in hook
    create = inspect.getsource(desk_splash.create_splash_window)
    assert '"shadow": False' in create
    assert '"background_color": "#2B2B2B"' in create
    assert "#F6F1E8" not in create
    reveal = inspect.getsource(desk_splash.open_desk)
    assert "_retry_caption(window)" in reveal
    keep = inspect.getsource(desk_splash.keep_caption_hidden)
    assert "ResizeEnd" in keep
    assert "_after_user_frame" in keep
    assert "_round_live" in keep
    assert "LocationChanged" not in keep
    run = inspect.getsource(desk_splash._run_on_gui)
    assert "wait" in run
    assert "Invoke" in run
    drag = inspect.getsource(desk_splash.WindowBridge.start_drag)
    assert "_pin_after_frame_change" in drag
    assert "wait=True" in drag
    fill = inspect.getsource(desk_splash._fill_browser)
    assert "PerformLayout" in fill
    assert "_notify_webview_moved" in fill
    notify = inspect.getsource(desk_splash._notify_webview_moved)
    assert "NotifyParentWindowPositionChanged" in notify
    assert "GetField" not in notify
    assert "_sync_webview" not in hide
    page = early_splash_html()
    assert "height: 100vh" not in page
    assert "width: 100%" in page


def test_window_bridge_js_api_does_not_expose_gui() -> None:
    """pywebview recursively inspects public js_api attrs. A public window
    pointer walks WinForms native.Bounds.Empty until the splash dies."""
    import inspect

    from backend.desk_splash import WindowBridge

    class Rect:
        __module__ = "System.Drawing"

        @property
        def Empty(self):
            return Rect()

    class Native:
        __module__ = "System.Windows.Forms"

        def __init__(self) -> None:
            self.Bounds = Rect()

    class FakeWindow:
        __module__ = "webview.window"

        def __init__(self) -> None:
            self.native = Native()

    bridge = WindowBridge()
    bridge.bind(FakeWindow())
    public = [name for name in dir(bridge) if not name.startswith("_")]
    assert "window" not in public

    exposed: list[int] = []
    functions: dict[str, bool] = {}

    def get_functions(obj: object, base_name: str = "") -> None:
        obj_id = id(obj)
        if obj_id in exposed:
            return
        exposed.append(obj_id)
        for name in dir(obj):
            if name.startswith("_"):
                continue
            attr = getattr(obj, name)
            full_name = f"{base_name}.{name}" if base_name else name
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                functions[full_name] = True
            elif inspect.isclass(attr) or (
                isinstance(attr, object) and not callable(attr) and hasattr(attr, "__module__")
            ):
                get_functions(attr, full_name)

    get_functions(bridge)
    assert "close" in functions
    assert "start_drag" in functions
    assert "start_resize" in functions
    assert "begin_resize" in functions
    assert "move_resize" in functions
    assert "end_resize" in functions
    assert "splash_ready" in functions
    assert not any("native" in key or "Empty" in key or "Bounds" in key for key in functions)


def test_hook_client_fill_skips_without_hwnd() -> None:
    from backend.desk_splash import (
        WM_NCCALCSIZE,
        WM_NCHITTEST,
        _NCCALC_HOOKS,
        _hook_client_fill,
        _style_matches,
    )

    before = dict(_NCCALC_HOOKS)
    _hook_client_fill(0)
    assert _NCCALC_HOOKS == before
    assert WM_NCCALCSIZE == 0x0083
    assert WM_NCHITTEST == 0x0084
    assert _style_matches(0, resizable=True) is False


def test_resize_hit_zones() -> None:
    from backend.desk_splash import (
        HTBOTTOM,
        HTBOTTOMLEFT,
        HTBOTTOMRIGHT,
        HTCLIENT,
        HTLEFT,
        HTRIGHT,
        HTTOP,
        HTTOPLEFT,
        HTTOPRIGHT,
        _resize_hit,
        _unpack_nchittest,
    )

    w, h, b = 400, 300, 8
    assert _resize_hit(0, 0, w, h, b) == HTTOPLEFT
    assert _resize_hit(w - 1, 0, w, h, b) == HTTOPRIGHT
    assert _resize_hit(0, h - 1, w, h, b) == HTBOTTOMLEFT
    assert _resize_hit(w - 1, h - 1, w, h, b) == HTBOTTOMRIGHT
    assert _resize_hit(0, 40, w, h, b) == HTLEFT
    assert _resize_hit(w - 1, 40, w, h, b) == HTRIGHT
    assert _resize_hit(40, 0, w, h, b) == HTTOP
    assert _resize_hit(40, h - 1, w, h, b) == HTBOTTOM
    # Caption buttons live here — must stay HTCLIENT so DWM does not draw OS chrome.
    assert _resize_hit(w - 24, 16, w, h, b) == HTCLIENT
    assert _resize_hit(200, 150, w, h, b) == HTCLIENT
    assert _resize_hit(10, 10, w, h, 0) == HTCLIENT
    x, y = _unpack_nchittest(0x000A0005)
    assert (x, y) == (5, 10)
    x, y = _unpack_nchittest(0xFFFEFFFF)
    assert (x, y) == (-1, -2)


def test_resize_rect_from_delta_keeps_min_and_anchor() -> None:
    from backend.desk_splash import (
        HTBOTTOM,
        HTLEFT,
        HTTOPLEFT,
        HTRIGHT,
        _resize_rect_from_delta,
    )

    # Right/bottom grow.
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTRIGHT, 50, 0, 200, 150)
    assert (x, y, w, h) == (100, 80, 450, 300)
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTBOTTOM, 0, 40, 200, 150)
    assert (x, y, w, h) == (100, 80, 400, 340)

    # Left/top keep the opposite edge.
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTLEFT, 30, 0, 200, 150)
    assert (x, y, w, h) == (130, 80, 370, 300)
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTTOPLEFT, 10, 20, 200, 150)
    assert (x, y, w, h) == (110, 100, 390, 280)

    # Shrinking past min restores the dragged edge.
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTLEFT, 300, 0, 200, 150)
    assert w == 200
    assert x + w == 500
    x, y, w, h = _resize_rect_from_delta(100, 80, 400, 300, HTTOPLEFT, 0, 400, 200, 150)
    assert h == 150
    assert y + h == 380


def test_live_resize_helpers_without_hwnd() -> None:
    from backend.desk_splash import (
        HTBOTTOMRIGHT,
        _LIVE_RESIZE,
        _begin_live_resize,
        _end_live_resize,
        _move_live_resize,
        _pin_after_frame_change,
    )

    _LIVE_RESIZE.clear()
    _begin_live_resize(object(), HTBOTTOMRIGHT, "se")
    _move_live_resize(object())
    _end_live_resize(object())
    _pin_after_frame_change(object())
    assert _LIVE_RESIZE == {}


def test_nc_hit_test_skips_when_not_resizable() -> None:
    import backend.desk_splash as desk_splash

    prev = desk_splash._CHROME_RESIZABLE
    try:
        desk_splash._CHROME_RESIZABLE = False
        assert desk_splash._nc_hit_test(1, 0) == desk_splash.HTCLIENT
    finally:
        desk_splash._CHROME_RESIZABLE = prev


def test_sync_webview_without_native() -> None:
    from backend.desk_splash import _notify_webview_moved, _request_webview_sync, _sync_webview

    _notify_webview_moved(object())
    _sync_webview(object())
    _request_webview_sync(object())


def test_keep_caption_hidden_binds_show_events() -> None:
    from backend.desk_splash import keep_caption_hidden

    class Event:
        def __init__(self) -> None:
            self.handlers = []

        def __iadd__(self, fn):
            self.handlers.append(fn)
            return self

    class Fake:
        def __init__(self) -> None:
            self.events = type("Events", (), {})()
            self.events.shown = Event()
            self.events.loaded = Event()

    fake = Fake()
    keep_caption_hidden(fake)
    assert len(fake.events.shown.handlers) == 1
    assert len(fake.events.loaded.handlers) == 1


def test_reveal_when_painted_shows_once() -> None:
    from backend.desk_splash import WindowBridge, reveal_when_painted

    class Event:
        def __init__(self) -> None:
            self.handlers = []

        def __iadd__(self, fn):
            self.handlers.append(fn)
            return self

    class Fake:
        def __init__(self) -> None:
            self.shown = 0
            self.events = type("Events", (), {})()
            self.events.shown = Event()
            self.events.loaded = Event()

        def show(self) -> None:
            self.shown += 1

    fake = Fake()
    reveal_when_painted(fake, timeout=60)
    assert fake.events.loaded.handlers == []
    bridge = WindowBridge()
    bridge.bind(fake)
    bridge.splash_ready()
    bridge.splash_ready()
    assert fake.shown == 1


def test_ensure_stdio_restores_isatty(monkeypatch) -> None:
    import io
    import sys

    from backend.desk_splash import ensure_stdio

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    ensure_stdio()
    assert sys.stdout is not None
    assert sys.stderr is not None
    assert hasattr(sys.stdout, "isatty")
    assert sys.stdout.isatty() in (True, False)
    if isinstance(sys.stdout, io.TextIOBase):
        sys.stdout.write("")


def test_after_gui_shows_splash_before_caption_hooks() -> None:
    """Caption COM on the webview.start callback deadlocks a hidden HWND."""
    import inspect

    from backend import __main__ as main

    src = inspect.getsource(main.main)
    after = src[src.index("def _after_gui") :]
    assert "window.show()" in after
    assert "vtm-chrome" in after
    assert "target=_chrome" in after
    assert after.index("window.show()") < after.index("keep_caption_hidden")


def test_wait_for_server_stops_when_api_thread_dies() -> None:
    import threading

    from backend import __main__ as main

    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    assert main._wait_for_server("127.0.0.1", 1, timeout=2.0, thread=dead) is False
