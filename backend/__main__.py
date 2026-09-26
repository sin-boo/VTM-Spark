"""App entry: local FastAPI + native pywebview window (or API-only / Vite-dev)."""

from __future__ import annotations

import codecs


def _idna_encode(s: str, errors: str = "strict") -> tuple[bytes, int]:
    return s.encode("ascii", errors), len(s)


def _idna_decode(b: bytes, errors: str = "strict") -> tuple[str, int]:
    return b.decode("ascii", errors), len(b)


codecs.register(
    lambda name: codecs.CodecInfo(
        name="idna", encode=_idna_encode, decode=_idna_decode
    )
    if name == "idna"
    else None
)

import argparse
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _log_path() -> Path:
    return _ROOT / "models" / "vtm_noble.log"


def _file_log(msg: str) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _port_in_use(host: str, port: int) -> bool:
    """True if something already accepts connections on host:port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.25)
        target = "127.0.0.1" if host in {"localhost", "127.0.0.1", "::1"} else host
        s.connect((target, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _port_open(host: str, port: int) -> bool:
    """True when the local API is accepting connections (server is up)."""
    return _port_in_use(host, port)


def _can_bind(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0", "::", "::1"} else host
        s.bind((bind_host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _port_free(host: str, port: int) -> bool:
    """Exclusive bind test — no SO_REUSEADDR, which on Windows binds over a live listener."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0", "::", "::1"} else host
        s.bind((bind_host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _pick_port(host: str, preferred: int, *, span: int = 1) -> int | None:
    """Pick preferred port, optionally trying preferred+1… within span.

    Default span=1 refuses a second instance (port hopping used to leave
    multi-GB orphan backends stacking until the machine froze).
    Returns None when no port in the span is free.
    """
    bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0", "::", "::1"} else host
    span = max(1, int(span))
    for port in range(int(preferred), int(preferred) + span):
        if _can_bind(bind_host, port):
            if port != preferred:
                _file_log(f"port {preferred} busy — using {port}")
            return port
    return None


def _blocked_by_existing(host: str, port: int) -> str | None:
    """Why this port cannot be claimed. None = free enough to try a bind."""
    from backend.single_instance import health_url, probe_existing_api

    if _port_free(host, port):
        # Nothing listening. Skips a refused HTTP probe (~0.7s on Windows).
        return None
    if probe_existing_api(host, port):
        url = health_url(host, port).rsplit("/api/", 1)[0]
        return (
            f"VTM Noble already running at {url}. "
            "Close the other window, or menu [K] Kill leftovers."
        )
    if _port_in_use(host, port):
        return (
            f"Port {port} is in use by another program (often ui/_mock_boot.py). "
            "Close that process, or menu [K] Kill leftovers, then retry."
        )
    return None


def _wait_for_server(
    host: str,
    port: int,
    timeout: float = 180.0,
    thread: threading.Thread | None = None,
) -> bool:
    """True when THIS desk's /api/health is up. A TCP listener is not enough —

    ui/_mock_boot.py occupies :8765 during Vite work and answers ``{}``, which
    WebView2 then pretty-prints as a blank page.
    """
    from backend.single_instance import probe_existing_api

    deadline = time.time() + timeout
    while time.time() < deadline:
        if thread is not None and not thread.is_alive():
            _file_log("ERROR: API thread exited before bind")
            return False
        try:
            if probe_existing_api(host, port):
                return True
        except Exception as exc:
            _file_log(f"port check error: {exc}")
        time.sleep(0.25)
    return False


def _run_uvicorn(host: str, port: int, *, mount_ui: bool) -> None:
    try:
        from backend.desk_splash import ensure_stdio

        ensure_stdio()
        _file_log("importing uvicorn / backend.api…")
        from backend.paths import (
            configure_torch_compile_cache,
            ensure_import_paths,
            torch_train_dir,
        )

        configure_torch_compile_cache()
        ensure_import_paths()
        _file_log(f"torch_train={torch_train_dir()}")

        import uvicorn

        _file_log("imported uvicorn")
        from backend.api import app, configure_runtime, mount_frontend

        _file_log("imported backend.api")
        configure_runtime()
        _file_log("runtime configured")
        if mount_ui:
            mount_frontend()
            _file_log("frontend mounted")
        _file_log(f"uvicorn binding {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            reload=False,
            use_colors=False,
        )
    except Exception:
        _file_log("uvicorn crashed:\n" + traceback.format_exc())


def _shutdown_all(*, exit_code: int = 0) -> None:
    """Best-effort cleanup, then hard-exit.

    ``shutdown_runtime()`` can hang on camera/CUDA teardown. Never block the
    process forever — orphans were holding tens of GB of RAM after close.
    """
    def _cleanup() -> None:
        try:
            from backend.desk_splash import kill_orphan_webview2

            kill_orphan_webview2()
        except Exception:
            pass
        try:
            from backend.lab_process import stop_owned_lab

            stop_owned_lab()
        except Exception:
            pass
        try:
            from backend.api import shutdown_runtime

            shutdown_runtime()
        except Exception:
            try:
                _file_log("shutdown_runtime failed:\n" + traceback.format_exc())
            except Exception:
                pass

    t = threading.Thread(target=_cleanup, name="vtm-shutdown", daemon=True)
    t.start()
    t.join(timeout=2.0)
    try:
        import os

        os._exit(int(exit_code))
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VTM Noble desktop app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="API port (default 8765; second instance is refused unless --port-span>1)",
    )
    parser.add_argument(
        "--port-span",
        type=int,
        default=1,
        help="Ports to try after --port (default 1 = single instance; set >1 only if you really need multi)",
    )
    parser.add_argument(
        "--ui",
        choices=("webview", "none", "dev"),
        default="webview",
        help="webview=native window, none=API only, dev=open Vite URL in webview",
    )
    parser.add_argument(
        "--vite-url",
        default="http://127.0.0.1:5173",
        help="Vite dev server URL when --ui=dev",
    )
    parser.add_argument("--width", type=int, default=1380)
    parser.add_argument("--height", type=int, default=920)
    parser.add_argument("--splash-width", type=int, default=1000)
    parser.add_argument("--splash-height", type=int, default=562)
    args = parser.parse_args(argv)

    from backend.desk_splash import ensure_stdio

    ensure_stdio()
    _file_log(f"starting ui={args.ui}")

    # Cap CPU thread oversubscription before torch/onnx import (helps RAM/CPU thrash).
    for key, val in (
        ("OMP_NUM_THREADS", "4"),
        ("MKL_NUM_THREADS", "4"),
        ("OPENBLAS_NUM_THREADS", "4"),
        ("NUMEXPR_NUM_THREADS", "4"),
        ("TORCH_NUM_THREADS", "4"),
    ):
        os.environ.setdefault(key, val)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.name == "nt":
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    wait_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    mount_ui = args.ui == "webview"
    if args.ui == "dev":
        mount_ui = False

    def _claim_desk() -> str | None:
        """Take the single-instance lock and bind port. None = ok."""
        from backend.single_instance import acquire_single_instance

        blocked = _blocked_by_existing(wait_host, args.port)
        if blocked:
            _file_log("ERROR: " + blocked)
            return blocked
        if not acquire_single_instance():
            msg = (
                "Another VTM Noble instance holds the single-instance lock. "
                "Close it or run backend\\packaging\\kill-orphans.ps1."
            )
            _file_log("ERROR: " + msg)
            return msg
        picked = _pick_port(args.host, args.port, span=max(1, int(args.port_span)))
        if picked is None:
            msg = (
                f"Port {args.port} busy and --port-span={args.port_span} has no free port. "
                "Use menu [K] Kill leftovers, then retry."
            )
            _file_log("ERROR: " + msg)
            return msg
        args.port = picked
        _file_log(f"API will bind {args.host}:{args.port}")
        return None

    if args.ui == "none":
        blocked = _claim_desk()
        if blocked:
            return 2
        server = threading.Thread(
            target=_run_uvicorn,
            args=(args.host, args.port),
            kwargs={"mount_ui": mount_ui},
            daemon=True,
            name="uvicorn",
        )

        def _wait_api_none() -> str | None:
            if not _wait_for_server(
                wait_host, args.port, timeout=180.0, thread=server
            ):
                _file_log("ERROR: FastAPI failed to start")
                return None
            base = f"http://{wait_host}:{args.port}"
            _file_log(f"API ready at {base}/api/health")
            return base

        server.start()
        if _wait_api_none() is None:
            _shutdown_all(exit_code=1)
            return 1
        _file_log("API-only mode. Press Ctrl+C to stop.")
        try:
            while server.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        _shutdown_all(exit_code=0)
        return 0

    try:
        import webview
    except ImportError:
        _file_log(
            "pywebview is not installed. Run install.bat.\n"
            "Meanwhile use --ui=none and open the API in a browser."
        )
        _shutdown_all(exit_code=1)
        return 1

    from backend.desk_splash import (
        kill_orphan_webview2,
        mute_webview_microphone,
        patch_webview2_no_microphone,
    )

    mute_webview_microphone()
    patch_webview2_no_microphone()
    killed_wv = kill_orphan_webview2()
    if killed_wv:
        _file_log(f"cleared leftover WebView2 pids={killed_wv}")

    blocked = _claim_desk()
    if blocked:
        _file_log(blocked)
        return 2

    # torch / FastAPI import on the uvicorn thread after the splash is up;
    # an import failure lands on the splash as "Desk API failed to start".
    try:
        from backend.desk_splash import (
            WindowBridge,
            create_splash_window,
            desk_min_size,
            early_splash_html,
            keep_caption_hidden,
            open_desk,
            work_area_size,
            write_early_splash,
        )

        splash_w = max(720, int(args.splash_width))
        splash_h = max(420, int(args.splash_height))
        min_w, min_h = desk_min_size()
        work_w, work_h = work_area_size()
        splash_w = min(splash_w, work_w)
        splash_h = min(splash_h, work_h)
        desk_w = min(work_w, max(min_w, int(args.width)))
        desk_h = min(work_h, max(min_h, int(args.height)))
        api_origin = f"http://{wait_host}:{args.port}"
        splash_path = write_early_splash(api_origin=api_origin)
        _file_log("opening webview splash")
        bridge = WindowBridge()
        if splash_path is not None:
            window = create_splash_window(
                webview,
                url=splash_path.as_uri(),
                width=splash_w,
                height=splash_h,
                js_api=bridge,
            )
        else:
            window = create_splash_window(
                webview,
                html=early_splash_html(api_origin=api_origin),
                width=splash_w,
                height=splash_h,
                js_api=bridge,
            )
        bridge.bind(window)

        def _on_window_closed() -> None:
            _file_log("webview closed — forcing process exit")
            _shutdown_all(exit_code=0)

        try:
            window.events.closed += _on_window_closed
        except Exception:
            try:
                window.events.closing += lambda: _on_window_closed()
            except Exception:
                pass

        server = threading.Thread(
            target=_run_uvicorn,
            args=(args.host, args.port),
            kwargs={"mount_ui": mount_ui},
            daemon=True,
            name="uvicorn",
        )

        def _wait_api() -> str | None:
            if not _wait_for_server(
                wait_host, args.port, timeout=180.0, thread=server
            ):
                _file_log("ERROR: FastAPI failed to start")
                return None
            base = f"http://{wait_host}:{args.port}"
            _file_log(f"API ready at {base}/api/health")
            return base

        def _after_gui() -> None:
            # Show first, then return to the WinForms pump. Caption/WebView2
            # COM from this callback (before the loop runs) leaves a hidden
            # HWND — taskbar flicker, then the desk never appears.
            try:
                window.show()
            except Exception:
                pass
            _file_log("splash shown")

            def _chrome() -> None:
                time.sleep(0.08)
                keep_caption_hidden(window, resizable=False)

            threading.Thread(target=_chrome, name="vtm-chrome", daemon=True).start()
            server.start()

            def _handoff() -> None:
                base = _wait_api()
                if base is None:
                    try:
                        window.load_html(
                            early_splash_html(error="Desk API failed to start")
                        )
                    except Exception:
                        pass
                    return
                url = args.vite_url if args.ui == "dev" else base
                _file_log(f"API ready — splash stays until boot finishes, then {url}")
                from backend.stream import get_runtime

                for _ in range(9000):
                    try:
                        if get_runtime().boot_snapshot().get("ready"):
                            break
                    except Exception:
                        pass
                    time.sleep(0.2)
                else:
                    _file_log("boot wait timed out — loading desk anyway")
                _file_log(f"splash handing off -> {url}")
                open_desk(window, url, desk_w, desk_h)

            threading.Thread(
                target=_handoff, name="vtm-splash-handoff", daemon=True
            ).start()

        webview.start(_after_gui)
    except Exception:
        _file_log("webview crashed:\n" + traceback.format_exc())
        _shutdown_all(exit_code=1)
        return 1
    _shutdown_all(exit_code=0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        _file_log("fatal:\n" + traceback.format_exc())
        _shutdown_all(exit_code=1)
        raise
