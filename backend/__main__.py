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
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "vtm_noble.log"
    return _ROOT / "data" / "vtm_noble.log"


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


def _wait_for_server(host: str, port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if _port_open(host, port):
                return True
        except Exception as exc:
            _file_log(f"port check error: {exc}")
        time.sleep(0.25)
    return False


def _run_uvicorn(host: str, port: int, *, mount_ui: bool) -> None:
    try:
        _file_log("importing uvicorn / backend.api…")
        from backend.paths import ensure_import_paths, torch_train_dir

        ensure_import_paths()
        _file_log(f"torch_train={torch_train_dir()}")

        import uvicorn

        from backend.api import app, configure_runtime, mount_frontend

        configure_runtime()
        if mount_ui:
            mount_frontend()
        _file_log(f"uvicorn binding {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning", reload=False)
    except Exception:
        _file_log("uvicorn crashed:\n" + traceback.format_exc())


def _shutdown_all(*, exit_code: int = 0) -> None:
    """Best-effort cleanup, then hard-exit.

    ``shutdown_runtime()`` can hang on camera/CUDA teardown. Never block the
    process forever — orphans were holding tens of GB of RAM after close.
    """
    def _cleanup() -> None:
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
    args = parser.parse_args(argv)

    _file_log(f"starting ui={args.ui} frozen={getattr(sys, 'frozen', False)}")

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

    from backend.single_instance import (
        acquire_single_instance,
        health_url,
        probe_existing_api,
    )

    wait_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    if probe_existing_api(wait_host, args.port):
        url = health_url(wait_host, args.port).rsplit("/api/", 1)[0]
        _file_log(
            f"ERROR: VTM Noble already running at {url} — refusing second instance "
            f"(this was stacking multi-GB backends and freezing the PC). "
            f"Close the other window, or menu [K] Kill leftovers."
        )
        if getattr(sys, "frozen", False) or args.ui == "webview":
            try:
                input("Already running. Press Enter to close…")
            except Exception:
                pass
        return 2

    if not acquire_single_instance():
        _file_log(
            "ERROR: another VTM Noble instance holds the single-instance lock. "
            "Close it or run packaging\\kill-orphans.ps1 ([K] in start.bat)."
        )
        if getattr(sys, "frozen", False) or args.ui == "webview":
            try:
                input("Already running. Press Enter to close…")
            except Exception:
                pass
        return 2

    mount_ui = args.ui == "webview"
    if args.ui == "dev":
        mount_ui = False

    picked = _pick_port(args.host, args.port, span=max(1, int(args.port_span)))
    if picked is None:
        _file_log(
            f"ERROR: port {args.port} busy and --port-span={args.port_span} has no free port. "
            f"Refusing to start another backend (prevents RAM pile-up). "
            f"Use menu [K] Kill leftovers, then retry."
        )
        if getattr(sys, "frozen", False) or args.ui == "webview":
            try:
                input("Port busy. Press Enter to close…")
            except Exception:
                pass
        return 2
    args.port = picked
    _file_log(f"API will bind {args.host}:{args.port}")

    server = threading.Thread(
        target=_run_uvicorn,
        args=(args.host, args.port),
        kwargs={"mount_ui": mount_ui},
        daemon=True,
        name="uvicorn",
    )
    server.start()

    if not _wait_for_server(wait_host, args.port, timeout=180.0):
        _file_log("ERROR: FastAPI failed to start within 180s")
        if getattr(sys, "frozen", False):
            input("Press Enter to close…")
        _shutdown_all(exit_code=1)
        return 1

    base = f"http://{wait_host}:{args.port}"
    _file_log(f"API ready at {base}/api/health")

    if args.ui == "none":
        _file_log("API-only mode. Press Ctrl+C to stop.")
        try:
            while server.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        _shutdown_all(exit_code=0)
        return 0

    url = args.vite_url if args.ui == "dev" else base
    try:
        import webview
    except ImportError:
        _file_log(
            "pywebview is not installed. Run: pip install pywebview\n"
            f"Meanwhile open {url} in a browser, or use --ui=none"
        )
        _shutdown_all(exit_code=1)
        return 1

    try:
        _file_log(f"opening webview -> {url}")
        window = webview.create_window(
            "VTM Noble",
            url=url,
            width=args.width,
            height=args.height,
            min_size=(1000, 720),
        )

        def _on_window_closed() -> None:
            _file_log("webview closed — forcing process exit")
            _shutdown_all(exit_code=0)

        # Ensure close always kills the process even if start() never returns.
        try:
            window.events.closed += _on_window_closed
        except Exception:
            try:
                window.events.closing += lambda: _on_window_closed()
            except Exception:
                pass

        webview.start()
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
        if getattr(sys, "frozen", False):
            try:
                input("Press Enter to close…")
            except Exception:
                time.sleep(10)
        raise
