"""Tiny hold window for a backend reload.

Stdlib only — never import backend / torch / webview. The desk process dies
while this one stays up, then it starts a fresh ``python -m backend``.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


@dataclass(frozen=True)
class HoldConfig:
    pid: int
    host: str
    port: int
    cwd: str
    python: str
    flags: tuple[str, ...]
    wait_dead: float = 30.0
    wait_up: float = 180.0


def parse_args(argv: list[str] | None = None) -> HoldConfig:
    parser = argparse.ArgumentParser(description="VTM Spark reload hold window")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("flags", nargs="*")
    args = parser.parse_args(argv)
    host = "127.0.0.1" if args.host in {"0.0.0.0", "::", "localhost"} else str(args.host)
    return HoldConfig(
        pid=int(args.pid),
        host=host,
        port=int(args.port),
        cwd=str(args.cwd),
        python=str(args.python),
        flags=tuple(str(a) for a in args.flags),
    )


def backend_command(python: str, flags: tuple[str, ...] | list[str]) -> list[str]:
    return [str(python), "-m", "backend", *[str(a) for a in flags]]


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def health_ok(host: str, port: int, timeout: float = 0.4) -> bool:
    url = f"http://{host}:{int(port)}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(64)
            return int(resp.status) == 200 and b"ok" in body
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def port_bound(host: str, port: int, timeout: float = 0.2) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def wait_until(pred, timeout: float, *, interval: float = 0.15) -> bool:
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def wait_old_desk_gone(cfg: HoldConfig) -> bool:
    dead = wait_until(lambda: not pid_alive(cfg.pid), cfg.wait_dead)
    down = wait_until(lambda: not health_ok(cfg.host, cfg.port), min(8.0, cfg.wait_dead))
    return dead and down


def spawn_backend(cfg: HoldConfig) -> subprocess.Popen[bytes]:
    cmd = backend_command(cfg.python, cfg.flags)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("VTM_NOBLE_ROOT", cfg.cwd)
    kw: dict = {
        "cwd": cfg.cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kw["creationflags"] = (
            CREATE_NO_WINDOW
            | CREATE_NEW_PROCESS_GROUP
            | DETACHED_PROCESS
            | CREATE_BREAKAWAY_FROM_JOB
        )
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def run_reload(cfg: HoldConfig, *, on_status=None, spawn=spawn_backend) -> int:
    say = on_status or (lambda _msg: None)
    say("Waiting for the old desk to exit…")
    if cfg.pid > 0 and not wait_old_desk_gone(cfg):
        say("Old desk did not exit.")
        return 2
    say("Starting backend…")
    try:
        spawn(cfg)
    except OSError as exc:
        say(f"Failed to start: {exc}")
        return 1
    say("Waiting for the desk…")
    if not wait_until(lambda: health_ok(cfg.host, cfg.port), cfg.wait_up):
        say("Backend did not come back.")
        return 3
    say("Desk is back.")
    return 0


def _show_window(cfg: HoldConfig) -> int:
    try:
        import tkinter as tk
    except Exception:
        return run_reload(cfg)

    root = tk.Tk()
    root.title("VTM Spark")
    root.configure(bg="#2b2b2b")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    width, height = 320, 92
    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
        root.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 3}")
    except Exception:
        root.geometry(f"{width}x{height}")

    label = tk.Label(
        root,
        text="Reloading backend…",
        bg="#2b2b2b",
        fg="#cfcfcf",
        font=("Segoe UI", 11),
        padx=16,
        pady=18,
    )
    label.pack(fill="both", expand=True)

    result = {"code": 0}

    def _work() -> None:
        result["code"] = run_reload(
            cfg,
            on_status=lambda msg: root.after(0, lambda m=msg: label.config(text=m)),
        )
        root.after(80, root.destroy)

    threading.Thread(target=_work, name="vtm-reload-hold", daemon=True).start()
    root.mainloop()
    return int(result["code"])


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    return _show_window(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
