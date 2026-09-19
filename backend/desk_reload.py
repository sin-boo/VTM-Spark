"""Start the out-of-process reload hold window, then exit this desk.

The hold script lives in ``backend/packaging/reload_hold.py`` and must not
import this package — it stays up while Python unloads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_requested = False


def hold_script() -> Path:
    return Path(__file__).resolve().parent / "packaging" / "reload_hold.py"


def current_flags(argv: list[str] | None = None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:2] == ["-m", "backend"]:
        raw = raw[2:]
    return [str(a) for a in raw]


def hold_command(
    *,
    python: str,
    pid: int,
    host: str,
    port: int,
    cwd: str,
    flags: list[str] | None = None,
) -> list[str]:
    cmd = [
        str(python),
        str(hold_script()),
        "--pid",
        str(int(pid)),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--cwd",
        str(cwd),
        "--python",
        str(python),
        "--",
        *list(flags if flags is not None else current_flags()),
    ]
    return cmd


def spawn_kwargs() -> dict:
    kw: dict = {
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
    return kw


def spawn_hold(
    *,
    python: str | None = None,
    pid: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    cwd: str | None = None,
    flags: list[str] | None = None,
    popen=subprocess.Popen,
) -> list[str]:
    from .paths import package_root

    exe = str(python or sys.executable)
    root = str(cwd or package_root())
    cmd = hold_command(
        python=exe,
        pid=int(pid if pid is not None else os.getpid()),
        host=host,
        port=port,
        cwd=root,
        flags=flags,
    )
    popen(cmd, cwd=root, env=os.environ.copy(), **spawn_kwargs())
    return cmd


def exit_for_reload() -> None:
    """Leave Track Lab alone. Process death releases the single-instance lock."""
    os._exit(0)


def request_reload(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    apply: bool = True,
    delay: float = 0.45,
    spawn=spawn_hold,
    exit_fn=exit_for_reload,
) -> dict:
    """Open the hold window, then quit so a new ``-m backend`` can import fresh code."""
    global _requested
    if _requested:
        return {"ok": True, "reloading": True}
    cmd = spawn(
        host=host,
        port=port,
    )
    _requested = True
    if apply:
        threading.Timer(max(0.05, float(delay)), exit_fn).start()
    return {"ok": True, "reloading": True, "hold": cmd}
