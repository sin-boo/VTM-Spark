"""python -m backend.pair — the lab API and UI as one process tree.

The API runs here; the Vite UI is its child. Close either and both stop:
- the UI exits → the API shuts down;
- the API stops (Ctrl+C, crash, window closed) → the UI is killed with it.

The second half is a Windows job object: this process joins a job that kills
everything in it when the last handle closes, which happens however we die.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .__main__ import claim_port
from .bind import probe_state
from .ports import HOST, PORT

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
_job: int | None = None


def _bind_to_job() -> None:
    """Kill every process this one starts when this one dies. Windows only."""
    global _job
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _Io),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        print("[track-lab] no job object; the UI may outlive the API", flush=True)
        return
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)
    )  # 9 = JobObjectExtendedLimitInformation
    ok = ok and kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job), wintypes.HANDLE(kernel32.GetCurrentProcess())
    )
    if not ok:
        print("[track-lab] could not join a job; the UI may outlive the API", flush=True)
        return
    _job = job  # Never closed: the handle dies with us, and takes the UI along.


def _find_npm() -> str | None:
    """Portable Node from install.bat first, then a system Node on PATH."""
    portable = UI_DIR.parents[1] / ".tools" / "node" / "npm.cmd"
    if portable.is_file():
        return str(portable)
    return shutil.which("npm")


def _start_ui() -> subprocess.Popen[bytes]:
    npm = _find_npm()
    if npm is None:
        raise FileNotFoundError("Node.js missing (.tools\\node); run install.bat")
    env = os.environ.copy()
    node_dir = str(Path(npm).parent)
    env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")
    return subprocess.Popen([npm, "run", "dev"], cwd=str(UI_DIR), env=env)


def _stop_ui(ui: subprocess.Popen[bytes] | None) -> None:
    if ui is None or ui.poll() is not None:
        return
    if os.name == "nt":
        # npm → cmd → node: kill the whole branch, not just npm.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(ui.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        ui.terminate()
    try:
        ui.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ui.kill()


def _ui_only() -> int:
    """An API this launch did not start is already up: run the UI against it."""
    print(
        f"[track-lab] API already running on http://{HOST}:{PORT} (started elsewhere). "
        "Running the UI only; closing it will not stop that API.",
        flush=True,
    )
    ui = _start_ui()
    try:
        return ui.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        _stop_ui(ui)


def main() -> int:
    _bind_to_job()
    if probe_state(HOST, PORT) == "lab":
        return _ui_only()
    code = claim_port()
    if code is not None:
        return code

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            "backend.server:app",
            host=HOST,
            port=PORT,
            log_level="info",
            # A line per /harness/frame poll buried everything else.
            access_log=False,
        )
    )
    holder: dict[str, subprocess.Popen[bytes]] = {}

    def _pair() -> None:
        # The proxy needs the API; start the UI once it is serving.
        while not server.started:
            if server.should_exit:
                return
            time.sleep(0.1)
        try:
            ui = _start_ui()
        except Exception as exc:
            print(f"[track-lab] UI failed to start: {exc}; stopping API", flush=True)
            server.should_exit = True
            return
        holder["ui"] = ui
        code = ui.wait()
        if not server.should_exit:
            print(f"[track-lab] UI stopped (code {code}); stopping API", flush=True)
            server.should_exit = True

    print(f"[track-lab] binding http://{HOST}:{PORT} with the UI attached", flush=True)
    threading.Thread(target=_pair, name="track-lab-ui", daemon=True).start()
    try:
        server.run()
    finally:
        _stop_ui(holder.get("ui"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
