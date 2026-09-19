"""Start Track Lab from the main .venv-build, and replace a stale 8780 listener."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .paths import package_root

LAB_PORT = 8780
HARNESS_STATUS = f"http://127.0.0.1:{LAB_PORT}/harness/status"
BOOT_WAIT = 90.0
CONNECT_WAIT = 3.0
HARNESS_PROBE = 1.2
WATCH_INTERVAL = 2.0
_lock = threading.Lock()
_owned: subprocess.Popen[Any] | None = None
_lab_stdio: Any = None
_watch_started = False
_probe_fail = ""
_stopped_foreign = False

WaitFn = Callable[[float], None]


def lab_root() -> Path:
    return package_root() / "track_lab"


def venv_python() -> Path:
    return package_root() / ".venv-build" / "Scripts" / "python.exe"


def lab_creationflags() -> int:
    """Hide spawned consoles (no INFO / port uvicorn window)."""
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))


def _hidden_flags(**extra: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {"stderr": subprocess.DEVNULL, **extra}
    flags = lab_creationflags()
    if flags:
        kw["creationflags"] = flags
    return kw


def _log(msg: str) -> None:
    print(f"[track-lab] {msg}", flush=True)


def _note_probe(msg: str) -> None:
    global _probe_fail
    if msg == _probe_fail:
        return
    _probe_fail = msg
    _log(msg)


def harness_ok(timeout: float = HARNESS_PROBE) -> bool:
    """True when /harness/status is a live Track Lab packet.

    Must read the full body. Status JSON is >4KB once cameras / mouth maps land;
    a truncated read looks like 'lab down' forever and the desk waits on splash.
    """
    global _probe_fail
    try:
        with urllib.request.urlopen(HARNESS_STATUS, timeout=timeout) as resp:
            raw = resp.read()
            status = int(resp.status)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        _note_probe(f"harness probe failed: {type(exc).__name__}: {exc}")
        return False
    if status != 200:
        _note_probe(f"harness probe HTTP {status} bytes={len(raw)}")
        return False
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _note_probe(f"harness probe not JSON bytes={len(raw)}: {exc}")
        return False
    if not isinstance(payload, dict):
        _note_probe(f"harness probe not an object bytes={len(raw)}")
        return False
    protocol = str(payload.get("protocol") or "")
    kind = str(payload.get("type") or "")
    ok = protocol.startswith("track_lab.harness") or kind in {"status", "frame", "ack"}
    if ok:
        if _probe_fail:
            _log(
                f"harness live protocol={protocol or '—'} type={kind or '—'} "
                f"bytes={len(raw)} loaded={payload.get('loaded')}"
            )
        _probe_fail = ""
        return True
    keys = ",".join(sorted(str(k) for k in payload.keys())[:8])
    _note_probe(
        f"harness probe foreign packet protocol={protocol!r} type={kind!r} keys={keys}"
    )
    return False


def _harness_http_code(timeout: float = HARNESS_PROBE) -> int | None:
    try:
        with urllib.request.urlopen(HARNESS_STATUS, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _owned_alive() -> bool:
    proc = _owned
    return proc is not None and proc.poll() is None


def _python_processes() -> list[tuple[int, int, str]]:
    try:
        import psutil
    except ImportError:
        return _python_processes_powershell()
    rows: list[tuple[int, int, str]] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            name = str(proc.info.get("name") or "").lower()
            if name not in {"python.exe", "pythonw.exe"}:
                continue
            cmd = " ".join(str(part) for part in (proc.info.get("cmdline") or []) if part)
            rows.append((int(proc.info["pid"]), int(proc.info.get("ppid") or 0), cmd))
        except (psutil.Error, TypeError, ValueError):
            continue
    return rows


def _python_processes_powershell() -> list[tuple[int, int, str]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            **_hidden_flags(),
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
    rows: list[tuple[int, int, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("ProcessId") or 0)
            parent = int(item.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            continue
        rows.append((pid, parent, str(item.get("CommandLine") or "")))
    return rows


def _listener_pids(port: int = LAB_PORT) -> list[int]:
    try:
        raw = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            **_hidden_flags(),
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: set[int] = set()
    needle = f":{int(port)}"
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        state = parts[3].upper()
        if state not in {"LISTENING", "LISTEN"}:
            continue
        local = parts[1]
        if not (local.endswith(needle) or local.endswith(f"]{int(port)}")):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 4:
            pids.add(pid)
    return sorted(pids)


def stale_pids(port: int = LAB_PORT) -> list[int]:
    """Listeners on 8780 plus their multiprocessing children, including ghost parents."""
    targets = set(_listener_pids(port))
    table = _python_processes()
    changed = True
    while changed:
        changed = False
        for pid, parent, cmd in table:
            if pid <= 4 or pid in targets:
                continue
            lower = cmd.lower()
            compact = lower.replace(" ", "")
            spawn = "multiprocessing.spawn" in lower
            if parent in targets:
                targets.add(pid)
                changed = True
                continue
            if spawn and any(f"parent_pid={item}" in compact for item in targets):
                targets.add(pid)
                changed = True
                continue
            if re.search(r"-m\s+backend(\s|$)", cmd) and "track_lab" in lower and "--ui" not in lower:
                targets.add(pid)
                changed = True
    return sorted(targets)


def kill_stale(port: int = LAB_PORT) -> list[int]:
    killed: list[int] = []
    for pid in stale_pids(port):
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                **_hidden_flags(stdout=subprocess.DEVNULL),
            )
            killed.append(pid)
        except OSError:
            pass
    return killed


def _copy_osf() -> None:
    root = lab_root()
    src_os = package_root() / "vendor" / "tools" / "openseeface"
    dst_os = root / "osf"
    dst_models = root / "models"
    dst_os.mkdir(parents=True, exist_ok=True)
    dst_models.mkdir(parents=True, exist_ok=True)
    for name in ("tracker.py", "retinaface.py", "similaritytransform.py", "remedian.py"):
        src = src_os / name
        if src.is_file():
            shutil.copy2(src, dst_os / name)
    src_models = src_os / "models"
    if src_models.is_dir():
        for item in src_models.iterdir():
            if item.is_file() and not (dst_models / item.name).is_file():
                shutil.copy2(item, dst_models / item.name)


def _start_lab() -> subprocess.Popen[Any]:
    global _lab_stdio
    py = venv_python()
    root = lab_root()
    if not py.is_file():
        raise FileNotFoundError(f"missing {py} — run install.bat")
    if not (root / "backend" / "__main__.py").is_file():
        raise FileNotFoundError(f"missing Track Lab at {root}")
    _copy_osf()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    log_path = package_root() / "models" / "vtm_noble.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _lab_stdio = log_path.open("a", encoding="utf-8")
    _log(f"starting hidden {py} -m backend cwd={root}")
    return subprocess.Popen(
        [str(py), "-m", "backend"],
        cwd=str(root),
        env=env,
        stdout=_lab_stdio,
        stderr=_lab_stdio,
        creationflags=lab_creationflags(),
    )


def _stop_foreign_lab_capture() -> None:
    """A leftover lab from the last close can still own the webcam audio pin."""
    try:
        from .lab_harness import lab as lab_harness

        packet = lab_harness.status(merge_frame=False)
        if not packet.get("online"):
            return
        lab_harness.command("stop")
        _log("released leftover lab camera")
    except Exception as exc:
        _log(f"leftover camera release skipped: {exc}")


def spawn_lab() -> bool:
    """Start the harness in the background. Does not wait, does not track.

    Never starts a second host on 8780. A booting lab looks 'down' to a short
    poll; killing it or spawning a duplicate makes the desk report
    'Track Lab did not start' while the real process is still coming up.
    """
    global _owned, _stopped_foreign
    release = False
    if harness_ok():
        if not _owned_alive() and not _stopped_foreign:
            _stopped_foreign = True
            release = True
        if release:
            _stop_foreign_lab_capture()
        return True
    with _lock:
        if harness_ok():
            if not _owned_alive() and not _stopped_foreign:
                _stopped_foreign = True
                release = True
            owned_ok = True
        elif _owned_alive():
            owned_ok = True
        else:
            owned_ok = None
            listeners = _listener_pids()
            if listeners:
                if harness_ok(timeout=HARNESS_PROBE):
                    if not _stopped_foreign:
                        _stopped_foreign = True
                        release = True
                    owned_ok = True
                else:
                    code = _harness_http_code()
                    if code == 404:
                        killed = kill_stale()
                        _log(f"cleared stale 8780 pids={killed or '—'}")
                    else:
                        _log(f"8780 occupied pids={listeners} http={code} — reusing")
                        owned_ok = True
            else:
                _log("port 8780 free — starting host")
            if owned_ok is None:
                try:
                    _owned = _start_lab()
                except Exception as exc:
                    _log(f"start failed: {exc}")
                    owned_ok = False
                else:
                    owned_ok = _owned is not None and _owned.poll() is None
    if release:
        _stop_foreign_lab_capture()
    return bool(owned_ok)


def watch_lab(*, interval: float = WATCH_INTERVAL) -> None:
    """Keep a host on 8780 for the life of the desk. Daemon-thread safe."""
    global _watch_started
    with _lock:
        if _watch_started:
            return
        _watch_started = True
    _log(f"watch started interval={interval:.1f}s")
    while True:
        try:
            spawn_lab()
        except Exception as exc:
            _log(f"watch failed: {exc}")
        time.sleep(max(0.5, float(interval)))


def connect_lab(
    *,
    timeout: float = BOOT_WAIT,
    on_wait: WaitFn | None = None,
) -> dict[str, Any]:
    """Start the harness if needed and ping it. Does not start tracking."""
    from .lab_harness import lab, offline_status

    if not ensure_lab(timeout=timeout, on_wait=on_wait):
        _log("connect failed — harness never answered")
        packet = offline_status("Track Lab did not start")
        packet["handshake"] = False
        return packet
    packet = lab.handshake()
    if not isinstance(packet, dict):
        packet = offline_status("Track Lab handshake failed")
    packet.setdefault("handshake", False)
    _log(
        f"connect handshake={packet.get('handshake')} online={packet.get('online')} "
        f"loaded={packet.get('loaded')} error={packet.get('error') or '—'}"
    )
    return packet


def ensure_lab(timeout: float = BOOT_WAIT, on_wait: WaitFn | None = None) -> bool:
    """Reuse a live harness, or kill the 404 leftover and start one from .venv-build."""
    global _owned
    if harness_ok():
        if on_wait is not None:
            on_wait(1.0)
        return True
    _log(f"ensure wait={timeout:.1f}s owned={_owned_alive()}")
    spawn_lab()
    end = time.monotonic() + max(0.0, float(timeout))
    start = time.monotonic()
    span = max(0.001, float(timeout))
    last_spawn = 0.0
    while time.monotonic() < end:
        if harness_ok():
            _log(f"harness live after {time.monotonic() - start:.1f}s")
            if on_wait is not None:
                on_wait(1.0)
            return True
        proc = _owned
        now = time.monotonic()
        if proc is not None and proc.poll() is not None:
            _log(f"lab exited code={proc.returncode}")
            with _lock:
                if _owned is proc:
                    _owned = None
            if harness_ok():
                if on_wait is not None:
                    on_wait(1.0)
                return True
            spawn_lab()
            last_spawn = now
        elif proc is None and (now - last_spawn) >= 2.0:
            spawn_lab()
            last_spawn = now
        if on_wait is not None:
            on_wait(min(1.0, (now - start) / span))
        time.sleep(0.25)
    if harness_ok():
        _log("harness live")
        if on_wait is not None:
            on_wait(1.0)
        return True
    _log(
        f"harness still starting after {time.monotonic() - start:.1f}s — "
        f"owned={_owned_alive()} leaving the process running"
    )
    if on_wait is not None:
        on_wait(min(1.0, (time.monotonic() - start) / span))
    return harness_ok()


def stop_owned_lab() -> None:
    """Kill the lab we started, and any leftover 8780 listener from last close."""
    global _owned, _stopped_foreign
    with _lock:
        proc = _owned
        _owned = None
        _stopped_foreign = False
    if proc is not None:
        pid = 0
        try:
            pid = int(proc.pid or 0)
        except (TypeError, ValueError):
            pid = 0
        try:
            proc.kill()
        except Exception:
            pass
        if pid > 4:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    check=False,
                    **_hidden_flags(stdout=subprocess.DEVNULL),
                )
            except OSError:
                pass
    kill_stale()


def main() -> int:
    ok = ensure_lab()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
