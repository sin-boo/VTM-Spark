"""Who owns port 8780: our lab, something else, or nobody.

stdlib only — start.ps1 and python -m backend call this before importing the bench.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Literal

from .ports import (
    HARNESS_PROTOCOL_PREFIX,
    HARNESS_STATUS_PATH,
    HEALTH_PATH,
    HOST,
    PORT,
    SERVICE,
)

PortState = Literal["lab", "stale", "busy", "free"]


def health_url(host: str = HOST, port: int = PORT) -> str:
    return f"http://{host}:{int(port)}{HEALTH_PATH}"


def harness_status_url(host: str = HOST, port: int = PORT) -> str:
    return f"http://{host}:{int(port)}{HARNESS_STATUS_PATH}"


def port_in_use(host: str = HOST, port: int = PORT) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.25)
        sock.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def looks_like_health(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("ok") is not True:
        return False
    return str(payload.get("service") or "") == SERVICE


def looks_like_harness(payload: object) -> bool:
    """True for a Track Lab harness packet. Health JSON is not enough."""
    if not isinstance(payload, dict):
        return False
    protocol = str(payload.get("protocol") or "")
    if protocol.startswith(HARNESS_PROTOCOL_PREFIX):
        return True
    return str(payload.get("type") or "") in {"status", "frame", "ack"}


def _get_json(url: str, timeout: float) -> object | None:
    """Read the full body. /harness/status is >4KB once cameras and mouth maps land."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            if resp.status != 200:
                return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def probe_health(host: str = HOST, port: int = PORT, timeout: float = 0.4) -> bool:
    return looks_like_health(_get_json(health_url(host, port), timeout))


def probe_harness(host: str = HOST, port: int = PORT, timeout: float = 0.4) -> bool:
    return looks_like_harness(_get_json(harness_status_url(host, port), timeout))


def _pid_image(pid: int) -> str:
    try:
        raw = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    if not line or line.lower().startswith("info:"):
        return ""
    return line.split(",")[0].strip('"').lower()


def python_listeners(port: int = PORT) -> list[int]:
    return [pid for pid in listener_pids(port) if _pid_image(pid).startswith("python")]


def harness_http_status(host: str = HOST, port: int = PORT, timeout: float = 0.4) -> int | None:
    url = harness_status_url(host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def probe_state(host: str = HOST, port: int = PORT) -> PortState:
    """lab = live harness, stale = old lab without /harness, busy = other, free = empty."""
    if probe_harness(host, port):
        return "lab"
    if not port_in_use(host, port):
        return "free"
    if (
        probe_health(host, port)
        or python_listeners(port)
        or harness_http_status(host, port) == 404
    ):
        return "stale"
    return "busy"


def occupied_message(port: int = PORT) -> str:
    return (
        f"Port {int(port)} is already in use by another app. "
        "Close that app, then start Track Lab again."
    )


def already_running_message(host: str = HOST, port: int = PORT) -> str:
    return f"Track Lab already running on http://{host}:{int(port)}"


def stale_message(port: int = PORT) -> str:
    return (
        f"Track Lab on port {int(port)} is an old build (no {HARNESS_STATUS_PATH}). "
        "Restarting it to load the harness."
    )


def _local_port(addr: str) -> int | None:
    text = str(addr or "").strip()
    if text.startswith("["):
        _, _, rest = text.rpartition("]:")
    else:
        _, _, rest = text.rpartition(":")
    try:
        return int(rest)
    except ValueError:
        return None


def listener_pids(port: int = PORT) -> list[int]:
    """PIDs listening on TCP port. Windows netstat; empty on failure."""
    try:
        raw = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: set[int] = set()
    want = int(port)
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0].upper() != "TCP":
            continue
        state = parts[3].upper() if len(parts) >= 5 else ""
        if state not in {"LISTENING", "LISTEN"}:
            continue
        if _local_port(parts[1]) != want:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 4:
            pids.add(pid)
    return sorted(pids)


def _python_process_table() -> list[tuple[int, int, str]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            stderr=subprocess.DEVNULL,
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


def _stale_pids(port: int = PORT) -> list[int]:
    targets = set(listener_pids(port))
    table = _python_process_table()
    changed = True
    while changed:
        changed = False
        for pid, parent, cmd in table:
            if pid in targets or pid <= 4:
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
    return sorted(pid for pid in targets if pid > 4)


def kill_listeners(port: int = PORT) -> list[int]:
    """Force-kill whatever is bound to the lab port, including reload children."""
    killed: list[int] = []
    for pid in _stale_pids(port):
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            killed.append(pid)
        except OSError:
            pass
    return killed


def _pause(message: str) -> None:
    print(message, flush=True)
    if not sys.stdin.isatty():
        return
    try:
        input("Press Enter to close…")
    except Exception:
        pass
