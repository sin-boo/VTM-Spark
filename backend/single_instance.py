"""Ensure only one VTM Studio backend owns the machine at a time.

Orphan / double-started backends were stacking multi-GB CUDA+torch heaps
(tens of GB → full system freeze on close/reopen).
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

_lock_handle = None
_mutex_handle = None
# Undo steps for the locks this process took (mutex, lock file).
_releases: list = []


def lock_path() -> Path:
    from .paths import data_dir

    return data_dir() / "vtm_noble.lock"


def _try_windows_mutex(name: str = "Local\\VTMNobleSingleInstance") -> bool:
    """Return True if we own the named mutex; False if another instance holds it."""
    global _mutex_handle
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ERROR_ALREADY_EXISTS = 183
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            return True  # fail open — port check still applies
        err = int(kernel32.GetLastError())
        _mutex_handle = handle
        if err == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            _mutex_handle = None
            return False

        def _release() -> None:
            global _mutex_handle
            h = _mutex_handle
            _mutex_handle = None
            if h:
                try:
                    kernel32.ReleaseMutex(h)
                    kernel32.CloseHandle(h)
                except Exception:
                    pass

        atexit.register(_release)
        _releases.append(_release)
        return True
    except Exception:
        return True


def _try_file_lock(path: Path) -> bool:
    """Cross-platform advisory lock via an exclusive lock file."""
    global _lock_handle
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(path, "a+b")
    except OSError:
        return True
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            if fh.read(1) == b"":
                fh.write(b"0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n".encode("ascii", errors="ignore"))
        fh.flush()
        _lock_handle = fh

        def _release() -> None:
            global _lock_handle
            h = _lock_handle
            _lock_handle = None
            if h is None:
                return
            try:
                if os.name == "nt":
                    import msvcrt

                    h.seek(0)
                    msvcrt.locking(h.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(h.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass

        atexit.register(_release)
        _releases.append(_release)
        return True
    except OSError:
        try:
            fh.close()
        except Exception:
            pass
        return False


def acquire_single_instance() -> bool:
    """Try to become the sole VTM Studio instance. False = another copy is alive."""
    if os.environ.get("VTM_ALLOW_MULTI", "").strip() in {"1", "true", "yes"}:
        return True
    if sys.platform.startswith("win"):
        if not _try_windows_mutex():
            return False
    return _try_file_lock(lock_path())


def release_single_instance() -> None:
    """Give up the instance locks now, before a slow exit.

    Freeing CUDA memory can keep a closed desk's process alive for seconds.
    Holding the locks that long made run.exe report a hidden running copy.
    """
    while _releases:
        step = _releases.pop()
        try:
            step()
        except Exception:
            pass


def health_url(host: str, port: int) -> str:
    h = "127.0.0.1" if host in {"0.0.0.0", "::", "localhost"} else host
    return f"http://{h}:{int(port)}/api/health"


def probe_existing_api(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if something already serves our /api/health on host:port."""
    import urllib.error
    import urllib.request

    url = health_url(host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(64)
            return resp.status == 200 and b"ok" in body
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
