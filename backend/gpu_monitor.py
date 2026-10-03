"""Live GPU utilisation for the desk's GPU readout, straight from the driver.

NVML ships with every NVIDIA driver (``nvml.dll`` / ``libnvidia-ml.so.1``), so
this needs no extra package. It reads the whole card — games and OBS count
too — which is what matters when deciding whether the avatar leaves room.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Any

# Sampling more often than NVML's own window just returns the same number.
SAMPLE_S = 0.5


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


_lock = threading.Lock()
_nvml: Any = None
_failed = False
_handles: dict[str, ctypes.c_void_p] = {}
_cache: dict[str, tuple[float, int | None]] = {}


def _load() -> Any:
    global _nvml, _failed
    if _nvml is not None or _failed:
        return _nvml
    try:
        lib = (
            ctypes.WinDLL("nvml.dll")
            if os.name == "nt"
            else ctypes.CDLL("libnvidia-ml.so.1")
        )
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        _nvml = lib
    except (OSError, AttributeError):
        _failed = True
    return _nvml


def _handle(lib: Any, uuid: str) -> ctypes.c_void_p | None:
    handle = _handles.get(uuid)
    if handle is None:
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"), ctypes.byref(handle)) != 0:
            return None
        _handles[uuid] = handle
    return handle


def device_uuid(device: Any = None) -> str:
    """NVML UUID ("GPU-…") of the CUDA device torch runs on, or ""."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        index = getattr(device, "index", None)
        props = torch.cuda.get_device_properties(0 if index is None else int(index))
        raw = str(getattr(props, "uuid", "") or "")
    except Exception:
        return ""
    if not raw:
        return ""
    return raw if raw.startswith("GPU-") else f"GPU-{raw}"


def gpu_utilization(uuid: str, *, now: float | None = None) -> int | None:
    """Percent of the last sample window the card was running kernels (0–100).

    None when there is no NVIDIA driver or the card cannot be read.
    """
    if not uuid:
        return None
    t = time.monotonic() if now is None else float(now)
    with _lock:
        hit = _cache.get(uuid)
        if hit is not None and t - hit[0] < SAMPLE_S:
            return hit[1]
        value: int | None = None
        lib = _load()
        if lib is not None:
            handle = _handle(lib, uuid)
            util = _Utilization()
            if handle is not None and lib.nvmlDeviceGetUtilizationRates(
                handle, ctypes.byref(util)
            ) == 0:
                value = int(util.gpu)
        _cache[uuid] = (t, value)
        return value


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def gpu_memory_used_mb(uuid: str) -> float | None:
    """Whole-card VRAM in use, MB (games count too), or None without NVML."""
    if not uuid:
        return None
    with _lock:
        lib = _load()
        if lib is None:
            return None
        handle = _handle(lib, uuid)
        mem = _Memory()
        if handle is None or lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        return mem.used / (1024 * 1024)
