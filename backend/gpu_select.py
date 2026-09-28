"""Which NVIDIA GPU the desk runs on.

The pick is saved to ``models/gpu.json`` and applied at process start by
pinning ``CUDA_VISIBLE_DEVICES`` to the GPU's UUID, so the chosen card is the
only one CUDA sees and every ``"cuda"`` / ``torch.cuda.*`` call lands on it —
worker threads, side streams and Track Lab included. Changing it needs a
restart (the desk's Reload backend). Stdlib-only: runs before torch imports.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

PREFS_NAME = "gpu.json"
# Marks a CUDA_VISIBLE_DEVICES value this module set, so a reload (which
# inherits our env) can replace it while a user-set value is left alone.
OWNED_ENV = "VTM_GPU_UUID"
CUDA_ENV = "CUDA_VISIBLE_DEVICES"

CREATE_NO_WINDOW = 0x08000000

# UUID pinned at start ("" = automatic). None until apply_saved_gpu() runs.
_applied: str | None = None


def prefs_path() -> Path:
    from .paths import data_dir

    return data_dir() / PREFS_NAME


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        for candidate in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def parse_smi_csv(text: str) -> list[dict[str, Any]]:
    """Rows of ``index, uuid, name, memory.total`` (MiB) from nvidia-smi."""
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[1].startswith("GPU-"):
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        try:
            memory_mb = int(float(parts[3]))
        except ValueError:
            memory_mb = 0
        gpus.append(
            {"index": index, "uuid": parts[1], "name": parts[2], "memory_mb": memory_mb}
        )
    return gpus


def list_gpus(run=subprocess.run) -> list[dict[str, Any]]:
    """Every NVIDIA GPU the driver sees, in PCI order. Empty when none / no driver.

    Uses nvidia-smi, not torch: once CUDA_VISIBLE_DEVICES is pinned, torch only
    sees the chosen card, but the picker still has to list all of them.
    """
    exe = _nvidia_smi()
    if not exe:
        return []
    kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 8}
    if os.name == "nt":
        kw["creationflags"] = CREATE_NO_WINDOW
    try:
        out = run(
            [
                exe,
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            **kw,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if getattr(out, "returncode", 1) != 0:
        return []
    return parse_smi_csv(out.stdout or "")


def load_gpu_pref(path: Path | None = None) -> str:
    """Saved GPU UUID, or "" for automatic."""
    target = path or prefs_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    uuid = str(raw.get("uuid") or "").strip() if isinstance(raw, dict) else ""
    return uuid if uuid.startswith("GPU-") else ""


def save_gpu_pref(uuid: str, path: Path | None = None) -> str:
    uuid = str(uuid or "").strip()
    if uuid and not uuid.startswith("GPU-"):
        raise ValueError(f"Not a GPU UUID: {uuid!r}")
    target = path or prefs_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"uuid": uuid}, indent=2), encoding="utf-8")
    return uuid


def external_pin(env: dict[str, str] | None = None) -> str:
    """CUDA_VISIBLE_DEVICES set outside the app ("" when unset or ours)."""
    env = os.environ if env is None else env
    current = env.get(CUDA_ENV)
    if current is None or current == env.get(OWNED_ENV):
        return ""
    return current


def apply_saved_gpu(
    env: dict[str, str] | None = None,
    *,
    saved: str | None = None,
    gpus: list[dict[str, Any]] | None = None,
    log=print,
) -> str:
    """Pin CUDA to the saved GPU. Call before anything touches CUDA.

    Returns the UUID pinned ("" = automatic). A CUDA_VISIBLE_DEVICES the user
    set themselves wins; a saved GPU that is no longer installed falls back to
    automatic rather than leaving CUDA with no device at all.
    """
    global _applied
    env = os.environ if env is None else env
    if external_pin(env):
        log(f"[gpu] {CUDA_ENV}={env.get(CUDA_ENV)} set outside the app — keeping it")
        _applied = ""
        return ""
    if env.get(OWNED_ENV) is not None:
        env.pop(CUDA_ENV, None)
        env.pop(OWNED_ENV, None)
    uuid = load_gpu_pref() if saved is None else saved
    if uuid:
        present = list_gpus() if gpus is None else gpus
        if any(g["uuid"] == uuid for g in present):
            env[CUDA_ENV] = uuid
            env[OWNED_ENV] = uuid
            name = next(g["name"] for g in present if g["uuid"] == uuid)
            log(f"[gpu] using {name} ({uuid})")
        else:
            log(f"[gpu] saved GPU {uuid} not found — using automatic")
            uuid = ""
    _applied = uuid
    return uuid


def applied_gpu() -> str:
    """UUID this process pinned at start ("" = automatic / not pinned)."""
    if _applied is not None:
        return _applied
    return os.environ.get(OWNED_ENV, "") if not external_pin() else ""


def gpu_in_use_name() -> str:
    """Name of the card torch is actually running on ("" without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


def gpu_snapshot() -> dict[str, Any]:
    """Everything the desk's GPU picker needs."""
    gpus = list_gpus()
    saved = load_gpu_pref()
    if saved and not any(g["uuid"] == saved for g in gpus):
        saved = ""
    active = applied_gpu()
    external = external_pin()
    return {
        "gpus": gpus,
        "selected": saved,
        "active": active,
        "in_use": gpu_in_use_name(),
        "external": external,
        "restart_needed": not external and saved != active,
    }
