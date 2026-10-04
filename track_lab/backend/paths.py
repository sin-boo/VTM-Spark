"""Shared weight locations. Track Lab reads the desk trees, not a private copy."""

import os
import shutil
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
# The performer's saved lab state (rest, shapes, feel, limiters, camera).
# Tests point TRACK_LAB_OUTPUT at a temp dir so they never touch it.
OUTPUT = Path(os.environ.get("TRACK_LAB_OUTPUT") or LAB_ROOT / "output")
# Shipped starting state (authored shapes, feel, limiters). Copied into OUTPUT
# only where a file is missing, so a performer's own edits always win.
DEFAULTS = Path(os.environ.get("TRACK_LAB_DEFAULTS") or LAB_ROOT / "defaults")
REPO = LAB_ROOT.parent
TRACKERS = REPO / "models" / "trackers"
OSF_MODELS = REPO / "vendor" / "tools" / "openseeface" / "models"
OSF_PYTHON = LAB_ROOT / "osf"


def seed_output() -> list[str]:
    """Copy each shipped default whose OUTPUT file is missing. Names copied."""
    if not DEFAULTS.is_dir():
        return []
    copied: list[str] = []
    for src in sorted(DEFAULTS.glob("*.json")):
        dest = OUTPUT / src.name
        if dest.exists():
            continue
        try:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        except OSError:
            continue
        copied.append(src.name)
    return copied
