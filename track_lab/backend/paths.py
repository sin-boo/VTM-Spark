"""Shared weight locations. Track Lab reads the desk trees, not a private copy."""

import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
# The performer's saved lab state (rest, shapes, feel, limiters, camera).
# Tests point TRACK_LAB_OUTPUT at a temp dir so they never touch it.
OUTPUT = Path(os.environ.get("TRACK_LAB_OUTPUT") or LAB_ROOT / "output")
REPO = LAB_ROOT.parent
TRACKERS = REPO / "models" / "trackers"
OSF_MODELS = REPO / "vendor" / "tools" / "openseeface" / "models"
OSF_PYTHON = LAB_ROOT / "osf"
