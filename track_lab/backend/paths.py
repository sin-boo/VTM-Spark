"""Shared weight locations. Track Lab reads the desk trees, not a private copy."""

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO = LAB_ROOT.parent
TRACKERS = REPO / "models" / "trackers"
OSF_MODELS = REPO / "vendor" / "tools" / "openseeface" / "models"
OSF_PYTHON = LAB_ROOT / "osf"
