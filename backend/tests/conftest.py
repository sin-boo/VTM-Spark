"""Shared pytest setup: vendor import paths (live-poser, torch_train, …)."""

import atexit
import os
import shutil
import tempfile

# Track Lab code reached from these tests (and any worker they start) saves
# lab state to a temp dir, never the performer's track_lab/output.
_LAB_OUTPUT = tempfile.mkdtemp(prefix="track_lab_test_output_")
os.environ["TRACK_LAB_OUTPUT"] = _LAB_OUTPUT
atexit.register(shutil.rmtree, _LAB_OUTPUT, True)

from backend.paths import ensure_import_paths  # noqa: E402

ensure_import_paths()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_lower_lid_detector(monkeypatch):
    """Loading a character reads its lower eyelids with the anime face
    detector; tests never load that model."""
    import backend.engine as engine_module

    monkeypatch.setattr(engine_module, "detect_reference_lower_lids", lambda *_a, **_k: None)
