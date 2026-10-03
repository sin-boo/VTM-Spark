"""Every test starts from the built-in feel and never writes it to disk.

Feel loads output/tracking_feel.json at import: tests read the performer's
own turn / look caps (22 deg, not 80) and whatever an earlier test left, and
each ``feel.update`` rewrote that file.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

# Before any backend import: saved lab state goes to a temp dir, never the
# performer's track_lab/output (rest, shapes, feel, limiters, camera).
_OUTPUT = tempfile.mkdtemp(prefix="track_lab_test_output_")
os.environ["TRACK_LAB_OUTPUT"] = _OUTPUT
atexit.register(shutil.rmtree, _OUTPUT, True)

import pytest  # noqa: E402

from backend.feel import DEFAULTS, feel  # noqa: E402


@pytest.fixture(autouse=True)
def _default_feel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(feel, "save", lambda: None)
    saved = dict(feel.values)
    feel.values = dict(DEFAULTS)
    yield
    feel.values = saved
