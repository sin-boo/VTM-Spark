"""Every test starts from the built-in feel and never writes it to disk.

Feel loads output/tracking_feel.json at import: tests read the performer's
own turn / look caps (22 deg, not 80) and whatever an earlier test left, and
each ``feel.update`` rewrote that file.
"""

from __future__ import annotations

import pytest

from backend.feel import DEFAULTS, feel


@pytest.fixture(autouse=True)
def _default_feel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(feel, "save", lambda: None)
    saved = dict(feel.values)
    feel.values = dict(DEFAULTS)
    yield
    feel.values = saved
