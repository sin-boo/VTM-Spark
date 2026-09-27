"""Opt-in JSON-lines trace for camera / tracking bugs.

Asleep unless ``VTM_DEBUG_LOG`` names a file. Callers check ``ENABLED`` first
so the tracking loops do no extra work while it is off.
"""

from __future__ import annotations

import json
import os
import time

PATH = os.environ.get("VTM_DEBUG_LOG", "").strip()
ENABLED = bool(PATH)


def log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not ENABLED:
        return
    try:
        payload = {
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
