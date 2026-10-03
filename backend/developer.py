"""Developer / test-mode switch for VTM Spark.

``VTM_DEVELOPER=1`` opens the Developer section in the UI (performance test,
Iris / Body / Drive pose, Auto sync track, Track FPS). The UI reads it from
status, so no rebuild is needed. Off, hidden options keep their product
defaults; users just cannot toggle them.
"""

from __future__ import annotations

import os

DEVELOPER = os.environ.get("VTM_DEVELOPER", "").strip().lower() in {"1", "true", "yes", "on"}
