"""Developer / test-mode switch for VTM Spark.

Flip ``DEVELOPER`` to ``True`` when you want the advanced operator controls
exposed in the UI (checkpoint picker, image path, Iris / Body / Drive pose,
Auto sync track, Track FPS). Compile stays available in the product UI.

Keep ``False`` for the normal product UI. Hidden options still apply with
their defaults — users just cannot toggle them.
"""

from __future__ import annotations

# Main switch — flip this for test / developer builds.
DEVELOPER = False
