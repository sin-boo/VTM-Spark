"""Fixed loopback bind for the lab API + harness.

Do not hop this port. Noble looks here; a second lab should reuse or refuse.
"""

from __future__ import annotations

HOST = "127.0.0.1"
PORT = 8780
HEALTH_PATH = "/api/health"
HARNESS_STATUS_PATH = "/harness/status"
SERVICE = "track_lab"
HARNESS_PROTOCOL_PREFIX = "track_lab.harness"
