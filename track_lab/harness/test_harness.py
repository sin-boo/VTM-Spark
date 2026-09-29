from __future__ import annotations

import json

import numpy as np

from harness.dispatch import bind, handle
from harness.hub import HarnessHub, LatestSlot, PacketMailbox, hub
from harness.pack import pack_frame, pack_keypoints, pack_status
from harness.protocol import COMMANDS, NUM_KEYPOINTS, ack, parse_command


def _face_points() -> list[list[float]]:
    pts = []
    for i in range(28):
        pts.append([100.0 + i, 200.0 + i * 2.0, 1.0])
    return pts


def _skeleton() -> list[dict[str, object]]:
    return [
        {"id": 31, "x": 114.0, "y": 260.0, "score": 1.0},
        {"id": 32, "x": 80.0, "y": 280.0, "score": 1.0},
        {"id": 33, "x": 70.0, "y": 320.0, "score": 1.0},
        {"id": 34, "x": 150.0, "y": 280.0, "score": 1.0},
        {"id": 35, "x": 160.0, "y": 320.0, "score": 1.0},
        {"id": 36, "x": 114.0, "y": 300.0, "score": 1.0},
    ]


def test_parse_command_accepts_op_and_ping() -> None:
    msg = parse_command({"op": "set_feel", "id": "a", "smoothing": 0.2})
    assert msg["op"] == "set_feel"
    assert msg["id"] == "a"
    assert msg["body"]["smoothing"] == 0.2
    nested = parse_command({"type": "command", "op": "stop", "id": 3, "body": {}})
    assert nested["op"] == "stop"
    assert parse_command("ping")["op"] == "ping"


def test_parse_command_rejects_unknown_op() -> None:
    try:
        parse_command({"op": "explode"})
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pack_keypoints_fills_iris_and_body() -> None:
    k = pack_keypoints(_face_points(), _skeleton())
    assert k.shape == (NUM_KEYPOINTS, 4)
    assert abs(float(k[0, 0]) - 100.0) < 1e-5
    assert float(k[31, 1]) == 260.0
    assert float(k[28, 3]) >= 0.5
    assert float(k[29, 3]) >= 0.5
    # IRIS.L (28) is the person-left / image-left eye midpoint (slots 11-13).
    expected_x = (111.0 + 112.0 + 113.0) / 3.0
    assert abs(float(k[28, 0]) - expected_x) < 1e-3


def test_pack_frame_is_character_space_schema() -> None:
    live = {
        "live": True,
        "tracker": "osf",
        "faces": 1,
        "ms": 8.2,
        "error": "",
        "source": "camera",
        "points": _face_points(),
        "skeleton": _skeleton(),
        "hair": [{"class": "hair_middle", "polygon": [[1, 2], [3, 4]]}],
        "weights": {"A": 0.4, "smile": 0.1},
        "head": {"pitch": 1.0, "yaw": -2.0, "roll": 0.5},
        "blink": {"l": 0.1, "r": 0.0},
        "calib": {"rest": True, "capturing": "", "progress": 1.0},
        "feel": {"smoothing": 0.3, "show_face": 1, "show_skeleton": 0},
    }
    packet = pack_frame(live, image_wh=(640, 480), generation=4, t=1.5)
    assert packet["type"] == "frame"
    assert packet["schema"] == "KEYPOINT_SCHEMA"
    assert packet["coord_space"] == "character_px"
    assert packet["image_wh"] == [640, 480]
    assert packet["generation"] == 4
    assert len(packet["keypoints"]) == NUM_KEYPOINTS
    assert packet["weights"]["A"] == 0.4
    assert packet["feel"]["show_skeleton"] == 0.0
    assert packet["feel"]["hair_pin"] == 0.7
    assert packet["loaded"] is True
    assert "invert_look" not in packet["feel"]
    assert "invert_yaw" not in packet["feel"]
    assert "invert_pitch" not in packet["feel"]
    assert packet["skeleton_method"] == "lab_follow"
    assert packet["hair_method"] == "lab_follow"
    assert packet["meta"]["hair_lost"] is False
    assert packet["calib"]["rest"] is True
    assert packet["iris_method"] == "eye_mid"


def test_pack_frame_keeps_tracked_iris() -> None:
    live = {
        "live": True,
        "tracker": "osf",
        "faces": 1,
        "points": _face_points(),
        "skeleton": [],
        "iris": [{"id": 28, "x": 333.0, "y": 111.0, "score": 0.95, "visible": True}],
        "iris_method": "iris_pose",
        "iris_cam": [{"side": "r", "x": 22.0, "y": 14.0, "score": 0.9, "visible": True, "method": "iris_pose"}],
        "look": {"x": 0.1, "y": -0.4},
        "weights": {},
        "head": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "blink": {"l": 0.0, "r": 0.0},
    }
    packet = pack_frame(live, image_wh=(640, 480), generation=1, t=1.0)
    by_i = {int(row["i"]): row for row in packet["keypoints"]}
    assert packet["iris_method"] == "iris_pose"
    assert packet["iris_cam"][0]["y"] == 14.0
    assert packet["look"] == {"x": 0.1, "y": -0.4}
    assert abs(float(by_i[28]["x"]) - 333.0) < 1e-5
    assert by_i[28]["visible"] is True
    expected = (117.0 + 118.0 + 119.0) / 3.0
    assert abs(float(by_i[29]["x"]) - expected) < 1e-3


def test_warming_packets_are_protocol_valid() -> None:
    from harness.pack import warming_frame, warming_status
    from harness.protocol import PROTOCOL

    status = warming_status(clients=2)
    assert status["type"] == "status"
    assert status["protocol"] == PROTOCOL
    assert status["loaded"] is False
    assert status["ok"] is True
    assert status["live"] is False
    assert status["error"] == ""
    assert "set_feel" in status["commands"]
    frame = warming_frame()
    assert frame["type"] == "frame"
    assert frame["loaded"] is False
    assert len(frame["keypoints"]) == NUM_KEYPOINTS
    assert frame["live"] is False


def test_pack_frame_marks_hair_lost_when_empty() -> None:
    packet = pack_frame(
        {
            "live": True,
            "points": _face_points(),
            "skeleton": [],
            "hair": [],
        },
        image_wh=(64, 64),
        t=1.0,
    )
    assert packet["hair_method"] == "none"
    assert packet["meta"]["hair_lost"] is True


def test_pack_status_advertises_commands() -> None:
    packet = pack_status(
        {
            "ok": True,
            "live": False,
            "feel": {"mouth": 0.5, "show_ids": 1},
            "camera_index": 2,
        },
        clients=3,
        rest=_face_points(),
    )
    assert packet["type"] == "status"
    assert packet["clients"] == 3
    assert packet["camera_index"] == 2
    assert sorted(packet["commands"]) == sorted(COMMANDS)
    assert packet["rest"][0][0] == 100.0


def test_pack_status_keeps_authored_mouth_shapes() -> None:
    rest = _face_points()
    smile = [row[:] for row in rest]
    smile[20] = [smile[20][0], smile[20][1] + 12.0, smile[20][2]]
    packet = pack_status(
        {
            "ok": True,
            "live": False,
            "active": "smile",
            "points": smile,
            "shapes": {"rest": rest, "smile": smile},
            "presets": [{"id": "rest", "label": "Rest", "ready": True}],
            "weights": {"smile": 0.0, "A": 0.0},
            "hair": [],
            "skeleton": _skeleton(),
        }
    )
    assert packet["active"] == "smile"
    assert packet["points"][20][1] == smile[20][1]
    assert packet["shapes"]["smile"][20][1] == smile[20][1]
    assert packet["shapes"]["rest"][20][1] == rest[20][1]
    assert packet["skeleton"][0]["id"] == 31


def test_pack_status_keeps_point_offsets() -> None:
    packet = pack_status(
        {
            "ok": True,
            "live": True,
            "point_offsets": [{"id": 28, "dx": 4.0, "dy": -2.5}],
            "iris_cam": [{"side": "r", "x": 22.0, "y": 14.0, "score": 0.9, "visible": True, "method": "iris_pose"}],
            "look": {"x": 0.0, "y": -0.5},
        }
    )
    assert packet["point_offsets"][0]["id"] == 28
    assert packet["point_offsets"][0]["dx"] == 4.0
    assert packet["iris_cam"][0]["side"] == "r"
    assert packet["look"]["y"] == -0.5


def test_hub_drops_old_frames_for_slow_consumer() -> None:
    local = HarnessHub()
    q = local.subscribe()
    local.publish({"type": "frame", "n": 1})
    local.publish({"type": "frame", "n": 2})
    local.publish({"type": "frame", "n": 3})
    first = q.get_nowait()
    second = q.get_nowait()
    try:
        q.get_nowait()
        overflowed = True
    except Exception:
        overflowed = False
    assert first["n"] in {2, 3} or second["n"] in {2, 3}
    assert local.latest_frame()["n"] == 3
    assert overflowed is False
    local.unsubscribe(q)


def test_latest_slot_drops_stale_items() -> None:
    slot = LatestSlot()
    slot.put({"n": 1})
    slot.put({"n": 2})
    slot.put({"n": 3})
    assert slot.take(0.0)["n"] == 3
    assert slot.take(0.0) is None


def test_packet_mailbox_keeps_status_beside_newer_frame() -> None:
    box = PacketMailbox()
    box.put({"type": "packet", "packet": {"type": "status", "source": "ifm"}})
    box.put({"type": "packet", "packet": {"type": "frame", "n": 1}})
    box.put({"type": "packet", "packet": {"type": "frame", "n": 2}})
    batch = box.take(0.0)
    assert [item["packet"]["type"] for item in batch] == ["status", "frame"]
    assert batch[0]["packet"]["source"] == "ifm"
    assert batch[1]["packet"]["n"] == 2
    assert box.take(0.0) == []


def test_hub_command_requires_bind() -> None:
    local = HarnessHub()
    reply = local.command({"op": "stop"})
    assert reply["ok"] is False
    assert "not bound" in reply["error"]
    ping = local.command("ping")
    assert ping["ok"] is True


class _FakeBench:
    def __init__(self) -> None:
        self.generation = 1
        self.source_bgr = np.zeros((40, 50, 3), dtype=np.uint8)
        self.rest_pts = np.asarray(_face_points(), dtype=np.float32)
        self.feel = {"smoothing": 0.48, "show_face": 1.0, "show_skeleton": 1.0}
        self.live = False
        self.camera_index = 0
        self.error = ""
        self.mirror = True
        self.points = {}
        self.cleared = None

    def status(self) -> dict[str, object]:
        return {
            "ok": True,
            "live": self.live,
            "error": self.error,
            "feel": dict(self.feel),
            "camera_index": self.camera_index,
            "mirror": self.mirror,
            "width": 50,
            "height": 40,
        }

    def live_status(self) -> dict[str, object]:
        return {
            "ok": True,
            "live": self.live,
            "tracker": "osf" if self.live else "anime",
            "faces": 1 if self.live else 0,
            "ms": 4.0,
            "error": self.error,
            "source": "camera",
            "points": _face_points(),
            "skeleton": _skeleton(),
            "hair": [],
            "weights": {"A": 0.0},
            "head": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "blink": {"l": 0.0, "r": 0.0},
            "feel": dict(self.feel),
        }

    def set_feel(self, body: object) -> dict[str, object]:
        if isinstance(body, dict):
            self.feel.update({k: float(v) for k, v in body.items()})
        self.error = ""
        return self.status()

    def start_live(self, camera=None, source=None, host=None, port=None, mirror=None) -> dict[str, object]:
        if camera is not None:
            self.camera_index = int(camera)
        if mirror is not None:
            self.mirror = bool(mirror)
        self.live = True
        self.error = ""
        return self.status()

    def set_mirror(self, on: bool) -> dict[str, object]:
        self.mirror = bool(on)
        return self.status()

    def stop_live(self) -> dict[str, object]:
        self.live = False
        return self.status()

    def set_camera(self, index: int) -> dict[str, object]:
        self.camera_index = int(index)
        return self.status()

    def set_source(self, payload, filename: str = "source.png") -> dict[str, object]:
        self.source_name = str(filename)
        self.source_bytes = len(payload)
        self.live = False
        self.error = ""
        return self.status()

    def set_point(self, body: object) -> dict[str, object]:
        if isinstance(body, dict):
            self.points[int(body.get("id", -1))] = (float(body.get("x")), float(body.get("y")))
        self.error = ""
        return self.status()

    def reset_points(self, body: object) -> dict[str, object]:
        if isinstance(body, dict) and body.get("id") not in (None, ""):
            self.cleared = int(body.get("id"))
            self.points.pop(self.cleared, None)
        else:
            self.cleared = "all"
            self.points = {}
        self.error = ""
        return self.status()

    def set_hair(self, body: object) -> dict[str, object]:
        if isinstance(body, dict):
            self.hair = body.get("hair")
        self.error = ""
        return self.status()


def test_dispatch_set_feel_and_start() -> None:
    bench = _FakeBench()
    reply = handle(bench, {"op": "set_feel", "id": "x", "body": {"smoothing": 0.1, "show_skeleton": 0}})
    assert reply["ok"] is True
    assert reply["id"] == "x"
    assert bench.feel["smoothing"] == 0.1
    assert bench.feel["show_skeleton"] == 0.0
    started = handle(bench, {"op": "start", "body": {"camera": 2}})
    assert started["ok"] is True
    assert bench.live is True
    assert bench.camera_index == 2
    assert started["status"]["live"] is True


def test_dispatch_set_mirror_and_start_mirror() -> None:
    bench = _FakeBench()
    reply = handle(bench, {"op": "set_mirror", "body": {"on": False}})
    assert reply["ok"] is True
    assert bench.mirror is False
    started = handle(bench, {"op": "start", "body": {"camera": 0, "mirror": True}})
    assert started["ok"] is True
    assert bench.mirror is True


def test_apply_mirror_flips_left_edge() -> None:
    from backend.osf_cam import apply_mirror

    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:, 0] = 255
    kept = apply_mirror(img, False)
    flipped = apply_mirror(img, True)
    assert np.array_equal(kept[:, 0], img[:, 0])
    assert np.array_equal(flipped[:, -1], img[:, 0])
    assert not np.array_equal(flipped[:, 0], img[:, 0])


def test_dispatch_set_source_reads_path(tmp_path) -> None:
    bench = _FakeBench()
    still = tmp_path / "char.png"
    still.write_bytes(b"\x89PNG fake")
    reply = handle(bench, {"op": "set_source", "body": {"path": str(still)}})
    assert reply["ok"] is True
    assert bench.source_name == "char.png"
    assert bench.source_bytes == len(b"\x89PNG fake")
    assert bench.live is False
    assert isinstance(reply.get("frame"), dict)
    assert reply["frame"].get("type") == "frame"
    assert reply["frame"].get("generation") == 1


def test_dispatch_set_point_and_reset(tmp_path) -> None:
    bench = _FakeBench()
    reply = handle(bench, {"op": "set_point", "body": {"id": 28, "x": 140.0, "y": 90.0}})
    assert reply["ok"] is True
    assert bench.points[28] == (140.0, 90.0)
    cleared = handle(bench, {"op": "reset_points", "body": {}})
    assert cleared["ok"] is True
    assert bench.cleared == "all"
    assert bench.points == {}


def test_dispatch_set_hair() -> None:
    bench = _FakeBench()
    hair = [{"class": "hair_middle", "polygon": [[0, 0], [10, 0], [10, 10]]}]
    reply = handle(bench, {"op": "set_hair", "body": {"hair": hair}})
    assert reply["ok"] is True
    assert bench.hair == hair


def test_lab_http_exposes_overlay_point() -> None:
    from backend.server import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/overlay-point" in paths
    assert "/api/overlay-points/reset" in paths
    assert "/api/generate" in paths


def test_bind_routes_through_global_hub() -> None:
    bench = _FakeBench()
    previous = hub._handler
    try:
        bind(bench)
        reply = hub.command({"op": "set_feel", "mouth": 1.25})
        assert reply["ok"] is True
        assert bench.feel["mouth"] == 1.25
        frame = hub.latest_status()
        assert frame is not None
        assert "set_feel" in frame["commands"]
    finally:
        hub.set_handler(previous)


def test_ack_shape() -> None:
    packet = ack(ident="1", ok=False, error="nope")
    assert packet["type"] == "ack"
    assert packet["ok"] is False
    assert packet["error"] == "nope"


def test_http_status_and_feel_command() -> None:
    import pytest

    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from harness import server as harness_server

    bench = _FakeBench()
    previous = hub._handler
    previous_bench = harness_server._bench
    try:
        app = FastAPI()
        harness_server.attach(app, bench)
        client = TestClient(app)
        status = client.get("/harness/status").json()
        assert status["type"] == "status"
        assert "set_feel" in status["commands"]
        reply = client.post("/harness/command", json={"op": "set_feel", "body": {"mouth": 1.5}}).json()
        assert reply["ok"] is True
        assert bench.feel["mouth"] == 1.5
        frame = client.get("/harness/frame").json()
        assert frame["type"] == "frame"
        assert frame["schema"] == "KEYPOINT_SCHEMA"
        assert len(frame["keypoints"]) == NUM_KEYPOINTS
        with client.websocket_connect("/harness/ws") as ws:
            first = ws.receive_json()
            assert first["type"] in {"status", "frame"}
            ws.send_text("ping")
            kinds = {first["type"]}
            saw_ack = False
            for _ in range(6):
                msg = ws.receive_json()
                kinds.add(msg["type"])
                if msg["type"] == "ack":
                    saw_ack = True
                    assert msg["ok"] is True
                    break
            assert saw_ack
            assert "status" in kinds or "frame" in kinds
    finally:
        hub.set_handler(previous)
        harness_server._bench = previous_bench


def test_host_warming_without_bench() -> None:
    from harness.bridge import WorkerBridge, publish_warming
    from harness.protocol import NUM_KEYPOINTS, PROTOCOL
    from harness.server import command, frame, status

    previous = hub._handler
    previous_status = hub._status
    previous_frame = hub._frame
    bridge = WorkerBridge()
    try:
        publish_warming()
        hub.set_handler(bridge.handle_command)
        # Routes return pre-encoded JSON.
        packet = json.loads(status().body)
        assert packet["protocol"] == PROTOCOL
        assert packet["loaded"] is False
        assert packet["ok"] is True
        assert packet["live"] is False
        ping = command({"op": "ping"})
        assert ping["ok"] is True
        queued = command({"op": "set_feel", "body": {"mouth": 0.2}})
        assert queued["ok"] is True
        assert len(bridge._queued) == 1
        packed = json.loads(frame().body)
        assert packed["type"] == "frame"
        assert packed["loaded"] is False
        assert len(packed["keypoints"]) == NUM_KEYPOINTS
    finally:
        hub.set_handler(previous)
        hub._status = previous_status
        hub._frame = previous_frame


def test_command_route_is_sync() -> None:
    import inspect

    from harness.server import command

    assert inspect.iscoroutinefunction(command) is False


def test_host_server_does_not_import_face() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "backend" / "server.py").read_text(encoding="utf-8")
    assert "from .face import" not in src
    assert "import face" not in src
    assert "backend.worker" in (Path(__file__).resolve().parents[1] / "harness" / "bridge.py").read_text(encoding="utf-8")


def test_capture_keeps_moving_rows_only() -> None:
    from harness.capture import _moved, row_of

    frame = pack_frame({"points": _face_points(), "skeleton": _skeleton(), "head": {"pitch": 1.0, "yaw": 2.0, "roll": 0.0}, "live": True}, image_wh=(800, 800))
    row = row_of(frame, 10.0)
    assert row["head"] == {"pitch": 1.0, "yaw": 2.0, "roll": 0.0}
    assert len(row["keypoints"]) == NUM_KEYPOINTS and len(row["keypoints"][0]) == 3
    # Same pose polled twice (60 fps phone, faster poll): one row, not two.
    assert _moved(row, None) is True
    assert _moved(row_of(frame, 10.01), row) is False
    moved = dict(frame, head={"pitch": 1.0, "yaw": 3.0, "roll": 0.0})
    assert _moved(row_of(moved, 10.02), row) is True
