from __future__ import annotations

from backend.lab_harness import (
    LabHarness,
    COMMAND_TIMEOUT,
    DEFAULT_FEEL,
    SLOW_OPS,
    hair_from_frame,
    looks_like_lab,
    merge_frame_into_status,
    occupied_error,
    offline_status,
    overlay_from_frame,
    tracker_loaded,
)


class _FakeClient(LabHarness):
    def __init__(self) -> None:
        super().__init__(base="http://127.0.0.1:9", timeout=0.05)
        self.calls: list[tuple[str, dict]] = []
        self._feel = dict(DEFAULT_FEEL)

    def status(self, **_kwargs) -> dict:
        return {
            "type": "status",
            "online": True,
            "ok": True,
            "live": True,
            "feel": dict(self._feel),
            "commands": ["set_feel", "calibrate"],
            "source": "camera",
            "camera_index": 0,
            "cameras": [{"index": 0, "name": "Cam"}],
        }

    def command(self, op: str, body: dict | None = None) -> dict:
        payload = dict(body or {})
        self.calls.append((op, payload))
        if op == "set_feel":
            self._feel.update({k: float(v) for k, v in payload.items()})
        return {"type": "ack", "ok": True, "online": True, "status": self.status()}


def test_offline_status_has_feel_defaults() -> None:
    packet = offline_status("down")
    assert packet["online"] is False
    assert packet["feel"]["smoothing"] == DEFAULT_FEEL["smoothing"]
    assert packet["feel"]["show_skeleton"] == 1.0
    assert packet["feel"]["hair_pin"] == 0.7
    assert packet["loaded"] is False
    assert packet["look"] is None
    assert packet["iris_cam"] == []
    assert packet["iris_method"] == "none"
    assert packet["shapes"] == {}
    assert packet["presets"] == []
    assert packet["active"] == ""


def test_status_fills_missing_feel_keys() -> None:
    client = LabHarness()
    client._get = lambda path: {"type": "status", "feel": {"mouth": 1.25}}  # type: ignore[method-assign]
    packet = client.status()
    assert packet["online"] is True
    assert packet["feel"]["mouth"] == 1.25
    assert packet["feel"]["response"] == DEFAULT_FEEL["response"]
    assert packet["feel"]["show_ids"] == 0.0


def test_status_marks_connection_refused() -> None:
    packet = LabHarness(base="http://127.0.0.1:1", timeout=0.05).status()
    assert packet["online"] is False
    assert packet["error"]


def test_short_error_404_names_the_url() -> None:
    import urllib.error

    from backend.lab_harness import _short_error

    exc = urllib.error.HTTPError(
        "http://127.0.0.1:8780/harness/status",
        404,
        "Not Found",
        hdrs=None,
        fp=None,
    )
    text = _short_error(exc)
    assert "old build" in text
    assert "404" in text
    assert "/harness/status" in text


def test_looks_like_lab_accepts_harness_packets() -> None:
    assert looks_like_lab({"type": "status", "feel": {}})
    assert looks_like_lab({"protocol": "track_lab.harness.v1", "ok": True})
    assert not looks_like_lab({"ok": True, "server": "nginx"})
    assert not looks_like_lab("<html></html>")


def test_status_rejects_foreign_listener() -> None:
    client = LabHarness(base="http://127.0.0.1:8780/harness")
    client._get = lambda path: {"ok": True, "name": "something-else"}  # type: ignore[method-assign]
    packet = client.status()
    assert packet["online"] is False
    assert "8780" in packet["error"]
    assert "another app" in packet["error"]
    assert occupied_error().startswith("Port")


def test_command_set_feel_roundtrip() -> None:
    fake = _FakeClient()
    reply = fake.command("set_feel", {"smoothing": 0.2, "show_face": 0})
    assert reply["ok"] is True
    assert fake.calls == [("set_feel", {"smoothing": 0.2, "show_face": 0})]
    assert fake._feel["smoothing"] == 0.2
    assert fake._feel["show_face"] == 0.0


def test_status_copies_overlay_and_calib_from_frame() -> None:
    client = LabHarness()
    packets = {
        "/status": {"type": "status", "live": False, "calib": {"rest": False}},
        "/frame": {
            "type": "frame",
            "live": True,
            "weights": {"smile": 0.4, "A": 0.2},
            "points": [[10.0, 20.0, 1.0]],
            "hair": [{"class": "hair_middle", "polygon": [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]}],
            "calib": {"rest": True, "capturing": "rest", "progress": 0.5},
        },
    }
    client._get = lambda path: packets[path]  # type: ignore[method-assign]
    packet = client.status()
    assert packet["live"] is True
    assert packet["weights"]["smile"] == 0.4
    assert packet["points"][0][0] == 10.0
    assert packet["calib"]["rest"] is True
    assert packet["calib"]["progress"] == 0.5
    assert packet["hair"][0]["class"] == "hair_middle"


def test_merge_frame_uses_live_calib() -> None:
    payload = {"calib": {"capturing": "rest", "progress": 0.0, "hint": "hold", "rest": False}}
    merge_frame_into_status(
        payload,
        {"live": True, "calib": {"rest": True, "capturing": "", "progress": 0.0}},
    )
    assert payload["calib"]["capturing"] == ""
    assert payload["calib"]["rest"] is True


def test_merge_frame_copies_iris_method() -> None:
    payload = {"iris_method": "none", "live": False}
    merge_frame_into_status(payload, {"live": True, "iris_method": "osf_gaze"})
    assert payload["iris_method"] == "osf_gaze"


def test_merge_frame_copies_look_and_iris_cam() -> None:
    payload = {"look": None, "iris_cam": [], "live": False}
    merge_frame_into_status(
        payload,
        {
            "live": True,
            "look": {"x": 0.42, "y": -0.2},
            "iris_cam": [
                {"side": "r", "x": 22.0, "y": 14.0, "score": 0.9, "visible": True, "method": "look"}
            ],
        },
    )
    assert payload["look"] == {"x": 0.42, "y": -0.2}
    assert payload["iris_cam"][0]["method"] == "look"


def test_overlay_from_frame_keeps_lab_iris() -> None:
    frame = {
        "image_wh": [100, 100],
        "keypoints": [
            {"i": 11, "x": 20.0, "y": 40.0, "score": 1.0, "visible": True},
            {"i": 13, "x": 40.0, "y": 40.0, "score": 1.0, "visible": True},
            {"i": 28, "ref": "IRIS.L", "x": 48.0, "y": 40.0, "score": 0.95, "visible": True},
        ],
    }
    k = overlay_from_frame(frame)
    assert k is not None
    assert abs(float(k[28, 0]) - (48.0 / 100.0 * 2.0 - 1.0)) < 1e-5
    assert float(k[28, 3]) >= 0.5


def test_overlay_from_frame_resolves_iris_by_ref() -> None:
    frame = {
        "image_wh": [100, 100],
        "keypoints": [
            {"ref": "IRIS.L", "x": 30.0, "y": 40.0, "score": 1.0, "visible": True},
            {"legacy": "left_iris", "x": 70.0, "y": 40.0, "score": 1.0, "visible": True},
        ],
    }
    k = overlay_from_frame(frame)
    assert k is not None
    assert abs(float(k[28, 0]) - (30.0 / 100.0 * 2.0 - 1.0)) < 1e-5
    assert abs(float(k[29, 0]) - (70.0 / 100.0 * 2.0 - 1.0)) < 1e-5


def test_merge_frame_copies_hair_method() -> None:
    payload = {"hair_method": "none", "live": False}
    merge_frame_into_status(payload, {"live": True, "hair_method": "lab_follow"})
    assert payload["hair_method"] == "lab_follow"
    payload = {"error": "hold still", "live": False}
    merge_frame_into_status(payload, {"live": True, "error": ""})
    assert payload["live"] is True
    assert payload["error"] == "hold still"


def test_merge_frame_does_not_resurrect_idle_start_gate_error() -> None:
    payload = {"error": "", "live": False}
    merge_frame_into_status(
        payload,
        {"live": False, "error": "Track a face first so rest exists"},
    )
    assert payload["error"] == ""


def test_overlay_from_frame_converts_character_px_to_norm() -> None:
    frame = {
        "image_wh": [100, 50],
        "keypoints": [
            {"i": 0, "x": 0.0, "y": 0.0, "score": 1.0, "visible": True},
            {"i": 21, "x": 50.0, "y": 25.0, "score": 1.0, "visible": True},
        ],
    }
    k = overlay_from_frame(frame)
    assert k is not None
    assert k.shape == (37, 4)
    assert abs(float(k[0, 0]) - -1.0) < 1e-5
    assert abs(float(k[0, 1]) - -1.0) < 1e-5
    assert abs(float(k[21, 0]) - 0.0) < 1e-5
    assert abs(float(k[21, 1]) - 0.0) < 1e-5
    assert float(k[21, 3]) >= 0.5


def test_overlay_from_frame_prefers_lab_image_wh() -> None:
    frame = {
        "image_wh": [100, 50],
        "keypoints": [
            {"i": 21, "x": 50.0, "y": 25.0, "score": 1.0, "visible": True},
        ],
    }
    k = overlay_from_frame(frame, width=200, height=100)
    assert k is not None
    assert abs(float(k[21, 0])) < 1e-5
    assert abs(float(k[21, 1])) < 1e-5


def test_hair_from_frame_prefers_lab_image_wh() -> None:
    frame = {
        "image_wh": [100, 50],
        "hair": [
            {
                "class": "hair_middle",
                "polygon": [[50.0, 25.0], [80.0, 25.0], [65.0, 40.0]],
            }
        ],
    }
    segs = hair_from_frame(frame, width=200, height=100)
    assert segs is not None
    assert abs(float(segs[0]["polygon"][0][0])) < 1e-5
    assert abs(float(segs[0]["polygon"][0][1])) < 1e-5


def test_hair_from_frame_converts_character_px_to_norm() -> None:
    frame = {
        "image_wh": [100, 50],
        "keypoints": [
            {"i": 21, "x": 50.0, "y": 25.0, "score": 1.0, "visible": True},
        ],
        "hair": [
            {
                "class": "hair_middle",
                "polygon": [[50.0, 25.0], [80.0, 25.0], [65.0, 40.0]],
            }
        ],
    }
    k = overlay_from_frame(frame)
    segs = hair_from_frame(frame)
    assert k is not None
    assert segs is not None
    assert segs[0]["class"] == "hair_middle"
    assert abs(float(segs[0]["polygon"][0][0]) - float(k[21, 0])) < 1e-5
    assert abs(float(segs[0]["polygon"][0][1]) - float(k[21, 1])) < 1e-5
    assert hair_from_frame({"image_wh": [100, 50]}) is None
    assert hair_from_frame({"hair": []}) == []


def test_slow_ops_cover_start_stop_track() -> None:
    assert SLOW_OPS["start"] >= 30
    assert SLOW_OPS["stop"] >= 8
    assert SLOW_OPS["track"] >= 30
    assert SLOW_OPS["set_source"] >= 8


def test_command_timeout_uses_slow_ops() -> None:
    seen: list[float] = []
    client = LabHarness(timeout=0.6)

    def fake_post(path, payload, timeout=None):
        seen.append(float(timeout))
        return {"type": "ack", "ok": True}

    client._post = fake_post  # type: ignore[method-assign]
    client.command("start", {"camera": 0})
    client.command("calibrate", {"id": "rest"})
    assert seen[0] == SLOW_OPS["start"]
    assert seen[1] == COMMAND_TIMEOUT


def test_put_source_falls_back_when_op_unknown(tmp_path) -> None:
    still = tmp_path / "char.png"
    still.write_bytes(b"png")
    client = LabHarness()
    posted: list[str] = []
    client.command = lambda op, body=None: {  # type: ignore[method-assign]
        "type": "ack",
        "ok": False,
        "error": "unknown op 'set_source'",
    }
    client._post_source_file = lambda path: posted.append(path) or {  # type: ignore[method-assign]
        "type": "ack",
        "ok": True,
        "online": True,
        "error": "",
    }
    reply = client.put_source(str(still))
    assert reply["ok"] is True
    assert posted == [str(still)]


def test_handshake_pings_without_start() -> None:
    fake = _FakeClient()
    out = fake.handshake(attempts=1, delay=0)
    assert out["handshake"] is True
    assert out["online"] is True
    assert fake.calls == [("ping", {})]


def test_handshake_retries_until_online() -> None:
    hits = {"n": 0}

    class Flaky(LabHarness):
        def __init__(self) -> None:
            super().__init__(base="http://127.0.0.1:9", timeout=0.01)

        def status(self, **_kwargs) -> dict:
            hits["n"] += 1
            if hits["n"] < 3:
                return offline_status("down")
            return {"type": "status", "online": True, "ok": True, "live": False}

        def command(self, op: str, body: dict | None = None) -> dict:
            assert op == "ping"
            return {"type": "ack", "ok": True, "online": True}

    out = Flaky().handshake(attempts=5, delay=0)
    assert out["handshake"] is True
    assert hits["n"] == 3


def test_handshake_hooks_in_when_ping_times_out() -> None:
    client = LabHarness(base="http://127.0.0.1:9", timeout=0.01)
    client.status = lambda **_kwargs: {  # type: ignore[method-assign]
        "type": "status",
        "online": True,
        "ok": True,
        "live": False,
        "loaded": True,
        "error": "",
    }
    client.command = lambda op, body=None: offline_status("Track Lab timed out")  # type: ignore[method-assign]
    out = client.handshake()
    assert out["online"] is True
    assert out["handshake"] is True
    assert out["ok"] is True


def test_handshake_skips_frame_and_allows_unloaded() -> None:
    paths: list[str] = []
    client = LabHarness()

    def fake_get(path: str, timeout=None):
        paths.append(path)
        return {
            "type": "status",
            "protocol": "track_lab.harness.v1",
            "loaded": False,
            "live": False,
            "ok": True,
        }

    client._get = fake_get  # type: ignore[method-assign]
    client.command = lambda op, body=None: {"type": "ack", "ok": True, "online": True}  # type: ignore[method-assign]
    out = client.handshake()
    assert out["handshake"] is True
    assert out["loaded"] is False
    assert "/status" in paths
    assert "/frame" not in paths


def test_tracker_loaded_distinguishes_warming() -> None:
    assert tracker_loaded({"online": True, "loaded": False}) is False
    assert tracker_loaded({"online": True, "loaded": True}) is True
    assert tracker_loaded({"online": True}) is True
    assert tracker_loaded({"online": False, "loaded": True}) is False


def test_wait_loaded_polls_until_true(monkeypatch) -> None:
    from backend import lab_harness

    hits = {"n": 0}

    def fake_status(merge_frame: bool = True) -> dict:
        hits["n"] += 1
        return {"online": True, "loaded": hits["n"] >= 3, "ok": True}

    monkeypatch.setattr(lab_harness.lab, "status", fake_status)
    out = lab_harness.wait_loaded(timeout=2.0)
    assert out["loaded"] is True
    assert hits["n"] >= 3


def _hair_runtime():
    import threading
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.Lock()
    rt._lab_drive = False
    rt._lab_seen_online = False
    rt._pose_frozen = False
    rt._mesh_edited = False
    rt._last_image = None
    rt._lab_image_wh = None
    rt._hair_rig = None
    rt._hair_capture_done = False
    rt._last_lab_hair = None
    rt._lab_rest_hair = None
    rt._last_overlay_kps = None
    rt._drag_slots = set()
    rt._drag_xy = None
    rt.engine = SimpleNamespace(_ref_keypoints=None)
    return rt


def test_followed_hair_copies_lab_polygons() -> None:
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    rt._last_lab_hair = [
        {"class": "hair_middle", "polygon": [[0.0, 0.0], [0.2, 0.0], [0.1, 0.2]]}
    ]
    segs = StreamRuntime._followed_hair_segments(rt)
    assert segs[0]["class"] == "hair_middle"
    assert segs[0]["polygon"][0] == [0.0, 0.0]


def test_rest_hair_uses_lab_snapshot() -> None:
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    rt._lab_drive = True
    rt._lab_rest_hair = [
        {"class": "hair_left", "polygon": [[-0.4, -0.2], [-0.2, -0.2], [-0.3, 0.0]]}
    ]
    rt._last_lab_hair = [
        {"class": "hair_left", "polygon": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]}
    ]
    segs = StreamRuntime._rest_hair_segments(rt)
    assert segs[0]["polygon"][0] == [-0.4, -0.2]


def test_adopt_lab_hair_skips_desk_animeseg(monkeypatch) -> None:
    from backend.lab_harness import hair_from_frame
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    frame = {
        "image_wh": [100, 50],
        "hair": [
            {
                "class": "hair_right",
                "polygon": [[80.0, 10.0], [90.0, 10.0], [85.0, 20.0]],
            }
        ],
    }

    class _Lab:
        def frame(self):
            return frame

        def status(self, merge_frame=True):
            return {"online": True, **frame, "width": 100, "height": 50}

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    captured = {"n": 0}

    def fail_capture(self, rest_keypoints=None):
        captured["n"] += 1

    monkeypatch.setattr(StreamRuntime, "_capture_character_hair_mesh", fail_capture)
    assert StreamRuntime._adopt_lab_hair(rt) is True
    assert rt._last_lab_hair[0]["class"] == "hair_right"
    expected = hair_from_frame(frame)
    assert rt._last_lab_hair[0]["polygon"][0] == expected[0]["polygon"][0]
    assert rt._lab_image_wh == (100, 50)
    assert rt._lab_rest_hair[0]["class"] == "hair_right"
    assert rt._hair_capture_done is True
    assert rt._hair_rig is None
    StreamRuntime._maybe_capture_hair(rt)
    assert captured["n"] == 0


def test_lab_overlay_keeps_lab_look(monkeypatch) -> None:
    from PIL import Image

    from backend.lab_harness import overlay_from_frame
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    rt._lab_drive = True
    rt._last_image = Image.new("RGB", (100, 100), (0, 0, 0))
    rt._status = {"travel_box": {"enabled": True, "eyes": True, "eye_x": 0.05, "eye_y": 0.05}}
    frame = {
        "image_wh": [100, 100],
        "keypoints": [
            {"i": 11, "x": 20.0, "y": 40.0, "score": 1.0, "visible": True},
            {"i": 12, "x": 30.0, "y": 40.0, "score": 1.0, "visible": True},
            {"i": 13, "x": 40.0, "y": 40.0, "score": 1.0, "visible": True},
            {"i": 28, "ref": "IRIS.L", "x": 48.0, "y": 40.0, "score": 1.0, "visible": True},
        ],
        "look": {"x": 0.7, "y": -0.4},
        "hair": [],
    }

    class _Lab:
        def frame(self):
            return frame

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())

    def fail_clamp(*_args, **_kwargs):
        raise AssertionError("desk travel box must not rewrite lab look")

    monkeypatch.setattr(StreamRuntime, "_clamp_travel_box", fail_clamp)
    expected = overlay_from_frame(frame, width=100, height=100)
    out = StreamRuntime._lab_overlay_keypoints(rt)
    assert expected is not None
    assert out is not None
    assert abs(float(out[28, 0]) - float(expected[28, 0])) < 1e-6
    assert abs(float(out[28, 1]) - float(expected[28, 1])) < 1e-6


def test_lab_overlay_uses_lab_wh_when_desk_differs(monkeypatch) -> None:
    from PIL import Image

    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    rt._lab_drive = True
    rt._last_image = Image.new("RGB", (200, 200), (0, 0, 0))
    rt._clamp_lab_walk = lambda driven, hair: (driven, hair)
    frame = {
        "image_wh": [100, 100],
        "keypoints": [
            {"i": 21, "x": 50.0, "y": 50.0, "score": 1.0, "visible": True},
        ],
        "hair": [],
    }

    class _Lab:
        def frame(self):
            return frame

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    out = StreamRuntime._lab_overlay_keypoints(rt)
    assert out is not None
    assert abs(float(out[21, 0])) < 1e-5
    assert abs(float(out[21, 1])) < 1e-5
    assert rt._lab_image_wh == (100, 100)


def test_lab_overlay_walk_stops_at_wall(monkeypatch) -> None:
    from PIL import Image

    from backend.engine import neutral_keypoints
    from backend.pose_controller import RIGHT_IRIS
    from backend.stream import StreamRuntime

    rest = neutral_keypoints()
    rest[11:14, 3] = 1.0
    rest[RIGHT_IRIS, 3] = 1.0
    rt = _hair_runtime()
    rt._lab_drive = True
    rt._last_image = Image.new("RGB", (100, 100), (0, 0, 0))
    rt._status = {"travel_box": {"enabled": True, "side": True, "left": 0.0, "right": 0.0}}
    rt.engine._ref_keypoints = rest
    live = rest.copy()
    live[:, 0] += 0.7
    look = 0.05
    live[RIGHT_IRIS, 0] = float(live[13, 0]) + look
    rows = []
    for i in range(37):
        if float(live[i, 3]) < 0.5:
            continue
        rows.append(
            {
                "i": i,
                "x": (float(live[i, 0]) + 1.0) * 50.0,
                "y": (float(live[i, 1]) + 1.0) * 50.0,
                "score": 1.0,
                "visible": True,
            }
        )
    frame = {"image_wh": [100, 100], "keypoints": rows, "hair": []}

    class _Lab:
        def frame(self):
            return frame

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    monkeypatch.setattr(
        StreamRuntime,
        "_clamp_travel_box",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("desk travel box must not rewrite lab look")
        ),
    )
    out = StreamRuntime._lab_overlay_keypoints(rt)
    assert out is not None
    assert float(out[4, 0]) < float(live[4, 0]) - 0.2
    assert abs(float(out[RIGHT_IRIS, 0] - out[13, 0]) - look) < 1e-5


def test_lab_keeps_authored_end_shapes() -> None:
    from backend.stream import lab_keeps_authored

    assert lab_keeps_authored({"ready": True, "shapes": {"A": [[0, 0, 1]]}})
    assert lab_keeps_authored({"ready": False, "shapes": {"smile": [[0, 0, 1]]}})
    assert lab_keeps_authored({"ready": True, "shapes": {}})
    assert lab_keeps_authored({"ready": False, "shapes": {}}) is False
    assert lab_keeps_authored(None) is False


def test_sync_lab_character_keeps_authored_shapes(monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    calls: list[str] = []

    class _Lab:
        def status(self, merge_frame=False):
            return {
                "online": True,
                "ready": True,
                "shapes": {"rest": [[0, 0, 1]], "A": [[0, 1, 1]]},
            }

        def put_source(self, path):
            calls.append(f"put:{path}")
            return {"ok": True, "status": {}}

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    rt._same_lab_still = lambda: True
    rt._lab_ack = lambda op, body=None: calls.append(op)
    rt._adopt_lab_hair = lambda: calls.append("hair")
    rt.adopt_lab_overlay = lambda packet=None, emit=False: calls.append("overlay")
    StreamRuntime._sync_lab_character(rt)
    assert "put" not in "".join(calls)
    assert "track" not in calls
    assert "hair" in calls
    assert "overlay" in calls


def test_sync_lab_character_replaces_when_still_changes(monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _hair_runtime()
    calls: list[str] = []

    class _Lab:
        def status(self, merge_frame=False):
            return {
                "online": True,
                "ready": True,
                "shapes": {"rest": [[0, 0, 1]], "A": [[0, 1, 1]]},
            }

        def put_source(self, path):
            calls.append(f"put:{path}")
            return {"ok": True, "status": {}}

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    rt._same_lab_still = lambda: False
    rt._write_lab_source = lambda: "track_lab/input/source.png"
    rt._lab_ack = lambda op, body=None: calls.append(op)
    rt._adopt_lab_hair = lambda: calls.append("hair")
    rt.adopt_lab_overlay = lambda packet=None, emit=False: calls.append("overlay")
    StreamRuntime._sync_lab_character(rt)
    assert any(item.startswith("put:") for item in calls)
    assert "track" in calls
    assert "hair" in calls


def test_current_keypoints_uses_lab_overlay_when_idle() -> None:
    from backend.engine import neutral_keypoints
    from backend.stream import StreamRuntime

    rest = neutral_keypoints()
    live = rest.copy()
    live[21, 1] += 0.2
    rt = _hair_runtime()
    rt._tracking = False
    rt._lab_drive = False
    rt._lab_seen_online = True
    rt._mesh_edited = False
    rt._pose_frozen = False
    rt._driven_keypoints = rest.copy()
    rt.status = lambda: {"drive_pose": True}
    rt._lab_overlay_keypoints = lambda frame=None: live.copy()
    out = StreamRuntime._current_keypoints(rt)
    assert out is not None
    assert abs(float(out[21, 1]) - float(live[21, 1])) < 1e-6


def test_lab_drive_generate_skips_sanitize() -> None:
    import queue

    from backend.engine import neutral_keypoints
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lab_drive = True
    rt._gen_queue = queue.Queue(maxsize=2)
    rest = neutral_keypoints()
    StreamRuntime._enqueue_generate(rt, steps=4, streaming=False, keypoints=rest)
    job = rt._gen_queue.get_nowait()
    assert job["sanitize"] == "none"


def test_apply_reference_reports_progress_and_resyncs_lab() -> None:
    import inspect

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime.apply_reference)
    assert "on_progress=None" not in src
    assert "Creating character…" in src
    assert "_sync_lab_character(replace=True)" in src
    assert "_set_progress" in src
    assert "Fitting overlay…" in src
    assert "_warm_overlay_tools" in src
    assert "Loading face detector" in inspect.getsource(StreamRuntime._warm_overlay_tools)
    create_src = inspect.getsource(StreamRuntime.create_character)
    assert "Saving character" in create_src
    assert "_reset_character_runtime" in src


def test_desk_calibrate_runs_when_desk_is_tracking() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rail = (root / "ui" / "src" / "components" / "ControlRail.tsx").read_text(
        encoding="utf-8"
    )
    app = (root / "ui" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "const canCalibrate = tracking || labLive" in rail
    assert "labOnline ? !labLive : !tracking" not in rail
    assert "tracking || labOnline" in rail
    assert "onLabCalibrate('rest')" in rail
    assert "await sendLab('calibrate', { id })" in app


def test_recenter_sends_lab_calibrate() -> None:
    import inspect

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime.recenter)
    assert "calibrate_lab_rest" in src
    assert "Use Calibrate to capture Track Lab rest" not in src
    apply = inspect.getsource(StreamRuntime.apply_lab_calibrate)
    assert "adopt_lab_overlay" in apply
    assert "_pose_frozen = False" in apply


def test_lab_command_copies_overlay_after_calibrate() -> None:
    from pathlib import Path

    fn = (
        Path(__file__).resolve().parents[1] / "api.py"
    ).read_text(encoding="utf-8").split("def lab_command")[1].split("@app.")[0]
    assert 'body.op == "calibrate"' in fn
    assert "apply_lab_calibrate" in fn


def test_apply_lab_calibrate_unfreezes_and_copies_overlay() -> None:
    rt = _hair_runtime()
    rt._pose_frozen = True
    rt._mesh_edited = True
    seen: list[object] = []

    def adopt(packet=None, emit=False):
        seen.append((packet, emit))
        return True

    rt.adopt_lab_overlay = adopt  # type: ignore[method-assign]
    rt._set_status = lambda **kwargs: None  # type: ignore[method-assign]
    from backend.stream import StreamRuntime

    StreamRuntime.apply_lab_calibrate(
        rt,
        {
            "ok": True,
            "status": {
                "calib": {"capturing": "rest", "progress": 0.1},
                "keypoints": [{"i": 0, "x": 1, "y": 2}],
            },
        },
    )
    assert rt._pose_frozen is False
    assert rt._mesh_edited is False
    assert seen and seen[0][1] is True


def test_desk_calibrate_stays_above_mouth_slider() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rail = (root / "ui" / "src" / "components" / "ControlRail.tsx").read_text(
        encoding="utf-8"
    )
    feel = (root / "ui" / "src" / "components" / "LabFeel.tsx").read_text(
        encoding="utf-8"
    )
    assert "Set rest again" not in rail
    assert "Set rest" not in rail
    assert rest_label_is_calibrate(rail)
    assert rail.index("track-run") < rail.index("</LabFeel>")
    assert feel.index("{props.actions}") < feel.index("<FeelSliders")


def test_tracking_buttons_stay_enabled_while_generating() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rail = (root / "ui" / "src" / "components" / "ControlRail.tsx").read_text(
        encoding="utf-8"
    )
    idx = rail.index("onClick={props.onToggleTracking}")
    chunk = rail[idx : idx + 220]
    assert "disabled={Boolean(s?.track_busy)}" in chunk
    assert "busy || streaming" not in chunk
    cal = rail[rail.index("onLabCalibrate('rest')") : rail.index("onLabCalibrate('rest')") + 420]
    assert "busy ||" not in cal


def test_lab_command_returns_ack() -> None:
    from pathlib import Path

    fn = (Path(__file__).resolve().parents[1] / "api.py").read_text(
        encoding="utf-8"
    ).split("def lab_command")[1].split("@app.")[0]
    assert "raise HTTPException(503" in fn
    assert "\n    return result\n" in fn
    assert "        return result" not in fn


def test_lab_status_skips_overlay_copy_while_tracking() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "api.py").read_text(encoding="utf-8")
    assert 'if not bool(getattr(rt, "_tracking", False)):' in text
    assert "emit=not bool" not in text


def test_desk_stops_boot_poll_when_ready() -> None:
    from pathlib import Path

    app = (Path(__file__).resolve().parents[2] / "ui" / "src" / "App.tsx").read_text(
        encoding="utf-8"
    )
    assert "if (deskReady) return" in app
    assert "[deskReady]" in app
    boot = app.split("useEffect(() => {")[1].split("}, [deskReady])")[0]
    assert "api.boot()" in boot
    assert "inflight" in boot


def rest_label_is_calibrate(rail: str) -> bool:
    return ": 'Calibrate'" in rail or ': "Calibrate"' in rail


def test_update_settings_sends_set_mirror_when_lab_is_seen() -> None:
    import threading
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    calls: list[tuple[str, dict]] = []
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.Lock()
    rt._status = {"mirror": False}
    rt._lab_drive = False
    rt._lab_seen_online = True
    rt._tracking = True
    rt.tracker = SimpleNamespace(mirror=False, reset_center=lambda: None)
    rt._listeners = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {})))
    rt._reset_live_origin = lambda reason="": None
    StreamRuntime.update_settings(rt, mirror=True)
    assert calls == [("set_mirror", {"on": True})]
    assert rt.tracker.mirror is True
    assert rt._status["mirror"] is True
