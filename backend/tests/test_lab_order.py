"""Track Lab packet order on the desk: restarts (session) and stale frames (seq)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.lab_harness import lab_packet_order, merge_frame_into_status
from backend.stream import StreamRuntime


def _runtime() -> StreamRuntime:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.Lock()
    rt._status = {}
    rt._streaming = False
    rt._frame_in_flight = False
    rt._gen_busy = False
    rt._tracking = True
    rt._lab_drive = True
    rt._lab_seen_online = True
    rt._lab_overlay_gen = 0
    rt._lab_seen_generation = 0
    rt._pose_frozen = False
    rt._mesh_edited = False
    rt._last_image = Image.new("RGB", (100, 100), (0, 0, 0))
    rt._lab_image_wh = None
    rt._hair_rig = None
    rt._hair_capture_done = False
    rt._last_lab_hair = None
    rt._lab_rest_hair = None
    rt._last_overlay_kps = None
    rt._driven_keypoints = None
    rt._last_good_keypoints = None
    rt._drag_slots = set()
    rt._drag_xy = None
    rt.engine = SimpleNamespace(_ref_keypoints=None)
    rt._emit = lambda msg: None
    rt._frame_payload = lambda image, kps: {}
    return rt


def _frame(session: str, seq: int, x: float, *, gen: int = 0) -> dict:
    return {
        "type": "frame",
        "generation": gen,
        "session": session,
        "seq": seq,
        "image_wh": [100, 100],
        "keypoints": [{"i": 21, "x": x, "y": 50.0, "score": 1.0, "visible": True}],
        "hair": [{"class": "hair_middle", "polygon": [[x, 10.0], [x + 10.0, 10.0], [x + 5.0, 20.0]]}],
    }


def _shown_x(rt: StreamRuntime) -> float:
    """Character-pixel x of point 21 on the desk overlay (100 px still)."""
    return round((float(rt._last_overlay_kps[21, 0]) + 1.0) * 50.0, 3)


def _hair_x(rt: StreamRuntime) -> float:
    return round((float(rt._last_lab_hair[0]["polygon"][0][0]) + 1.0) * 50.0, 3)


def test_an_older_frame_never_replaces_a_newer_one() -> None:
    """Track poll, DiT feed and command acks all apply lab frames. One fetched
    before a newer one was applied put the older pose and hair back."""
    rt = _runtime()
    assert StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 5, 60.0)) is not None
    assert StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 4, 20.0)) is None
    assert _shown_x(rt) == 60.0
    assert round((float(rt._driven_keypoints[21, 0]) + 1.0) * 50.0, 3) == 60.0
    assert _hair_x(rt) == 60.0
    # Same seq (a command ack of the same pose) and newer ones still land.
    assert StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 5, 61.0)) is not None
    assert StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 6, 70.0)) is not None
    assert _shown_x(rt) == 70.0


def test_a_limiter_edit_is_not_undone_by_a_poll_fetched_before_it() -> None:
    """Dragging a travel slider while tracking: the set_travel ack shows the
    re-capped pose, then a /frame fetched before it snapped it back a frame."""
    rt = _runtime()
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 7, 30.0))
    StreamRuntime._emit_live_limiter_pose(rt, {"ok": True, "frame": _frame("s1", 9, 70.0)})
    assert _shown_x(rt) == 70.0
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 8, 30.0))
    assert _shown_x(rt) == 70.0


def test_lab_hair_is_ordered_with_the_overlay() -> None:
    rt = _runtime()
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 5, 60.0))
    assert StreamRuntime._adopt_lab_hair(rt, _frame("s1", 3, 20.0)) is False
    assert _hair_x(rt) == 60.0
    assert StreamRuntime._adopt_lab_hair(rt, _frame("s1", 6, 65.0)) is True
    assert _hair_x(rt) == 65.0


def test_frames_without_an_order_still_apply() -> None:
    """An older Track Lab sends no session / seq: nothing changes for it."""
    rt = _runtime()
    old = _frame("", 0, 40.0)
    del old["session"], old["seq"]
    assert lab_packet_order(old) is None
    assert StreamRuntime._lab_overlay_keypoints(rt, old) is not None
    assert StreamRuntime._lab_overlay_keypoints(rt, dict(old, keypoints=[dict(old["keypoints"][0], x=45.0)])) is not None
    assert _shown_x(rt) == 45.0


def test_a_restarted_lab_is_followed_not_held_frozen() -> None:
    """A new Track Lab worker starts generation (and seq) at 0. The desk kept
    the old ones and dropped every new frame: "Tracking on" over a frozen pose."""
    rt = _runtime()
    rt._lab_overlay_gen = 5
    StreamRuntime._lab_overlay_keypoints(rt, _frame("old", 900, 30.0, gen=5))
    assert rt._lab_seen_generation == 5
    assert getattr(rt, "_lab_session_restore", False) is False
    fresh = _frame("new", 1, 55.0, gen=0)
    assert StreamRuntime._lab_overlay_keypoints(rt, fresh) is not None
    assert _shown_x(rt) == 55.0
    assert rt._lab_overlay_gen == 0 and rt._lab_seen_generation == 0
    assert rt._lab_session_restore is True
    # A late packet from the tracker that is gone stays out.
    assert StreamRuntime._lab_overlay_keypoints(rt, _frame("old", 901, 30.0, gen=5)) is None
    assert _shown_x(rt) == 55.0


def test_the_first_lab_seen_needs_no_restore() -> None:
    rt = _runtime()
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 3, 30.0))
    assert rt._lab_session == "s1"
    assert getattr(rt, "_lab_session_restore", False) is False


def test_character_sync_after_a_restart_does_not_fail_on_the_old_generation(monkeypatch) -> None:
    """The restarted lab's set_source came back as generation 1 against a seen
    7, and the desk said "Track Lab did not load the new character still"."""
    rt = _runtime()
    StreamRuntime._lab_overlay_keypoints(rt, _frame("old", 50, 30.0, gen=7))
    assert rt._lab_seen_generation == 7
    calls: list[str] = []

    class _Lab:
        def status(self, merge_frame=False):
            return {"online": True, "ready": True, "shapes": {}, "generation": 0, "session": "new", "seq": 2}

        def put_source(self, path):
            calls.append("put")
            return {"ok": True, "status": {"generation": 1, "session": "new", "seq": 3}}

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    rt._same_lab_still = lambda: False
    rt._write_lab_source = lambda: "track_lab/input/source.png"
    rt._lab_ack = lambda op, body=None: calls.append(op) or {
        "ok": True,
        "status": {"generation": 1, "session": "new", "seq": 3},
        "frame": _frame("new", 3, 40.0, gen=1),
    }
    rt._maybe_capture_hair = lambda rest_keypoints=None: None
    StreamRuntime._sync_lab_character(rt)
    assert calls == ["put", "track"]
    assert rt._lab_overlay_gen == 1
    assert _shown_x(rt) == 40.0


def _restore_runtime(monkeypatch, *, source: str = "camera"):
    rt = _runtime()
    rt._status = {"mirror": True}
    rt.preferred_camera = lambda: 2
    rt._lab_session_restore = True
    calls: list[tuple] = []
    rt._sync_lab_character = lambda replace=False: calls.append(("sync",))
    monkeypatch.setattr(
        "backend.lab_harness.lab",
        SimpleNamespace(status=lambda merge_frame=False: {"online": True, "source": source}),
    )
    return rt, calls


def test_restore_gives_a_restarted_lab_the_character_and_starts_it(monkeypatch) -> None:
    """Nothing re-sent start to a respawned worker: tracking never came back."""
    rt, calls = _restore_runtime(monkeypatch)
    messages: list[str] = []
    rt._set_track_status = lambda **kw: messages.append(str(kw.get("track_message")))
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    StreamRuntime._restore_lab_session(rt)
    assert calls == [("sync",), ("start", {"source": "camera", "camera": 2, "mirror": True})]
    assert rt._lab_session_restore is False
    # Not "Tracking on" while the new tracker is brought back.
    assert messages and "resuming" in messages[0]
    rt, calls = _restore_runtime(monkeypatch, source="ifm")
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    StreamRuntime._restore_lab_session(rt)
    assert calls[-1] == ("start", {"source": "ifm"})


def test_restore_only_resyncs_when_not_tracking(monkeypatch) -> None:
    rt, calls = _restore_runtime(monkeypatch)
    rt._tracking = False
    rt._lab_ack = lambda op, body=None: pytest.fail("no start while tracking is off")
    StreamRuntime._restore_lab_session(rt)
    assert calls == [("sync",)]


def test_restore_that_cannot_start_turns_tracking_off(monkeypatch) -> None:
    rt, calls = _restore_runtime(monkeypatch)
    stopped: list[bool] = []
    rt.stop_tracking = lambda: stopped.append(True)

    def no_camera(op, body=None):
        raise RuntimeError("No camera at index 2")

    rt._lab_ack = no_camera
    StreamRuntime._restore_lab_session(rt)  # does not raise into the poll loop
    assert stopped == [True]


def test_track_poll_runs_the_restore(monkeypatch) -> None:
    rt = _runtime()
    rt._tracking = False
    rt._track_stop = threading.Event()
    ran: list[bool] = []

    def restore() -> None:
        ran.append(True)
        rt._lab_session_restore = False
        rt._track_stop.set()

    rt._restore_lab_session = restore
    rt._lab_session_restore = True
    loop = threading.Thread(target=StreamRuntime._track_poll_loop, args=(rt,), daemon=True)
    loop.start()
    loop.join(timeout=2.0)
    rt._track_stop.set()
    loop.join(timeout=1.0)
    assert ran == [True]


def test_nudge_names_the_frame_on_screen() -> None:
    """The lab measures the nudge on that frame's pose, not its newest one."""
    rt = _runtime()
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 12, 60.0))
    StreamRuntime._lab_overlay_keypoints(rt, _frame("s1", 11, 20.0))  # stale, never shown
    calls: list[tuple] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True, "status": {}}
    StreamRuntime._nudge_lab_point(rt, 21, 40.0, 60.0)
    assert calls == [("set_point", {"id": 21, "x": 40.0, "y": 60.0, "session": "s1", "seq": 12})]


def test_merged_status_carries_the_frames_order() -> None:
    status = {"session": "s1", "seq": 3, "generation": 1}
    merge_frame_into_status(status, _frame("s1", 8, 10.0, gen=1))
    assert lab_packet_order(status) == ("s1", 8)


def test_restore_skips_a_tracker_another_flow_already_synced(monkeypatch) -> None:
    """Start tracking (or a load) synced the new tracker while the restore
    waited: syncing again stopped the tracking it had started."""
    rt, calls = _restore_runtime(monkeypatch)
    rt._lab_restore_sync_n = 3
    rt._lab_sync_n = 4
    rt._lab_ack = lambda op, body=None: pytest.fail("the flow that synced it also started it")
    StreamRuntime._restore_lab_session(rt)
    assert calls == []
    assert rt._lab_session_restore is False


def test_a_restart_notes_the_sync_it_follows() -> None:
    rt = _runtime()
    rt._lab_sync_n = 5
    StreamRuntime._note_lab_session(rt, _frame("s1", 1, 10.0))
    assert not getattr(rt, "_lab_session_restore", False)  # the first tracker seen
    StreamRuntime._note_lab_session(rt, _frame("s2", 1, 10.0))
    assert rt._lab_session_restore is True
    assert rt._lab_restore_sync_n == 5


def test_character_syncs_take_turns() -> None:
    """The restore runs on the track poll thread, Start tracking on another:
    their set_source / set_rest / track must not interleave."""
    rt = _runtime()
    inside: list[int] = []
    overlaps: list[int] = []
    started = threading.Barrier(2)

    def locked(*, replace: bool) -> None:
        inside.append(1)
        if len(inside) > 1:
            overlaps.append(len(inside))
        threading.Event().wait(0.05)
        inside.pop()

    rt._sync_lab_character_locked = locked

    def sync() -> None:
        started.wait()
        StreamRuntime._sync_lab_character(rt)

    workers = [threading.Thread(target=sync, daemon=True) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=2.0)
    assert not overlaps
    assert rt._lab_sync_n == 2


def test_a_stale_reply_never_writes_its_rest_into_the_character() -> None:
    """A late reply for the previous still (or a restarted tracker) carried
    that face's rest rows; they were copied into this character's rest."""
    import numpy as np

    rt = _runtime()
    held = np.zeros((37, 4), dtype=np.float32)
    held[:, 3] = 1.0
    rt._rest_kps = held.copy()
    rt._lab_overlay_gen = 5
    rows = [[80.0, 80.0]] * 28

    def reply(**extra) -> dict:
        return {"image_wh": [100, 100], "rest": rows, **extra}

    for stale in (reply(generation=4), reply(generation=5, session="old")):
        rt._lab_old_sessions = ("old",)
        rest = StreamRuntime._lab_rest(rt, stale)
        assert np.allclose(rest, held), stale
    rest = StreamRuntime._lab_rest(rt, reply(generation=5))
    assert np.allclose(rest[:28, :2], 0.6)  # 80 px of 100 -> +0.6
