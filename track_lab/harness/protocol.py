"""Harness packet and command schema.

A consumer (later: VTM Spark) talks to Track Lab over this protocol instead
of embedding the tracker. Frames carry the character-space overlay mesh.
Commands mutate the same settings the lab UI already exposes.
"""

from __future__ import annotations

from typing import Any

from .points import (
    BODY_END,
    BODY_START,
    FACE_COUNT,
    KEYPOINT_NAMES,
    KEYPOINT_REFS,
    LEFT_EYE_SLOTS,
    LEFT_IRIS,
    NUM_KEYPOINTS,
    RIGHT_EYE_SLOTS,
    RIGHT_IRIS,
)

PROTOCOL = "track_lab.harness.v1"
SCHEMA = "KEYPOINT_SCHEMA"

# Frames and status carry ``session`` (one id per tracker process; generation
# restarts with it) and ``seq`` (bumped each time the live pose changes). In
# one session a lower seq is an older pose. Packets without them are unordered.

# Slot 28 / IRIS.L sits in EYE.L (11-13). Person-left blink drives that eye.
# The old packet name right_iris is in Point.legacy only.

# Lid amount (0 open, 1 shut) at which an eye's pupil is dropped from the
# packet. The model draws a pupil wherever it gets one, so a shut eye must
# not carry an iris, not even the eye-centre fallback.
IRIS_HIDE_BLINK = 0.85

COMMANDS: dict[str, str] = {
    "ping": "Heartbeat. Returns ok.",
    "start": "Start live tracking. body: camera?, source? (camera|ifm), host?, port?",
    "stop": "Stop live tracking.",
    "track": "Fit the rest mesh on the still (same as Track).",
    "set_rest": "Install a packaged rest mesh on the still without detection. body: points (28 x [x, y, score]), iris, skeleton, hair? (hair is detected when omitted), point_offsets? (the character's saved nudges). Character pixels.",
    "set_source": "Load the character still. body: path (file the lab can read)",
    "reset": "Tear down the tracker and restore rest.",
    "set_camera": "Pick a webcam. body: index (must be in the cameras list)",
    "refresh_cameras": "Re-list capture devices and re-find the chosen camera by name. No-op while OSF runs.",
    "set_input": "Choose camera or iFacialMocap. body: source (camera|ifm)",
    "set_ifm": "iFacialMocap bind. body: host?, port?",
    "set_mirror": "Left/right rule. off = reflection (person-left on screen-left), on = anatomical copy. Swaps L/R pairs and negates X for every source; no recenter. body: on (bool)",
    "set_feel": "Live feel / overlay flags. body: response, smoothing, mouth, hair_pin, hair_width, gaze_gain, gaze_smooth, use_visemes, show_face, show_skeleton, show_hair, show_ids, max_yaw_left, max_yaw_right, max_roll_left, max_roll_right (max_yaw / max_roll set both sides), max_pitch_up, max_pitch_down, max_size, max_look_x, max_look_y, head_sway (iFacialMocap: 0 rotates the head in place, 1 swings it round the neck like a webcam sees, 2 double)",
    "set_travel": "Character limiters, fixed to the rest still. body (partial ok): enabled, left, right, up, down (head room), body_left, body_right, body_up, body_down (body room), yaw, roll, pitch_up, pitch_down, eye, size. Room 0..1.2 face heights, eye 0..1, yaw/roll 0..80, pitch_up 0..50, pitch_down 0..32, size 0..0.7 (grow / shrink from rest when you step toward or away from the camera). The whole character moves as one piece and stops at the first wall. Merges onto current; no-op when unchanged. Ack status includes full travel_box; feel caps follow.",
    "fit_travel": "Fit the limiters to the loaded still: head and body room from the free space to the picture's edges, turn / tilt centred on the pose the still is drawn in. body: from? (default = start from the built-in limits, for a character with none yet; otherwise look up / down, eye range, size and enabled stay as they are). Ack status includes full travel_box; feel caps follow.",
    "calibrate": "Hold and capture a shape. body: id (rest|smile|sad|A|I|U|E|O|...)",
    "reset_calibrate": "Clear captured viseme rest.",
    "apply_preset": "Apply an authored mouth or eye shape. body: id",
    "set_mouth": "Write a shape's own slots (lips, or eyes for eye_open / eye_closed). body: id, mouth",
    "move_key": "Slide a saved in-between along its pair. body: id, t (0–1). The shape stays.",
    "drop_key": "Remove a saved in-between. body: id. End shapes stay.",
    "set_mouth_point": "Toggle / remap an OSF lip landmark. body: id, on?, to?",
    "set_eye_point": "Toggle / remap an OSF eye landmark. body: id, on?, to?",
    "set_skeleton_point": "Nudge a rest skeleton joint. body: id, x, y",
    "set_rest_point": "Move one rest face point (0-27) or iris (28-29) on the still and drop its overlay nudge. body: id, x, y (character pixels).",
    "set_hair": "Replace rest hair polygons. body: hair: [{class, polygon}] in character pixels.",
    "set_point": "Nudge any overlay point. body: id, x, y (character pixels), seq? + session? (the frame the point was lined up on: the offset is measured on that frame's pose, else on the current one; another session's nudge is refused). Offset rides on live tracking.",
    "reset_points": "Clear overlay nudges. body: id? (omit = all).",
    "set_offsets": "Replace every overlay nudge, e.g. the ones a character saved. body: point_offsets: [{id, dx, dy}] (character pixels).",
    "generate": "Run the DiT once on the current overlay (still + points + hair). body: points?, hair?, skeleton?, iris?",
    "record": "Record live character movement for the benchmark. body: on (bool). Needs a reference still and live tracking.",
}

FEEL_KEYS = (
    "response",
    "smoothing",
    "mouth",
    "use_visemes",
    "show_face",
    "show_skeleton",
    "show_hair",
    "show_ids",
    "hair_pin",
    "hair_width",
    "max_yaw_left",
    "max_yaw_right",
    "max_roll_left",
    "max_roll_right",
    "max_pitch_up",
    "max_pitch_down",
    "max_size",
    "max_look_x",
    "max_look_y",
    "gaze_gain",
    "gaze_smooth",
    "head_sway",
)


def parse_command(raw: object) -> dict[str, Any]:
    """Normalize a client message into {id, op, body}."""
    if raw == "ping" or raw == b"ping":
        return {"id": None, "op": "ping", "body": {}}
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == "ping":
            return {"id": None, "op": "ping", "body": {}}
        raise ValueError("command must be a JSON object")
    if not isinstance(raw, dict):
        raise ValueError("command must be a JSON object")
    op = str(raw.get("op") or raw.get("type") or "").strip()
    if op in {"command", "cmd"}:
        op = str(raw.get("op") or "").strip()
    if not op:
        raise ValueError("missing op")
    if op not in COMMANDS:
        raise ValueError(f"unknown op '{op}'")
    body = raw.get("body")
    if body is None:
        body = {key: value for key, value in raw.items() if key not in {"type", "op", "id", "body"}}
    if not isinstance(body, dict):
        body = {}
    return {"id": raw.get("id"), "op": op, "body": body}


def ack(
    *,
    ident: object = None,
    ok: bool = True,
    error: str = "",
    status: dict[str, Any] | None = None,
    frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "ack",
        "protocol": PROTOCOL,
        "id": ident,
        "ok": bool(ok),
        "error": str(error or ""),
    }
    if status is not None:
        payload["status"] = status
    if frame is not None:
        payload["frame"] = frame
    return payload
