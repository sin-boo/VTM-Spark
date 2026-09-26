"""Harness packet and command schema.

A consumer (later: VTM Noble) talks to Track Lab over this protocol instead
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

# Slot 28 / IRIS.L sits in EYE.L (11-13). Person-left blink drives that eye.
# The old packet name right_iris is in Point.legacy only.

COMMANDS: dict[str, str] = {
    "ping": "Heartbeat. Returns ok.",
    "start": "Start live tracking. body: camera?, source? (camera|ifm), host?, port?",
    "stop": "Stop live tracking.",
    "track": "Fit the rest mesh on the still (same as Track).",
    "set_source": "Load the character still. body: path (file the lab can read)",
    "reset": "Tear down the tracker and restore rest.",
    "set_camera": "Pick a webcam. body: index",
    "set_input": "Choose camera or iFacialMocap. body: source (camera|ifm)",
    "set_ifm": "iFacialMocap bind. body: host?, port?",
    "set_mirror": "Left/right rule. off = reflection (person-left on screen-left), on = anatomical copy. Swaps L/R pairs and negates X for every source; no recenter. body: on (bool)",
    "set_feel": "Live feel / overlay flags. body: response, smoothing, mouth, hair_pin, hair_width, gaze_gain, gaze_smooth, use_visemes, show_face, show_skeleton, show_hair, show_ids, max_yaw_left, max_yaw_right, max_roll_left, max_roll_right (max_yaw / max_roll set both sides), max_pitch_up, max_pitch_down, max_size, max_look_x, max_look_y",
    "set_travel": "Character limiters, fixed to the rest still. body (partial ok): enabled, left, right, up, down (head room), body_left, body_right, body_up, body_down (body room), yaw, roll, pitch_up, pitch_down, eye, size. Room 0..1.2 face heights, eye 0..1, yaw/roll 0..80, pitch_up 0..50, pitch_down 0..32, size 0..0.7 (grow / shrink from rest when you step toward or away from the camera). The whole character moves as one piece and stops at the first wall. Merges onto current; no-op when unchanged. Ack status includes full travel_box; feel caps follow.",
    "calibrate": "Hold and capture a shape. body: id (rest|smile|sad|A|I|U|E|O|...)",
    "reset_calibrate": "Clear captured viseme rest.",
    "apply_preset": "Apply an authored mouth shape. body: id",
    "set_mouth": "Write mouth slots on a preset. body: id, mouth",
    "move_key": "Slide a saved in-between along its pair. body: id, t (0–1). The mouth shape stays.",
    "drop_key": "Remove a saved in-between. body: id. End shapes stay.",
    "set_mouth_point": "Toggle / remap an OSF lip landmark. body: id, on?, to?",
    "set_eye_point": "Toggle / remap an OSF eye landmark. body: id, on?, to?",
    "set_skeleton_point": "Nudge a rest skeleton joint. body: id, x, y",
    "set_hair": "Replace rest hair polygons. body: hair: [{class, polygon}] in character pixels.",
    "set_point": "Nudge any overlay point. body: id, x, y (character pixels). Offset rides on live tracking.",
    "reset_points": "Clear overlay nudges. body: id? (omit = all).",
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
