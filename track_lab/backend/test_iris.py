from __future__ import annotations

import numpy as np

from backend.iris import (
    IrisHit,
    catchlight_pupil,
    eye_crop_box,
    from_look,
    map_crop_to_frame,
    map_into_eye,
    match_to_eyes,
    match_to_osf,
    merge_hits,
    osf_gaze_hits,
    pupil_in_eye,
    raw_debug,
    resolve_weights,
    rest_look_from_cam,
    retarget,
    track_still,
)
from harness.pack import pack_keypoints
from harness.protocol import LEFT_IRIS, RIGHT_IRIS


def _char() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[11] = [100.0, 100.0, 1.0]
    pts[12] = [120.0, 70.0, 1.0]
    pts[13] = [140.0, 100.0, 1.0]
    pts[17] = [200.0, 100.0, 1.0]
    pts[18] = [220.0, 70.0, 1.0]
    pts[19] = [240.0, 100.0, 1.0]
    return pts


def test_resolve_weights_finds_iris_pose() -> None:
    path = resolve_weights()
    assert path is not None
    assert path.name == "iris_pose.pt"
    assert path.is_file()
    assert "trackers" in path.parts


def test_map_into_eye_preserves_look_offset() -> None:
    src = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]])
    dest = np.array([[100.0, 100.0], [140.0, 100.0], [140.0, 120.0], [100.0, 120.0]])
    x, y = map_into_eye((20.0, 12.0), src, dest)
    assert abs(x - 120.0) < 1e-6
    assert abs(y - 104.0) < 1e-6


def test_retarget_uses_iris_pose_hits() -> None:
    char = _char()
    lms = np.zeros((48, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    lms[36:42, 0] = np.linspace(10, 30, 6)
    lms[36:42, 1] = 12.0
    lms[42:48, 0] = np.linspace(50, 70, 6)
    lms[42:48, 1] = 12.0
    cam = [
        {"side": "r", "x": 20.0, "y": 11.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 13.0, "score": 0.8, "visible": True},
    ]
    rows, method = retarget(char, cam_lms=lms, cam_iris=cam)
    assert method == "iris_pose"
    by_id = {int(row["id"]): row for row in rows}
    assert RIGHT_IRIS in by_id
    assert LEFT_IRIS in by_id
    # Person-left maps onto the image-left character eye (slots 11-13 / iris 28).
    assert float(by_id[RIGHT_IRIS]["x"]) < float(by_id[LEFT_IRIS]["x"])


def test_look_moves_pupils_inside_the_eye() -> None:
    char = _char()
    rest = from_look(char, 0.0, 0.0)
    left = from_look(char, 1.0, 0.0)
    rest_x = {int(row["id"]): float(row["x"]) for row in rest}
    left_x = {int(row["id"]): float(row["x"]) for row in left}
    assert left_x[RIGHT_IRIS] > rest_x[RIGHT_IRIS]


def test_look_up_moves_pupils_toward_the_lid() -> None:
    char = _char()
    rest = from_look(char, 0.0, 0.0)
    up = from_look(char, 0.0, -1.0)
    rest_y = {int(row["id"]): float(row["y"]) for row in rest}
    up_y = {int(row["id"]): float(row["y"]) for row in up}
    assert up_y[RIGHT_IRIS] < rest_y[RIGHT_IRIS]
    assert up_y[LEFT_IRIS] < rest_y[LEFT_IRIS]


def test_selfie_flips_gaze_x() -> None:
    char = _char()
    rest = from_look(char, 0.0, 0.0)
    right = from_look(char, 1.0, 0.0)
    flipped = from_look(char, 1.0, 0.0, selfie=True)
    rest_x = {int(row["id"]): float(row["x"]) for row in rest}
    right_x = {int(row["id"]): float(row["x"]) for row in right}
    flip_x = {int(row["id"]): float(row["x"]) for row in flipped}
    assert right_x[RIGHT_IRIS] > rest_x[RIGHT_IRIS]
    assert right_x[LEFT_IRIS] > rest_x[LEFT_IRIS]
    assert flip_x[RIGHT_IRIS] < rest_x[RIGHT_IRIS]
    assert flip_x[LEFT_IRIS] < rest_x[LEFT_IRIS]


def test_max_look_x_shortens_gaze() -> None:
    char = _char()
    rest = from_look(char, 0.0, 0.0)
    full = from_look(char, 1.0, 0.0)
    tight = from_look(char, 1.0, 0.0, max_look_x=0.5)
    rest_x = {int(row["id"]): float(row["x"]) for row in rest}
    full_x = {int(row["id"]): float(row["x"]) for row in full}
    tight_x = {int(row["id"]): float(row["x"]) for row in tight}
    assert full_x[RIGHT_IRIS] > rest_x[RIGHT_IRIS]
    assert tight_x[RIGHT_IRIS] > rest_x[RIGHT_IRIS]
    assert (tight_x[RIGHT_IRIS] - rest_x[RIGHT_IRIS]) < (
        full_x[RIGHT_IRIS] - rest_x[RIGHT_IRIS]
    ) - 1e-3


def test_left_wink_hides_image_left_iris() -> None:
    char = _char()
    rest = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    rows = from_look(char, 0.0, 0.0, blink={"l": 0.9, "r": 0.0}, rest_iris=rest)
    ids = {int(row["id"]) for row in rows}
    assert RIGHT_IRIS not in ids
    assert LEFT_IRIS in ids
    lms = _osf_lms()
    cam = [
        {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
    ]
    cam_rows, method = retarget(
        char, cam_lms=lms, cam_iris=cam, rest_iris=rest, blink={"l": 0.9, "r": 0.0}
    )
    assert method == "iris_pose"
    cam_ids = {int(row["id"]) for row in cam_rows}
    assert RIGHT_IRIS not in cam_ids
    assert LEFT_IRIS in cam_ids


def test_right_wink_hides_image_right_iris() -> None:
    char = _char()
    rest = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    rows = from_look(char, 0.0, 0.0, blink={"l": 0.0, "r": 0.9}, rest_iris=rest)
    ids = {int(row["id"]) for row in rows}
    assert LEFT_IRIS not in ids
    assert RIGHT_IRIS in ids
    lms = _osf_lms()
    cam = [
        {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
    ]
    cam_rows, method = retarget(
        char, cam_lms=lms, cam_iris=cam, rest_iris=rest, blink={"l": 0.0, "r": 0.9}
    )
    assert method == "iris_pose"
    cam_ids = {int(row["id"]) for row in cam_rows}
    assert LEFT_IRIS not in cam_ids
    assert RIGHT_IRIS in cam_ids


def test_pack_prefers_tracked_iris() -> None:
    face = []
    for i in range(28):
        face.append([100.0 + i, 200.0 + i, 1.0])
    iris = [{"id": 28, "x": 333.0, "y": 111.0, "score": 0.95, "visible": True}]
    k = pack_keypoints(face, [], iris)
    assert abs(float(k[28, 0]) - 333.0) < 1e-5
    assert float(k[28, 3]) >= 0.5
    # Left iris still falls back to the image-right eye mid.
    expected = (117.0 + 118.0 + 119.0) / 3.0
    assert abs(float(k[29, 0]) - expected) < 1e-3


def test_match_to_osf_picks_nearest_eye() -> None:
    lms = np.zeros((48, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    lms[36:42, 0] = np.linspace(10, 20, 6)
    lms[36:42, 1] = 10.0
    lms[42:48, 0] = np.linspace(80, 90, 6)
    lms[42:48, 1] = 10.0
    dets = [
        {"cx": 15.0, "cy": 10.0, "pupil": (16.0, 9.0), "score": 0.9, "visible": True},
        {"cx": 85.0, "cy": 10.0, "pupil": (86.0, 11.0), "score": 0.8, "visible": True},
    ]
    right, left = match_to_osf(lms, dets)
    assert right.visible and abs(right.x - 16.0) < 1e-6
    assert left.visible and abs(left.x - 86.0) < 1e-6


def test_pupil_beside_the_eye_is_rejected() -> None:
    lms = np.zeros((48, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    lms[36:42, 0] = np.linspace(10, 20, 6)
    lms[36:42, 1] = 10.0
    lms[42:48, 0] = np.linspace(80, 90, 6)
    lms[42:48, 1] = 10.0
    assert pupil_in_eye((16.0, 9.0), lms, tuple(range(36, 42)))
    assert pupil_in_eye((15.0, 8.5), lms, tuple(range(36, 42)))
    assert not pupil_in_eye((2.0, 10.0), lms, tuple(range(36, 42)))
    dets = [
        {"cx": 15.0, "cy": 10.0, "pupil": (2.0, 10.0), "score": 0.9, "visible": True},
        {"cx": 85.0, "cy": 10.0, "pupil": (86.0, 9.0), "score": 0.8, "visible": True},
    ]
    right, left = match_to_osf(lms, dets)
    assert not right.visible
    assert left.visible and abs(left.x - 86.0) < 1e-6


def test_track_still_uses_iris_pose_on_the_mesh() -> None:
    char = _char()
    dets = [
        {
            "cx": 120.0,
            "cy": 100.0,
            "pupil": (118.0, 108.0),
            "score": 0.9,
            "visible": True,
        },
        {
            "cx": 220.0,
            "cy": 100.0,
            "pupil": (222.0, 108.0),
            "score": 0.8,
            "visible": True,
        },
    ]
    right, left = match_to_eyes(char, dets, (11, 12, 13), (17, 18, 19))
    assert right.visible and abs(right.x - 118.0) < 1e-6
    assert left.visible and abs(left.x - 222.0) < 1e-6


def test_still_iris_keeps_model_pupil(monkeypatch) -> None:
    char = _char()
    monkeypatch.setattr(
        "backend.iris.detect",
        lambda _image: [
            {
                "cx": 120.0,
                "cy": 100.0,
                "pupil": (118.0, 108.0),
                "score": 0.9,
                "visible": True,
            },
            {
                "cx": 220.0,
                "cy": 100.0,
                "pupil": (222.0, 108.0),
                "score": 0.8,
                "visible": True,
            },
        ],
    )
    rows, method = track_still(np.zeros((8, 8, 3), dtype=np.uint8), char)
    assert method == "iris_pose"
    by_id = {int(row["id"]): row for row in rows}
    assert abs(float(by_id[RIGHT_IRIS]["x"]) - 118.0) < 1e-6
    assert abs(float(by_id[RIGHT_IRIS]["y"]) - 108.0) < 1e-6
    assert abs(float(by_id[LEFT_IRIS]["x"]) - 222.0) < 1e-6
    assert abs(float(by_id[LEFT_IRIS]["y"]) - 108.0) < 1e-6


def test_still_iris_does_not_sit_on_the_lid(monkeypatch) -> None:
    char = _char()
    monkeypatch.setattr(
        "backend.iris.detect",
        lambda _image: [
            {
                "cx": 120.0,
                "cy": 70.0,
                "pupil": (120.0, 70.0),
                "score": 0.9,
                "visible": True,
            },
            {
                "cx": 220.0,
                "cy": 70.0,
                "pupil": (220.0, 70.0),
                "score": 0.8,
                "visible": True,
            },
        ],
    )
    rows, _method = track_still(np.zeros((8, 8, 3), dtype=np.uint8), char)
    by_id = {int(row["id"]): row for row in rows}
    assert float(by_id[RIGHT_IRIS]["y"]) > float(char[12, 1])
    assert float(by_id[LEFT_IRIS]["y"]) > float(char[18, 1])
    assert abs(float(by_id[RIGHT_IRIS]["x"]) - 120.0) < 15
    assert abs(float(by_id[LEFT_IRIS]["x"]) - 220.0) < 15


def test_closed_eye_keeps_the_box_and_drops_the_dot(monkeypatch) -> None:
    char = _char()
    monkeypatch.setattr(
        "backend.iris.detect",
        lambda _image: [
            {
                "cx": 120.0,
                "cy": 100.0,
                "bbox": (100.0, 80.0, 140.0, 110.0),
                "pupil": None,
                "score": 0.8,
                "visible": False,
            },
            {
                "cx": 220.0,
                "cy": 100.0,
                "bbox": (200.0, 80.0, 240.0, 110.0),
                "pupil": (222.0, 108.0),
                "score": 0.9,
                "visible": True,
            },
        ],
    )
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(image, (120, 97), 5, (255, 255, 255), -1)
    rows, method = track_still(image, char)
    assert method == "iris_pose"
    by_id = {int(row["id"]): row for row in rows}
    assert by_id[RIGHT_IRIS]["visible"] is False
    assert by_id[RIGHT_IRIS]["box"] == [100.0, 80.0, 140.0, 110.0]
    assert by_id[LEFT_IRIS]["visible"] is True
    assert abs(float(by_id[LEFT_IRIS]["x"]) - 222.0) < 1e-6


def test_track_still_omits_a_hidden_pupil(monkeypatch) -> None:
    char = _char()
    monkeypatch.setattr(
        "backend.iris.detect",
        lambda _image: [
            {
                "cx": 120.0,
                "cy": 100.0,
                "pupil": (118.0, 108.0),
                "score": 0.9,
                "visible": True,
            },
            {
                "cx": 220.0,
                "cy": 100.0,
                "pupil": None,
                "score": 0.4,
                "visible": False,
            },
        ],
    )
    rows, method = track_still(np.zeros((8, 8, 3), dtype=np.uint8), char)
    assert method == "iris_pose"
    assert {int(row["id"]) for row in rows} == {RIGHT_IRIS}


def test_catchlight_marks_the_open_pupil_and_skips_a_shut_lid() -> None:
    char = _char()
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(image, (120, 97), 5, (255, 255, 255), -1)
    spot = catchlight_pupil(image, char, (11, 12, 13))
    assert spot is not None
    assert abs(spot[0] - 120.0) < 4.0
    shut = char.copy()
    shut[12, 1] = 96.0
    assert catchlight_pupil(image, shut, (11, 12, 13)) is None


def test_track_still_falls_back_to_eye_mid(monkeypatch) -> None:
    char = _char()
    monkeypatch.setattr("backend.iris.detect", lambda _image: [])
    rows, method = track_still(np.zeros((8, 8, 3), dtype=np.uint8), char)
    assert method == "eye_mid"
    by_id = {int(row["id"]): row for row in rows}
    assert abs(float(by_id[RIGHT_IRIS]["x"]) - 120.0) < 1e-6
    assert abs(float(by_id[LEFT_IRIS]["x"]) - 220.0) < 1e-6
    assert float(by_id[RIGHT_IRIS]["y"]) > float(char[12, 1]) + 15


def test_retarget_falls_back_to_eye_mid() -> None:
    char = _char()
    rows, method = retarget(char)
    assert method == "eye_mid"
    assert {int(row["id"]) for row in rows} == {RIGHT_IRIS, LEFT_IRIS}


def test_retarget_keeps_still_iris_at_rest_gaze() -> None:
    char = _char()
    rest = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    lms = np.zeros((48, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    lms[36:42, 0] = np.linspace(10, 30, 6)
    lms[36:42, 1] = 12.0
    lms[42:48, 0] = np.linspace(50, 70, 6)
    lms[42:48, 1] = 12.0
    cam = [
        {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
    ]
    rows, method = retarget(char, cam_lms=lms, cam_iris=cam, rest_iris=rest)
    assert method == "iris_pose"
    by_id = {int(row["id"]): row for row in rows}
    assert abs(float(by_id[RIGHT_IRIS]["x"]) - 118.0) < 4
    assert abs(float(by_id[RIGHT_IRIS]["y"]) - 108.0) < 4
    assert abs(float(by_id[LEFT_IRIS]["x"]) - 222.0) < 4
    assert abs(float(by_id[LEFT_IRIS]["y"]) - 108.0) < 4


def _osf_lms() -> np.ndarray:
    lms = np.zeros((68, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    lms[36:42, 0] = np.linspace(10, 30, 6)
    lms[36:42, 1] = 12.0
    lms[42:48, 0] = np.linspace(50, 70, 6)
    lms[42:48, 1] = 12.0
    return lms


def test_crop_maps_pupil_back_to_frame() -> None:
    x, y = map_crop_to_frame((80.0, 40.0), 2.0, (100.0, 50.0))
    assert abs(x - 140.0) < 1e-6
    assert abs(y - 70.0) < 1e-6


def test_eye_crop_box_pads_osf_eye() -> None:
    lms = _osf_lms()
    box = eye_crop_box(lms, tuple(range(36, 42)), (640, 480))
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 10
    assert x1 >= 30
    assert y0 <= 12
    assert y1 >= 12
    assert (x1 - x0) > 20


def test_merge_prefers_yolo_then_osf_gaze() -> None:
    yolo_r = IrisHit(x=16.0, y=9.0, score=0.9, visible=True, side="r", method="iris_pose")
    yolo_l = IrisHit(side="l", method="iris_pose")
    osf_r = IrisHit(x=18.0, y=10.0, score=0.7, visible=True, side="r", method="osf_gaze")
    osf_l = IrisHit(x=86.0, y=11.0, score=0.8, visible=True, side="l", method="osf_gaze")
    right, left, method = merge_hits((yolo_r, yolo_l), (osf_r, osf_l))
    assert method == "mixed"
    assert right.visible and abs(right.x - 16.0) < 1e-6
    assert right.method == "iris_pose"
    assert left.visible and abs(left.x - 86.0) < 1e-6
    assert left.method == "osf_gaze"


def test_osf_gaze_hits_from_swapped_lms() -> None:
    lms = np.zeros((68, 3), dtype=np.float32)
    lms[66] = [22.0, 14.0, 0.9]
    lms[67] = [62.0, 15.0, 0.8]
    right, left = osf_gaze_hits(lms)
    assert right.visible and abs(right.x - 22.0) < 1e-6
    assert left.visible and abs(left.x - 62.0) < 1e-6
    dummy = np.zeros((68, 3), dtype=np.float32)
    blank_r, blank_l = osf_gaze_hits(dummy)
    assert not blank_r.visible and not blank_l.visible


def test_raw_debug_keeps_camera_iris() -> None:
    packed = raw_debug(
        [
            {"side": "r", "x": 22.4567, "y": 14.1, "score": 0.91, "visible": True, "method": "iris_pose"},
            {"side": "l", "x": 62.0, "y": 15.0, "score": 0.8, "visible": True, "method": "osf_gaze"},
        ],
        {"x": 0.12, "y": -0.54},
    )
    hits = packed["iris_cam"]
    assert isinstance(hits, list) and len(hits) == 2
    assert hits[0] == {
        "side": "r",
        "x": 22.457,
        "y": 14.1,
        "score": 0.91,
        "visible": True,
        "method": "iris_pose",
    }
    assert packed["look"] == {"x": 0.12, "y": -0.54}
    empty = raw_debug(None, None)
    assert empty["iris_cam"] == []
    assert empty["look"] is None


def test_rest_look_delta_moves_from_still_iris() -> None:
    char = _char()
    rest = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    lms = _osf_lms()
    rest_look = rest_look_from_cam(
        lms,
        IrisHit(x=20.0, y=12.0, visible=True, side="r", method="iris_pose"),
        IrisHit(x=60.0, y=12.0, visible=True, side="l", method="iris_pose"),
    )
    rest_rows, rest_method = retarget(
        char,
        cam_lms=lms,
        cam_iris=[
            {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
            {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
        ],
        rest_iris=rest,
        rest_look=rest_look,
    )
    assert rest_method == "iris_pose"
    rest_by = {int(row["id"]): row for row in rest_rows}
    assert abs(float(rest_by[RIGHT_IRIS]["x"]) - 118.0) < 1.5
    live_rows, _method = retarget(
        char,
        cam_lms=lms,
        cam_iris=[
            {"side": "r", "x": 26.0, "y": 12.0, "score": 0.9, "visible": True},
            {"side": "l", "x": 66.0, "y": 12.0, "score": 0.8, "visible": True},
        ],
        rest_iris=rest,
        rest_look=rest_look,
    )
    live_by = {int(row["id"]): row for row in live_rows}
    assert float(live_by[RIGHT_IRIS]["x"]) > float(rest_by[RIGHT_IRIS]["x"]) + 4
    assert float(live_by[LEFT_IRIS]["x"]) > float(rest_by[LEFT_IRIS]["x"]) + 4


def test_blink_half_shows_iris_shut_hides() -> None:
    char = _char()
    lms = _osf_lms()
    cam = [
        {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
    ]
    rest = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    half, _method = retarget(char, cam_lms=lms, cam_iris=cam, rest_iris=rest, blink={"r": 0.5, "l": 0.5})
    assert {int(row["id"]) for row in half} == {RIGHT_IRIS, LEFT_IRIS}
    shut, method = retarget(char, cam_lms=lms, cam_iris=cam, rest_iris=rest, blink={"r": 0.9, "l": 0.9})
    assert method == "none"
    assert shut == []
    look_half = from_look(char, 0.4, 0.0, blink={"r": 0.5, "l": 0.5}, rest_iris=rest)
    assert {int(row["id"]) for row in look_half} == {RIGHT_IRIS, LEFT_IRIS}
    look_shut = from_look(char, 0.4, 0.0, blink={"r": 0.9, "l": 0.9}, rest_iris=rest)
    assert look_shut == []


def test_rest_iris_follows_a_yawed_eye() -> None:
    """Still pupils ride the posed sockets. A look-with-the-head is not camera-forward."""
    rest = _char()
    rest_iris = [
        {"id": RIGHT_IRIS, "x": 118.0, "y": 108.0, "score": 0.9, "visible": True},
        {"id": LEFT_IRIS, "x": 222.0, "y": 108.0, "score": 0.8, "visible": True},
    ]
    posed = rest.copy()
    posed[11:14, 0] += 22.0
    posed[17:20, 0] += 10.0
    stuck, _method = retarget(posed, rest_iris=rest_iris, look={"x": 0.0, "y": 0.0})
    stuck_x = {int(row["id"]): float(row["x"]) for row in stuck}
    assert abs(stuck_x[RIGHT_IRIS] - 118.0) < 2
    rows, method = retarget(
        posed,
        rest_iris=rest_iris,
        rest_pts=rest,
        look={"x": 0.0, "y": 0.0},
    )
    assert method == "look"
    by_id = {int(row["id"]): row for row in rows}
    assert float(by_id[RIGHT_IRIS]["x"]) > 130.0
    assert float(by_id[LEFT_IRIS]["x"]) > 225.0
    lms = _osf_lms()
    cam = [
        {"side": "r", "x": 20.0, "y": 12.0, "score": 0.9, "visible": True},
        {"side": "l", "x": 60.0, "y": 12.0, "score": 0.8, "visible": True},
    ]
    cam_rows, cam_method = retarget(
        posed,
        cam_lms=lms,
        cam_iris=cam,
        rest_iris=rest_iris,
        rest_pts=rest,
    )
    assert cam_method == "iris_pose"
    cam_by = {int(row["id"]): row for row in cam_rows}
    assert float(cam_by[RIGHT_IRIS]["x"]) > 130.0
    assert abs(float(cam_by[RIGHT_IRIS]["x"]) - 118.0) > 10


def test_feel_dump_includes_gaze_keys() -> None:
    from backend.feel import DEFAULTS
    from harness.protocol import FEEL_KEYS

    assert "gaze_gain" in DEFAULTS
    assert "gaze_smooth" in DEFAULTS
    assert DEFAULTS["gaze_gain"] == 1.0
    assert abs(float(DEFAULTS["gaze_smooth"]) - 0.28) < 1e-9
    assert "gaze_gain" in FEEL_KEYS
    assert "gaze_smooth" in FEEL_KEYS
    # Left/right is Mirror, not a feel knob.
    for key in ("invert_look", "invert_yaw", "invert_pitch"):
        assert key not in DEFAULTS
        assert key not in FEEL_KEYS
    assert "max_look_x" in DEFAULTS
    assert "max_look_y" in FEEL_KEYS


def test_selfie_flips_head_turn() -> None:
    from backend.feel import feel
    from backend.rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0] = [10.0, 32.0, 1.0]
    rest[4] = [90.0, 32.0, 1.0]
    rest[11] = [26.0, 38.0, 1.0]
    rest[12] = [30.0, 28.0, 1.0]
    rest[13] = [34.0, 38.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[17] = [66.0, 38.0, 1.0]
    rest[18] = [70.0, 28.0, 1.0]
    rest[19] = [74.0, 38.0, 1.0]
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    head = {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}
    prev = feel.payload()

    def drive(selfie: bool) -> np.ndarray:
        rig = FaceRig()
        rig.selfie = selfie
        rig.apply(rest, rest, head0, origin)
        out = rest
        for _ in range(12):
            out = rig.apply(rest, rest, head, origin)
        assert out is not None
        return out

    try:
        feel.update({"smoothing": 0.0})
        natural = drive(False)
        flipped = drive(True)
    finally:
        feel.update(prev)
    near = abs(float(natural[12, 1] - natural[11, 1]))
    far = abs(float(natural[18, 1] - natural[17, 1]))
    flip_near = abs(float(flipped[12, 1] - flipped[11, 1]))
    flip_far = abs(float(flipped[18, 1] - flipped[17, 1]))
    assert near > far
    assert flip_far > flip_near
