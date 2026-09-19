"""Named overlay slots for Track Lab / desk tracking.

Slot ids 0–36 stay KEYPOINT_SCHEMA. The letter in a ref (JAW.R, EYE.L) is
the training name and is NOT a consistent side: jaw / nose / mouth / body
use the character's own side (R = screen-left on a facing still) while
brow / eye / iris use image side (L = screen-left). Never drive by letter.

Drive by ``screen``: the side of the still the slot sits on. The rule for
which camera side feeds a slot lives in backend/sides.py — canonical is
image-left camera → screen-left slot, and Mirror OFF (selfie) swaps it.

``osf`` is the canonical camera source (image-left landmarks first).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    id: int
    ref: str
    name: str
    group: str
    role: str
    side: str
    osf: tuple[int, ...] = ()
    legacy: str = ""
    screen: str = "c"


# Jaw follows OSF 0→16 (image-left → image-right).
# Brows / eyes follow FACE_SOURCES in backend/retarget.py.
# Mouth follows mouth_bits DEFAULT_TO (OSF inner lips).
POINTS: tuple[Point, ...] = (
    Point(0, "JAW.R", "jaw_r", "jaw", "side", "r", (0,), "face_0", "l"),
    Point(1, "JAW.R.MID", "jaw_r_mid", "jaw", "mid", "r", (4,), "face_1", "l"),
    Point(2, "CHIN", "chin", "jaw", "chin", "c", (8,), "face_2"),
    Point(3, "JAW.L.MID", "jaw_l_mid", "jaw", "mid", "l", (12,), "face_3", "r"),
    Point(4, "JAW.L", "jaw_l", "jaw", "side", "l", (16,), "face_4", "r"),
    Point(5, "BROW.L.OUT", "brow_l_outer", "brow", "outer", "l", (17, 18), "face_5", "l"),
    Point(6, "BROW.L", "brow_l", "brow", "mid", "l", (18, 19, 20), "face_6", "l"),
    Point(7, "BROW.L.IN", "brow_l_inner", "brow", "inner", "l", (20, 21), "face_7", "l"),
    Point(8, "BROW.R.IN", "brow_r_inner", "brow", "inner", "r", (22, 23), "face_8", "r"),
    Point(9, "BROW.R", "brow_r", "brow", "mid", "r", (23, 24, 25), "face_9", "r"),
    Point(10, "BROW.R.OUT", "brow_r_outer", "brow", "outer", "r", (25, 26), "face_10", "r"),
    # Slots sweep left→right (training flip pairs 11↔19, 13↔17): 11 is the
    # OUTER corner and 13 the INNER one. The refs are legacy names kept
    # for saved data; trust ``role``.
    Point(11, "EYE.L.IN", "eye_l_inner", "eye", "outer", "l", (36,), "face_11", "l"),
    Point(12, "EYE.L.LID", "eye_l_lid", "eye", "lid", "l", (37, 38, 41, 40), "face_12", "l"),
    Point(13, "EYE.L.OUT", "eye_l_outer", "eye", "inner", "l", (39,), "face_13", "l"),
    Point(14, "NOSE.R", "nose_r", "nose", "ala", "r", (31,), "face_14", "l"),
    Point(15, "NOSE", "nose", "nose", "tip", "c", (30,), "face_15"),
    Point(16, "NOSE.L", "nose_l", "nose", "ala", "l", (35,), "face_16", "r"),
    Point(17, "EYE.R.IN", "eye_r_inner", "eye", "inner", "r", (42,), "face_17", "r"),
    Point(18, "EYE.R.LID", "eye_r_lid", "eye", "lid", "r", (43, 44, 47, 46), "face_18", "r"),
    Point(19, "EYE.R.OUT", "eye_r_outer", "eye", "outer", "r", (45,), "face_19", "r"),
    Point(20, "MOUTH.U.R", "mouth_upper_r", "mouth", "upper", "r", (59,), "face_20", "l"),
    Point(21, "MOUTH.U", "mouth_upper", "mouth", "upper", "c", (60,), "face_21"),
    Point(22, "MOUTH.U.L", "mouth_upper_l", "mouth", "upper", "l", (61,), "face_22", "r"),
    Point(23, "MOUTH.R", "mouth_corner_r", "mouth", "corner", "r", (58,), "face_23", "l"),
    Point(24, "MOUTH.D.R", "mouth_lower_r", "mouth", "lower", "r", (65,), "face_24", "l"),
    Point(25, "MOUTH.D", "mouth_lower", "mouth", "lower", "c", (64,), "face_25"),
    Point(26, "MOUTH.L", "mouth_corner_l", "mouth", "corner", "l", (62,), "face_26", "r"),
    Point(27, "MOUTH.D.L", "mouth_lower_l", "mouth", "lower", "l", (63,), "face_27", "r"),
    Point(28, "IRIS.L", "iris_l", "iris", "pupil", "l", (), "right_iris", "l"),
    Point(29, "IRIS.R", "iris_r", "iris", "pupil", "r", (), "left_iris", "r"),
    Point(30, "BODY.NOSE", "body_nose", "body", "nose", "c", (), "nose"),
    Point(31, "NECK", "neck", "body", "neck", "c", (), "neck"),
    Point(32, "SHO.R", "shoulder_r", "body", "shoulder", "r", (), "right_shoulder", "l"),
    Point(33, "ELB.R", "elbow_r", "body", "elbow", "r", (), "right_elbow", "l"),
    Point(34, "SHO.L", "shoulder_l", "body", "shoulder", "l", (), "left_shoulder", "r"),
    Point(35, "ELB.L", "elbow_l", "body", "elbow", "l", (), "left_elbow", "r"),
    Point(36, "CHEST", "chest", "body", "chest", "c", (), "chest"),
)

CAM_POINTS: tuple[Point, ...] = (
    Point(36, "CAM.EYE.R.OUT", "osf_eye_r_outer", "osf_eye", "outer", "r"),
    Point(37, "CAM.EYE.R.UP.O", "osf_eye_r_up_outer", "osf_eye", "upper", "r"),
    Point(38, "CAM.EYE.R.UP.I", "osf_eye_r_up_inner", "osf_eye", "upper", "r"),
    Point(39, "CAM.EYE.R.IN", "osf_eye_r_inner", "osf_eye", "inner", "r"),
    Point(40, "CAM.EYE.R.DN.I", "osf_eye_r_dn_inner", "osf_eye", "lower", "r"),
    Point(41, "CAM.EYE.R.DN.O", "osf_eye_r_dn_outer", "osf_eye", "lower", "r"),
    Point(42, "CAM.EYE.L.IN", "osf_eye_l_inner", "osf_eye", "inner", "l"),
    Point(43, "CAM.EYE.L.UP.I", "osf_eye_l_up_inner", "osf_eye", "upper", "l"),
    Point(44, "CAM.EYE.L.UP.O", "osf_eye_l_up_outer", "osf_eye", "upper", "l"),
    Point(45, "CAM.EYE.L.OUT", "osf_eye_l_outer", "osf_eye", "outer", "l"),
    Point(46, "CAM.EYE.L.DN.O", "osf_eye_l_dn_outer", "osf_eye", "lower", "l"),
    Point(47, "CAM.EYE.L.DN.I", "osf_eye_l_dn_inner", "osf_eye", "lower", "l"),
    Point(48, "CAM.LIP.OUT.R", "osf_lip_out_r", "osf_mouth", "outer", "r"),
    Point(49, "CAM.LIP.OUT.UR", "osf_lip_out_ur", "osf_mouth", "outer", "r"),
    Point(50, "CAM.LIP.OUT.U", "osf_lip_out_u", "osf_mouth", "outer", "c"),
    Point(51, "CAM.LIP.OUT.UL", "osf_lip_out_ul", "osf_mouth", "outer", "l"),
    Point(52, "CAM.LIP.OUT.L", "osf_lip_out_l", "osf_mouth", "outer", "l"),
    Point(53, "CAM.LIP.OUT.LL", "osf_lip_out_ll", "osf_mouth", "outer", "l"),
    Point(54, "CAM.LIP.OUT.D", "osf_lip_out_d", "osf_mouth", "outer", "c"),
    Point(55, "CAM.LIP.OUT.LR", "osf_lip_out_lr", "osf_mouth", "outer", "r"),
    Point(56, "CAM.LIP.OUT.DR", "osf_lip_out_dr", "osf_mouth", "outer", "r"),
    Point(57, "CAM.LIP.OUT.DL", "osf_lip_out_dl", "osf_mouth", "outer", "l"),
    Point(58, "CAM.LIP.R", "osf_lip_corner_r", "osf_mouth", "corner", "r"),
    Point(59, "CAM.LIP.U.R", "osf_lip_upper_r", "osf_mouth", "inner", "r"),
    Point(60, "CAM.LIP.U", "osf_lip_upper", "osf_mouth", "inner", "c"),
    Point(61, "CAM.LIP.U.L", "osf_lip_upper_l", "osf_mouth", "inner", "l"),
    Point(62, "CAM.LIP.L", "osf_lip_corner_l", "osf_mouth", "corner", "l"),
    Point(63, "CAM.LIP.D.L", "osf_lip_lower_l", "osf_mouth", "inner", "l"),
    Point(64, "CAM.LIP.D", "osf_lip_lower", "osf_mouth", "inner", "c"),
    Point(65, "CAM.LIP.D.R", "osf_lip_lower_r", "osf_mouth", "inner", "r"),
    Point(66, "CAM.LID.R", "osf_lid_r", "osf_eye", "lid_mid", "r"),
    Point(67, "CAM.LID.L", "osf_lid_l", "osf_eye", "lid_mid", "l"),
)

HAIR_PARTS: tuple[str, ...] = ("hair_left", "hair_middle", "hair_right")

NUM_KEYPOINTS = 37
FACE_COUNT = 28
IRIS_L = 28
IRIS_R = 29
BODY_START = 30
BODY_END = 36
BODY_NOSE = 30
EYE_L = (11, 12, 13)
EYE_R = (17, 18, 19)
BROW_L = (5, 6, 7)
BROW_R = (8, 9, 10)
MOUTH = tuple(range(20, 28))
JAW = (0, 1, 2, 3, 4)
NOSE = (14, 15, 16)

# Legacy slot names kept so iris / pack imports do not change.
RIGHT_IRIS = IRIS_L
LEFT_IRIS = IRIS_R
LEFT_EYE_SLOTS = EYE_L
RIGHT_EYE_SLOTS = EYE_R

BY_ID: dict[int, Point] = {p.id: p for p in POINTS}
BY_REF: dict[str, Point] = {p.ref: p for p in POINTS}
BY_NAME: dict[str, Point] = {p.name: p for p in POINTS}
CAM_BY_ID: dict[int, Point] = {p.id: p for p in CAM_POINTS}
KEYPOINT_NAMES: tuple[str, ...] = tuple(p.name for p in POINTS)
KEYPOINT_REFS: tuple[str, ...] = tuple(p.ref for p in POINTS)


def point_of(slot: int) -> Point | None:
    return BY_ID.get(int(slot))


def ref_of(slot: int) -> str:
    point = BY_ID.get(int(slot))
    return point.ref if point else str(slot)


def cam_ref_of(index: int) -> str:
    point = CAM_BY_ID.get(int(index))
    return point.ref if point else str(index)


def row_meta(slot: int) -> dict[str, str]:
    point = BY_ID.get(int(slot))
    if point is None:
        return {"name": f"slot_{slot}", "ref": str(slot), "legacy": ""}
    return {"name": point.name, "ref": point.ref, "legacy": point.legacy}
