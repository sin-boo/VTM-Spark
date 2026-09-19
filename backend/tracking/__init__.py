"""Per-model live trackers, merged into KEYPOINT_SCHEMA for the generator.

Layout
------
tracking/
  merger.py     combine face + iris + body + hair → (37,4) + hair
  face/         OpenSeeFace landmarks → Label28 (incl. mouth map)
  iris/         iris_pose.pt + OpenSeeFace gaze
  hair/         animeseg_hair3 / hair_seg
  body/         MediaPipe / YOLO / face-synth upper body
"""

from __future__ import annotations

from ..paths import ensure_import_paths

ensure_import_paths()

from .body import BodyTrack, BodyTracker
from .face import FaceTrack, FaceTracker
from .hair import HairTrack, HairTracker
from .iris import IrisTrack, IrisTracker
from .merger import MergedTrack, merge_tracks

__all__ = [
    "BodyTrack",
    "BodyTracker",
    "FaceTrack",
    "FaceTracker",
    "HairTrack",
    "HairTracker",
    "IrisTrack",
    "IrisTracker",
    "MergedTrack",
    "merge_tracks",
]
