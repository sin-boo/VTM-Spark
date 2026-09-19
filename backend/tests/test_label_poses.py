from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.label_poses import (
    LabelPoseError,
    list_label_files,
    load_label_for_generate,
    resolve_label_file,
)


def _write_stack(folder: Path, name: str, *, pose_id: str = "020") -> Path:
    path = folder / name
    path.write_text(
        json.dumps(
            {
                "character_id": "00001",
                "pose_id": pose_id,
                "image_width": 200,
                "image_height": 100,
                "faces": [
                    {
                        "face_id": 0,
                        "keypoints": [
                            {
                                "id": 2,
                                "label": "face_outline",
                                "x": 100.0,
                                "y": 80.0,
                                "score": 0.9,
                                "visible": True,
                            },
                            {
                                "id": 15,
                                "label": "nose",
                                "x": 100.0,
                                "y": 40.0,
                                "score": 0.8,
                                "visible": True,
                            },
                        ],
                    }
                ],
                "iris": {
                    "faces": [
                        {
                            "face_id": 0,
                            "irises": {
                                "right_iris": {
                                    "x": 80.0,
                                    "y": 35.0,
                                    "score": 0.7,
                                    "visible": True,
                                },
                                "left_iris": {
                                    "x": 120.0,
                                    "y": 35.0,
                                    "score": 0.7,
                                    "visible": True,
                                },
                            },
                        }
                    ]
                },
                "upper_body_pose": {
                    "people": [
                        {
                            "joints": {
                                "neck": {
                                    "x": 100.0,
                                    "y": 70.0,
                                    "score": 0.9,
                                    "visible": True,
                                }
                            }
                        }
                    ]
                },
                "hair": {
                    "segments": [
                        {
                            "class": "hair_middle",
                            "polygon": [[90, 10], [110, 10], [110, 30], [90, 30]],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_list_label_files_names_stems(tmp_path: Path) -> None:
    _write_stack(tmp_path, "020_full_stack.json")
    _write_stack(tmp_path, "base_full_stack.json", pose_id="base")
    listed = list_label_files(dest_dir=tmp_path)
    assert [row["id"] for row in listed] == [
        "020_full_stack.json",
        "base_full_stack.json",
    ]
    assert listed[0]["name"] == "020"
    assert listed[1]["name"] == "base"


def test_list_label_files_missing_dir(tmp_path: Path) -> None:
    assert list_label_files(dest_dir=tmp_path / "missing") == []


def test_resolve_label_file_uses_basename_only(tmp_path: Path) -> None:
    dest = _write_stack(tmp_path, "020_full_stack.json")
    assert resolve_label_file("../020_full_stack.json", dest_dir=tmp_path) == dest.resolve()
    with pytest.raises(LabelPoseError):
        resolve_label_file("nope.json", dest_dir=tmp_path)


def test_load_label_converts_pixels_to_norm_crop(tmp_path: Path) -> None:
    _write_stack(tmp_path, "020_full_stack.json")
    packed = load_label_for_generate("020_full_stack.json", dest_dir=tmp_path, image_size=64)
    kps = packed["keypoints"]
    assert kps.shape == (37, 4)
    assert kps[2, 3] == 1.0
    assert kps[15, 3] == 1.0
    assert kps[28, 3] == 1.0
    assert kps[31, 3] == 1.0
    vis = kps[:, 3] >= 0.5
    assert float(np.abs(kps[vis, :2]).max()) <= 1.5
    assert packed["hair_maps"].shape[0] == 3
    assert float(packed["hair_maps"][0].max()) > 0
    assert packed["name"] == "020"
