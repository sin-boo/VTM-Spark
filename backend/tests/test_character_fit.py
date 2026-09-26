from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.character_fit import (
    SKELETON_LABELS,
    build_fit_view,
    read_character_fit,
    replace_pack_keypoints,
    update_character_fit,
)
from backend.character_pack import read_character_pack, write_character_pack
from backend.engine import neutral_keypoints
from backend.travel_box import default_travel_box


def _tiny_pack_payload() -> dict:
    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (40, 80, 60)
    return {
        "name": "Fit",
        "preview_rgb": preview,
        "keypoints": neutral_keypoints(),
        "ref_latent": np.zeros((1, 4, 8, 8), dtype=np.float16),
        "ref_face_latent": np.zeros((1, 4, 4, 4), dtype=np.float16),
        "image_size": 768,
        "skip_crop": True,
        "source_name": "fit.png",
    }


def test_build_fit_view_puts_skeleton_and_body_in_pixels() -> None:
    kps = neutral_keypoints()
    width, height = 200, 160
    view = build_fit_view(
        width=width,
        height=height,
        keypoints=kps,
        hair_norm=[],
        box=default_travel_box(),
    )
    assert view["width"] == width
    assert view["height"] == height
    assert len(view["skeleton"]) == len(SKELETON_LABELS)
    for joint in view["skeleton"]:
        assert 0.0 <= joint["x"] <= width
        assert 0.0 <= joint["y"] <= height
        assert joint["id"] in SKELETON_LABELS
    body = view["boxes"]["body_tight"]
    assert body is not None
    assert len(body) == 4
    assert body[0] < body[2]
    assert body[1] < body[3]
    assert 0.0 <= body[0] <= width
    assert 0.0 <= body[3] <= height


def test_update_character_fit_roundtrip(tmp_path: Path) -> None:
    update_character_fit("hero", {"hair": [{"class": "hair_middle", "polygon": [[0, 0], [1, 0], [0, 1]]}]}, dest_dir=tmp_path)
    update_character_fit("hero", {"skeleton": [{"id": 31, "x": 0.1, "y": 0.2}]}, dest_dir=tmp_path)
    saved = read_character_fit("hero", dest_dir=tmp_path)
    assert saved["hair"][0]["class"] == "hair_middle"
    assert saved["skeleton"][0]["id"] == 31
    assert saved["skeleton"][0]["x"] == 0.1


def test_replace_pack_keypoints_swaps_npy(tmp_path: Path) -> None:
    dest = tmp_path / "fit.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    edited = neutral_keypoints().copy()
    edited[31, 0] = 0.42
    edited[31, 1] = -0.17
    replace_pack_keypoints(dest, edited)
    pack = read_character_pack(dest)
    np.testing.assert_allclose(pack.keypoints[31, :2], (0.42, -0.17), atol=1e-5)
    assert pack.keypoints.shape == (37, 4)
