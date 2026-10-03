"""Setup-only models leave the GPU once the still is set up."""

from __future__ import annotations

import numpy as np

from backend import anime, hair


def test_detect_hair_drops_the_model_after_use(monkeypatch) -> None:
    def fake_detect(image):
        hair._hair3 = {"model": object()}
        return []

    monkeypatch.setattr(hair, "_detect_hair3", fake_detect)
    assert hair.detect_hair(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    assert hair._hair3 is None


def test_detect_hair_drops_the_model_when_it_fails(monkeypatch) -> None:
    def boom(image):
        hair._hair3 = {"model": object()}
        raise RuntimeError("out of memory")

    monkeypatch.setattr(hair, "_detect_hair3", boom)
    assert hair.detect_hair(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    assert hair._hair3 is None


def test_fit_mesh_drops_the_face_models_and_loads_them_again(monkeypatch) -> None:
    made = []

    class FakeMesh:
        def __init__(self) -> None:
            made.append(self)

        def detect(self, bgr):
            return np.zeros((28, 3), dtype=np.float32), np.zeros(5, dtype=np.float32)

    monkeypatch.setattr(anime, "AnimeFaceMesh", FakeMesh)
    monkeypatch.setattr(anime, "_detector", None)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    anime.fit_mesh(image)
    assert anime._detector is None
    anime.fit_mesh(image)
    assert anime._detector is None
    assert len(made) == 2
