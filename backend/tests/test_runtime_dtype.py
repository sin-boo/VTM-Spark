import inspect

import torch

from inference_keypoint import build_keypoint_model, preferred_sd_vae_dtype


def test_live_dtype_is_float32() -> None:
    assert preferred_sd_vae_dtype(torch.device("cpu")) is torch.float32
    assert preferred_sd_vae_dtype(torch.device("cuda")) is torch.float32
    signature = inspect.signature(build_keypoint_model)
    assert signature.parameters["dtype"].default is torch.float32
