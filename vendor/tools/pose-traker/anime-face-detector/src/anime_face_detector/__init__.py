"""Anime face detector package.

Pure PyTorch implementation using YOLOv8 and HRNetV2.
"""

import pathlib
import warnings

import torch

from .detector import LandmarkDetector

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def get_checkpoint_path(model_name: str) -> pathlib.Path:
    """Get checkpoint path for pre-trained models.

    Args:
        model_name: Model name ('hrnetv2')

    Returns:
        Path to checkpoint file
    """
    assert model_name == 'hrnetv2'
    file_name = f'mmpose_anime-face_{model_name}.pth'

    model_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / file_name
    if not model_path.exists():
        url = f'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/{file_name}'
        torch.hub.download_url_to_file(url, model_path.as_posix())

    return model_path


def create_detector(
    face_detector_checkpoint_path: str | pathlib.Path | None = None,
    landmark_checkpoint_path: str | pathlib.Path | None = None,
    device: str = 'cuda:0',
    box_scale_factor: float = 1.25,
) -> LandmarkDetector:
    """Create a landmark detector with face detection.

    Args:
        face_detector_checkpoint_path: Path to YOLOv8 face detector checkpoint (.pt file).
                                       If None, downloads default face_yolov8n.pt from HuggingFace.
        landmark_checkpoint_path: Path to HRNetV2 landmark detector checkpoint (.pth file).
                                  If None, downloads pre-trained model from GitHub.
        device: Device to run models on ('cuda:0' or 'cpu')
        box_scale_factor: Scale factor for detected face boxes (default: 1.25)

    Returns:
        LandmarkDetector instance

    Examples:
        # Use default models (auto-download)
        detector = create_detector()

        # Use custom YOLOv8 model
        detector = create_detector(
            face_detector_checkpoint_path='path/to/custom_yolov8.pt'
        )

        # Use custom HRNetV2 model
        detector = create_detector(
            landmark_checkpoint_path='path/to/custom_hrnetv2.pth'
        )

        # Run on CPU
        detector = create_detector(device='cpu')
    """
    # Auto-download face_yolov8n.pt if not provided
    if face_detector_checkpoint_path is None:
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                'huggingface_hub is required for default YOLOv8 model. '
                'Install it with: pip install huggingface-hub ultralytics'
            )
        # Download face_yolov8n from Hugging Face
        model_path = hf_hub_download('Bingsu/adetailer', 'face_yolov8n.pt')
        face_detector_checkpoint_path = pathlib.Path(model_path)

    # Auto-download HRNetV2 if not provided
    if landmark_checkpoint_path is None:
        landmark_checkpoint_path = get_checkpoint_path('hrnetv2')

    model = LandmarkDetector(
        landmark_detector_checkpoint_path=landmark_checkpoint_path,
        face_detector_checkpoint_path=face_detector_checkpoint_path,
        device=device,
        box_scale_factor=box_scale_factor,
    )
    return model


def get_config_path(model_name: str, *, deprecated_ok: bool = False) -> pathlib.Path:
    """Get the path to a built-in config file.

    Deprecated:
        This function is deprecated. Prefer importing configs from
        `anime_face_detector.configs` (or whichever new API you standardize on).

    Args:
        model_name: Name of the model config.
        deprecated_ok: Set True to acknowledge the deprecation and suppress the warning.

    Returns:
        Path to the config file.
    """
    if not deprecated_ok:
        warnings.warn(
            "get_config_path() is deprecated as part of the legacy api and will be removed in a future release. "
            "Pass deprecated_ok=True to acknowledge for now, and migrate to the new API.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    assert model_name in ['faster-rcnn', 'yolov3', 'hrnetv2']

    package_path = pathlib.Path(__file__).parent.resolve()
    if model_name in ['faster-rcnn', 'yolov3']:
        config_dir = package_path / 'configs' / 'mmdet'
    else:
        config_dir = package_path / 'configs' / 'mmpose'
    return config_dir / f'{model_name}.py'


__all__ = ['create_detector', 'LandmarkDetector']
