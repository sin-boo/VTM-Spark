"""Anime face detector and landmark detector.

Pure PyTorch implementation without mmdet/mmpose dependencies.
"""

from __future__ import annotations

import pathlib
import warnings

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .models.hrnetv2 import HRNetV2
from .models.inference import inference_batch


class LandmarkDetector:
    """Anime face landmark detector using YOLOv8 and HRNetV2.

    Args:
        landmark_detector_checkpoint_path: Path to HRNetV2 checkpoint (.pth file)
        face_detector_checkpoint_path: Path to YOLOv8 checkpoint (.pt file)
        device: Device to run inference on ('cuda:0' or 'cpu')
        box_scale_factor: Scale factor for bounding box expansion (default: 1.25)
    """

    def __init__(
        self,
        landmark_detector_checkpoint_path: str | pathlib.Path,
        face_detector_checkpoint_path: str | pathlib.Path | None = None,
        device: str = 'cuda:0',
        box_scale_factor: float = 1.25,
    ):
        self.device = device
        self.box_scale_factor = box_scale_factor

        # Initialize HRNetV2 landmark detector
        self.landmark_detector = self._init_hrnetv2(
            landmark_detector_checkpoint_path, device
        )

        # Initialize YOLOv8 face detector
        self.face_detector = self._init_yolo_detector(
            face_detector_checkpoint_path, device
        )

    def _init_hrnetv2(
        self, checkpoint_path: str | pathlib.Path, device: str
    ) -> HRNetV2:
        """Initialize HRNetV2 model.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on

        Returns:
            HRNetV2 model
        """
        if isinstance(checkpoint_path, pathlib.Path):
            checkpoint_path = checkpoint_path.as_posix()

        # Create model
        model = HRNetV2(num_keypoints=28, pretrained=checkpoint_path)
        model.to(device)
        model.eval()

        return model

    def _init_yolo_detector(
        self, checkpoint_path: str | pathlib.Path | None, device: str
    ) -> YOLO | None:
        """Initialize YOLOv8 face detector.

        Args:
            checkpoint_path: Path to .pt model file (e.g., face_yolov8n.pt)
            device: Device to run on ('cuda:0' or 'cpu')

        Returns:
            YOLO model instance or None
        """
        if checkpoint_path is None:
            return None

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                'Ultralytics is not installed. Install it with: pip install ultralytics'
            )

        if isinstance(checkpoint_path, pathlib.Path):
            checkpoint_path = checkpoint_path.as_posix()

        # Load YOLO model
        model = YOLO(checkpoint_path)

        # Set device
        if device.startswith('cuda'):
            model.to('cuda')
        else:
            model.to('cpu')

        return model

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect faces using YOLOv8.

        Args:
            image: Input image in BGR format (OpenCV format)

        Returns:
            List of bounding boxes with format [x0, y0, x1, y1, score]
        """
        if self.face_detector is None:
            raise ValueError('Face detector is not initialized.')
        # Run inference
        results = self.face_detector(image, verbose=False)

        # Extract bounding boxes
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            # boxes_data.xyxy: tensor of shape (N, 4) with [x0, y0, x1, y1]
            # boxes_data.conf: tensor of shape (N,) with confidence scores
            for i in range(len(boxes_data)):
                xyxy = boxes_data.xyxy[i].cpu().numpy()  # [x0, y0, x1, y1]
                conf = boxes_data.conf[i].cpu().numpy()  # confidence score
                # Combine to [x0, y0, x1, y1, score]
                box = np.append(xyxy, conf)
                boxes.append(box)

        # Note: box scaling is applied in inference_batch via scale_factor parameter
        return boxes

    def _update_pred_box(self, pred_boxes: list[np.ndarray]) -> list[np.ndarray]:
        """Scale bounding boxes by scale factor.

        Args:
            pred_boxes: List of bounding boxes [x0, y0, x1, y1, score]

        Returns:
            Scaled bounding boxes
        """
        boxes = []
        for pred_box in pred_boxes:
            box = pred_box[:4].copy()
            size = box[2:] - box[:2] + 1
            new_size = size * self.box_scale_factor
            center = (box[:2] + box[2:]) / 2
            tl = center - new_size / 2
            br = tl + new_size
            pred_box_copy = pred_box.copy()
            pred_box_copy[:4] = np.concatenate([tl, br])
            boxes.append(pred_box_copy)
        return boxes

    @staticmethod
    def _load_image(
        image_or_path: np.ndarray | str | pathlib.Path,
    ) -> np.ndarray:
        """Load image from path or return numpy array.

        Args:
            image_or_path: Image array or path to image file

        Returns:
            Image array in BGR format
        """
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(image_or_path.as_posix())
        else:
            raise ValueError('Invalid image type')

        if image is None:
            raise ValueError('Failed to load image')
        return image

    def __call__(
        self,
        image_or_path: np.ndarray | str | pathlib.Path,
        boxes: list[np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Detect face landmarks.

        Args:
            image_or_path: An image with BGR channel order or an image path.
            boxes: A list of bounding boxes for faces. Each bounding box
                should be of the form [x0, y0, x1, y1, [score]].

        Returns: A list of detection results. Each detection result has
            bounding box of the form [x0, y0, x1, y1, [score]], and landmarks
            of the form [x, y, score].
        """
        image = self._load_image(image_or_path)

        # Detect faces if boxes not provided
        if boxes is None:
            if self.face_detector is not None:
                boxes = self._detect_faces(image)
            else:
                warnings.warn(
                    'Neither the face detector nor the bounding box is '
                    'specified. So the entire image is treated as the face '
                    'region.'
                )
                h, w = image.shape[:2]
                boxes = [np.array([0, 0, w - 1, h - 1, 1])]

        # Detect landmarks
        results = inference_batch(
            self.landmark_detector,
            image,
            boxes,
            device=self.device,
            input_size=(256, 256),
            scale_factor=self.box_scale_factor,
        )

        return results
