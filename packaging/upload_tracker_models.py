"""One-shot upload of tracker + OpenSeeFace weights to Hugging Face."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "sinBoo1/VTM-Elf-0.01"


def main() -> int:
    api = HfApi()
    uploads = [
        (ROOT / "models/trackers/iris_pose.pt", "trackers/iris_pose.pt"),
        (ROOT / "models/trackers/dwpose_v2.pt", "trackers/dwpose_v2.pt"),
        (
            ROOT / "models/trackers/pose_landmarker_lite.task",
            "trackers/pose_landmarker_lite.task",
        ),
        (
            ROOT / "vendor/tools/openseeface/models/lm_model3_opt.onnx",
            "openseeface/lm_model3_opt.onnx",
        ),
        (
            ROOT / "vendor/tools/openseeface/models/retinaface_640x640_opt.onnx",
            "openseeface/retinaface_640x640_opt.onnx",
        ),
        (
            ROOT / "vendor/tools/openseeface/models/mnv3_detection_opt.onnx",
            "openseeface/mnv3_detection_opt.onnx",
        ),
        (
            ROOT / "vendor/tools/openseeface/models/mnv3_gaze32_split_opt.onnx",
            "openseeface/mnv3_gaze32_split_opt.onnx",
        ),
        (
            ROOT / "vendor/tools/openseeface/models/priorbox_640x640.json",
            "openseeface/priorbox_640x640.json",
        ),
    ]

    for src, dest in uploads:
        if not src.is_file():
            print(f"Missing: {src}")
            return 1
        print(f"Uploading {src.name} ({src.stat().st_size} bytes) -> {REPO}:{dest}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=REPO,
            repo_type="model",
            commit_message=f"Add {dest}",
        )
        print(f"  OK {dest}")

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
