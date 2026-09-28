# Third-party model notices

Apache License 2.0 in [LICENSE](LICENSE) covers **VTM Spark source code** and **our original training work** (the DiT checkpoint and the tracker fine-tunes we trained). It does **not** re-license anyone else’s weights.

A third-party model you add later keeps the license of the person or org that published it. Apache here does not attach to that file.

This file lists the weights we actually load, the official license from the eligible publisher, and where that text already lives in the tree.

## Ours (Apache-2.0)

Eligible publisher: this project / [sinBoo1/VTM-Elf-0.01](https://huggingface.co/sinBoo1/VTM-Elf-0.01) (`license: apache-2.0` on that card).

| File | What it is |
|------|------------|
| `models/dit/VTM-1.5.1.pt` | Our DiT generation checkpoint |
| `models/trackers/iris_pose.pt` | Our YOLO-pose iris / pupil fine-tune |
| `models/trackers/dwpose_v2.pt` (+ `.onnx`) | Our YOLO-pose body fine-tune |
| `models/trackers/animeseg_hair3.pt` | Our Mask2Former hair-part fine-tune |

Our training work is Apache. The **base weights those fine-tunes started from** are not. Ultralytics says a YOLO model trained from their pretrained checkpoints stays under AGPL-3.0 unless you have an Enterprise license or you trained from scratch without their weights. Iris and `dwpose_v2` are YOLO-pose fine-tunes.

`animeseg_hair3.pt` loads config from `facebook/mask2former-swin-base-ade-semantic`. Facebook Research’s model zoo licenses those downloaded Mask2Former weights **CC BY-NC 4.0** (non-commercial). Hugging Face tags the conversion `license: other`. Our hair labels on top do not turn that base into Apache.

## Third-party weights we ship or download

| File / id | Eligible publisher | Official license | Where the text is |
|-----------|--------------------|------------------|-------------------|
| OpenSeeFace ONNX (`lm_model3_opt.onnx`, RetinaFace, `mnv3_detection`, `mnv3_gaze32`, `priorbox`, and the unused extra `lm_model*` in the same folder) | Emiliana / [emilianavt/OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) | BSD 2-Clause (code **and** models) | `vendor/tools/openseeface/LICENSE` |
| OSF binary library notices (onnxruntime, RetinaFace, etc.) | Each listed author | MIT / BSD / Apache as in that folder | `vendor/tools/openseeface/Licenses/` |
| `pose_landmarker_lite.task` | Google / MediaPipe Authors | Apache-2.0 | [MediaPipe #6306](https://github.com/google-ai-edge/mediapipe/issues/6306); same terms as [LICENSE](LICENSE) |
| `face_yolov8n.pt` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | Apache-2.0 on the **weights card** | That Hugging Face card (`license: apache-2.0`). The ADetailer **extension** repo is AGPL-3.0; we use the weight file, not that extension. |
| `mmpose_anime-face_hrnetv2.pth` | hysts / [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector) | MIT | [hysts/anime-face-detector-hrnetv2](https://huggingface.co/hysts/anime-face-detector-hrnetv2) (`license: mit`); code LICENSE is MIT, Copyright (c) 2021 hysts |
| Ultralytics YOLO runtime + pretrained start for iris / dwpose_v2 | Ultralytics | AGPL-3.0 | `licenses/AGPL-3.0.txt` (copy of [ultralytics/ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.244/LICENSE)); policy: [ultralytics.com/license](https://www.ultralytics.com/license) |
| Mask2Former ADE20k base used by `animeseg_hair3.pt` | Meta / Facebook Research | CC BY-NC 4.0 for **model-zoo weights**; code is MIT | `licenses/CC-BY-NC-4.0.txt`; [MODEL_ZOO.md](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) |
| `stabilityai/sd-vae-ft-mse` | Stability AI | MIT | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) (`license: mit`) |
| `cqyan/hybrid-sd-tinyvae` | cqyan / ByteDance Hybrid-SD; fine-tune of TAESD | **Not declared on the weight card.** Hybrid-SD *code* is Apache-2.0. TAESD *weights* are MIT (Ollin Boer Bohan). | [cqyan/hybrid-sd-tinyvae](https://huggingface.co/cqyan/hybrid-sd-tinyvae) (no `license:` field); [bytedance/Hybrid-SD](https://github.com/bytedance/Hybrid-SD); [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) |
| `openai/clip-vit-large-patch14` (HF cache on first generation setup) | OpenAI | MIT | [openai/CLIP LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE), Copyright (c) 2021 OpenAI |

AnimeSeg ([suzukimain/AnimeSeg](https://github.com/suzukimain/AnimeSeg)) has **no license** on GitHub (`license: null`). Do not treat it as Apache.

## Gaps

1. **YOLO fine-tunes** — Apache on our training does not replace AGPL on Ultralytics pretrained starts. Shipping `iris_pose.pt` or `dwpose_v2.pt` inside a closed app is the case Ultralytics says needs AGPL source-offer or an Enterprise license.
2. **Hair Mask2Former** — CC BY-NC 4.0 on the Facebook zoo weights is the opposite of a commercial Apache grant. Keep `animeseg_hair3.pt` out of a paid/redistributed build unless Meta (or a later official relicense) says otherwise.
3. **TinyVAE** — no official weight license from cqyan. Runtime download is convenient; redistributing those bytes needs a statement from that publisher.
4. **Hugging Face pack** `sinBoo1/VTM-Elf-0.01` is tagged Apache-2.0 for the **repo**, but the uploaded zip also contains OpenSeeFace (BSD) and MediaPipe (Apache) files. That tag does not rewrite those licenses. Keep this notices file next to any redistributed pack.
5. **Future models** — add a row here with the publisher’s own LICENSE / Hugging Face `license:` field. Do not assume Apache.

## Runtime vs git

DiT and some trackers download from Hugging Face on setup (`models/model_sources.json`). OpenSeeFace extra landmark nets (`lm_model0`–`4`, `T`, `U`, `V`) stay under the same BSD grant as `lm_model3`. Unused live-poser copies under `vendor/tools/live-poser/models/` are the same tracker files, not a different license.
