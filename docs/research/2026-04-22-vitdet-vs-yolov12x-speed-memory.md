# ViTDet vs YOLO12x: Speed and Memory Notes

## Scope

This note compares:

- `YOLO12x` using the official YOLOv12 repository's published detection results
- `ViTDet-H` as it is effectively wired in this repository

The goal is practical detector selection for this repo's offline/refined pipeline, with a focus on:

- inference speed
- memory / VRAM pressure

## Source Boundaries

There is an important asymmetry in the available official data:

- The official YOLOv12 repo publishes detection latency, parameter count, and FLOPs for `YOLO12x`.
- The official ViTDet paper and Detectron2 project do **not** publish a directly comparable "T4 TensorRT10 latency + params + VRAM" table for `ViTDet-H`.

Because of that, this note separates statements into two groups:

1. **Officially published numbers**
2. **Engineering inferences** derived from the ViTDet-H config that this repo actually uses

## What This Repo Actually Uses for ViTDet

This repository's ViTDet path loads `cascade_mask_rcnn_vitdet_h_75ep.py`, i.e. the `ViTDet-H` Cascade Mask R-CNN variant:

- local detector entrypoint: [models/sam_3d_body/tools/build_detector.py](/E:/Project/sam-body4d-master/models/sam_3d_body/tools/build_detector.py#L110)
- local config file: [models/sam_3d_body/tools/cascade_mask_rcnn_vitdet_h_75ep.py](/E:/Project/sam-body4d-master/models/sam_3d_body/tools/cascade_mask_rcnn_vitdet_h_75ep.py#L108)

From that config:

- test image size is `1024`
- backbone checkpoint is `mae_pretrain_vit_huge_p14to16.pth`
- backbone shape is `embed_dim=1280`, `depth=32`, `num_heads=16`
- box predictor threshold is `0.05`

In other words, the ViTDet branch here is not a small ViTDet-B; it is the much larger `ViTDet-H`.

## Officially Published Numbers

### YOLO12x

From the official YOLOv12 repository:

- `YOLO12x` (Turbo/default) at `640` resolution:
  - `55.4` COCO mAP 50-95
  - `10.38 ms` speed on `T4 TensorRT10`
  - `59.3M` parameters
  - `184.6G` FLOPs
- `YOLO12x` (v1.0) at `640` resolution:
  - `55.2` COCO mAP 50-95
  - `11.79 ms` speed on `T4 TensorRT10`
  - `59.1M` parameters
  - `199.0G` FLOPs

### ViTDet

From the official ViTDet paper:

- ViTDet is a plain ViT-based detector that reaches up to `61.3 AP_box` on COCO with MAE pretraining.

From the official Detectron2 ViTDet config:

- the `cascade_mask_rcnn_vitdet_h_75ep.py` config uses `ViT-Huge`
- the official config's `test_score_thresh` is `0.05`

### Direct Comparison Table

| Item | YOLO12x | ViTDet-H in this repo |
|---|---:|---:|
| Official speed table available | Yes | No equivalent table found |
| Published inference speed | `10.38 ms` (Turbo) or `11.79 ms` (v1.0), T4 TensorRT10 | Not published in the same format |
| Published params | `59.3M` (Turbo) / `59.1M` (v1.0) | Not published in a directly comparable table |
| Published FLOPs | `184.6G` (Turbo) / `199.0G` (v1.0) | Not published in a directly comparable table |
| Published test resolution | `640` | config uses `1024` |

## Engineering Inference for Memory

### YOLO12x weight memory

Using the official `59.3M` parameter count:

- FP32 weights only: about `0.22 GiB`
- FP16 weights only: about `0.11 GiB`

This is **weights only**, not full runtime VRAM.

### ViTDet-H backbone weight memory

Using the local `ViTDet-H` config (`embed_dim=1280`, `depth=32`, `num_heads=16`) and a rough ViT-H parameter derivation, the backbone alone is approximately:

- `~630.8M` parameters

That implies roughly:

- FP32 weights only: about `2.35 GiB`
- FP16 weights only: about `1.17 GiB`

This estimate is for the transformer backbone only and does **not** include:

- FPN-style simple pyramid layers
- RPN / ROI heads
- activation tensors
- Detectron2 framework overhead

### What this means in practice

Even before activation memory is considered:

- `ViTDet-H` backbone weights are about `10.6x` the size of `YOLO12x` weights
- this repo's ViTDet path also runs at `1024` test size instead of `640`

The image-area ratio alone is:

- `(1024 x 1024) / (640 x 640) = 2.56x`

So, for inference VRAM, `ViTDet-H` should be expected to be **substantially** heavier than `YOLO12x`, due to a combination of:

- much larger backbone
- higher test resolution
- two-stage Detectron2 pipeline
- transformer activations / attention blocks

For training VRAM, the gap should be even larger.

## Engineering Inference for Speed

Because there is no official apples-to-apples latency table for `ViTDet-H` in the same format as the YOLOv12 repo, the speed conclusion here is architectural:

- `YOLO12x` is explicitly positioned and benchmarked as a real-time detector, with official `T4 TensorRT10` latency around `10-12 ms`.
- `ViTDet-H` in this repo uses a huge ViT backbone, a `1024` test size, and a Cascade Mask R-CNN / Detectron2 inference path.

So for this repo's detector stage:

- `YOLO12x` should be expected to be **much faster**
- `ViTDet-H` should be expected to be **meaningfully slower** and less deployment-friendly

This is not a quoted official latency number for ViTDet-H; it is an engineering inference from the model class and config.

## One More Practical Caveat About YOLO12x

The official YOLOv12 repo includes a note on `2025-06-17` saying users should prefer that repo over the Ultralytics implementation because the Ultralytics path is "inefficient" and "requires more memory".

That matters here because this repository's YOLO path is built around the Ultralytics API. So if you point this repo's YOLO backend at `YOLO12x` weights:

- your local VRAM behavior may be worse than the official YOLOv12 repo's intended implementation
- official YOLOv12 speed numbers are still useful for model-level comparison, but they are **not** a guaranteed measurement of this exact local integration

## Practical Recommendation for This Repo

### If you care about throughput and VRAM first

Prefer `YOLO12x`.

Why:

- official real-time latency is published
- parameter count is much smaller
- deployment and debug iteration will be easier

### If you care about close-overlap recall more than speed

Consider `ViTDet-H`, but only if:

- you can afford substantially more VRAM
- detector throughput is not your bottleneck
- you are willing to benchmark on your actual clips

### My recommendation for this repo

For batch/offline initialization:

- start with `YOLO12x` if you want a stronger YOLO than the smaller defaults
- use `ViTDet-H` only when you have evidence it materially improves overlapping-person recall on your data enough to justify the runtime cost

## Minimal Benchmarking Plan

If you want exact apples-to-apples numbers on your own machine, the most reliable next step is:

1. Run the same 50 to 100 representative frames through [debug_human_detection.py](/E:/Project/sam-body4d-master/scripts/debug_human_detection.py)
2. Measure:
   - wall-clock per frame
   - peak VRAM
   - number of detected people in close-overlap scenes
3. Compare:
   - `ViTDet-H`
   - `YOLO12x`
   - your current default detector

That local benchmark will be more decision-useful than any paper-vs-paper comparison.

## Sources

- Official YOLOv12 repository README:
  - https://github.com/sunsmarterjie/yolov12
  - raw table used above: https://raw.githubusercontent.com/sunsmarterjie/yolov12/main/README.md
- Official YOLOv12 paper page:
  - https://arxiv.org/abs/2502.12524
- Official ViTDet paper page:
  - https://arxiv.org/abs/2203.16527
- Official Detectron2 repository:
  - https://github.com/facebookresearch/detectron2
- Detectron2 tools README showing the official benchmark entrypoint:
  - https://github.com/facebookresearch/detectron2/blob/main/tools/README.md
- This repo's ViTDet wiring:
  - [models/sam_3d_body/tools/build_detector.py](/E:/Project/sam-body4d-master/models/sam_3d_body/tools/build_detector.py)
  - [models/sam_3d_body/tools/cascade_mask_rcnn_vitdet_h_75ep.py](/E:/Project/sam-body4d-master/models/sam_3d_body/tools/cascade_mask_rcnn_vitdet_h_75ep.py)
