# Offline Refined Mask Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note:** the current workspace does not contain `.git` metadata. The commit commands below should be run after these changes are synced into the target repository checkout, such as `lililuya/sam4dbodyForAnimate`.

**Goal:** Build a new refined offline pipeline that keeps the existing `scripts/offline_app.py` unchanged while adding YOLO-based automatic prompting, stronger occlusion-aware mask refinement, chunked tracking, and automatic re-prompt recovery.

**Architecture:** The implementation adds a new `scripts/offline_app_refined.py` entrypoint plus focused helper modules for YOLO detection, mask refinement, and re-prompt heuristics. The new path reuses the existing model stack and the stronger two-stage mask logic already present in `app.py`, while isolating all refined behavior behind a new config and new output directories.

**Tech Stack:** Python 3.12, PyTorch, OmegaConf, OpenCV, SAM3, SAM-3D-Body, Diffusion-VAS, Ultralytics YOLO, Python `unittest`

---

## File Map

### New Files

- `configs/body4d_refined.yaml`
  - refined runtime config with detector, tracking, re-prompt, and debug sections
- `scripts/offline_app_refined.py`
  - refined CLI, sample orchestration, chunked tracking, refined 4D flow
- `scripts/offline_refined_helpers.py`
  - pure helper functions for temporal IoU filtering and occlusion-window selection
- `scripts/offline_reprompt.py`
  - drift metrics, trigger rules, and detection-to-track matching
- `models/sam_3d_body/tools/build_detector_yolo.py`
  - YOLO loading and bbox inference adapter
- `tests/__init__.py`
  - unittest package marker
- `tests/refined/__init__.py`
  - refined test package marker
- `tests/refined/test_offline_app_refined.py`
  - CLI and config smoke tests
- `tests/refined/test_detector_yolo.py`
  - YOLO filtering and sorting tests
- `tests/refined/test_mask_refinement.py`
  - temporal mask-selection helper tests
- `tests/refined/test_reprompt.py`
  - re-prompt trigger and matcher tests
- `tests/refined/test_refined_pipeline_output.py`
  - output layout and chunk-manifest smoke tests

### Modified Files

- `pyproject.toml`
  - add the YOLO runtime dependency
- `models/sam_3d_body/tools/build_detector.py`
  - register the YOLO backend in `HumanDetector`
- `README.md`
  - document the new refined offline command and config expectations

## Task 1: Scaffold the Refined CLI and Config

**Files:**
- Create: `configs/body4d_refined.yaml`
- Create: `scripts/offline_app_refined.py`
- Create: `tests/__init__.py`
- Create: `tests/refined/__init__.py`
- Create: `tests/refined/test_offline_app_refined.py`
- Test: `python -m unittest tests.refined.test_offline_app_refined -v`

- [ ] **Step 1: Write the failing CLI/config tests**

```python
# tests/refined/test_offline_app_refined.py
import unittest


class RefinedCliTests(unittest.TestCase):
    def test_parser_accepts_refined_arguments(self):
        from scripts.offline_app_refined import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--input_video",
                "sample.mp4",
                "--config",
                "configs/body4d_refined.yaml",
                "--detector_backend",
                "yolo",
                "--track_chunk_size",
                "96",
                "--save_debug_metrics",
            ]
        )

        self.assertEqual(args.input_video, "sample.mp4")
        self.assertEqual(args.config, "configs/body4d_refined.yaml")
        self.assertEqual(args.detector_backend, "yolo")
        self.assertEqual(args.track_chunk_size, 96)
        self.assertTrue(args.save_debug_metrics)

    def test_load_refined_config_reads_detector_backend(self):
        from scripts.offline_app_refined import load_refined_config

        cfg = load_refined_config("configs/body4d_refined.yaml")
        self.assertEqual(cfg.detector.backend, "yolo")
        self.assertTrue(hasattr(cfg, "tracking"))
        self.assertTrue(hasattr(cfg, "reprompt"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined -v
```

Expected:

```text
ImportError: No module named 'scripts.offline_app_refined'
```

- [ ] **Step 3: Add the refined config file and CLI skeleton**

```yaml
# configs/body4d_refined.yaml
paths:
  ckpt_root: "path to global checkpoint root"

sam3:
  ckpt_path: ${paths.ckpt_root}/sam3/sam3.pt

sam_3d_body:
  ckpt_path: ${paths.ckpt_root}/sam-3d-body-dinov3/model.ckpt
  mhr_path: ${paths.ckpt_root}/sam-3d-body-dinov3/assets/mhr_model.pt
  fov_path: ${paths.ckpt_root}/moge-2-vitl-normal/model.pt
  batch_size: 32
  detector_path: ""
  segmentor_path: ""

runtime:
  output_dir: ./outputs_refined
  continue_on_error: false

completion:
  enable: true
  detection_resolution: [256, 512]
  completion_resolution: [512, 1024]
  model_path_mask: ${paths.ckpt_root}/diffusion-vas-amodal-segmentation
  model_path_rgb: ${paths.ckpt_root}/diffusion-vas-content-completion
  model_path_depth: ${paths.ckpt_root}/depth_anything_v2_vitl.pth
  depth_encoder: vitl
  max_occ_len: 25

detector:
  backend: yolo
  weights_path: ""
  bbox_thresh: 0.35
  iou_thresh: 0.50
  max_det: 20

tracking:
  chunk_size: 180

reprompt:
  enable: true
  empty_mask_patience: 3
  area_drop_ratio: 0.35
  edge_touch_ratio: 0.40
  iou_low_threshold: 0.55

debug:
  save_metrics: true
```

```python
# scripts/offline_app_refined.py
import argparse
import os

from omegaconf import OmegaConf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refined offline 4D body generation with YOLO and occlusion-aware recovery"
    )
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "body4d_refined.yaml"),
    )
    parser.add_argument("--detector_backend", type=str, default="yolo")
    parser.add_argument("--track_chunk_size", type=int, default=None)
    parser.add_argument("--disable_auto_reprompt", action="store_true")
    parser.add_argument("--save_debug_metrics", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def load_refined_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_refined_config(args.config)
    if args.detector_backend:
        cfg.detector.backend = args.detector_backend
    if args.track_chunk_size is not None:
        cfg.tracking.chunk_size = args.track_chunk_size
    if args.disable_auto_reprompt:
        cfg.reprompt.enable = False
    if args.save_debug_metrics:
        cfg.debug.save_metrics = True
    print(f"Loaded refined config for detector backend: {cfg.detector.backend}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to confirm the scaffold passes**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the scaffold**

```bash
git add configs/body4d_refined.yaml scripts/offline_app_refined.py tests/__init__.py tests/refined/__init__.py tests/refined/test_offline_app_refined.py
git commit -m "feat: scaffold refined offline pipeline cli"
```

## Task 2: Add the YOLO Detector Backend

**Files:**
- Create: `models/sam_3d_body/tools/build_detector_yolo.py`
- Create: `tests/refined/test_detector_yolo.py`
- Modify: `models/sam_3d_body/tools/build_detector.py`
- Modify: `pyproject.toml`
- Test: `python -m unittest tests.refined.test_detector_yolo -v`

- [ ] **Step 1: Write the failing detector tests**

```python
# tests/refined/test_detector_yolo.py
import unittest
import numpy as np


class YoloDetectorTests(unittest.TestCase):
    def test_sort_boxes_xyxy_is_deterministic(self):
        from models.sam_3d_body.tools.build_detector_yolo import sort_boxes_xyxy

        boxes = np.array(
            [
                [20, 10, 40, 30],
                [0, 0, 10, 10],
                [20, 5, 40, 25],
            ],
            dtype=np.float32,
        )
        sorted_boxes = sort_boxes_xyxy(boxes)
        expected = np.array(
            [
                [0, 0, 10, 10],
                [20, 5, 40, 25],
                [20, 10, 40, 30],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(sorted_boxes, expected)

    def test_extract_person_boxes_filters_by_class_and_threshold(self):
        from models.sam_3d_body.tools.build_detector_yolo import extract_person_boxes

        class FakeBoxes:
            def __init__(self):
                self.xyxy = np.array(
                    [[0, 0, 50, 100], [10, 10, 30, 30], [5, 5, 40, 90]],
                    dtype=np.float32,
                )
                self.conf = np.array([0.90, 0.95, 0.20], dtype=np.float32)
                self.cls = np.array([0, 2, 0], dtype=np.float32)

        boxes = extract_person_boxes(FakeBoxes(), bbox_thr=0.30, det_cat_id=0)
        expected = np.array([[0, 0, 50, 100]], dtype=np.float32)
        np.testing.assert_allclose(boxes, expected)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the detector tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_detector_yolo -v
```

Expected:

```text
ERROR: ModuleNotFoundError for models.sam_3d_body.tools.build_detector_yolo
```

- [ ] **Step 3: Implement the YOLO adapter and register it**

```python
# models/sam_3d_body/tools/build_detector_yolo.py
import numpy as np


def sort_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    sorted_indices = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[sorted_indices].astype(np.float32)


def extract_person_boxes(boxes, bbox_thr: float, det_cat_id: int = 0) -> np.ndarray:
    xyxy = np.asarray(boxes.xyxy, dtype=np.float32)
    conf = np.asarray(boxes.conf, dtype=np.float32)
    cls = np.asarray(boxes.cls, dtype=np.float32)
    keep = (cls == float(det_cat_id)) & (conf >= float(bbox_thr))
    filtered = xyxy[keep]
    return sort_boxes_xyxy(filtered)


def load_ultralytics_yolo(path: str = ""):
    if not path:
        raise FileNotFoundError("YOLO weights_path must be set for offline refined runs")
    from ultralytics import YOLO

    return YOLO(path)


def run_ultralytics_yolo(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.35,
    nms_thr: float = 0.50,
    default_to_full_image: bool = False,
    max_det: int = 20,
):
    height, width = img.shape[:2]
    results = detector.predict(
        source=img,
        conf=bbox_thr,
        iou=nms_thr,
        max_det=max_det,
        verbose=False,
    )
    boxes = extract_person_boxes(results[0].boxes, bbox_thr=bbox_thr, det_cat_id=det_cat_id)
    if boxes.size == 0 and default_to_full_image:
        return np.array([[0, 0, width, height]], dtype=np.float32)
    return boxes
```

```python
# models/sam_3d_body/tools/build_detector.py
class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device

        if name == "vitdet":
            print("########### Using human detector: ViTDet")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet
            self.detector = self.detector.to(self.device)
            self.detector.eval()
        elif name in {"yolo", "yolo11"}:
            print("########### Using human detector: YOLO")
            from .build_detector_yolo import load_ultralytics_yolo, run_ultralytics_yolo

            self.detector = load_ultralytics_yolo(**kwargs)
            self.detector_func = run_ultralytics_yolo
        else:
            raise NotImplementedError
```

```toml
# pyproject.toml
dependencies = [
    "gradio~=6.0.0",
    "opencv-python~=4.12.0.88",
    "einops~=0.8.1",
    "decord~=0.6.0",
    "pycocotools~=2.0.10",
    "psutil~=7.1.3",
    "braceexpand~=0.1.7",
    "roma~=1.5.4",
    "omegaconf~=2.3.0",
    "pytorch_lightning",
    "yacs~=0.1.8",
    "matplotlib~=3.10.7",
    "cloudpickle~=3.1.2",
    "fvcore~=0.1.5.post20221221",
    "pyrender~=0.1.45",
    "termcolor~=3.2.0",
    "diffusers==0.29.1",
    "transformers~=4.57.3",
    "accelerate~=1.12.0",
    "imageio[ffmpeg]",
    "ultralytics~=8.3.0",
    "MoGe @ git+https://github.com/microsoft/MoGe.git",
]
```

- [ ] **Step 4: Run the YOLO detector tests**

Run:

```bash
python -m unittest tests.refined.test_detector_yolo -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the YOLO backend**

```bash
git add pyproject.toml models/sam_3d_body/tools/build_detector.py models/sam_3d_body/tools/build_detector_yolo.py tests/refined/test_detector_yolo.py
git commit -m "feat: add yolo detector backend for refined pipeline"
```

## Task 3: Add Pure Mask-Refinement Helpers

**Files:**
- Create: `scripts/offline_refined_helpers.py`
- Create: `tests/refined/test_mask_refinement.py`
- Test: `python -m unittest tests.refined.test_mask_refinement -v`

- [ ] **Step 1: Write the failing helper tests**

```python
# tests/refined/test_mask_refinement.py
import unittest


class MaskRefinementHelperTests(unittest.TestCase):
    def test_cap_consecutive_ones_by_iou_keeps_top_scores(self):
        from scripts.offline_refined_helpers import cap_consecutive_ones_by_iou

        flags = [1, 1, 1, 0, 1, 1]
        ious = [0.1, 0.8, 0.6, 0.0, 0.2, 0.9]
        result = cap_consecutive_ones_by_iou(flags, ious, max_keep=2)
        self.assertEqual(result, [0, 1, 1, 1, 1, 1])

    def test_find_occlusion_window_applies_padding(self):
        from scripts.offline_refined_helpers import find_occlusion_window

        ious = [0.9, 0.4, 0.3, 0.8, 0.95]
        result = find_occlusion_window(ious, threshold=0.55, total_frames=5, pad=1)
        self.assertEqual(result, (0, 3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the helper tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_mask_refinement -v
```

Expected:

```text
ERROR: ModuleNotFoundError for scripts.offline_refined_helpers
```

- [ ] **Step 3: Implement the pure helper module**

```python
# scripts/offline_refined_helpers.py
from typing import List, Optional, Sequence, Tuple


def cap_consecutive_ones_by_iou(
    flag: Sequence[int],
    iou: Sequence[float],
    max_keep: int = 3,
) -> List[int]:
    n = len(flag)
    if len(iou) != n:
        raise ValueError(f"len(flag)={n} != len(iou)={len(iou)}")

    out = [1 if flag[i] == 0 else 0 for i in range(n)]
    i = 0
    while i < n:
        if flag[i] != 1:
            i += 1
            continue
        j = i
        while j < n and flag[j] == 1:
            j += 1
        run_idx = list(range(i, j))
        if len(run_idx) <= max_keep:
            for k in run_idx:
                out[k] = 1
        else:
            top = sorted(run_idx, key=lambda k: (-float(iou[k]), k))[:max_keep]
            for k in top:
                out[k] = 1
        i = j
    return out


def smooth_equal_threshold_hits(iou_values: Sequence[float], threshold: float) -> List[float]:
    arr = [float(x) for x in iou_values]
    for idx in range(1, len(arr) - 1):
        if arr[idx] == threshold and arr[idx - 1] < threshold and arr[idx + 1] < threshold:
            arr[idx] = 0.0
    return arr


def find_occlusion_window(
    iou_values: Sequence[float],
    threshold: float,
    total_frames: int,
    pad: int = 2,
) -> Optional[Tuple[int, int]]:
    low = [idx for idx, value in enumerate(iou_values) if float(value) < float(threshold)]
    if not low:
        return None
    start = max(0, low[0] - pad)
    end = min(total_frames - 1, low[-1] + pad)
    return start, end
```

- [ ] **Step 4: Run the helper tests**

Run:

```bash
python -m unittest tests.refined.test_mask_refinement -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the helper module**

```bash
git add scripts/offline_refined_helpers.py tests/refined/test_mask_refinement.py
git commit -m "feat: add refined mask helper utilities"
```

## Task 4: Integrate the Refined Output Layout and Two-Stage 4D Flow

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Create: `tests/refined/test_refined_pipeline_output.py`
- Test: `python -m unittest tests.refined.test_refined_pipeline_output -v`

- [ ] **Step 1: Write the failing output-layout smoke test**

```python
# tests/refined/test_refined_pipeline_output.py
import os
import tempfile
import unittest


class RefinedOutputLayoutTests(unittest.TestCase):
    def test_prepare_output_dirs_creates_refined_directories(self):
        from scripts.offline_app_refined import prepare_output_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_output_dirs(tmpdir, [1, 2], save_debug_metrics=True)
            expected = [
                "images",
                "masks_raw",
                "masks_refined",
                os.path.join("completion_refined", "images"),
                os.path.join("completion_refined", "masks"),
                "rendered_frames",
                os.path.join("debug_metrics"),
            ]
            for rel_path in expected:
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, rel_path)), rel_path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the smoke test to confirm it fails**

Run:

```bash
python -m unittest tests.refined.test_refined_pipeline_output -v
```

Expected:

```text
ERROR: cannot import name 'prepare_output_dirs' from scripts.offline_app_refined
```

- [ ] **Step 3: Implement the refined output layout and orchestration hooks**

```python
# scripts/offline_app_refined.py
import json
import os
from typing import Dict, Iterable, List

from scripts.offline_refined_helpers import cap_consecutive_ones_by_iou, find_occlusion_window


def prepare_output_dirs(output_dir: str, obj_ids: Iterable[int], save_debug_metrics: bool) -> Dict[str, str]:
    paths = {
        "images": os.path.join(output_dir, "images"),
        "masks_raw": os.path.join(output_dir, "masks_raw"),
        "masks_refined": os.path.join(output_dir, "masks_refined"),
        "completion_images": os.path.join(output_dir, "completion_refined", "images"),
        "completion_masks": os.path.join(output_dir, "completion_refined", "masks"),
        "rendered_frames": os.path.join(output_dir, "rendered_frames"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    for obj_id in obj_ids:
        os.makedirs(os.path.join(output_dir, "mesh_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "focal_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rendered_frames_individual", str(obj_id)), exist_ok=True)
    if save_debug_metrics:
        debug_dir = os.path.join(output_dir, "debug_metrics")
        os.makedirs(debug_dir, exist_ok=True)
        paths["debug_metrics"] = debug_dir
    return paths


def write_chunk_manifest(debug_dir: str, chunk_records: List[dict]) -> None:
    if not debug_dir:
        return
    path = os.path.join(debug_dir, "chunk_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(chunk_records, handle, indent=2)
```

```python
# scripts/offline_app_refined.py
class RefinedOfflineApp:
    def __init__(self, config_path: str):
        self.CONFIG = load_refined_config(config_path)
        self.chunk_records = []

    def prepare_sample_output(self, output_dir: str, obj_ids: Iterable[int]):
        self.output_paths = prepare_output_dirs(
            output_dir,
            obj_ids,
            save_debug_metrics=bool(self.CONFIG.debug.save_metrics),
        )
        return self.output_paths

    def finalize_sample(self):
        debug_dir = self.output_paths.get("debug_metrics", "")
        write_chunk_manifest(debug_dir, self.chunk_records)
```

- [ ] **Step 4: Run the output-layout smoke test**

Run:

```bash
python -m unittest tests.refined.test_refined_pipeline_output -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the refined output layout**

```bash
git add scripts/offline_app_refined.py tests/refined/test_refined_pipeline_output.py
git commit -m "feat: add refined output layout orchestration"
```

## Task 5: Add Re-Prompt Drift Rules and Matching

**Files:**
- Create: `scripts/offline_reprompt.py`
- Create: `tests/refined/test_reprompt.py`
- Modify: `scripts/offline_app_refined.py`
- Test: `python -m unittest tests.refined.test_reprompt -v`

- [ ] **Step 1: Write the failing re-prompt tests**

```python
# tests/refined/test_reprompt.py
import unittest
import numpy as np


class RePromptTests(unittest.TestCase):
    def test_should_trigger_reprompt_for_repeated_empty_masks(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {
            "empty_run": 4,
            "area_ratio": 0.25,
            "edge_touch_ratio": 0.10,
            "iou_value": 0.40,
        }
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.40,
            "iou_low_threshold": 0.55,
        }
        self.assertTrue(should_trigger_reprompt(metrics, thresholds))

    def test_match_detection_to_track_prefers_highest_overlap(self):
        from scripts.offline_reprompt import match_detection_to_track

        prev_box = np.array([10, 10, 30, 50], dtype=np.float32)
        candidates = np.array(
            [
                [0, 0, 20, 40],
                [12, 12, 28, 48],
                [40, 40, 80, 90],
            ],
            dtype=np.float32,
        )
        matched = match_detection_to_track(prev_box, candidates)
        np.testing.assert_allclose(matched, np.array([12, 12, 28, 48], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the re-prompt tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_reprompt -v
```

Expected:

```text
ERROR: ModuleNotFoundError for scripts.offline_reprompt
```

- [ ] **Step 3: Implement the re-prompt helper module and integrate it**

```python
# scripts/offline_reprompt.py
from typing import Dict

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def should_trigger_reprompt(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    if metrics["empty_run"] >= thresholds["empty_mask_patience"]:
        return True
    if metrics["area_ratio"] <= thresholds["area_drop_ratio"]:
        return True
    if metrics["edge_touch_ratio"] >= thresholds["edge_touch_ratio"]:
        return True
    if metrics["iou_value"] <= thresholds["iou_low_threshold"]:
        return True
    return False


def match_detection_to_track(prev_box: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        raise ValueError("No candidate detections available for re-prompt")
    best_idx = max(range(len(candidates)), key=lambda idx: iou_xyxy(prev_box, candidates[idx]))
    return candidates[best_idx].astype(np.float32)
```

```python
# scripts/offline_app_refined.py
from scripts.offline_reprompt import match_detection_to_track, should_trigger_reprompt


def build_reprompt_thresholds(cfg) -> dict:
    return {
        "empty_mask_patience": int(cfg.reprompt.empty_mask_patience),
        "area_drop_ratio": float(cfg.reprompt.area_drop_ratio),
        "edge_touch_ratio": float(cfg.reprompt.edge_touch_ratio),
        "iou_low_threshold": float(cfg.reprompt.iou_low_threshold),
    }
```

- [ ] **Step 4: Run the re-prompt tests**

Run:

```bash
python -m unittest tests.refined.test_reprompt -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the re-prompt helpers**

```bash
git add scripts/offline_reprompt.py scripts/offline_app_refined.py tests/refined/test_reprompt.py
git commit -m "feat: add refined re-prompt heuristics"
```

## Task 6: Wire the Full Refined Flow and Document It

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Modify: `README.md`
- Test: `python -m unittest tests.refined.test_offline_app_refined tests.refined.test_detector_yolo tests.refined.test_mask_refinement tests.refined.test_refined_pipeline_output tests.refined.test_reprompt -v`

- [ ] **Step 1: Add the refined pipeline integration points**

```python
# scripts/offline_app_refined.py
def run_refined_pipeline(args) -> None:
    cfg = load_refined_config(args.config)
    if args.detector_backend:
        cfg.detector.backend = args.detector_backend
    if args.track_chunk_size is not None:
        cfg.tracking.chunk_size = args.track_chunk_size
    if args.disable_auto_reprompt:
        cfg.reprompt.enable = False
    if args.save_debug_metrics:
        cfg.debug.save_metrics = True

    app = RefinedOfflineApp(args.config)
    sample = app.prepare_input(args.input_video, args.output_dir, args.skip_existing)
    initial_targets = app.detect_initial_targets(sample)
    app.prepare_sample_output(sample["output_dir"], initial_targets["obj_ids"])

    for chunk in app.iter_chunks(sample["frames"], cfg.tracking.chunk_size):
        raw_chunk = app.track_chunk(chunk, initial_targets)
        refined_chunk = app.refine_chunk_masks(raw_chunk)
        final_chunk = app.maybe_reprompt_chunk(chunk, refined_chunk, initial_targets)
        app.write_chunk_outputs(chunk, raw_chunk, final_chunk)

    app.run_refined_4d_generation()
    app.finalize_sample()
```

```python
# scripts/offline_app_refined.py
class RefinedOfflineApp:
    def prepare_input(self, input_video: str, output_dir: str | None, skip_existing: bool) -> dict:
        raise NotImplementedError("prepare_input must collect frames and choose the sample output directory")

    def detect_initial_targets(self, sample: dict) -> dict:
        raise NotImplementedError("detect_initial_targets must run detector-driven initial prompting")

    def iter_chunks(self, frames: list[str], chunk_size: int):
        raise NotImplementedError("iter_chunks must yield bounded tracking windows")

    def track_chunk(self, chunk: dict, initial_targets: dict) -> dict:
        raise NotImplementedError("track_chunk must write raw masks and images for the current chunk")

    def refine_chunk_masks(self, raw_chunk: dict) -> dict:
        raise NotImplementedError("refine_chunk_masks must run the two-stage occlusion refinement flow")

    def maybe_reprompt_chunk(self, chunk: dict, refined_chunk: dict, initial_targets: dict) -> dict:
        raise NotImplementedError("maybe_reprompt_chunk must apply drift checks and re-prompt when needed")

    def write_chunk_outputs(self, chunk: dict, raw_chunk: dict, final_chunk: dict) -> None:
        raise NotImplementedError("write_chunk_outputs must persist masks, metrics, and per-chunk diagnostics")

    def run_refined_4d_generation(self) -> None:
        raise NotImplementedError("run_refined_4d_generation must run mesh reconstruction from refined masks")
```

- [ ] **Step 2: Update the README with refined usage**

````md
## Refined Auto Run

Run the refined end-to-end offline pipeline with automatic YOLO prompting and stronger occlusion-aware mask refinement:

```bash
python scripts/offline_app_refined.py --input_video <path> --config configs/body4d_refined.yaml
```

Key differences from the baseline script:

- preserves `scripts/offline_app.py` as the original path
- supports a YOLO detector backend for automatic prompts
- runs two-stage occlusion-aware mask refinement
- saves raw masks, refined masks, chunk manifests, and re-prompt diagnostics

For fully offline execution, set `detector.weights_path` in `configs/body4d_refined.yaml` to a local YOLO weights file.
````

- [ ] **Step 3: Run the refined script help and unit suite**

Run:

```bash
python scripts/offline_app_refined.py --help
python -m unittest tests.refined.test_offline_app_refined tests.refined.test_detector_yolo tests.refined.test_mask_refinement tests.refined.test_refined_pipeline_output tests.refined.test_reprompt -v
```

Expected:

```text
usage: offline_app_refined.py [-h] --input_video INPUT_VIDEO [--output_dir OUTPUT_DIR] [--config CONFIG]
OK
```

- [ ] **Step 4: Do one smoke run against a local sample**

Run:

```bash
python scripts/offline_app_refined.py --input_video path/to/sample.mp4 --config configs/body4d_refined.yaml --save_debug_metrics
```

Expected:

```text
Loaded refined config for detector backend: yolo
Created refined output directories
Wrote debug_metrics/chunk_manifest.json
```

- [ ] **Step 5: Commit the integrated refined flow**

```bash
git add README.md scripts/offline_app_refined.py
git commit -m "feat: integrate refined offline mask pipeline"
```
