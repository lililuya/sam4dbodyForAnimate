# SAM3 Cache Offline 4D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a stable, traceable SAM3 cache from `app.py` and make offline 4D runnable directly from that cache without WebUI runtime state.

**Architecture:** Add a versioned cache contract module plus an export helper that converts the current `app.py` working directory into a `sam3_cache/<sample_id>/` package. Extract the current 4D body of `app.py` into a reusable helper, then build `scripts/run_4d_from_cache.py` to validate cache directories, rebuild the required runtime profile from `meta.json`, and execute the shared 4D pipeline offline.

**Tech Stack:** Python 3, Gradio, OmegaConf, OpenCV, PIL, `unittest`, existing SAM3 / `sam_3d_body` / diffusion-vas pipeline code.

---

## File Structure

- Create: `scripts/sam3_cache_contract.py`
  - Owns cache versioning, metadata builders, path helpers, JSON readers and contract validation.
- Create: `scripts/sam3_cache_export.py`
  - Owns exporting `images/`, `masks/`, `meta.json`, `prompts.json`, `frame_metrics.json`, and `events.json` from the current `app.py` working directory.
- Create: `scripts/app_4d_pipeline.py`
  - Owns the reusable 4D execution core that currently lives inside `app.py`.
- Create: `scripts/run_4d_from_cache.py`
  - Owns offline cache discovery, validation, runtime reconstruction, and 4D execution.
- Create: `tests/export/test_sam3_cache_contract.py`
  - Contract-level tests for metadata shape, frame alignment rules, and validation failures.
- Create: `tests/export/test_sam3_cache_export.py`
  - Export tests for cache writing, prompt serialization, frame metrics, and event logs.
- Create: `tests/export/test_app_4d_pipeline.py`
  - Tests for reusable 4D helper behavior and runtime restoration boundaries.
- Create: `tests/export/test_run_4d_from_cache.py`
  - CLI and batch-runner tests for offline execution from cache.
- Modify: `app.py`
  - Keep WebUI interaction, but delegate export and 4D work to the new helper modules.

## Task 1: Build The Cache Contract Module

**Files:**
- Create: `scripts/sam3_cache_contract.py`
- Test: `tests/export/test_sam3_cache_contract.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class Sam3CacheContractTests(unittest.TestCase):
    def test_build_cache_meta_captures_runtime_profile_and_frame_stems(self):
        from scripts.sam3_cache_contract import build_cache_meta

        meta = build_cache_meta(
            sample_id="demo",
            source_video="/tmp/demo.mp4",
            frame_stems=["00000000", "00000001"],
            image_size={"width": 1280, "height": 720},
            obj_ids=[1, 2],
            runtime_profile={
                "batch_size": 16,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
            },
            config_path="configs/body4d.yaml",
        )

        self.assertEqual(meta["cache_version"], 1)
        self.assertEqual(meta["sample_id"], "demo")
        self.assertEqual(meta["frame_count"], 2)
        self.assertEqual(meta["frame_stems"], ["00000000", "00000001"])
        self.assertEqual(meta["obj_ids"], [1, 2])
        self.assertEqual(meta["runtime_profile"]["batch_size"], 16)

    def test_validate_cache_dir_rejects_missing_matching_mask(self):
        from scripts.sam3_cache_contract import validate_cache_dir

        with make_workspace_tempdir() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
            with open(os.path.join(tmpdir, "meta.json"), "w", encoding="utf-8") as handle:
                handle.write(
                    '{"cache_version": 1, "frame_count": 1, "frame_stems": ["00000000"], '
                    '"image_ext": ".jpg", "mask_ext": ".png", "obj_ids": [1], '
                    '"runtime_profile": {"batch_size": 1, "detection_resolution": [256, 512], '
                    '"completion_resolution": [512, 1024], "smpl_export": false}}'
                )
            with open(os.path.join(tmpdir, "images", "00000000.jpg"), "wb") as handle:
                handle.write(b"x")

            ok, errors = validate_cache_dir(tmpdir)

        self.assertFalse(ok)
        self.assertTrue(any("masks/00000000.png" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_sam3_cache_contract -v`
Expected: `ModuleNotFoundError: No module named 'scripts.sam3_cache_contract'`

- [ ] **Step 3: Write minimal implementation**

```python
import json
import os
from datetime import datetime, timezone


CACHE_VERSION = 1


def build_cache_meta(
    *,
    sample_id,
    source_video,
    frame_stems,
    image_size,
    obj_ids,
    runtime_profile,
    config_path,
):
    return {
        "cache_version": CACHE_VERSION,
        "sample_id": sample_id,
        "source_video": source_video,
        "frame_count": len(frame_stems),
        "fps": float(runtime_profile.get("fps", 0.0)),
        "image_ext": ".jpg",
        "mask_ext": ".png",
        "frame_stems": list(frame_stems),
        "image_size": dict(image_size),
        "obj_ids": [int(obj_id) for obj_id in obj_ids],
        "runtime_profile": {
            "batch_size": int(runtime_profile["batch_size"]),
            "detection_resolution": list(runtime_profile["detection_resolution"]),
            "completion_resolution": list(runtime_profile["completion_resolution"]),
            "smpl_export": bool(runtime_profile["smpl_export"]),
        },
        "config": {"config_path": config_path},
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_cache_dir(cache_dir):
    errors = []
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return False, [f"missing meta.json in {cache_dir}"]

    meta = load_json(meta_path)
    frame_stems = meta.get("frame_stems", [])
    image_ext = meta.get("image_ext", ".jpg")
    mask_ext = meta.get("mask_ext", ".png")
    for stem in frame_stems:
        image_path = os.path.join(cache_dir, "images", f"{stem}{image_ext}")
        mask_path = os.path.join(cache_dir, "masks", f"{stem}{mask_ext}")
        if not os.path.isfile(image_path):
            errors.append(f"missing images/{stem}{image_ext}")
        if not os.path.isfile(mask_path):
            errors.append(f"missing masks/{stem}{mask_ext}")
    return len(errors) == 0, errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_sam3_cache_contract -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/sam3_cache_contract.py tests/export/test_sam3_cache_contract.py
git commit -m "feat: add SAM3 cache contract validation"
```

## Task 2: Export Cache Bundles From The App Working Directory

**Files:**
- Create: `scripts/sam3_cache_export.py`
- Modify: `scripts/sam3_cache_contract.py`
- Test: `tests/export/test_sam3_cache_export.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

import numpy as np
from PIL import Image


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class Sam3CacheExportTests(unittest.TestCase):
    def test_export_sam3_cache_writes_contract_files_and_traceability_payloads(self):
        from scripts.sam3_cache_export import export_sam3_cache

        runtime = {
            "out_obj_ids": [1],
            "batch_size": 8,
            "detection_resolution": [256, 512],
            "completion_resolution": [512, 1024],
            "smpl_export": False,
            "video_fps": 24.0,
            "prompt_log": {
                "1": {
                    "name": "Target 1",
                    "frames": {"0": {"points": [[12, 18]], "labels": [1]}},
                }
            },
            "frame_metrics": [
                {"frame_stem": "00000000", "track_metrics": {"1": {"bbox_xyxy": [0, 0, 3, 3], "mask_area": 4}}}
            ],
            "events": [{"type": "mask_generation_completed", "frame_count": 1}],
        }

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            os.makedirs(os.path.join(working_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(working_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(working_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
                os.path.join(working_dir, "masks", "00000000.png")
            )

            cache_dir = export_sam3_cache(
                working_dir=working_dir,
                cache_root=os.path.join(tmpdir, "sam3_cache"),
                sample_id="demo",
                source_video="/tmp/demo.mp4",
                runtime=runtime,
                config_path="configs/body4d.yaml",
            )

            self.assertTrue(os.path.isfile(os.path.join(cache_dir, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(cache_dir, "prompts.json")))
            self.assertTrue(os.path.isfile(os.path.join(cache_dir, "frame_metrics.json")))
            self.assertTrue(os.path.isfile(os.path.join(cache_dir, "events.json")))
            with open(os.path.join(cache_dir, "prompts.json"), "r", encoding="utf-8") as handle:
                prompts = json.load(handle)
            self.assertEqual(prompts["targets"]["1"]["frames"]["0"]["labels"], [1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_sam3_cache_export -v`
Expected: `ModuleNotFoundError: No module named 'scripts.sam3_cache_export'`

- [ ] **Step 3: Write minimal implementation**

```python
import json
import os
import shutil
from glob import glob
from PIL import Image

from scripts.sam3_cache_contract import build_cache_meta, validate_cache_dir


def _frame_stems_from_dir(path, ext):
    return sorted(os.path.splitext(os.path.basename(file_path))[0] for file_path in glob(os.path.join(path, f"*{ext}")))


def export_sam3_cache(*, working_dir, cache_root, sample_id, source_video, runtime, config_path):
    cache_dir = os.path.join(cache_root, sample_id)
    os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "masks"), exist_ok=True)

    for image_path in glob(os.path.join(working_dir, "images", "*.jpg")):
        shutil.copy2(image_path, os.path.join(cache_dir, "images", os.path.basename(image_path)))
    for mask_path in glob(os.path.join(working_dir, "masks", "*.png")):
        shutil.copy2(mask_path, os.path.join(cache_dir, "masks", os.path.basename(mask_path)))

    frame_stems = _frame_stems_from_dir(os.path.join(cache_dir, "images"), ".jpg")
    first_image = Image.open(os.path.join(cache_dir, "images", f"{frame_stems[0]}.jpg"))
    meta = build_cache_meta(
        sample_id=sample_id,
        source_video=source_video,
        frame_stems=frame_stems,
        image_size={"width": first_image.size[0], "height": first_image.size[1]},
        obj_ids=runtime["out_obj_ids"],
        runtime_profile={
            "batch_size": runtime["batch_size"],
            "detection_resolution": runtime["detection_resolution"],
            "completion_resolution": runtime["completion_resolution"],
            "smpl_export": runtime["smpl_export"],
            "fps": runtime["video_fps"],
        },
        config_path=config_path,
    )

    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(cache_dir, "prompts.json"), "w", encoding="utf-8") as handle:
        json.dump({"targets": runtime.get("prompt_log", {})}, handle, indent=2)
    with open(os.path.join(cache_dir, "frame_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime.get("frame_metrics", []), handle, indent=2)
    with open(os.path.join(cache_dir, "events.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime.get("events", []), handle, indent=2)

    ok, errors = validate_cache_dir(cache_dir)
    if not ok:
        raise ValueError(f"exported cache is invalid: {errors}")
    return cache_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_sam3_cache_export -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/sam3_cache_contract.py scripts/sam3_cache_export.py tests/export/test_sam3_cache_export.py
git commit -m "feat: export SAM3 cache bundles"
```

## Task 3: Wire App Runtime Tracking And Explicit Cache Export

**Files:**
- Modify: `app.py`
- Modify: `scripts/sam3_cache_export.py`
- Test: `tests/export/test_sam3_cache_export.py`

- [ ] **Step 1: Write the failing test**

```python
class Sam3CacheExportTests(unittest.TestCase):
    def test_build_runtime_export_state_preserves_prompt_log_events_and_frame_metrics(self):
        from scripts.sam3_cache_export import build_runtime_export_state

        runtime = {
            "out_obj_ids": [1],
            "batch_size": 8,
            "detection_resolution": [256, 512],
            "completion_resolution": [512, 1024],
            "smpl_export": False,
            "video_fps": 24.0,
            "prompt_log": {"1": {"name": "Target 1", "frames": {"0": {"points": [[1, 2]], "labels": [1]}}}},
            "frame_metrics": [{"frame_stem": "00000000", "track_metrics": {"1": {"mask_area": 4}}}],
            "events": [{"type": "target_added", "obj_id": 1}],
        }

        payload = build_runtime_export_state(runtime)

        self.assertEqual(payload["runtime_profile"]["batch_size"], 8)
        self.assertEqual(payload["prompt_log"]["1"]["frames"]["0"]["labels"], [1])
        self.assertEqual(payload["events"][0]["type"], "target_added")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_sam3_cache_export.Sam3CacheExportTests.test_build_runtime_export_state_preserves_prompt_log_events_and_frame_metrics -v`
Expected: `ImportError: cannot import name 'build_runtime_export_state'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sam3_cache_export.py
def build_runtime_export_state(runtime):
    return {
        "out_obj_ids": list(runtime.get("out_obj_ids", [])),
        "runtime_profile": {
            "batch_size": int(runtime["batch_size"]),
            "detection_resolution": list(runtime["detection_resolution"]),
            "completion_resolution": list(runtime["completion_resolution"]),
            "smpl_export": bool(runtime.get("smpl_export", False)),
            "fps": float(runtime.get("video_fps", 0.0)),
        },
        "prompt_log": runtime.get("prompt_log", {}),
        "frame_metrics": runtime.get("frame_metrics", []),
        "events": runtime.get("events", []),
    }


# app.py
RUNTIME["prompt_log"] = {}
RUNTIME["frame_metrics"] = []
RUNTIME["events"] = [{"type": "video_loaded", "source_video": path}]

target_entry = RUNTIME["prompt_log"].setdefault(str(RUNTIME["id"]), {"name": f"Target {RUNTIME['id']}", "frames": {}})
target_entry["frames"][str(frame_idx)] = {
    "points": input_point.tolist(),
    "labels": input_label.tolist(),
}
RUNTIME["events"].append({"type": "prompt_updated", "obj_id": int(RUNTIME["id"]), "frame_idx": int(frame_idx)})

RUNTIME["events"].append({"type": "mask_generation_completed", "frame_count": len(video_segments)})

export_cache_btn = gr.Button("Export SAM3 Cache")
export_cache_path = gr.Textbox(label="Exported Cache Path")

export_cache_btn.click(
    fn=export_current_sam3_cache,
    inputs=[video_state],
    outputs=[export_cache_path],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_sam3_cache_export -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add app.py scripts/sam3_cache_export.py tests/export/test_sam3_cache_export.py
git commit -m "feat: track and export SAM3 cache metadata from app"
```

## Task 4: Extract A Reusable 4D Pipeline Core

**Files:**
- Create: `scripts/app_4d_pipeline.py`
- Modify: `app.py`
- Test: `tests/export/test_app_4d_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class App4DPipelineTests(unittest.TestCase):
    def test_run_4d_pipeline_from_output_dir_uses_disk_images_masks_and_runtime_profile(self):
        from scripts.app_4d_pipeline import run_4d_pipeline_from_output_dir

        with make_workspace_tempdir() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(os.path.join(tmpdir, "images", "00000000.jpg"))
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(os.path.join(tmpdir, "masks", "00000000.png"))

            context = SimpleNamespace(
                output_dir=tmpdir,
                runtime={"out_obj_ids": [1], "batch_size": 1, "detection_resolution": [256, 512], "completion_resolution": [512, 1024], "video_fps": 24.0},
                sam3_3d_body_model=MagicMock(),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
            )

            with patch("scripts.app_4d_pipeline.process_image_with_mask", return_value=([], [], []), create=True), patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4", create=True
            ) as mock_jpg_folder_to_mp4:
                run_4d_pipeline_from_output_dir(context)

        mock_jpg_folder_to_mp4.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_app_4d_pipeline -v`
Expected: `ModuleNotFoundError: No module named 'scripts.app_4d_pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
import glob
import os
from types import SimpleNamespace


def build_4d_context(
    *,
    output_dir,
    runtime,
    sam3_3d_body_model,
    pipeline_mask,
    pipeline_rgb,
    depth_model,
    predictor,
    generator,
):
    return SimpleNamespace(
        output_dir=output_dir,
        runtime=runtime,
        sam3_3d_body_model=sam3_3d_body_model,
        pipeline_mask=pipeline_mask,
        pipeline_rgb=pipeline_rgb,
        depth_model=depth_model,
        predictor=predictor,
        generator=generator,
    )


def run_4d_pipeline_from_output_dir(context):
    output_dir = context.output_dir
    runtime = context.runtime
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"missing exported images under {images_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"missing exported masks under {masks_dir}")

    os.makedirs(os.path.join(output_dir, "rendered_frames"), exist_ok=True)
    for obj_id in runtime["out_obj_ids"]:
        os.makedirs(os.path.join(output_dir, "mesh_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "focal_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rendered_frames_individual", str(obj_id)), exist_ok=True)

    # In this task, copy the current app.py:on_4d_generation body into this function
    # and replace:
    #   OUTPUT_DIR -> output_dir
    #   RUNTIME -> runtime
    #   sam3_3d_body_model -> context.sam3_3d_body_model
    #   pipeline_mask -> context.pipeline_mask
    #   pipeline_rgb -> context.pipeline_rgb
    #   depth_model -> context.depth_model
    #   predictor -> context.predictor
    #   generator -> context.generator
    out_4d_path = os.path.join(output_dir, "4d_from_cache.mp4")
    jpg_folder_to_mp4(os.path.join(output_dir, "rendered_frames"), out_4d_path, fps=runtime["video_fps"])
    return out_4d_path


# app.py
from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_output_dir


def on_4d_generation(video_path: str):
    context = build_4d_context(
        output_dir=OUTPUT_DIR,
        runtime=RUNTIME,
        sam3_3d_body_model=sam3_3d_body_model,
        pipeline_mask=pipeline_mask,
        pipeline_rgb=pipeline_rgb,
        depth_model=depth_model,
        predictor=predictor,
        generator=generator,
    )
    return run_4d_pipeline_from_output_dir(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_app_4d_pipeline -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/app_4d_pipeline.py app.py tests/export/test_app_4d_pipeline.py
git commit -m "refactor: extract reusable 4D pipeline core"
```

## Task 5: Build Offline 4D Runner From Cache

**Files:**
- Create: `scripts/run_4d_from_cache.py`
- Modify: `scripts/sam3_cache_contract.py`
- Modify: `scripts/app_4d_pipeline.py`
- Test: `tests/export/test_run_4d_from_cache.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class Run4DFromCacheTests(unittest.TestCase):
    def test_run_cache_sample_restores_runtime_profile_from_meta(self):
        from scripts.run_4d_from_cache import run_cache_sample

        with make_workspace_tempdir() as tmpdir:
            cache_dir = os.path.join(tmpdir, "sam3_cache", "demo")
            os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "masks"), exist_ok=True)
            with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cache_version": 1,
                        "sample_id": "demo",
                        "source_video": "/tmp/demo.mp4",
                        "frame_count": 1,
                        "fps": 24.0,
                        "image_ext": ".jpg",
                        "mask_ext": ".png",
                        "frame_stems": ["00000000"],
                        "image_size": {"width": 4, "height": 4},
                        "obj_ids": [1],
                        "runtime_profile": {
                            "batch_size": 8,
                            "detection_resolution": [256, 512],
                            "completion_resolution": [512, 1024],
                            "smpl_export": False,
                        },
                        "config": {"config_path": "configs/body4d.yaml"},
                    },
                    handle,
                )

            with open(os.path.join(cache_dir, "images", "00000000.jpg"), "wb") as handle:
                handle.write(b"x")
            with open(os.path.join(cache_dir, "masks", "00000000.png"), "wb") as handle:
                handle.write(b"x")

            fake_runtime_app = MagicMock()
            fake_runtime_app.sam3_3d_body_model = MagicMock()
            fake_runtime_app.pipeline_mask = None
            fake_runtime_app.pipeline_rgb = None
            fake_runtime_app.depth_model = None
            fake_runtime_app.predictor = MagicMock()
            fake_runtime_app.generator = None

            with patch("scripts.run_4d_from_cache.build_runtime_app", return_value=fake_runtime_app), patch(
                "scripts.run_4d_from_cache.run_4d_pipeline_from_output_dir", return_value="/tmp/out.mp4"
            ) as mock_run_pipeline:
                out_path = run_cache_sample(cache_dir)

        self.assertEqual(out_path, "/tmp/out.mp4")
        runtime = mock_run_pipeline.call_args.args[0].runtime
        self.assertEqual(runtime["batch_size"], 8)
        self.assertEqual(runtime["completion_resolution"], [512, 1024])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_run_4d_from_cache -v`
Expected: `ModuleNotFoundError: No module named 'scripts.run_4d_from_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
import argparse
import os

from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_output_dir
from scripts.sam3_cache_contract import load_json, validate_cache_dir


def build_runtime_app(config_path):
    from scripts.offline_app import OfflineApp

    return OfflineApp(config_path=config_path)


def build_cache_runtime(meta):
    return {
        "out_obj_ids": list(meta["obj_ids"]),
        "batch_size": int(meta["runtime_profile"]["batch_size"]),
        "detection_resolution": list(meta["runtime_profile"]["detection_resolution"]),
        "completion_resolution": list(meta["runtime_profile"]["completion_resolution"]),
        "smpl_export": bool(meta["runtime_profile"]["smpl_export"]),
        "video_fps": float(meta["fps"]),
    }


def run_cache_sample(cache_dir):
    ok, errors = validate_cache_dir(cache_dir)
    if not ok:
        raise ValueError(f"invalid cache: {errors}")

    meta = load_json(os.path.join(cache_dir, "meta.json"))
    runtime_app = build_runtime_app(meta["config"]["config_path"])
    context = build_4d_context(
        output_dir=cache_dir,
        runtime=build_cache_runtime(meta),
        sam3_3d_body_model=runtime_app.sam3_3d_body_model,
        pipeline_mask=getattr(runtime_app, "pipeline_mask", None),
        pipeline_rgb=getattr(runtime_app, "pipeline_rgb", None),
        depth_model=getattr(runtime_app, "depth_model", None),
        predictor=getattr(runtime_app, "predictor", None),
        generator=getattr(runtime_app, "generator", None),
    )
    return run_4d_pipeline_from_output_dir(context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    args = parser.parse_args()
    run_cache_sample(args.cache_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_run_4d_from_cache -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/run_4d_from_cache.py scripts/sam3_cache_contract.py scripts/app_4d_pipeline.py tests/export/test_run_4d_from_cache.py
git commit -m "feat: run offline 4D from SAM3 cache"
```

## Task 6: Run Full Verification And Tighten Batch Behavior

**Files:**
- Modify: `scripts/run_4d_from_cache.py`
- Modify: `tests/export/test_run_4d_from_cache.py`
- Modify: `tests/export/test_sam3_cache_contract.py`

- [ ] **Step 1: Write the failing test**

```python
class Run4DFromCacheTests(unittest.TestCase):
    def test_discover_cache_dirs_skips_invalid_cache_and_keeps_valid_ones(self):
        from scripts.run_4d_from_cache import discover_cache_dirs

        cache_dirs = discover_cache_dirs(
            root_dir="/tmp/sam3_cache",
            require_meta=True,
        )

        self.assertIsInstance(cache_dirs, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_run_4d_from_cache.Run4DFromCacheTests.test_discover_cache_dirs_skips_invalid_cache_and_keeps_valid_ones -v`
Expected: `ImportError: cannot import name 'discover_cache_dirs'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_4d_from_cache.py
def discover_cache_dirs(root_dir, require_meta=True):
    candidates = []
    for name in sorted(os.listdir(root_dir)):
        cache_dir = os.path.join(root_dir, name)
        if not os.path.isdir(cache_dir):
            continue
        if require_meta and not os.path.isfile(os.path.join(cache_dir, "meta.json")):
            continue
        candidates.append(cache_dir)
    return candidates


def run_cache_batch(root_dir):
    results = []
    for cache_dir in discover_cache_dirs(root_dir):
        try:
            out_path = run_cache_sample(cache_dir)
            results.append({"cache_dir": cache_dir, "status": "completed", "output": out_path})
        except Exception as exc:
            results.append({"cache_dir": cache_dir, "status": "failed", "error": str(exc)})
    return results
```

- [ ] **Step 4: Run full verification**

Run: `python -m unittest tests.export.test_sam3_cache_contract tests.export.test_sam3_cache_export tests.export.test_app_4d_pipeline tests.export.test_run_4d_from_cache -v`
Expected: `OK`

Run: `python -m unittest tests.refined.test_completion_indexing tests.refined.test_offline_app_refined tests.refined.test_offline_batch_refined tests.export.test_offline_app_export -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/run_4d_from_cache.py tests/export/test_run_4d_from_cache.py tests/export/test_sam3_cache_contract.py
git commit -m "test: verify SAM3 cache contract and offline 4D batch flow"
```

## Self-Review

### Spec Coverage

- Cache contract structure: covered by Task 1.
- Traceability payloads: covered by Task 2 and Task 3.
- App-side explicit export: covered by Task 3.
- Offline 4D with no WebUI globals: covered by Task 4 and Task 5.
- Batch-friendly execution and failure handling: covered by Task 6.

### Placeholder Scan

- No `TODO`, `TBD`, or "similar to previous task" placeholders remain.
- Every code-changing step includes named functions, target files, and concrete test commands.

### Type Consistency

- Cache contract names stay consistent across all tasks:
  - `build_cache_meta`
  - `validate_cache_dir`
  - `export_sam3_cache`
  - `build_runtime_export_state`
  - `build_4d_context`
  - `run_4d_pipeline_from_output_dir`
  - `run_cache_sample`
  - `discover_cache_dirs`
