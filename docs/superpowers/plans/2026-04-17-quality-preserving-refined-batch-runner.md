# Quality-Preserving Refined Batch Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note:** the current workspace does not contain `.git` metadata. The commit commands below should be run after these changes are synced into the target repository checkout, such as `lililuya/sam4dbodyForAnimate`.

**Goal:** Add a batch-safe refined offline runner that reuses one loaded model stack, processes one sample at a time, and preserves per-sample quality through sample-local retries and diagnostics.

**Architecture:** The implementation adds a new batch entrypoint plus a pure helper module for sample discovery, retry profiles, and batch-result bookkeeping. It also extends `scripts/offline_app_refined.py` with sample-level state reset and `run_sample(...)` orchestration so batch mode can call the same refined per-sample engine without mutating global config between samples.

**Tech Stack:** Python 3.12, OmegaConf, Python `unittest`, JSON, existing refined offline pipeline modules

---

## File Map

### New Files

- `scripts/offline_batch_helpers.py`
  - pure helper functions for sample discovery, sample ids, retry profiles, and batch manifests
- `scripts/offline_batch_refined.py`
  - CLI parsing and batch orchestration
- `tests/refined/test_offline_batch_refined.py`
  - batch helper and batch runner smoke tests

### Modified Files

- `configs/body4d_refined.yaml`
  - add batch defaults for search window and quality-safe retries
- `scripts/offline_app_refined.py`
  - add sample-state reset, per-sample debug summary writes, and `run_sample(...)`
- `tests/refined/test_offline_app_refined.py`
  - cover sample-state reset and `run_sample(...)`
- `README.md`
  - document the new batch runner and its quality-preserving behavior

## Task 1: Add Batch Config Defaults and Pure Batch Helpers

**Files:**
- Modify: `configs/body4d_refined.yaml`
- Create: `scripts/offline_batch_helpers.py`
- Create: `tests/refined/test_offline_batch_refined.py`
- Test: `python -m unittest tests.refined.test_offline_batch_refined.BatchHelperTests -v`

- [ ] **Step 1: Write the failing helper tests**

```python
# tests/refined/test_offline_batch_refined.py
import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from omegaconf import OmegaConf


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_batch_refined_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class BatchHelperTests(unittest.TestCase):
    def test_discover_samples_supports_input_root_and_jsonl_manifest(self):
        from scripts.offline_batch_helpers import discover_samples

        with make_workspace_tempdir() as tmpdir:
            input_root = os.path.join(tmpdir, "inputs")
            os.makedirs(input_root, exist_ok=True)

            video_path = os.path.join(input_root, "clip_a.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"mp4")

            frames_dir = os.path.join(input_root, "frames_b")
            os.makedirs(frames_dir, exist_ok=True)
            with open(os.path.join(frames_dir, "00000001.jpg"), "wb") as handle:
                handle.write(b"jpg")

            manifest_path = os.path.join(tmpdir, "inputs.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"input": video_path, "sample_id": "video_a"}) + "\n")
                handle.write(frames_dir + "\n")

            from_root = discover_samples(input_root=input_root)
            from_manifest = discover_samples(input_list=manifest_path)

        self.assertEqual(
            from_root,
            [
                {"input": video_path, "sample_id": "clip_a"},
                {"input": frames_dir, "sample_id": "frames_b"},
            ],
        )
        self.assertEqual(
            from_manifest,
            [
                {"input": video_path, "sample_id": "video_a"},
                {"input": frames_dir, "sample_id": "frames_b"},
            ],
        )

    def test_build_retry_profiles_keeps_quality_safe_defaults(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
            }
        )

        class Args:
            retry_mode = None
            track_chunk_size = None

        profiles = build_retry_profiles(cfg, Args())

        self.assertEqual(
            profiles,
            [
                {
                    "retry_index": 0,
                    "reason": "base",
                    "tracking.chunk_size": 180,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 1,
                    "reason": "safer_tracking",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 2,
                    "reason": "safer_reconstruction",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 24,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 3,
                    "reason": "search_expansion",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 24,
                    "initial_search_frames": 48,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the helper tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchHelperTests -v
```

Expected:

```text
ERROR: ModuleNotFoundError for scripts.offline_batch_helpers
```

- [ ] **Step 3: Add batch config defaults and the pure helper module**

```yaml
# configs/body4d_refined.yaml
batch:
  initial_search_frames: 24
  retry_mode: quality_safe
  retry_chunk_sizes: [120, 96]
  retry_batch_sizes: [24, 16]
```

```python
# scripts/offline_batch_helpers.py
import json
import os
from typing import Dict, List


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sample_id_from_input(input_path: str) -> str:
    normalized = os.path.normpath(input_path)
    base_name = os.path.basename(normalized)
    stem, ext = os.path.splitext(base_name)
    return stem if ext else base_name


def is_frame_directory(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for entry in sorted(os.listdir(path)):
        if entry.lower().endswith(IMAGE_EXTENSIONS):
            return True
    return False


def load_input_list(input_list: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(input_list, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                data = json.loads(line)
                input_path = data["input"]
                sample_id = data.get("sample_id", sample_id_from_input(input_path))
            else:
                input_path = line
                sample_id = sample_id_from_input(input_path)
            records.append({"input": input_path, "sample_id": sample_id})
    return records


def discover_samples(input_root: str = "", input_list: str = "", max_samples: int | None = None) -> List[Dict[str, str]]:
    if bool(input_root) == bool(input_list):
        raise ValueError("Exactly one of input_root or input_list must be provided")

    if input_list:
        samples = load_input_list(input_list)
    else:
        samples = []
        for entry in sorted(os.listdir(input_root)):
            full_path = os.path.join(input_root, entry)
            if os.path.isfile(full_path) and full_path.lower().endswith(".mp4"):
                samples.append({"input": full_path, "sample_id": sample_id_from_input(full_path)})
            elif is_frame_directory(full_path):
                samples.append({"input": full_path, "sample_id": sample_id_from_input(full_path)})

    if max_samples is not None:
        return samples[: int(max_samples)]
    return samples


def build_retry_profiles(cfg, args) -> List[Dict[str, int | str]]:
    retry_mode = args.retry_mode or cfg.batch.retry_mode
    base_chunk_size = int(args.track_chunk_size or cfg.tracking.chunk_size)
    base_batch_size = int(cfg.sam_3d_body.batch_size)
    initial_search_frames = int(cfg.batch.initial_search_frames)

    chunk_sizes = [base_chunk_size] + [int(value) for value in cfg.batch.retry_chunk_sizes]
    batch_sizes = [base_batch_size] + [int(value) for value in cfg.batch.retry_batch_sizes]

    if retry_mode == "never":
        return [
            {
                "retry_index": 0,
                "reason": "base",
                "tracking.chunk_size": base_chunk_size,
                "sam_3d_body.batch_size": base_batch_size,
                "initial_search_frames": initial_search_frames,
            }
        ]

    safe_chunk_size = chunk_sizes[1]
    safe_batch_size = batch_sizes[1]
    aggressive_chunk_size = chunk_sizes[-1] if retry_mode == "aggressive_safe" else safe_chunk_size
    aggressive_batch_size = batch_sizes[-1] if retry_mode == "aggressive_safe" else safe_batch_size

    return [
        {
            "retry_index": 0,
            "reason": "base",
            "tracking.chunk_size": base_chunk_size,
            "sam_3d_body.batch_size": base_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 1,
            "reason": "safer_tracking",
            "tracking.chunk_size": safe_chunk_size,
            "sam_3d_body.batch_size": base_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 2,
            "reason": "safer_reconstruction",
            "tracking.chunk_size": aggressive_chunk_size,
            "sam_3d_body.batch_size": safe_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 3,
            "reason": "search_expansion",
            "tracking.chunk_size": aggressive_chunk_size,
            "sam_3d_body.batch_size": aggressive_batch_size,
            "initial_search_frames": initial_search_frames * 2,
        },
    ]
```

- [ ] **Step 4: Run the helper tests to confirm they pass**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchHelperTests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the batch helper scaffold**

```bash
git add configs/body4d_refined.yaml scripts/offline_batch_helpers.py tests/refined/test_offline_batch_refined.py
git commit -m "feat: add refined batch helper scaffold"
```

## Task 2: Extend RefinedOfflineApp with Sample Reset and Sample Execution Hooks

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Modify: `tests/refined/test_offline_app_refined.py`
- Test: `python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests -v`

- [ ] **Step 1: Add failing tests for sample-local orchestration**

```python
# tests/refined/test_offline_app_refined.py
    def test_reset_sample_state_clears_sample_local_fields(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "reprompt": {
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.chunk_records = [{"chunk_id": 0}]
        app.output_paths = {"debug_metrics": "./debug_metrics"}
        app.reprompt_events = [{"frame_idx": 12}]
        app.sample_summary = {"status": "failed"}

        app.reset_sample_state()

        self.assertEqual(app.chunk_records, [])
        self.assertEqual(app.output_paths, {})
        self.assertEqual(app.reprompt_events, [])
        self.assertEqual(app.sample_summary, {})

    def test_run_sample_uses_runtime_profile_without_mutating_global_config(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "reprompt": {
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )

        class StubApp(RefinedOfflineApp):
            def prepare_input(self, input_video, output_dir, skip_existing):
                return {"frames": ["f0.png"], "output_dir": "./sample_out", "input_video": input_video}

            def detect_initial_targets(self, sample):
                return {"obj_ids": [1]}

            def iter_chunks(self, frames, chunk_size):
                self.observed_chunk_size = chunk_size
                return [{"chunk_id": 0}]

            def track_chunk(self, chunk, initial_targets):
                self.chunk_records.append({"chunk_id": chunk["chunk_id"]})
                return {"raw": True}

            def refine_chunk_masks(self, raw_chunk):
                return {"refined": True}

            def maybe_reprompt_chunk(self, chunk, refined_chunk, initial_targets):
                return {"final": True}

            def write_chunk_outputs(self, chunk, raw_chunk, final_chunk):
                return None

            def run_refined_4d_generation(self):
                return None

        app = StubApp("configs/body4d_refined.yaml", config=config)
        summary = app.run_sample(
            input_video="sample.mp4",
            output_dir="./batch_out",
            skip_existing=False,
            runtime_profile={
                "tracking.chunk_size": 96,
                "sam_3d_body.batch_size": 24,
                "initial_search_frames": 48,
                "retry_index": 1,
                "reason": "safer_tracking",
            },
        )

        self.assertEqual(app.observed_chunk_size, 96)
        self.assertEqual(config.tracking.chunk_size, 180)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["runtime_profile"]["tracking.chunk_size"], 96)
```

- [ ] **Step 2: Run the refined app tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests -v
```

Expected:

```text
AttributeError: 'RefinedOfflineApp' object has no attribute 'reset_sample_state'
```

- [ ] **Step 3: Add sample reset, sample debug writes, and run_sample(...)**

```python
# scripts/offline_app_refined.py
def write_debug_json(debug_dir: str, filename: str, payload) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class RefinedOfflineApp:
    def __init__(self, config_path: str, config=None):
        self.config_path = config_path
        self.CONFIG = config if config is not None else load_refined_config(config_path)
        self.reprompt_thresholds = build_reprompt_thresholds(self.CONFIG) if hasattr(self.CONFIG, "reprompt") else {}
        self.should_trigger_reprompt = should_trigger_reprompt
        self.match_detection_to_track = match_detection_to_track
        self.reset_sample_state()

    def reset_sample_state(self) -> None:
        self.chunk_records = []
        self.output_paths = {}
        self.reprompt_events = []
        self.sample_summary = {}

    def write_sample_debug(self, runtime_profile: dict) -> None:
        debug_dir = self.output_paths.get("debug_metrics", "")
        write_debug_json(debug_dir, "sample_summary.json", self.sample_summary)
        write_debug_json(debug_dir, "runtime_profile.json", runtime_profile)
        write_debug_json(debug_dir, "reprompt_events.json", self.reprompt_events)

    def run_sample(self, input_video: str, output_dir: Optional[str], skip_existing: bool, runtime_profile: Optional[dict] = None):
        runtime_profile = dict(runtime_profile or {})
        self.reset_sample_state()

        sample = self.prepare_input(input_video, output_dir, skip_existing)
        initial_targets = self.detect_initial_targets(sample)
        self.prepare_sample_output(sample["output_dir"], initial_targets["obj_ids"])

        chunk_size = int(runtime_profile.get("tracking.chunk_size", self.CONFIG.tracking.chunk_size))
        self.sample_summary = {
            "input_video": input_video,
            "output_dir": sample["output_dir"],
            "status": "running",
            "retry_index": int(runtime_profile.get("retry_index", 0)),
            "runtime_profile": runtime_profile,
        }

        try:
            for chunk in self.iter_chunks(sample["frames"], chunk_size):
                raw_chunk = self.track_chunk(chunk, initial_targets)
                refined_chunk = self.refine_chunk_masks(raw_chunk)
                final_chunk = self.maybe_reprompt_chunk(chunk, refined_chunk, initial_targets)
                self.write_chunk_outputs(chunk, raw_chunk, final_chunk)
            self.run_refined_4d_generation()
            self.sample_summary["status"] = "completed"
        except Exception as exc:
            self.sample_summary["status"] = "failed"
            self.sample_summary["error"] = str(exc)
            raise
        finally:
            self.sample_summary["chunk_count"] = len(self.chunk_records)
            self.write_sample_debug(runtime_profile)
            self.finalize_sample()

        return self.sample_summary
```

- [ ] **Step 4: Run the refined app tests to confirm they pass**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the sample-level refined app hooks**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "feat: add refined sample execution hooks"
```

## Task 3: Add the Batch Runner CLI and Happy-Path Orchestration

**Files:**
- Create: `scripts/offline_batch_refined.py`
- Modify: `tests/refined/test_offline_batch_refined.py`
- Test: `python -m unittest tests.refined.test_offline_batch_refined.BatchRunnerTests -v`

- [ ] **Step 1: Add failing batch-runner tests**

```python
# tests/refined/test_offline_batch_refined.py
class BatchRunnerTests(unittest.TestCase):
    def test_batch_parser_accepts_quality_preserving_arguments(self):
        from scripts.offline_batch_refined import build_batch_parser

        parser = build_batch_parser()
        args = parser.parse_args(
            [
                "--input_root",
                "./inputs",
                "--output_dir",
                "./outputs_batch",
                "--config",
                "configs/body4d_refined.yaml",
                "--retry_mode",
                "quality_safe",
                "--skip_existing",
            ]
        )

        self.assertEqual(args.input_root, "./inputs")
        self.assertEqual(args.output_dir, "./outputs_batch")
        self.assertEqual(args.retry_mode, "quality_safe")
        self.assertTrue(args.skip_existing)

    def test_run_batch_calls_refined_app_once_per_sample(self):
        import scripts.offline_batch_refined as offline_batch_refined

        args = unittest.mock.Mock(
            input_root="./inputs",
            input_list="",
            output_dir="./outputs_batch",
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            continue_on_error=False,
            save_debug_metrics=False,
            skip_existing=False,
            max_samples=None,
            retry_mode="never",
        )
        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        samples = [
            {"input": "./inputs/a.mp4", "sample_id": "a"},
            {"input": "./inputs/b.mp4", "sample_id": "b"},
        ]
        app = unittest.mock.MagicMock()
        app.run_sample.side_effect = [
            {"status": "completed", "output_dir": "./outputs_batch/a"},
            {"status": "completed", "output_dir": "./outputs_batch/b"},
        ]

        with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_batch_refined, "discover_samples", return_value=samples
        ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
            results = offline_batch_refined.run_batch(args)

        self.assertEqual(len(results), 2)
        self.assertEqual(app.run_sample.call_count, 2)
```

- [ ] **Step 2: Run the batch-runner tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchRunnerTests -v
```

Expected:

```text
ERROR: ModuleNotFoundError for scripts.offline_batch_refined
```

- [ ] **Step 3: Implement the batch runner CLI and happy-path run_batch(...)**

```python
# scripts/offline_batch_refined.py
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.offline_app_refined import RefinedOfflineApp, apply_runtime_overrides, load_refined_config
from scripts.offline_batch_helpers import build_retry_profiles, discover_samples


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quality-preserving batch runner for refined offline 4D body generation"
    )
    parser.add_argument("--input_root", type=str, default="")
    parser.add_argument("--input_list", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "body4d_refined.yaml"))
    parser.add_argument("--detector_backend", type=str, default=None)
    parser.add_argument("--track_chunk_size", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--save_debug_metrics", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--retry_mode",
        type=str,
        choices=["never", "quality_safe", "aggressive_safe"],
        default=None,
    )
    return parser


def sample_output_dir(batch_output_dir: str, sample_id: str) -> str:
    return os.path.join(batch_output_dir, sample_id)


def write_batch_manifest(batch_output_dir: str, samples: list[dict]) -> None:
    os.makedirs(batch_output_dir, exist_ok=True)
    with open(os.path.join(batch_output_dir, "batch_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)


def append_batch_result(batch_output_dir: str, record: dict) -> None:
    os.makedirs(batch_output_dir, exist_ok=True)
    with open(os.path.join(batch_output_dir, "batch_results.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_batch(args):
    cfg = apply_runtime_overrides(args, load_refined_config(args.config))
    samples = discover_samples(args.input_root, args.input_list, args.max_samples)
    write_batch_manifest(args.output_dir, samples)

    app = RefinedOfflineApp(args.config, config=cfg)
    results = []
    for sample in samples:
        runtime_profile = build_retry_profiles(cfg, args)[0]
        output_dir = sample_output_dir(args.output_dir, sample["sample_id"])
        summary = app.run_sample(
            input_video=sample["input"],
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            runtime_profile=runtime_profile,
        )
        record = {
            "sample_id": sample["sample_id"],
            "input": sample["input"],
            "output_dir": output_dir,
            "status": summary["status"],
            "retry_index": runtime_profile["retry_index"],
        }
        append_batch_result(args.output_dir, record)
        results.append(record)
    return results


def main() -> None:
    parser = build_batch_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the batch-runner tests to confirm they pass**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchRunnerTests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit the batch runner CLI**

```bash
git add scripts/offline_batch_refined.py tests/refined/test_offline_batch_refined.py
git commit -m "feat: add refined batch runner cli"
```

## Task 4: Add Quality-Safe Retries, Skip-Existing, and Batch Result Isolation

**Files:**
- Modify: `scripts/offline_batch_refined.py`
- Modify: `tests/refined/test_offline_batch_refined.py`
- Test: `python -m unittest tests.refined.test_offline_batch_refined.BatchRetryTests -v`

- [ ] **Step 1: Add failing retry and isolation tests**

```python
# tests/refined/test_offline_batch_refined.py
class BatchRetryTests(unittest.TestCase):
    def test_run_batch_skips_existing_sample_when_sample_summary_exists(self):
        import scripts.offline_batch_refined as offline_batch_refined

        args = unittest.mock.Mock(
            input_root="./inputs",
            input_list="",
            output_dir="",
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            continue_on_error=True,
            save_debug_metrics=False,
            skip_existing=True,
            max_samples=None,
            retry_mode="quality_safe",
        )

        with make_workspace_tempdir() as tmpdir:
            args.output_dir = os.path.join(tmpdir, "outputs_batch")
            sample_output = os.path.join(args.output_dir, "a", "debug_metrics")
            os.makedirs(sample_output, exist_ok=True)
            with open(os.path.join(sample_output, "sample_summary.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")

            cfg = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": [120, 96],
                        "retry_batch_sizes": [24, 16],
                    },
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": True},
                }
            )
            samples = [{"input": "./inputs/a.mp4", "sample_id": "a"}]
            app = unittest.mock.MagicMock()

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
                results = offline_batch_refined.run_batch(args)

        self.assertEqual(results[0]["status"], "skipped")
        app.run_sample.assert_not_called()

    def test_run_batch_retries_failed_sample_with_next_quality_safe_profile(self):
        import scripts.offline_batch_refined as offline_batch_refined

        args = unittest.mock.Mock(
            input_root="./inputs",
            input_list="",
            output_dir="./outputs_batch",
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            continue_on_error=True,
            save_debug_metrics=False,
            skip_existing=False,
            max_samples=None,
            retry_mode="quality_safe",
        )
        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        samples = [{"input": "./inputs/a.mp4", "sample_id": "a"}]
        app = unittest.mock.MagicMock()
        app.run_sample.side_effect = [
            RuntimeError("oom"),
            {"status": "completed", "output_dir": "./outputs_batch/a"},
        ]

        with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_batch_refined, "discover_samples", return_value=samples
        ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
            results = offline_batch_refined.run_batch(args)

        self.assertEqual(results[0]["status"], "retry_succeeded")
        self.assertEqual(app.run_sample.call_args_list[0].kwargs["runtime_profile"]["reason"], "base")
        self.assertEqual(app.run_sample.call_args_list[1].kwargs["runtime_profile"]["reason"], "safer_tracking")

    def test_run_batch_raises_on_failed_sample_when_continue_on_error_is_false(self):
        import scripts.offline_batch_refined as offline_batch_refined

        args = unittest.mock.Mock(
            input_root="./inputs",
            input_list="",
            output_dir="./outputs_batch",
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            continue_on_error=False,
            save_debug_metrics=False,
            skip_existing=False,
            max_samples=None,
            retry_mode="never",
        )
        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        samples = [{"input": "./inputs/a.mp4", "sample_id": "a"}]
        app = unittest.mock.MagicMock()
        app.run_sample.side_effect = RuntimeError("still failing")

        with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_batch_refined, "discover_samples", return_value=samples
        ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
            with self.assertRaisesRegex(RuntimeError, "Stopping batch on failed sample: a"):
                offline_batch_refined.run_batch(args)

    def test_run_batch_does_not_mutate_next_sample_profile_after_retry(self):
        import scripts.offline_batch_refined as offline_batch_refined

        args = unittest.mock.Mock(
            input_root="./inputs",
            input_list="",
            output_dir="./outputs_batch",
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            continue_on_error=True,
            save_debug_metrics=False,
            skip_existing=False,
            max_samples=None,
            retry_mode="quality_safe",
        )
        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        samples = [
            {"input": "./inputs/a.mp4", "sample_id": "a"},
            {"input": "./inputs/b.mp4", "sample_id": "b"},
        ]
        app = unittest.mock.MagicMock()
        app.run_sample.side_effect = [
            RuntimeError("oom"),
            {"status": "completed", "output_dir": "./outputs_batch/a"},
            {"status": "completed", "output_dir": "./outputs_batch/b"},
        ]

        with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_batch_refined, "discover_samples", return_value=samples
        ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
            offline_batch_refined.run_batch(args)

        self.assertEqual(app.run_sample.call_args_list[2].kwargs["runtime_profile"]["reason"], "base")
        self.assertEqual(app.run_sample.call_args_list[2].kwargs["runtime_profile"]["tracking.chunk_size"], 180)
```

- [ ] **Step 2: Run the retry tests to confirm they fail**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchRetryTests -v
```

Expected:

```text
FAIL: expected retry_succeeded, got completed
```

- [ ] **Step 3: Add per-sample retries and failure recording**

```python
# scripts/offline_batch_refined.py
def should_skip_sample(sample_output_dir: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    return os.path.isfile(os.path.join(sample_output_dir, "debug_metrics", "sample_summary.json"))


def run_sample_with_retries(app, sample: dict, batch_output_dir: str, cfg, args) -> dict:
    output_dir = sample_output_dir(batch_output_dir, sample["sample_id"])
    if should_skip_sample(output_dir, args.skip_existing):
        return {
            "sample_id": sample["sample_id"],
            "input": sample["input"],
            "output_dir": output_dir,
            "status": "skipped",
            "retry_index": 0,
        }

    last_error = None
    profiles = build_retry_profiles(cfg, args)
    for profile in profiles:
        try:
            summary = app.run_sample(
                input_video=sample["input"],
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                runtime_profile=profile,
            )
            return {
                "sample_id": sample["sample_id"],
                "input": sample["input"],
                "output_dir": output_dir,
                "status": "completed" if profile["retry_index"] == 0 else "retry_succeeded",
                "retry_index": profile["retry_index"],
                "runtime_profile": profile,
                "summary_status": summary["status"],
            }
        except Exception as exc:
            last_error = str(exc)

    return {
        "sample_id": sample["sample_id"],
        "input": sample["input"],
        "output_dir": output_dir,
        "status": "failed",
        "retry_index": profiles[-1]["retry_index"],
        "runtime_profile": profiles[-1],
        "error": last_error,
    }


def run_batch(args):
    cfg = apply_runtime_overrides(args, load_refined_config(args.config))
    samples = discover_samples(args.input_root, args.input_list, args.max_samples)
    write_batch_manifest(args.output_dir, samples)

    app = RefinedOfflineApp(args.config, config=cfg)
    results = []
    for sample in samples:
        record = run_sample_with_retries(app, sample, args.output_dir, cfg, args)
        append_batch_result(args.output_dir, record)
        results.append(record)
        if record["status"] == "failed" and not args.continue_on_error:
            raise RuntimeError(f"Stopping batch on failed sample: {sample['sample_id']}")
    return results
```

- [ ] **Step 4: Run the retry tests to confirm they pass**

Run:

```bash
python -m unittest tests.refined.test_offline_batch_refined.BatchRetryTests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit retries and batch result isolation**

```bash
git add scripts/offline_batch_refined.py tests/refined/test_offline_batch_refined.py
git commit -m "feat: add quality-safe batch retries"
```

## Task 5: Document the Batch Runner and Run Full Verification

**Files:**
- Modify: `README.md`
- Test: `python scripts/offline_batch_refined.py --help`
- Test: `python -m unittest tests.refined.test_offline_app_refined tests.refined.test_offline_batch_refined tests.refined.test_detector_yolo tests.refined.test_mask_refinement tests.refined.test_refined_pipeline_output tests.refined.test_reprompt -v`

- [ ] **Step 1: Update the README with batch usage**

````md
## Refined Batch Run

Run the quality-preserving refined batch runner:

```bash
python scripts/offline_batch_refined.py --input_root <path> --output_dir <batch-output> --config configs/body4d_refined.yaml
```

Key properties of this batch runner:

- reuses one loaded refined model stack across samples
- still processes one sample at a time to preserve per-video quality
- retries only with quality-safe adjustments such as smaller chunk size or smaller reconstruction batch size
- writes `batch_manifest.json`, `batch_results.jsonl`, and per-sample debug summaries

For resumable runs, use `--skip_existing`. For large batches where you want failure isolation, add `--continue_on_error`.
````

- [ ] **Step 2: Run the batch runner help command**

Run:

```bash
python scripts/offline_batch_refined.py --help
```

Expected:

```text
usage: offline_batch_refined.py [-h] [--input_root INPUT_ROOT] [--input_list INPUT_LIST] --output_dir OUTPUT_DIR
```

- [ ] **Step 3: Run the full refined and batch unit suite**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined tests.refined.test_offline_batch_refined tests.refined.test_detector_yolo tests.refined.test_mask_refinement tests.refined.test_refined_pipeline_output tests.refined.test_reprompt -v
```

Expected:

```text
OK
```

- [ ] **Step 4: Commit the batch docs and verification changes**

```bash
git add README.md
git commit -m "docs: add refined batch runner usage"
```
