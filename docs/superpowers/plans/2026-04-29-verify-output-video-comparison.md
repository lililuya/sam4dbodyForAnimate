# Verify Output Video Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script under `verify_output/` that stitches `target.mp4`, `4d.mp4`, and optional auxiliary videos from one target result directory into a single high-quality comparison `mp4`.

**Architecture:** Add a small, testable video-composition module with pure helpers for discovery, padding, grid assembly, and overlay generation, then wrap it in a CLI that validates required inputs and delegates final encoding to `ffmpeg`. Keep `target.mp4` as the canonical tile size and master timebase.

**Tech Stack:** Python, OpenCV, NumPy, subprocess `ffmpeg`, unittest

---

### Task 1: Add failing tests for target video comparison helpers

**Files:**
- Create: `verify_output/concat_compare_videos.py`
- Create: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

import numpy as np


class CompareVideoHelpersTests(unittest.TestCase):
    def test_pad_frame_to_canvas_centers_without_resizing(self):
        from verify_output.concat_compare_videos import pad_frame_to_canvas

        frame = np.full((2, 4, 3), 255, dtype=np.uint8)
        padded = pad_frame_to_canvas(frame, target_width=8, target_height=6)

        self.assertEqual(padded.shape, (6, 8, 3))
        self.assertTrue((padded[2:4, 2:6] == 255).all())
        self.assertTrue((padded[:2] == 0).all())

    def test_build_overlay_frame_blends_target_and_rendered(self):
        from verify_output.concat_compare_videos import build_overlay_frame

        target = np.zeros((2, 2, 3), dtype=np.uint8)
        rendered = np.full((2, 2, 3), 200, dtype=np.uint8)
        overlay = build_overlay_frame(target, rendered, alpha=0.5)

        self.assertEqual(overlay.shape, (2, 2, 3))
        self.assertTrue((overlay == 100).all())

    def test_compose_grid_frame_uses_target_resolution_tiles(self):
        from verify_output.concat_compare_videos import compose_grid_frame

        tile = np.zeros((4, 6, 3), dtype=np.uint8)
        grid = compose_grid_frame(
            {
                "target": tile,
                "4d": tile,
                "overlay": tile,
                "src_face": tile,
                "src_pose": tile,
                "src_bg": tile,
                "src_mask": tile,
                "src_mask_detail": tile,
                "blank": tile,
            }
        )

        self.assertEqual(grid.shape, (12, 18, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: FAIL with `ModuleNotFoundError` or missing helper functions

- [ ] **Step 3: Write minimal implementation**

```python
def pad_frame_to_canvas(frame, target_width, target_height):
    ...


def build_overlay_frame(target_frame, rendered_frame, alpha):
    ...


def compose_grid_frame(named_tiles):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "test: add verify output video composition helpers"
```

### Task 2: Add failing tests for input discovery and validation

**Files:**
- Modify: `verify_output/concat_compare_videos.py`
- Modify: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import tempfile


def test_resolve_input_videos_requires_target_and_4d(self):
    from verify_output.concat_compare_videos import resolve_input_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "target.mp4"), "wb").close()
        with self.assertRaisesRegex(FileNotFoundError, "4d.mp4"):
            resolve_input_videos(tmpdir)


def test_resolve_input_videos_collects_optional_panels(self):
    from verify_output.concat_compare_videos import resolve_input_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ("target.mp4", "4d.mp4", "src_face.mp4"):
            open(os.path.join(tmpdir, name), "wb").close()
        resolved = resolve_input_videos(tmpdir)

    self.assertTrue(resolved["target"].endswith("target.mp4"))
    self.assertTrue(resolved["4d"].endswith("4d.mp4"))
    self.assertTrue(resolved["src_face"].endswith("src_face.mp4"))
    self.assertIsNone(resolved["src_pose"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: FAIL with missing `resolve_input_videos`

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_input_videos(sample_dir):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "test: add verify output input discovery coverage"
```

### Task 3: Add failing tests for video metadata validation

**Files:**
- Modify: `verify_output/concat_compare_videos.py`
- Modify: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the failing test**

```python
def test_validate_required_videos_rejects_frame_count_mismatch(self):
    from verify_output.concat_compare_videos import validate_required_videos

    with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
        validate_required_videos(
            target_info={"width": 8, "height": 8, "fps": 25.0, "frame_count": 10},
            rendered_info={"width": 8, "height": 8, "fps": 25.0, "frame_count": 9},
        )


def test_validate_panel_size_rejects_oversized_panel(self):
    from verify_output.concat_compare_videos import validate_panel_size

    with self.assertRaisesRegex(RuntimeError, "exceeds target tile size"):
        validate_panel_size(
            panel_name="src_face",
            panel_info={"width": 16, "height": 8},
            target_width=8,
            target_height=8,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: FAIL with missing validation helpers

- [ ] **Step 3: Write minimal implementation**

```python
def validate_required_videos(target_info, rendered_info):
    ...


def validate_panel_size(panel_name, panel_info, target_width, target_height):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "test: add verify output validation coverage"
```

### Task 4: Add failing tests for ffmpeg encoding handoff

**Files:**
- Modify: `verify_output/concat_compare_videos.py`
- Modify: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch


def test_encode_frames_to_mp4_invokes_ffmpeg_with_libx264(self):
    from verify_output.concat_compare_videos import encode_frames_to_mp4

    with patch("verify_output.concat_compare_videos.subprocess.run") as mock_run:
        encode_frames_to_mp4(
            frames_dir="frames_dir",
            fps=25.0,
            output_path="out.mp4",
        )

    command = mock_run.call_args.args[0]
    self.assertIn("ffmpeg", command[0].lower())
    self.assertIn("libx264", command)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: FAIL with missing encoder helper

- [ ] **Step 3: Write minimal implementation**

```python
def encode_frames_to_mp4(frames_dir, fps, output_path):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "test: add verify output ffmpeg handoff coverage"
```

### Task 5: Implement the CLI entrypoint and end-to-end assembly

**Files:**
- Modify: `verify_output/concat_compare_videos.py`
- Modify: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
from unittest.mock import patch


def test_main_writes_compare_video_to_verify_output(self):
    from verify_output import concat_compare_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = os.path.join(tmpdir, "sample_target1")
        os.makedirs(sample_dir, exist_ok=True)
        for name in ("target.mp4", "4d.mp4"):
            open(os.path.join(sample_dir, name), "wb").close()

        with patch.object(concat_compare_videos, "build_comparison_video") as mock_build:
            concat_compare_videos.main(["--input", sample_dir])

        output_path = mock_build.call_args.kwargs["output_path"]
        self.assertTrue(output_path.endswith("_compare.mp4"))
        self.assertIn("verify_output", output_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: FAIL because `main` does not support CLI execution yet

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv=None):
    ...


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.export.test_concat_compare_videos`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "feat: add verify output comparison video cli"
```

### Task 6: Run verification and document usage

**Files:**
- Create: `verify_output/README.md`
- Modify: `verify_output/concat_compare_videos.py`
- Modify: `tests/export/test_concat_compare_videos.py`

- [ ] **Step 1: Write the README**

```markdown
# Verify Output

## Compare One Exported Target Directory

```bash
python verify_output/concat_compare_videos.py --input path/to/sample_target1
```

This writes:

- `verify_output/<sample_dir_name>_compare.mp4`
```

- [ ] **Step 2: Run focused tests**

Run: `python -m unittest tests.export.test_concat_compare_videos -v`
Expected: PASS

- [ ] **Step 3: Run neighboring regression tests**

Run: `python -m unittest tests.export.test_app_4d_pipeline tests.export.test_wan_export_integration -v`
Expected: PASS

- [ ] **Step 4: Run syntax verification**

Run: `python -m py_compile verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verify_output/README.md verify_output/concat_compare_videos.py tests/export/test_concat_compare_videos.py
git commit -m "docs: add verify output comparison usage"
```
