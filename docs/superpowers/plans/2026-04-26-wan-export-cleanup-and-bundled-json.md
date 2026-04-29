# Wan Export Cleanup And Bundled JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After successful Wan export, bundle each target with `4d.mp4`, `pose_meta_sequence.json`, and `smpl_sequence.json`, then clean heavy SAM4D intermediates from the sample working directory while preserving runtime/debug artifacts.

**Architecture:** Extend `WanSampleExporter` so each exported target writes aggregated pose/SMPL JSON and returns finalized target metadata. Then extend the refined offline pipeline to package the final sample-level `4d_*.mp4` into each target directory and remove intermediate sample artifacts only after successful export with at least one target. Keep the implementation test-first and preserve existing summary/ledger behavior.

**Tech Stack:** Python, `unittest`, JSON, OpenCV (`cv2`), existing refined offline runtime helpers, Wan export helpers

---

### Task 1: Add Failing Tests For Bundled JSON Outputs

**Files:**
- Modify: `tests/export/test_wan_sample_export.py`
- Test: `tests/export/test_wan_sample_export.py`

- [ ] **Step 1: Write the failing aggregated-pose test**

Add a test that creates a minimal `WanSampleExporter`, disables per-frame pose JSON, runs `finalize()`, and expects the exported target directory to contain `pose_meta_sequence.json`.

Assert the aggregated file contains:

- `sample_id`
- `track_id`
- `frame_count`
- ordered `records`

- [ ] **Step 2: Write the failing aggregated-SMPL test**

In the same test or a neighboring test, assert that `smpl_sequence.json` exists and contains:

- `sample_id`
- `track_id`
- `frame_count`
- `records`

Assert that `records[0]["person_output"]["pred_keypoints_2d"]` is JSON-safe list data rather than a NumPy array.

- [ ] **Step 3: Run the focused Wan export test to verify RED**

Run:

```bash
python -m unittest tests.export.test_wan_sample_export.WanSampleExportTests.test_finalize_writes_bundled_pose_and_smpl_sequences -v
```

Expected: FAIL because the exporter does not yet write the bundled JSON files.

- [ ] **Step 4: Commit**

```bash
git add tests/export/test_wan_sample_export.py
git commit -m "test: cover bundled wan export json outputs"
```

### Task 2: Add Failing Tests For 4D Copy And Cleanup

**Files:**
- Modify: `tests/refined/test_offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Write the failing successful-cleanup test**

Add a refined test that:

- prepares a sample output directory with:
  - `sample_runtime.json`
  - `debug_metrics/`
  - heavy intermediate directories such as `images/`, `masks_raw/`, `masks_refined/`, `rendered_frames/`, `mesh_4d_individual/`, `completion_refined/`
  - a final sample-level `4d_123456.mp4`
- configures `wan_export.enable = True`
- configures:
  - `copy_rendered_4d_to_targets = True`
  - `cleanup_sample_workdir_after_export = True`
- patches the export finalization result to include one target directory under the external Wan export root
- runs the packaging/cleanup path

Assert:

- `<target_dir>/4d.mp4` exists
- sample-level `sample_runtime.json` still exists
- `debug_metrics/` still exists
- heavy directories were removed
- sample-level `4d_123456.mp4` was removed

- [ ] **Step 2: Write the failing no-targets-no-cleanup test**

Add a refined test that:

- sets `cleanup_sample_workdir_after_export = True`
- returns zero exported target directories

Assert:

- no cleanup happens
- the sample-level `4d_*.mp4` remains
- heavy intermediate directories remain

- [ ] **Step 3: Run the focused refined cleanup tests to verify RED**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_packages_4d_into_targets_and_cleans_workdir tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_skips_cleanup_when_no_wan_targets_exported -v
```

Expected: FAIL because the refined pipeline does not yet package `4d.mp4` into targets or clean the workdir.

- [ ] **Step 4: Commit**

```bash
git add tests/refined/test_offline_app_refined.py
git commit -m "test: cover wan export cleanup packaging"
```

### Task 3: Implement Bundled JSON Support In Wan Exporter

**Files:**
- Modify: `scripts/wan_sample_export.py`
- Modify: `scripts/wan_sample_types.py`
- Test: `tests/export/test_wan_sample_export.py`

- [ ] **Step 1: Add typed config fields**

Extend `WanExportConfig` with:

- `copy_rendered_4d_to_targets: bool = True`
- `cleanup_sample_workdir_after_export: bool = True`

and coerce them from runtime config.

- [ ] **Step 2: Add JSON-safe payload conversion**

In `scripts/wan_sample_export.py`, add a helper that converts nested NumPy arrays/scalars inside `person_output` payloads into plain JSON-safe Python structures.

- [ ] **Step 3: Always write aggregated `pose_meta_sequence.json`**

After pose metadata for a target is built, write one aggregated JSON file containing:

- sample metadata
- track metadata
- ordered pose records

Do this even when `save_pose_meta_json` is disabled.

- [ ] **Step 4: Always write aggregated `smpl_sequence.json`**

For the same target, write one aggregated JSON file containing ordered `frame_stem` plus raw `person_output` records converted to JSON-safe structures.

- [ ] **Step 5: Return finalized target metadata**

Extend the exporter so the refined pipeline can know, per exported target:

- target directory path
- track id
- frame count

Preserve the existing `finalize()` return contract for current callers by returning the target directories as before, but store richer finalized target metadata on the exporter instance or expose a helper for the refined pipeline to read it safely.

- [ ] **Step 6: Run the focused Wan export test again**

Run:

```bash
python -m unittest tests.export.test_wan_sample_export.WanSampleExportTests.test_finalize_writes_bundled_pose_and_smpl_sequences -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/wan_sample_export.py scripts/wan_sample_types.py tests/export/test_wan_sample_export.py
git commit -m "feat: bundle wan export pose and smpl json"
```

### Task 4: Implement 4D Packaging And Sample Cleanup

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Add a final-4D discovery helper**

Add a helper in `scripts/offline_app_refined.py` to locate the sample-level final `4d_*.mp4` inside the working directory.

- [ ] **Step 2: Add a target-packaging helper**

Add a helper that, given exported target directories:

- copies the sample-level final `4d_*.mp4` into each target as `4d.mp4`
- skips the copy when:
  - Wan export is disabled
  - `copy_rendered_4d_to_targets = False`
  - no final `4d_*.mp4` exists
  - no target directories were exported

- [ ] **Step 3: Add a cleanup helper**

Add a helper that removes these paths from the sample working directory if they exist:

- `images/`
- `masks/`
- `masks_raw/`
- `masks_refined/`
- `rendered_frames/`
- `rendered_frames_individual/`
- `mesh_4d_individual/`
- `focal_4d_individual/`
- `completion_refined/`
- sample-local fallback `wan_export/`
- final sample-level `4d_*.mp4`

Preserve:

- `sample_runtime.json`
- `debug_metrics/`

- [ ] **Step 4: Hook packaging/cleanup after successful export**

At the point where the refined sample has:

- completed 4D generation
- finalized Wan export
- produced at least one target directory

invoke:

- target `4d.mp4` packaging
- workdir cleanup when `cleanup_sample_workdir_after_export = True`

If no targets were exported, do not clean.

- [ ] **Step 5: Keep failure behavior strict**

If packaging or cleanup fails:

- raise the error
- let the existing sample failure path record the failure in the issue ledger

- [ ] **Step 6: Run the focused refined cleanup tests again**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_packages_4d_into_targets_and_cleans_workdir tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_skips_cleanup_when_no_wan_targets_exported -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "feat: package wan targets and clean workdir"
```

### Task 5: Update Configs And Run Regression Verification

**Files:**
- Modify: `configs/body4d_refined.yaml`
- Modify: `configs/body4d_refined_low_memory.yaml`
- Modify: `configs/body4d_refined_80g_fast.yaml`
- Test: `tests/export/test_wan_sample_export.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Add new config keys to refined config variants**

Under `wan_export`, add:

```yaml
copy_rendered_4d_to_targets: true
cleanup_sample_workdir_after_export: true
```

- [ ] **Step 2: Run the Wan export suite**

Run:

```bash
python -m unittest tests.export.test_wan_sample_export -v
```

Expected: PASS

- [ ] **Step 3: Run the refined suite**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined -v
```

Expected: PASS

- [ ] **Step 4: Run the combined regression suite**

Run:

```bash
python -m unittest tests.export.test_wan_reference_compat tests.export.test_wan_sample_export tests.refined.test_offline_app_refined -v
```

Expected: PASS

- [ ] **Step 5: Review plan coverage**

Confirm the implementation now provides:

- `4d.mp4` copied into each exported target
- aggregated `pose_meta_sequence.json` per target
- aggregated `smpl_sequence.json` per target
- heavy refined intermediates cleaned after successful export
- preserved `sample_runtime.json` and `debug_metrics/`
- no cleanup when no Wan targets were exported

- [ ] **Step 6: Commit**

```bash
git add configs/body4d_refined.yaml configs/body4d_refined_low_memory.yaml configs/body4d_refined_80g_fast.yaml scripts/offline_app_refined.py scripts/wan_sample_export.py scripts/wan_sample_types.py tests/export/test_wan_sample_export.py tests/refined/test_offline_app_refined.py
git commit -m "feat: clean refined outputs after wan export"
```
