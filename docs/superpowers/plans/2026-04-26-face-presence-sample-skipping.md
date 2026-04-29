# Face Presence Sample Skipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip Wan-export samples early when sampled face visibility is too low, avoid leaving per-sample directories for those skips, and append all skipped or failed cases to a shared ledger.

**Architecture:** Delay working-directory materialization until after a face-presence precheck passes. Keep sample-level skip and failure reporting in `offline_app_refined.py`, centralize append-only ledger helpers in `wan_sample_export.py`, and keep Wan target-level skip reporting compatible with the current UUID summary files.

**Tech Stack:** Python, `unittest`, OpenCV (`cv2`), PIL, existing refined offline runtime helpers, existing Wan summary helpers

---

### Task 1: Add Failing Tests For Sample Skip And Ledger Behavior

**Files:**
- Modify: `tests/refined/test_offline_app_refined.py`
- Modify: `tests/export/test_wan_sample_export.py`
- Test: `tests/refined/test_offline_app_refined.py`
- Test: `tests/export/test_wan_sample_export.py`

- [ ] **Step 1: Write the failing refined skip test**

Add a refined test that:

- configures `wan_export.enable = True`
- configures `skip_sample_without_face = True`
- configures `face_presence_stride = 5`
- configures `max_no_face_ratio = 0.80`
- patches the face-presence probe to report `checked_frame_count = 5`, `no_face_frame_count = 4`, `face_detected_frame_count = 1`, `no_face_ratio = 0.80`
- calls `run_sample(...)`
- expects:
  - returned summary status is `skipped`
  - `prepare_sample_output()` is not called
  - `detect_initial_targets()` is not called
  - `run_refined_4d_generation()` is not called
  - the planned sample output directory does not exist
  - `sample_issue_ledger.json` exists under the configured Wan export root
  - the last ledger item has `event_type = "sample_skipped_no_face"` and `reason = "face_presence_below_threshold"`

- [ ] **Step 2: Write the failing sample-failure ledger test**

Add a refined test that:

- configures a Wan export root
- forces `detect_initial_targets()` to raise `RuntimeError("tracker failed")`
- expects:
  - `run_sample(...)` raises
  - ledger file exists
  - the last ledger item has `event_type = "sample_failed"`
  - `details.error_type = "RuntimeError"`
  - `details.error_message = "tracker failed"`

- [ ] **Step 3: Write the failing Wan target skipped ledger test**

Add an export test that:

- creates a Wan exporter with `min_valid_face_ratio = 0.6`
- provides no face detections
- runs `finalize()`
- expects:
  - `<sample_uuid>_skipped.json` still exists
  - `sample_issue_ledger.json` exists under external export root
  - the ledger contains one `wan_target_skipped` event with the existing target skip reason

- [ ] **Step 4: Run tests to verify RED**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_skips_face_sparse_sample_without_creating_output_dir tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_records_failed_case_in_issue_ledger tests.export.test_wan_sample_export.WanSampleExportTests.test_finalize_appends_wan_target_skip_to_issue_ledger -v
```

Expected: FAIL because the precheck gate, delayed output-dir creation, and issue ledger behavior do not yet exist.

- [ ] **Step 5: Commit**

```bash
git add tests/refined/test_offline_app_refined.py tests/export/test_wan_sample_export.py
git commit -m "test: cover face-presence sample skipping"
```

### Task 2: Implement Shared Ledger Helpers And Config Types

**Files:**
- Modify: `scripts/wan_sample_export.py`
- Modify: `scripts/wan_sample_types.py`
- Test: `tests/export/test_wan_sample_export.py`

- [ ] **Step 1: Add typed Wan config fields**

Extend `WanExportConfig` with:

- `skip_sample_without_face: bool = True`
- `face_presence_stride: int = 5`
- `max_no_face_ratio: float = 0.80`

Update `from_runtime()` to coerce these fields from runtime config.

- [ ] **Step 2: Add append-only ledger helpers**

In `scripts/wan_sample_export.py`, add:

- `ISSUE_LEDGER_FILENAME = "sample_issue_ledger.json"`
- `get_wan_issue_ledger_path(root_dir: str) -> str`
- `append_wan_issue_records(root_dir: str, records: list[dict]) -> dict`

Use the same read/merge/write style as the existing summary helpers, but always append into an `items` list.

- [ ] **Step 3: Append Wan target skip records**

When `finalize()` writes `skipped_targets`, also append one ledger record per skipped target when `export_root` and `sample_uuid` are available.

- [ ] **Step 4: Run the focused Wan export test**

Run:

```bash
python -m unittest tests.export.test_wan_sample_export.WanSampleExportTests.test_finalize_appends_wan_target_skip_to_issue_ledger -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/wan_sample_export.py scripts/wan_sample_types.py tests/export/test_wan_sample_export.py
git commit -m "feat: add issue ledger helpers"
```

### Task 3: Implement Sample Precheck, Skip Gate, And Delayed Output Creation

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Delay output directory materialization**

Update `prepare_input()` so it only resolves and stores the planned sample output path. Do not create the directory there.

Ensure `prepare_sample_output()` becomes the first place that materializes the working directory and updates the base app output path.

- [ ] **Step 2: Add sample identity and ledger-root helpers**

Add helpers in `scripts/offline_app_refined.py` to:

- resolve `sample_id` and `source_path`
- resolve the ledger root from `wan_export.output_dir` or the refined output root
- append sample-level issue records via the shared Wan ledger helper

- [ ] **Step 3: Add face-presence probe helpers**

Add helpers that:

- lazily construct an `InsightFaceBackend`
- sample frame indices by `face_presence_stride`
- detect whether each sampled frame contains at least one face
- return the summary payload:

```python
{
    "checked_frame_count": 5,
    "face_detected_frame_count": 1,
    "no_face_frame_count": 4,
    "no_face_ratio": 0.80,
    "face_presence_stride": 5,
    "max_no_face_ratio": 0.80,
    "skip_sample_without_face": True,
}
```

- [ ] **Step 4: Add the skip decision**

In `run_sample()`, after `prepare_input()` and before `detect_initial_targets()`:

- compute `fps_summary`
- compute `face_presence`
- if the sample should be skipped:
  - set status to `skipped`
  - set skip reason
  - append a `sample_skipped_no_face` ledger entry
  - return without creating the sample output directory

- [ ] **Step 5: Record failed samples**

In the `except` path of `run_sample()`, append a `sample_failed` ledger entry before re-raising.

- [ ] **Step 6: Route top-level execution through `run_sample()`**

Simplify `run_refined_pipeline(args)` so it configures the app, sets `reprompt_thresholds`, calls `app.run_sample(...)`, and returns the resolved config.

- [ ] **Step 7: Run the focused refined tests**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_skips_face_sparse_sample_without_creating_output_dir tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_records_failed_case_in_issue_ledger -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "feat: skip face-sparse samples before tracking"
```

### Task 4: Update Configs And Run Regression Verification

**Files:**
- Modify: `configs/body4d_refined.yaml`
- Modify: `configs/body4d_refined_low_memory.yaml`
- Modify: `configs/body4d_refined_80g_fast.yaml`
- Test: `tests/refined/test_offline_app_refined.py`
- Test: `tests/export/test_wan_sample_export.py`

- [ ] **Step 1: Add new config keys to refined config variants**

Under `wan_export`, add:

```yaml
skip_sample_without_face: true
face_presence_stride: 5
max_no_face_ratio: 0.80
```

- [ ] **Step 2: Run focused refined and Wan suites**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined tests.export.test_wan_sample_export -v
```

Expected: PASS

- [ ] **Step 3: Run broader regression verification**

Run:

```bash
python -m unittest tests.export.test_wan_reference_compat tests.export.test_wan_sample_export tests.refined.test_offline_app_refined -v
```

Expected: PASS

- [ ] **Step 4: Review coverage against the design**

Confirm the implementation now provides:

- configurable sampled face-presence gating
- early sample skipping before working-directory creation
- append-only logging for skipped and failed samples
- append-only logging for Wan target skips
- unchanged Wan per-sample summary and skipped-target report behavior

- [ ] **Step 5: Commit**

```bash
git add configs/body4d_refined.yaml configs/body4d_refined_low_memory.yaml configs/body4d_refined_80g_fast.yaml scripts/offline_app_refined.py scripts/wan_sample_export.py scripts/wan_sample_types.py tests/refined/test_offline_app_refined.py tests/export/test_wan_sample_export.py
git commit -m "feat: add face-presence sample skipping"
```
