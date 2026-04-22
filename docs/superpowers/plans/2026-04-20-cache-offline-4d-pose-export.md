# Cache Offline 4D Pose Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend cache-based offline 4D so it exports per-track OpenPose JSON and per-track SMPL JSON alongside the existing outputs.

**Architecture:** Add a shared pose export helper that serializes one frame of per-person outputs into traceable JSON trees. Wire the helper into the existing shared 4D pipeline through the `frame_writer` hook from `scripts/run_4d_from_cache.py`, then verify the helper and runner with focused tests.

**Tech Stack:** Python, `unittest`, NumPy, existing offline 4D pipeline helpers

---

### Task 1: Add failing export tests

**Files:**
- Create: `tests/export/test_pose_json_export.py`
- Modify: `tests/export/test_run_4d_from_cache.py`

- [ ] Add a failing helper test for per-track OpenPose and SMPL JSON export.
- [ ] Add a failing cache-runner test that proves `run_cache_sample(...)` provides a working `frame_writer`.
- [ ] Run the focused export tests and confirm they fail for the expected missing behavior.

### Task 2: Implement shared pose export helpers

**Files:**
- Create: `scripts/pose_json_export.py`
- Modify: `scripts/openpose_export.py`

- [ ] Add JSON-safe conversion helpers for tensor-like model outputs.
- [ ] Add per-track OpenPose JSON writing helpers.
- [ ] Add per-track SMPL JSON writing helpers that also preserve OpenPose-converted keypoints.
- [ ] Keep the existing frame-level OpenPose helper behavior unchanged.

### Task 3: Wire cache offline 4D to use the export hook

**Files:**
- Modify: `scripts/run_4d_from_cache.py`

- [ ] Build a `frame_writer` closure for cache-based offline 4D outputs.
- [ ] Pass that closure into `build_4d_context(...)`.
- [ ] Keep output-root, overwrite, and summary behavior unchanged.

### Task 4: Verify and document

**Files:**
- Modify: `README.md` if the new export trees need user-facing documentation

- [ ] Run the focused export tests and confirm they pass.
- [ ] Run the broader cache/offline export regression tests.
- [ ] Update docs only if the new output trees change user expectations materially.
