# Face-First Wan Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a face-first preprocessing stage that cuts stable per-face `clip.mp4` inputs before `sam4d`, then make the refined pipeline consume those clip packages and keep final Wan export output under the configured `wan_export.output_dir`.

**Architecture:** Introduce a new clip-extraction module that produces deterministic `sample_uuid` and `clip_id` packages from raw videos using conservative adjacent-frame face association. Extend the refined offline path with a clip-package input mode, face-guided single-target body binding, and Wan export ID reuse so the final exported sample directory is keyed by the first-stage clip identity.

**Tech Stack:** Python, OpenCV, InsightFace backend wrapper, OmegaConf config plumbing, unittest-based regression coverage.

---

### Task 1: Add Face-First Config And Identity Contract

**Files:**
- Create: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\export\test_face_clip_pipeline.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\wan_sample_types.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\configs\body4d_refined_low_memory.yaml`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\configs\body4d_refined.yaml`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\configs\body4d_refined_80g_fast.yaml`

- [ ] **Step 1: Write the failing config-coercion tests**

```python
class FaceClipConfigTests(unittest.TestCase):
    def test_face_clip_config_coerces_runtime_values(self):
        cfg = WanExportConfig.from_runtime(
            {
                "enable": True,
                "output_dir": "./WanExport",
                "face_clip": {
                    "enable": True,
                    "output_dir": "./face_clips",
                    "min_clip_seconds": 5,
                    "debug_save_face_crop_video": False,
                },
            }
        )

        self.assertTrue(cfg.face_clip_enable)
        self.assertEqual(cfg.face_clip_output_dir, "./face_clips")
        self.assertEqual(cfg.face_clip_min_clip_seconds, 5.0)
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `python -m unittest tests.export.test_face_clip_pipeline.FaceClipConfigTests -v`  
Expected: `AttributeError` or assertion failure because `WanExportConfig` does not yet expose the new face-clip fields.

- [ ] **Step 3: Add minimal typed config support**

```python
@dataclass(frozen=True)
class WanExportConfig:
    ...
    face_clip_enable: bool = False
    face_clip_output_dir: str | None = None
    face_clip_min_clip_seconds: float = 5.0
    face_clip_debug_save_face_crop_video: bool = False
```

- [ ] **Step 4: Wire `from_runtime()` to read the nested face-clip payload**

```python
face_clip = dict(payload.get("face_clip") or {})
return cls(
    ...,
    face_clip_enable=_coerce_bool(face_clip.get("enable", False)),
    face_clip_output_dir=None if face_clip.get("output_dir") in {None, ""} else str(face_clip.get("output_dir")),
    face_clip_min_clip_seconds=float(face_clip.get("min_clip_seconds", 5.0)),
    face_clip_debug_save_face_crop_video=_coerce_bool(face_clip.get("debug_save_face_crop_video", False)),
)
```

- [ ] **Step 5: Add config keys to refined YAML variants**

```yaml
wan_export:
  face_clip:
    enable: false
    output_dir: ""
    min_clip_seconds: 5.0
    debug_save_face_crop_video: false
```

- [ ] **Step 6: Re-run the config tests**

Run: `python -m unittest tests.export.test_face_clip_pipeline.FaceClipConfigTests -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/export/test_face_clip_pipeline.py scripts/wan_sample_types.py configs/body4d_refined_low_memory.yaml configs/body4d_refined.yaml configs/body4d_refined_80g_fast.yaml
git commit -m "feat: add face-first clip config contract"
```

### Task 2: Build Stable Face Clip Extraction Module

**Files:**
- Create: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\face_clip_pipeline.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\export\test_face_clip_pipeline.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\wan_sample_export.py`

- [ ] **Step 1: Write the first failing segment-extraction test**

```python
def test_extract_face_clips_keeps_only_segments_with_face_on_every_frame(self):
    frames = [
        [det((10, 10, 20, 20))],
        [det((11, 10, 21, 20))],
        [],
        [det((12, 10, 22, 20))],
    ]

    result = extract_face_tracks_from_detections(
        frames,
        fps=1.0,
        min_clip_seconds=2.0,
    )

    self.assertEqual(len(result.kept_segments), 1)
    self.assertEqual(result.kept_segments[0].start_frame, 0)
    self.assertEqual(result.kept_segments[0].end_frame, 1)
```

- [ ] **Step 2: Run the segment test to verify it fails**

Run: `python -m unittest tests.export.test_face_clip_pipeline.FaceClipSegmentationTests.test_extract_face_clips_keeps_only_segments_with_face_on_every_frame -v`  
Expected: FAIL because `extract_face_tracks_from_detections` does not yet exist.

- [ ] **Step 3: Add the minimal extraction dataclasses and conservative association helpers**

```python
@dataclass(frozen=True)
class FaceClipRecord:
    frame_index_in_source: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class FaceClipSegment:
    face_track_index: int
    segment_index: int
    start_frame: int
    end_frame: int
    records: list[FaceClipRecord]
```

- [ ] **Step 4: Implement adjacent-frame-only extraction with “ambiguity means break” semantics**

```python
def extract_face_tracks_from_detections(frame_detections, fps: float, min_clip_seconds: float):
    live_tracks = []
    completed_segments = []
    for frame_index, detections in enumerate(frame_detections):
        live_tracks, finalized = step_face_tracks(live_tracks, detections, frame_index)
        completed_segments.extend(finalized)
    completed_segments.extend(finalize_live_tracks(live_tracks))
    return filter_segments_by_duration(completed_segments, fps=fps, min_clip_seconds=min_clip_seconds)
```

- [ ] **Step 5: Write the next failing tests for ambiguity and deterministic IDs**

```python
def test_extract_face_clips_breaks_segment_on_ambiguous_assignment(self):
    ...

def test_resolve_or_create_face_clip_sample_uuid_reuses_mapping(self):
    ...
```

- [ ] **Step 6: Reuse the existing source-to-UUID mapping helpers from `wan_sample_export.py`**

```python
sample_uuid = resolve_or_create_wan_sample_uuid(
    export_root,
    source_path,
    sample_id=sample_id,
    working_output_dir=working_output_dir,
)
clip_id = f"{sample_uuid}_face{face_track_index:02d}_seg{segment_index:03d}"
```

- [ ] **Step 7: Add `clip.mp4`, `track.json`, and `meta.json` packaging helpers**

```python
def write_face_clip_package(output_root: str, clip_id: str, frames: list[np.ndarray], meta: dict, records: list[dict]) -> str:
    clip_dir = os.path.join(output_root, "clips", clip_id)
    os.makedirs(clip_dir, exist_ok=True)
    write_mp4(os.path.join(clip_dir, "clip.mp4"), frames, fps=meta["fps"])
    write_json(os.path.join(clip_dir, "track.json"), {"clip_id": clip_id, "records": records})
    write_json(os.path.join(clip_dir, "meta.json"), meta)
    return clip_dir
```

- [ ] **Step 8: Run the new face-clip tests**

Run: `python -m unittest tests.export.test_face_clip_pipeline -v`  
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/face_clip_pipeline.py tests/export/test_face_clip_pipeline.py scripts/wan_sample_export.py
git commit -m "feat: add stable face clip extraction stage"
```

### Task 3: Make Refined Pipeline Consume Clip Packages

**Files:**
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\offline_app_refined.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\refined\test_offline_app_refined.py`

- [ ] **Step 1: Write the failing clip-package input test**

```python
def test_prepare_input_accepts_clip_package_directory(self):
    sample = app.prepare_input(clip_dir, None, False)
    self.assertEqual(sample["input_type"], "clip_package")
    self.assertEqual(sample["clip_id"], "sampleuuid_face01_seg001")
    self.assertEqual(sample["input_video"], os.path.join(clip_dir, "clip.mp4"))
```

- [ ] **Step 2: Run the clip-package input test to verify it fails**

Run: `python -m unittest tests.refined.test_offline_app_refined.RefinedOfflineAppTests.test_prepare_input_accepts_clip_package_directory -v`  
Expected: FAIL because `prepare_input()` does not yet recognize a clip package.

- [ ] **Step 3: Extend `prepare_input()` with clip-package detection**

```python
if os.path.isdir(input_video) and os.path.isfile(os.path.join(input_video, "clip.mp4")):
    meta = load_json(os.path.join(input_video, "meta.json"))
    return {
        "input_type": "clip_package",
        "input_video": os.path.join(input_video, "clip.mp4"),
        "clip_dir": os.path.abspath(input_video),
        "clip_id": str(meta["clip_id"]),
        "sample_uuid": str(meta["sample_uuid"]),
        ...
    }
```

- [ ] **Step 4: Write the failing face-guided body-binding test**

```python
def test_detect_initial_targets_uses_face_guided_single_target_binding_for_clip_package(self):
    ...
    self.assertEqual(targets["obj_ids"], [1])
    self.assertEqual(len(mock_predictor.add_new_points_or_box.call_args_list), 1)
```

- [ ] **Step 5: Implement a face-guided single-target initialization branch**

```python
if sample["input_type"] == "clip_package":
    return self.detect_clip_package_target(sample)
```

```python
def detect_clip_package_target(self, sample: dict) -> dict:
    body_box = self._bind_face_track_to_body_box(sample, window_frames=12)
    return self._initialize_single_target(sample, body_box, start_frame_idx=0)
```

- [ ] **Step 6: Re-run the targeted refined tests**

Run: `python -m unittest tests.refined.test_offline_app_refined -v`  
Expected: PASS for the new clip-package tests and no regressions in existing refined-path tests.

- [ ] **Step 7: Commit**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "feat: add clip-package input mode for refined pipeline"
```

### Task 4: Reuse Face-Clip IDs In Wan Export

**Files:**
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\wan_sample_export.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\offline_app_refined.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\export\test_wan_sample_export.py`
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\refined\test_offline_app_refined.py`

- [ ] **Step 1: Write the failing Wan-export ID reuse test**

```python
def test_wan_export_reuses_clip_id_for_target_directory(self):
    ...
    self.assertEqual(sample_dirs, [os.path.join(export_root, "sampleuuid_face01_seg001_target1")])
```

- [ ] **Step 2: Run the ID reuse test to verify it fails**

Run: `python -m unittest tests.export.test_wan_sample_export.WanSampleExportTests.test_wan_export_reuses_clip_id_for_target_directory -v`  
Expected: FAIL because the exporter still allocates or resolves a sample-level UUID on its own.

- [ ] **Step 3: Add optional pre-resolved identity inputs to `WanSampleExporter`**

```python
WanSampleExporter(
    ...,
    sample_uuid=clip_meta.get("sample_uuid"),
    clip_id=clip_meta.get("clip_id"),
)
```

- [ ] **Step 4: Make exported target directories and metadata prefer `clip_id`**

```python
def _resolve_sample_dir(self, track_id: int) -> str:
    if self.export_root and self.clip_id:
        return os.path.join(self.export_root, f"{self.clip_id}_target{int(track_id)}")
```

- [ ] **Step 5: Update summary payloads to record both `sample_uuid` and `clip_id`**

```python
update_wan_sample_summary(
    metadata_root,
    sample_uuid,
    {
        "sample_uuid": sample_uuid,
        "clip_id": self.clip_id,
        "exported_targets": written_targets,
    },
)
```

- [ ] **Step 6: Run Wan-export and refined regression tests**

Run: `python -m unittest tests.export.test_wan_sample_export tests.refined.test_offline_app_refined -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/wan_sample_export.py scripts/offline_app_refined.py tests/export/test_wan_sample_export.py tests/refined/test_offline_app_refined.py
git commit -m "feat: reuse face-first clip ids in wan export"
```

### Task 5: End-To-End Verification

**Files:**
- Modify: `e:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\tests\export\test_wan_export_integration.py`

- [ ] **Step 1: Add a lightweight integration test for the face-first path**

```python
def test_face_first_clip_package_can_flow_into_wan_export_contract(self):
    ...
    self.assertTrue(os.path.isfile(os.path.join(final_target_dir, "target.mp4")))
    self.assertEqual(meta["clip_id"], "sampleuuid_face01_seg001")
```

- [ ] **Step 2: Run the integration test to verify it fails before the final wiring is complete**

Run: `python -m unittest tests.export.test_wan_export_integration -v`  
Expected: FAIL until the face-first path is fully wired.

- [ ] **Step 3: Complete any remaining glue needed for the integration flow**

```python
if face_clip_cfg.enable:
    clip_packages = extract_face_clip_packages(...)
    for clip_dir in clip_packages:
        self.run_sample(clip_dir, output_dir=None, skip_existing=skip_existing)
```

- [ ] **Step 4: Run the focused test suites**

Run: `python -m unittest tests.export.test_face_clip_pipeline tests.export.test_wan_sample_export tests.export.test_wan_export_integration tests.refined.test_offline_app_refined -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/export/test_wan_export_integration.py scripts/offline_app_refined.py
git commit -m "feat: wire face-first clips into wan export flow"
```
