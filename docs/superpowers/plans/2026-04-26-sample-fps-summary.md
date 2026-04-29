# Sample FPS Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist sample-level FPS summary metadata for refined runs without changing inference or export behavior.

**Architecture:** Add a small sample-FPS resolver in `offline_app_refined.py`, attach an `fps_summary` object to `sample_summary`, and persist it into `sample_runtime.json`, `debug_metrics/sample_summary.json`, and external Wan summary updates. Cover both video-input and image-sequence cases with targeted unit tests written first.

**Tech Stack:** Python, `unittest`, OpenCV (`cv2`), existing refined offline runtime helpers, Wan summary helpers

---

### Task 1: Add Failing Tests For FPS Summary Payloads

**Files:**
- Modify: `tests/refined/test_offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Write the failing tests**

Add two tests to `tests/refined/test_offline_app_refined.py`:

```python
    def test_run_sample_writes_video_source_fps_into_runtime_and_wan_summary(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        with make_workspace_tempdir() as tmpdir:
            export_root = os.path.join(tmpdir, "WanExport")
            sample_output_dir = os.path.join(tmpdir, "sample_out")
            os.makedirs(sample_output_dir, exist_ok=True)

            config = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "tracking": {"chunk_size": 180},
                    "wan_export": {"enable": True, "output_dir": export_root, "fps": 25},
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                }
            )
            app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
            sample = {
                "frames": [],
                "output_dir": sample_output_dir,
                "input_video": os.path.join(tmpdir, "sample.mp4"),
                "input_type": "video",
            }

            app.prepare_input = unittest.mock.MagicMock(return_value=sample)
            app.detect_initial_targets = unittest.mock.MagicMock(return_value={"obj_ids": [1]})
            app.prepare_sample_output = unittest.mock.MagicMock(
                return_value={"images": os.path.join(sample_output_dir, "images")}
            )
            app.iter_chunks = unittest.mock.MagicMock(return_value=[])
            app.track_chunk = unittest.mock.MagicMock()
            app.refine_chunk_masks = unittest.mock.MagicMock()
            app.maybe_reprompt_chunk = unittest.mock.MagicMock()
            app.write_chunk_outputs = unittest.mock.MagicMock()
            app.run_refined_4d_generation = unittest.mock.MagicMock()

            with patch("scripts.offline_app_refined.time.perf_counter", side_effect=[10.0, 16.5]), patch(
                "scripts.wan_sample_export.uuid.uuid4",
                return_value=SimpleNamespace(hex="runtimeuuid123456"),
            ), patch(
                "scripts.offline_app_refined.resolve_video_fps",
                return_value=29.97,
            ):
                app.run_sample("sample.mp4", "./custom_out", skip_existing=False, runtime_profile=None)

            runtime_path = os.path.join(sample_output_dir, "sample_runtime.json")
            with open(runtime_path, "r", encoding="utf-8") as handle:
                runtime_payload = json.load(handle)

            self.assertEqual(
                runtime_payload["fps_summary"],
                {
                    "source_fps": 29.97,
                    "source_fps_source": "video_metadata",
                    "rendered_4d_fps": 25.0,
                    "wan_target_fps": 25,
                },
            )

            summary_path = os.path.join(export_root, "runtimeuuid123456_summary.json")
            with open(summary_path, "r", encoding="utf-8") as handle:
                wan_summary = json.load(handle)

            self.assertEqual(wan_summary["fps_summary"], runtime_payload["fps_summary"])

    def test_run_sample_marks_image_sequence_source_fps_as_unavailable(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        with make_workspace_tempdir() as tmpdir:
            sample_output_dir = os.path.join(tmpdir, "sample_out")
            os.makedirs(sample_output_dir, exist_ok=True)

            config = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "tracking": {"chunk_size": 180},
                    "wan_export": {"enable": False},
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                }
            )
            app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
            sample = {
                "frames": [],
                "output_dir": sample_output_dir,
                "input_video": os.path.join(tmpdir, "frames_dir"),
                "input_type": "images",
            }

            app.prepare_input = unittest.mock.MagicMock(return_value=sample)
            app.detect_initial_targets = unittest.mock.MagicMock(return_value={"obj_ids": [1]})
            app.prepare_sample_output = unittest.mock.MagicMock(
                return_value={"images": os.path.join(sample_output_dir, "images")}
            )
            app.iter_chunks = unittest.mock.MagicMock(return_value=[])
            app.track_chunk = unittest.mock.MagicMock()
            app.refine_chunk_masks = unittest.mock.MagicMock()
            app.maybe_reprompt_chunk = unittest.mock.MagicMock()
            app.write_chunk_outputs = unittest.mock.MagicMock()
            app.run_refined_4d_generation = unittest.mock.MagicMock()

            with patch("scripts.offline_app_refined.time.perf_counter", side_effect=[20.0, 22.0]):
                app.run_sample("frames_dir", "./custom_out", skip_existing=False, runtime_profile=None)

            runtime_path = os.path.join(sample_output_dir, "sample_runtime.json")
            with open(runtime_path, "r", encoding="utf-8") as handle:
                runtime_payload = json.load(handle)

            self.assertEqual(
                runtime_payload["fps_summary"],
                {
                    "source_fps": None,
                    "source_fps_source": "image_sequence",
                    "rendered_4d_fps": 25.0,
                    "wan_target_fps": None,
                },
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_writes_video_source_fps_into_runtime_and_wan_summary tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_marks_image_sequence_source_fps_as_unavailable -v
```

Expected: FAIL because `sample_runtime.json` does not yet include `fps_summary`, and `offline_app_refined` does not yet provide `resolve_video_fps`.

- [ ] **Step 3: Write minimal implementation**

Do not start implementation until the two tests fail for the expected reason.

- [ ] **Step 4: Run test to verify it passes**

Re-run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_writes_video_source_fps_into_runtime_and_wan_summary tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_marks_image_sequence_source_fps_as_unavailable -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/refined/test_offline_app_refined.py scripts/offline_app_refined.py
git commit -m "feat: record sample fps summaries"
```

### Task 2: Implement FPS Resolution And Runtime Persistence

**Files:**
- Modify: `scripts/offline_app_refined.py`
- Modify: `tests/refined/test_offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Add a focused source-FPS resolver**

Add a small helper near `count_video_frames()` in `scripts/offline_app_refined.py`:

```python
def resolve_video_fps(video_path: str) -> Optional[float]:
    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps > 0:
        return fps
    return None
```

- [ ] **Step 2: Add a sample-level FPS summary helper**

Add a helper method on `RefinedOfflineApp`:

```python
    def _build_sample_fps_summary(self, sample: dict) -> dict:
        input_type = str(sample.get("input_type") or "")
        source_fps = None
        source_fps_source = "unavailable"
        if input_type == "video":
            source_fps = resolve_video_fps(str(sample.get("input_video") or ""))
            source_fps_source = "video_metadata" if source_fps is not None else "unavailable"
        elif input_type == "images":
            source_fps_source = "image_sequence"

        wan_export_cfg = to_plain_runtime_dict(cfg_get(self.CONFIG, "wan_export", {}))
        wan_target_fps = None
        if bool(wan_export_cfg.get("enable", False)):
            wan_target_fps = int(wan_export_cfg.get("fps", 25))

        return {
            "source_fps": source_fps,
            "source_fps_source": source_fps_source,
            "rendered_4d_fps": float(getattr(self, "RUNTIME", {}).get("video_fps", 25) or 25),
            "wan_target_fps": wan_target_fps,
        }
```

- [ ] **Step 3: Attach the FPS summary during `run_sample()`**

After `sample = self.prepare_input(...)`, attach the new payload:

```python
            self.sample_summary["fps_summary"] = self._build_sample_fps_summary(sample)
```

- [ ] **Step 4: Include FPS summary in runtime persistence**

Update `_persist_sample_runtime()` so `runtime_payload` includes:

```python
        runtime_payload["fps_summary"] = copy.deepcopy(self.sample_summary.get("fps_summary", {}))
```

Also include the same payload in the Wan summary update:

```python
                    "pipeline_runtime": runtime_payload,
                    "fps_summary": copy.deepcopy(self.sample_summary.get("fps_summary", {})),
```

- [ ] **Step 5: Keep debug summary behavior unchanged except for the new field**

Do not add extra files. Rely on the existing `sample_summary.json` write path so the added `fps_summary` rides along naturally.

- [ ] **Step 6: Run the focused tests again**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_writes_video_source_fps_into_runtime_and_wan_summary tests.refined.test_offline_app_refined.RefinedCliTests.test_run_sample_marks_image_sequence_source_fps_as_unavailable -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "feat: persist sample fps summaries"
```

### Task 3: Run Regression Verification For Refined Runtime Metadata

**Files:**
- Modify: `tests/refined/test_offline_app_refined.py`
- Test: `tests/refined/test_offline_app_refined.py`

- [ ] **Step 1: Run the broader refined test group**

Run:

```bash
python -m unittest tests.refined.test_offline_app_refined -v
```

Expected: PASS with no regressions in existing refined runtime behavior.

- [ ] **Step 2: Run the existing Wan export and refined metadata regression suite**

Run:

```bash
python -m unittest tests.export.test_wan_reference_compat tests.export.test_wan_sample_export tests.refined.test_offline_app_refined -v
```

Expected: PASS

- [ ] **Step 3: Review plan coverage**

Confirm the implementation now satisfies the spec requirements:
- source video FPS is recorded when available
- image-sequence source FPS is explicitly unavailable
- sample runtime includes the new `fps_summary`
- external Wan summary includes the same `fps_summary`
- no resampling or runtime behavior change was introduced

- [ ] **Step 4: Commit**

```bash
git add scripts/offline_app_refined.py tests/refined/test_offline_app_refined.py
git commit -m "test: verify fps summary reporting"
```
