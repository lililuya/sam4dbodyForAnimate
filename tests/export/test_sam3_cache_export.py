import copy
import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

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
    def test_start_sam3_export_session_resets_tracking_and_rotates_output_dir(self):
        from scripts.sam3_cache_export import start_sam3_export_session

        with make_workspace_tempdir() as tmpdir:
            runtime = {
                "batch_size": 8,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "video_fps": 24.0,
                "prompt_log": {"old": "value"},
                "frame_metrics": [{"frame_stem": "00000000"}],
                "events": [{"type": "old_event"}],
                "session_output_dir": "stale",
                "mask_generation_completed": True,
            }

            output_dir = start_sam3_export_session(
                runtime=runtime,
                output_root=tmpdir,
                source_video="video_b.mp4",
                id_factory=lambda: "session_b",
            )

        self.assertEqual(output_dir, os.path.join(tmpdir, "session_b"))
        self.assertEqual(runtime["prompt_log"], {})
        self.assertEqual(runtime["frame_metrics"], [])
        self.assertEqual(runtime["events"], [{"type": "video_loaded", "source_video": "video_b.mp4"}])
        self.assertEqual(runtime["session_video_path"], "video_b.mp4")
        self.assertEqual(runtime["session_output_dir"], output_dir)
        self.assertFalse(runtime["mask_generation_completed"])

    def test_ensure_sam3_export_ready_rejects_stale_or_incomplete_session(self):
        from scripts.sam3_cache_export import ensure_sam3_export_ready

        with make_workspace_tempdir() as tmpdir:
            output_dir = os.path.join(tmpdir, "session_a")
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
            runtime = {
                "session_video_path": "video_a.mp4",
                "session_output_dir": output_dir,
                "mask_generation_completed": False,
            }

            with self.assertRaisesRegex(ValueError, "Run Mask Generation"):
                ensure_sam3_export_ready(
                    runtime=runtime,
                    video_path="video_a.mp4",
                    output_dir=output_dir,
                )

            runtime["mask_generation_completed"] = True
            with self.assertRaisesRegex(ValueError, "current video session"):
                ensure_sam3_export_ready(
                    runtime=runtime,
                    video_path="video_b.mp4",
                    output_dir=output_dir,
                )

    def test_build_frame_metrics_from_video_segments_reports_mask_area_and_bbox(self):
        import numpy as np

        from scripts.sam3_cache_export import build_frame_metrics_from_video_segments

        video_segments = {
            0: {
                1: np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
                2: np.array([[[0, 0], [0, 0]]], dtype=np.uint8),
            }
        }

        frame_metrics = build_frame_metrics_from_video_segments(video_segments)

        self.assertEqual(frame_metrics[0]["frame_idx"], 0)
        self.assertEqual(frame_metrics[0]["frame_stem"], "00000000")
        self.assertEqual(frame_metrics[0]["track_metrics"]["1"]["mask_area"], 2)
        self.assertEqual(frame_metrics[0]["track_metrics"]["1"]["bbox_xyxy"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(frame_metrics[0]["track_metrics"]["2"]["mask_area"], 0)
        self.assertIsNone(frame_metrics[0]["track_metrics"]["2"]["bbox_xyxy"])

    def test_record_prompt_update_invalidates_completed_masks_and_clears_metrics(self):
        import numpy as np

        from scripts.sam3_cache_export import record_prompt_update

        runtime = {
            "prompt_log": {},
            "frame_metrics": [{"frame_stem": "00000000"}],
            "events": [{"type": "mask_generation_completed", "frame_count": 1}],
            "mask_generation_completed": True,
        }

        record_prompt_update(
            runtime,
            obj_id=2,
            frame_idx=7,
            input_point=np.array([[11, 13], [17, 19]], dtype=np.int32),
            input_label=np.array([1, 0], dtype=np.int32),
        )

        self.assertFalse(runtime["mask_generation_completed"])
        self.assertEqual(runtime["frame_metrics"], [])
        self.assertEqual(
            runtime["prompt_log"]["2"]["frames"]["7"],
            {"points": [[11, 13], [17, 19]], "labels": [1, 0]},
        )
        self.assertEqual(runtime["events"][-1]["type"], "prompt_updated")
        self.assertEqual(runtime["events"][-1]["obj_id"], 2)

    def test_build_runtime_export_state_preserves_prompt_log_events_and_frame_metrics(self):
        from scripts.sam3_cache_export import build_runtime_export_state

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
                    "frames": {"0": {"points": [[1, 2]], "labels": [1]}},
                }
            },
            "frame_metrics": [
                {"frame_stem": "00000000", "track_metrics": {"1": {"mask_area": 4}}}
            ],
            "events": [{"type": "target_added", "obj_id": 1}],
        }

        payload = build_runtime_export_state(runtime)

        self.assertEqual(payload["out_obj_ids"], [1])
        self.assertEqual(payload["runtime_profile"]["batch_size"], 8)
        self.assertEqual(payload["runtime_profile"]["detection_resolution"], [256, 512])
        self.assertEqual(payload["runtime_profile"]["completion_resolution"], [512, 1024])
        self.assertEqual(payload["runtime_profile"]["smpl_export"], False)
        self.assertEqual(payload["runtime_profile"]["fps"], 24.0)
        self.assertEqual(payload["prompt_log"]["1"]["frames"]["0"]["labels"], [1])
        self.assertEqual(payload["frame_metrics"][0]["track_metrics"]["1"]["mask_area"], 4)
        self.assertEqual(payload["events"][0]["type"], "target_added")

    def test_export_sam3_cache_writes_contract_files_and_runtime_payloads(self):
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            images_dir = os.path.join(working_dir, "images")
            masks_dir = os.path.join(working_dir, "masks")
            cache_root = os.path.join(tmpdir, "cache")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(cache_root, exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(images_dir, "00000000.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(masks_dir, "00000000.png")
            )

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
                        "frames": {
                            "0": {"points": [[12, 18], [8, 6]], "labels": [1, 0]},
                            "1": {"points": [[4, 5]], "labels": [1]},
                        },
                    }
                },
                "frame_metrics": [
                    {
                        "frame_stem": "00000000",
                        "track_metrics": {
                            "1": {"bbox_xyxy": [0, 0, 3, 3], "mask_area": 4}
                        },
                    }
                ],
                "events": [
                    {
                        "type": "mask_generation_completed",
                        "frame_count": 1,
                        "task_id": "task-1",
                    }
                ],
                "session_debug": {"token": "secret"},
            }
            expected_prompts = {"targets": copy.deepcopy(runtime["prompt_log"])}
            expected_frame_metrics = copy.deepcopy(runtime["frame_metrics"])
            expected_events = copy.deepcopy(runtime["events"])

            cache_dir = export_sam3_cache(
                working_dir=working_dir,
                cache_root=cache_root,
                sample_id="sample_001",
                source_video="inputs/source.mp4",
                runtime=runtime,
                config_path="configs/body4d.yaml",
            )
            runtime["frame_metrics"][0]["track_metrics"]["1"]["mask_area"] = 999
            runtime["events"][0]["type"] = "mutated_after_export"

            meta_path = os.path.join(cache_dir, "meta.json")
            prompts_path = os.path.join(cache_dir, "prompts.json")
            frame_metrics_path = os.path.join(cache_dir, "frame_metrics.json")
            events_path = os.path.join(cache_dir, "events.json")

            self.assertTrue(os.path.isfile(meta_path))
            self.assertTrue(os.path.isfile(prompts_path))
            self.assertTrue(os.path.isfile(frame_metrics_path))
            self.assertTrue(os.path.isfile(events_path))

            with open(prompts_path, "r", encoding="utf-8") as handle:
                prompts_data = json.load(handle)
            self.assertEqual(prompts_data, expected_prompts)

            with open(meta_path, "r", encoding="utf-8") as handle:
                meta_data = json.load(handle)
            with open(frame_metrics_path, "r", encoding="utf-8") as handle:
                frame_metrics_data = json.load(handle)
            with open(events_path, "r", encoding="utf-8") as handle:
                events_data = json.load(handle)
            self.assertEqual(meta_data["frame_count"], 1)
            self.assertEqual(meta_data["obj_ids"], [1])
            self.assertEqual(meta_data["fps"], 24.0)
            self.assertEqual(meta_data["config_path"], "configs/body4d.yaml")
            self.assertNotIn("session_debug", meta_data)
            self.assertEqual(frame_metrics_data, expected_frame_metrics)
            self.assertEqual(events_data, expected_events)

    def test_validate_cache_dir_rejects_missing_traceability_payloads(self):
        from scripts.sam3_cache_contract import validate_cache_dir
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            images_dir = os.path.join(working_dir, "images")
            masks_dir = os.path.join(working_dir, "masks")
            cache_root = os.path.join(tmpdir, "cache")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(cache_root, exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(images_dir, "00000000.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(masks_dir, "00000000.png")
            )

            cache_dir = export_sam3_cache(
                working_dir=working_dir,
                cache_root=cache_root,
                sample_id="sample_001",
                source_video="inputs/source.mp4",
                runtime={
                    "out_obj_ids": [1],
                    "batch_size": 8,
                    "detection_resolution": [256, 512],
                    "completion_resolution": [512, 1024],
                    "smpl_export": False,
                    "video_fps": 24.0,
                },
                config_path="configs/body4d.yaml",
            )

            os.remove(os.path.join(cache_dir, "prompts.json"))
            ok, errors = validate_cache_dir(cache_dir)

        self.assertFalse(ok)
        self.assertTrue(any("prompts.json" in error for error in errors))

    def test_export_sam3_cache_replaces_stale_cache_contents_for_same_sample_id(self):
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            images_dir = os.path.join(working_dir, "images")
            masks_dir = os.path.join(working_dir, "masks")
            cache_root = os.path.join(tmpdir, "cache")
            cache_dir = os.path.join(cache_root, "sample_001")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "masks"), exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(images_dir, "00000000.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(masks_dir, "00000000.png")
            )
            Image.new("RGB", (4, 4), color=(0, 0, 255)).save(
                os.path.join(cache_dir, "images", "00000001.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(cache_dir, "masks", "00000001.png")
            )
            with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({"frame_stems": ["00000001"]}, handle)
            with open(os.path.join(cache_dir, "prompts.json"), "w", encoding="utf-8") as handle:
                json.dump({"targets": {}}, handle)
            with open(os.path.join(cache_dir, "frame_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with open(os.path.join(cache_dir, "events.json"), "w", encoding="utf-8") as handle:
                json.dump([], handle)

            exported_cache_dir = export_sam3_cache(
                working_dir=working_dir,
                cache_root=cache_root,
                sample_id="sample_001",
                source_video="inputs/source.mp4",
                runtime={
                    "out_obj_ids": [1],
                    "batch_size": 8,
                    "detection_resolution": [256, 512],
                    "completion_resolution": [512, 1024],
                    "smpl_export": False,
                    "video_fps": 24.0,
                },
                config_path="configs/body4d.yaml",
            )

            with open(os.path.join(exported_cache_dir, "meta.json"), "r", encoding="utf-8") as handle:
                meta_data = json.load(handle)
            stale_image_exists = os.path.exists(
                os.path.join(exported_cache_dir, "images", "00000001.jpg")
            )
            stale_mask_exists = os.path.exists(
                os.path.join(exported_cache_dir, "masks", "00000001.png")
            )

        self.assertEqual(meta_data["frame_count"], 1)
        self.assertEqual(meta_data["frame_stems"], ["00000000"])
        self.assertFalse(stale_image_exists)
        self.assertFalse(stale_mask_exists)

    def test_export_sam3_cache_raises_when_export_is_invalid(self):
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            images_dir = os.path.join(working_dir, "images")
            cache_root = os.path.join(tmpdir, "cache")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(cache_root, exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(images_dir, "00000000.jpg")
            )

            with self.assertRaisesRegex(ValueError, "exported cache is invalid"):
                export_sam3_cache(
                    working_dir=working_dir,
                    cache_root=cache_root,
                    sample_id="sample_001",
                    source_video="inputs/source.mp4",
                    runtime={
                        "out_obj_ids": [1],
                        "batch_size": 8,
                        "detection_resolution": [256, 512],
                        "completion_resolution": [512, 1024],
                        "smpl_export": False,
                        "video_fps": 24.0,
                    },
                    config_path="configs/body4d.yaml",
                )

    def test_export_sam3_cache_rejects_unsafe_sample_id(self):
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            working_dir = os.path.join(tmpdir, "working")
            images_dir = os.path.join(working_dir, "images")
            masks_dir = os.path.join(working_dir, "masks")
            cache_root = os.path.join(tmpdir, "cache")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(cache_root, exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(images_dir, "00000000.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(masks_dir, "00000000.png")
            )

            with self.assertRaisesRegex(ValueError, "unsafe sample_id"):
                export_sam3_cache(
                    working_dir=working_dir,
                    cache_root=cache_root,
                    sample_id="..\\other_sample",
                    source_video="inputs/source.mp4",
                    runtime={
                        "out_obj_ids": [1],
                        "batch_size": 8,
                        "detection_resolution": [256, 512],
                        "completion_resolution": [512, 1024],
                        "smpl_export": False,
                        "video_fps": 24.0,
                    },
                    config_path="configs/body4d.yaml",
                )

    def test_export_sam3_cache_keeps_last_valid_cache_when_rerun_fails(self):
        from scripts.sam3_cache_export import export_sam3_cache

        with make_workspace_tempdir() as tmpdir:
            cache_root = os.path.join(tmpdir, "cache")
            valid_working_dir = os.path.join(tmpdir, "working_valid")
            invalid_working_dir = os.path.join(tmpdir, "working_invalid")
            for base_dir in (valid_working_dir, invalid_working_dir):
                os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, "masks"), exist_ok=True)
            os.makedirs(cache_root, exist_ok=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
                os.path.join(valid_working_dir, "images", "00000000.jpg")
            )
            Image.new("L", (4, 4), color=255).save(
                os.path.join(valid_working_dir, "masks", "00000000.png")
            )
            cache_dir = export_sam3_cache(
                working_dir=valid_working_dir,
                cache_root=cache_root,
                sample_id="sample_001",
                source_video="inputs/source.mp4",
                runtime={
                    "out_obj_ids": [1],
                    "batch_size": 8,
                    "detection_resolution": [256, 512],
                    "completion_resolution": [512, 1024],
                    "smpl_export": False,
                    "video_fps": 24.0,
                },
                config_path="configs/body4d.yaml",
            )

            Image.new("RGB", (4, 4), color=(0, 255, 0)).save(
                os.path.join(invalid_working_dir, "images", "00000000.jpg")
            )

            with self.assertRaisesRegex(ValueError, "exported cache is invalid"):
                export_sam3_cache(
                    working_dir=invalid_working_dir,
                    cache_root=cache_root,
                    sample_id="sample_001",
                    source_video="inputs/source.mp4",
                    runtime={
                        "out_obj_ids": [1],
                        "batch_size": 8,
                        "detection_resolution": [256, 512],
                        "completion_resolution": [512, 1024],
                        "smpl_export": False,
                        "video_fps": 24.0,
                    },
                    config_path="configs/body4d.yaml",
                )

            with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as handle:
                meta_data = json.load(handle)
            mask_still_exists = os.path.isfile(
                os.path.join(cache_dir, "masks", "00000000.png")
            )

        self.assertEqual(meta_data["frame_stems"], ["00000000"])
        self.assertTrue(mask_still_exists)


if __name__ == "__main__":
    unittest.main()
