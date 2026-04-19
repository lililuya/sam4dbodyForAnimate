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
                        "frames": {"0": {"points": [[12, 18]], "labels": [1]}},
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
                "events": [{"type": "mask_generation_completed", "frame_count": 1}],
            }

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
            self.assertEqual(
                prompts_data["targets"]["1"]["frames"]["0"]["labels"],
                [1],
            )

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
            self.assertEqual(frame_metrics_data[0]["track_metrics"]["1"]["mask_area"], 4)
            self.assertEqual(events_data[0]["type"], "mask_generation_completed")

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


if __name__ == "__main__":
    unittest.main()
