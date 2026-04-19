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
                "fps": 24.0,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
            },
            config_path="configs/body4d.yaml",
        )

        self.assertEqual(meta["cache_version"], 1)
        self.assertEqual(meta["sample_id"], "demo")
        self.assertEqual(meta["source_video"], "/tmp/demo.mp4")
        self.assertEqual(meta["frame_count"], 2)
        self.assertEqual(meta["frame_stems"], ["00000000", "00000001"])
        self.assertEqual(meta["image_size"], {"width": 1280, "height": 720})
        self.assertEqual(meta["image_ext"], ".jpg")
        self.assertEqual(meta["mask_ext"], ".png")
        self.assertEqual(meta["obj_ids"], [1, 2])
        self.assertIn("exported_at", meta)
        self.assertIsInstance(meta["exported_at"], str)
        self.assertTrue(meta["exported_at"])
        self.assertEqual(meta["fps"], 24.0)
        self.assertEqual(meta["config_path"], "configs/body4d.yaml")
        self.assertNotIn("config", meta)
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

    def test_validate_cache_dir_rejects_unexpected_cache_version(self):
        from scripts.sam3_cache_contract import validate_cache_dir

        with make_workspace_tempdir() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
            with open(os.path.join(tmpdir, "meta.json"), "w", encoding="utf-8") as handle:
                handle.write(
                    '{"cache_version": 2, "frame_count": 0, "frame_stems": [], '
                    '"image_ext": ".jpg", "mask_ext": ".png", "obj_ids": [], '
                    '"runtime_profile": {"batch_size": 1, "detection_resolution": [256, 512], '
                    '"completion_resolution": [512, 1024], "smpl_export": false}}'
                )

            ok, errors = validate_cache_dir(tmpdir)

        self.assertFalse(ok)
        self.assertTrue(any("cache_version" in error for error in errors))

    def test_validate_cache_dir_handles_non_dict_json_root(self):
        from scripts.sam3_cache_contract import validate_cache_dir

        with make_workspace_tempdir() as tmpdir:
            with open(os.path.join(tmpdir, "meta.json"), "w", encoding="utf-8") as handle:
                handle.write('["not", "an", "object"]')

            ok, errors = validate_cache_dir(tmpdir)

        self.assertFalse(ok)
        self.assertTrue(any("top-level" in error for error in errors))

    def test_build_cache_meta_detaches_runtime_profile_nested_values(self):
        from scripts.sam3_cache_contract import build_cache_meta

        runtime_profile = {
            "batch_size": 16,
            "detection_resolution": [256, 512],
            "completion_resolution": [512, 1024],
            "smpl_export": False,
        }
        meta = build_cache_meta(
            sample_id="demo",
            source_video="/tmp/demo.mp4",
            frame_stems=["00000000"],
            image_size={"width": 1280, "height": 720},
            obj_ids=[1],
            runtime_profile=runtime_profile,
            config_path="configs/body4d.yaml",
        )

        runtime_profile["detection_resolution"][0] = 999
        runtime_profile["completion_resolution"][0] = 888

        self.assertEqual(meta["runtime_profile"]["detection_resolution"], [256, 512])
        self.assertEqual(meta["runtime_profile"]["completion_resolution"], [512, 1024])


if __name__ == "__main__":
    unittest.main()
