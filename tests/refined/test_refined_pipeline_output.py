import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

from omegaconf import OmegaConf


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_refined_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class RefinedOutputLayoutTests(unittest.TestCase):
    def test_prepare_sample_output_honors_runtime_debug_override(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "body4d_refined.yaml")
        )
        runtime_cfg = OmegaConf.create({"debug": {"save_metrics": False}})

        with make_workspace_tempdir() as tmpdir:
            app = RefinedOfflineApp(config_path, config=runtime_cfg)
            output_paths = app.prepare_sample_output(tmpdir, [7])
            app.chunk_records.append({"chunk_id": 0, "frame_count": 8})
            app.finalize_sample()

            self.assertNotIn("debug_metrics", output_paths)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "debug_metrics")))

    def test_prepare_output_dirs_creates_refined_directories(self):
        from scripts.offline_app_refined import prepare_output_dirs

        with make_workspace_tempdir() as tmpdir:
            prepare_output_dirs(tmpdir, [1, 2], save_debug_metrics=True)
            expected = [
                "images",
                "masks_raw",
                "masks_refined",
                os.path.join("completion_refined", "images"),
                os.path.join("completion_refined", "masks"),
                "rendered_frames",
                os.path.join("debug_metrics"),
                os.path.join("mesh_4d_individual", "1"),
                os.path.join("mesh_4d_individual", "2"),
                os.path.join("focal_4d_individual", "1"),
                os.path.join("focal_4d_individual", "2"),
                os.path.join("rendered_frames_individual", "1"),
                os.path.join("rendered_frames_individual", "2"),
            ]
            for rel_path in expected:
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, rel_path)), rel_path)

    def test_write_chunk_manifest_persists_json_records(self):
        from scripts.offline_app_refined import write_chunk_manifest

        chunk_records = [
            {"chunk_id": 0, "start_frame": 0, "end_frame": 31},
            {"chunk_id": 1, "start_frame": 32, "end_frame": 63},
        ]
        with make_workspace_tempdir() as tmpdir:
            debug_dir = os.path.join(tmpdir, "debug_metrics")
            write_chunk_manifest(debug_dir, chunk_records)

            manifest_path = os.path.join(debug_dir, "chunk_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), chunk_records)

    def test_finalize_sample_writes_chunk_manifest_when_debug_enabled(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "body4d_refined.yaml")
        )
        with make_workspace_tempdir() as tmpdir:
            app = RefinedOfflineApp(config_path)
            app.prepare_sample_output(tmpdir, [3])
            app.chunk_records.extend(
                [
                    {"chunk_id": 0, "frame_count": 24},
                    {"chunk_id": 1, "frame_count": 18},
                ]
            )

            app.finalize_sample()

            manifest_path = os.path.join(tmpdir, "debug_metrics", "chunk_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), app.chunk_records)

    def test_finalize_sample_uses_passed_runtime_config_over_file_defaults(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        with make_workspace_tempdir() as tmpdir:
            config_path = os.path.join(tmpdir, "body4d_refined_disabled.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("debug:\n  save_metrics: false\n")

            runtime_cfg = OmegaConf.create({"debug": {"save_metrics": True}})
            app = RefinedOfflineApp(config_path, config=runtime_cfg)
            app.prepare_sample_output(tmpdir, [5])
            app.chunk_records.append({"chunk_id": 0, "frame_count": 12})

            app.finalize_sample()

            manifest_path = os.path.join(tmpdir, "debug_metrics", "chunk_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))


if __name__ == "__main__":
    unittest.main()
