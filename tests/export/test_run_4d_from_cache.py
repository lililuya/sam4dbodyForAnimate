import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


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


class Run4DFromCacheTests(unittest.TestCase):
    def test_run_cache_sample_restores_runtime_profile_and_writes_to_outputs_root(self):
        from scripts.run_4d_from_cache import run_cache_sample

        with make_workspace_tempdir() as tmpdir:
            cache_dir = os.path.join(tmpdir, "sam3_cache", "demo")
            output_root = os.path.join(tmpdir, "outputs_4d")
            os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "masks"), exist_ok=True)
            with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cache_version": 1,
                        "sample_id": "demo",
                        "source_video": "input.mp4",
                        "frame_count": 1,
                        "fps": 24.0,
                        "image_ext": ".jpg",
                        "mask_ext": ".png",
                        "frame_stems": ["00000000"],
                        "image_size": {"width": 4, "height": 4},
                        "obj_ids": [1],
                        "runtime_profile": {
                            "batch_size": 8,
                            "detection_resolution": [256, 512],
                            "completion_resolution": [512, 1024],
                            "smpl_export": False,
                        },
                        "config_path": "configs/body4d.yaml",
                        "exported_at": "2026-04-19T12:00:00Z",
                    },
                    handle,
                )
            with open(os.path.join(cache_dir, "prompts.json"), "w", encoding="utf-8") as handle:
                json.dump({"targets": {}}, handle)
            with open(os.path.join(cache_dir, "frame_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with open(os.path.join(cache_dir, "events.json"), "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with open(os.path.join(cache_dir, "images", "00000000.jpg"), "wb") as handle:
                handle.write(b"x")
            with open(os.path.join(cache_dir, "masks", "00000000.png"), "wb") as handle:
                handle.write(b"x")

            runtime_app = MagicMock()
            runtime_app.sam3_3d_body_model = MagicMock()
            runtime_app.pipeline_mask = None
            runtime_app.pipeline_rgb = None
            runtime_app.depth_model = None
            runtime_app.predictor = MagicMock()
            runtime_app.generator = None

            with patch("scripts.run_4d_from_cache.build_runtime_app", return_value=runtime_app), patch(
                "scripts.run_4d_from_cache.run_4d_pipeline_from_context",
                return_value=os.path.join(output_root, "demo", "4d.mp4"),
            ) as mock_run:
                out_path = run_cache_sample(cache_dir=cache_dir, output_root=output_root, overwrite=False)

            self.assertEqual(out_path, os.path.join(output_root, "demo", "4d.mp4"))
            context = mock_run.call_args.args[0]
            self.assertEqual(context.input_dir, cache_dir)
            self.assertEqual(context.output_dir, os.path.join(output_root, "demo"))
            self.assertEqual(context.runtime["batch_size"], 8)
            self.assertEqual(context.runtime["completion_resolution"], [512, 1024])

    def test_prepare_output_dir_refuses_to_overwrite_without_flag(self):
        from scripts.run_4d_from_cache import prepare_output_dir

        with make_workspace_tempdir() as tmpdir:
            output_root = os.path.join(tmpdir, "outputs_4d")
            os.makedirs(os.path.join(output_root, "demo"), exist_ok=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_output_dir("demo", output_root=output_root, overwrite=False)

    def test_write_run_summary_persists_summary_json(self):
        from scripts.run_4d_from_cache import write_run_summary

        with make_workspace_tempdir() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            write_run_summary(
                output_dir=output_dir,
                summary={
                    "sample_id": "demo",
                    "cache_dir": os.path.join(tmpdir, "sam3_cache", "demo"),
                    "status": "completed",
                    "config_path": "configs/body4d.yaml",
                    "started_at": "2026-04-19T12:00:00Z",
                    "finished_at": "2026-04-19T12:01:00Z",
                    "output_video": os.path.join(output_dir, "4d.mp4"),
                    "error": None,
                },
            )

            with open(os.path.join(output_dir, "run_summary.json"), "r", encoding="utf-8") as handle:
                summary = json.load(handle)

        self.assertEqual(summary["sample_id"], "demo")
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["output_video"].endswith("4d.mp4"))

    def test_discover_cache_dirs_skips_directories_without_meta(self):
        from scripts.run_4d_from_cache import discover_cache_dirs

        with make_workspace_tempdir() as tmpdir:
            cache_root = os.path.join(tmpdir, "sam3_cache")
            valid_cache = os.path.join(cache_root, "demo")
            invalid_cache = os.path.join(cache_root, "missing_meta")
            os.makedirs(valid_cache, exist_ok=True)
            os.makedirs(invalid_cache, exist_ok=True)
            with open(os.path.join(valid_cache, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({"sample_id": "demo"}, handle)

            cache_dirs = discover_cache_dirs(cache_root)

        self.assertEqual(cache_dirs, [valid_cache])

    def test_run_cache_batch_collects_successes_and_failures(self):
        from scripts.run_4d_from_cache import run_cache_batch

        with patch(
            "scripts.run_4d_from_cache.discover_cache_dirs",
            return_value=["/tmp/cache_a", "/tmp/cache_b"],
        ), patch(
            "scripts.run_4d_from_cache.run_cache_sample",
            side_effect=["/tmp/out_a.mp4", RuntimeError("boom")],
        ):
            results = run_cache_batch("/tmp/sam3_cache", output_root="/tmp/outputs_4d")

        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[0]["output"], "/tmp/out_a.mp4")
        self.assertEqual(results[1]["status"], "failed")
        self.assertIn("boom", results[1]["error"])


if __name__ == "__main__":
    unittest.main()
