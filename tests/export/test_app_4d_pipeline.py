import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
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


class App4DPipelineTests(unittest.TestCase):
    def test_run_4d_pipeline_uses_explicit_input_and_output_dirs(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(input_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                os.path.join(input_dir, "masks", "00000000.png")
            )

            runtime = {
                "out_obj_ids": [1],
                "batch_size": 1,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
                "video_fps": 24.0,
            }
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
            )

            with patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                return_value=([], [], [0]),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ), patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ) as mock_video, patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                out_path = run_4d_pipeline_from_context(context)
                mock_video.assert_called_once()
                self.assertEqual(os.path.dirname(out_path), output_dir)
                self.assertTrue(os.path.isdir(os.path.join(output_dir, "rendered_frames")))
                self.assertTrue(os.path.isdir(os.path.join(output_dir, "mesh_4d_individual", "1")))
                self.assertFalse(os.path.isdir(os.path.join(input_dir, "rendered_frames")))


if __name__ == "__main__":
    unittest.main()
