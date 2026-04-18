import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
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


class OfflineAppExportCliTests(unittest.TestCase):
    def test_parser_accepts_export_arguments(self):
        from scripts import offline_app_export

        parser = offline_app_export.build_parser()
        args = parser.parse_args(
            [
                "--input_video",
                "sample.mp4",
                "--output_dir",
                "./export_out",
                "--mask_video_fps",
                "12",
            ]
        )

        self.assertEqual(args.input_video, "sample.mp4")
        self.assertEqual(args.output_dir, "./export_out")
        self.assertEqual(args.mask_video_fps, 12)

    def test_resolve_mask_video_fps_prefers_override_and_falls_back_to_default(self):
        from scripts.offline_app_export import resolve_mask_video_fps

        self.assertEqual(resolve_mask_video_fps("sample.mp4", override_fps=9), 9)
        self.assertEqual(resolve_mask_video_fps("frames_dir", override_fps=None, default_fps=25), 25)

    def test_run_export_pipeline_calls_mask_generation_video_export_and_4d_generation(self):
        from scripts.offline_app_export import run_export_pipeline

        app = MagicMock()
        app.OUTPUT_DIR = "./sample_out"
        app.RUNTIME = {"out_obj_ids": [3, 1]}

        with patch("scripts.offline_app_export.export_binary_mask_videos") as mock_export_binary_mask_videos:
            run_export_pipeline(app, "sample.mp4", fps=15)

        app.on_mask_generation.assert_called_once_with(start_frame_idx=0)
        mock_export_binary_mask_videos.assert_called_once_with(
            os.path.join("./sample_out", "masks"),
            os.path.join("./sample_out", "mask_videos"),
            [3, 1],
            fps=15,
        )
        app.on_4d_generation.assert_called_once_with(video_path="sample.mp4")

    def test_inference_uses_lazy_base_module_loading(self):
        from scripts import offline_app_export

        fake_base_module = object()
        fake_app_class = MagicMock()
        fake_app = MagicMock()
        fake_app.RUNTIME = {"out_obj_ids": []}
        fake_app_class.return_value = fake_app
        args = SimpleNamespace(input_video="sample.mp4", output_dir="./custom_out", mask_video_fps=10)

        with patch.object(offline_app_export, "load_base_offline_module", return_value=fake_base_module), patch.object(
            offline_app_export, "build_export_app_class", return_value=fake_app_class
        ) as mock_build_export_app_class, patch.object(offline_app_export, "prepare_initial_tracking"), patch.object(
            offline_app_export, "run_export_pipeline"
        ) as mock_run_export_pipeline:
            offline_app_export.inference(args)

        mock_build_export_app_class.assert_called_once_with(fake_base_module)
        fake_app_class.assert_called_once_with()
        mock_run_export_pipeline.assert_called_once_with(fake_app, "sample.mp4", fps=10)
        self.assertEqual(fake_app.OUTPUT_DIR, "./custom_out")


class OfflineAppExportFrameWriterTests(unittest.TestCase):
    def test_export_override_writes_openpose_frames_and_video(self):
        from scripts.offline_app_export import build_export_app_class

        class FakeBaseModule:
            class OfflineApp:
                def __init__(self):
                    self.OUTPUT_DIR = ""
                    self.RUNTIME = {
                        "out_obj_ids": [1],
                        "batch_size": 1,
                        "detection_resolution": [256, 512],
                        "completion_resolution": [512, 1024],
                    }
                    self.pipeline_mask = None
                    self.depth_model = None
                    self.generator = None
                    self.sam3_3d_body_model = SimpleNamespace(
                        faces=np.array([[0, 1, 2]], dtype=np.int32)
                    )

                def on_4d_generation(self, video_path=None):
                    raise NotImplementedError("base implementation should be overridden")

        app_class = build_export_app_class(FakeBaseModule)
        app = app_class()

        with make_workspace_tempdir() as tmpdir:
            app.OUTPUT_DIR = tmpdir
            images_dir = os.path.join(tmpdir, "images")
            masks_dir = os.path.join(tmpdir, "masks")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(images_dir, "00000000.jpg")
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                os.path.join(masks_dir, "00000000.png")
            )

            mask_outputs = [
                [
                    {
                        "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
                    }
                ]
            ]
            id_batch = [[1]]

            with patch.object(app, "_write_openpose_frame") as mock_write_openpose_frame, patch(
                "scripts.offline_app_export.process_image_with_mask",
                return_value=(mask_outputs, id_batch, []),
                create=True,
            ), patch(
                "scripts.offline_app_export.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
                create=True,
            ), patch(
                "scripts.offline_app_export.visualize_sample",
                return_value=[np.zeros((4, 4, 3), dtype=np.uint8)],
                create=True,
            ), patch(
                "scripts.offline_app_export.save_mesh_results",
                create=True,
            ), patch(
                "scripts.offline_app_export.jpg_folder_to_mp4",
                create=True,
            ) as mock_jpg_folder_to_mp4, patch(
                "scripts.offline_app_export.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.offline_app_export.cv2.imwrite",
                return_value=True,
            ):
                out_path = app.on_4d_generation(video_path="sample.mp4")

        mock_write_openpose_frame.assert_called_once_with(
            os.path.join(images_dir, "00000000.jpg"),
            mask_outputs[0],
            id_batch[0],
        )
        self.assertTrue(out_path.endswith(".mp4"))
        self.assertIn("4d_", os.path.basename(out_path))
        mock_jpg_folder_to_mp4.assert_called_once()


if __name__ == "__main__":
    unittest.main()
