import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf
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


class WanExportIntegrationTests(unittest.TestCase):
    def test_offline_app_refined_syncs_wan_export_runtime(self):
        import scripts.offline_app_refined as offline_app_refined

        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "yolo"},
                "tracking": {"chunk_size": 120},
                "sam_3d_body": {"batch_size": 16},
                "completion": {
                    "detection_resolution": [192, 384],
                    "completion_resolution": [256, 512],
                },
                "wan_export": {
                    "enable": True,
                    "fps": 25,
                    "resolution_area": [512, 768],
                    "face_resolution": [512, 512],
                },
            }
        )
        app = offline_app_refined.RefinedOfflineApp("configs/body4d_refined.yaml", config=cfg)
        runtime_app = MagicMock()
        runtime_app.RUNTIME = {}
        runtime_app.sam3_3d_body_model = MagicMock()
        runtime_app.sam3_3d_body_model.device = "cuda"
        runtime_app.sam3_3d_body_model.detector = None

        with patch.object(app, "_configure_detector"):
            app._sync_base_app_runtime(runtime_app, "./sample_out")

        self.assertEqual(runtime_app.RUNTIME["wan_export"]["enable"], True)
        self.assertEqual(runtime_app.RUNTIME["wan_export"]["fps"], 25)

    def test_run_4d_pipeline_calls_frame_writer_finalize(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        class _RecordingWriter:
            def __init__(self):
                self.calls = 0
                self.finalized = 0

            def __call__(self, image_path, mask_output, id_current):
                del image_path, mask_output, id_current
                self.calls += 1
                return None

            def finalize(self):
                self.finalized += 1
                return []

        with make_workspace_tempdir() as temp_dir:
            input_dir = os.path.join(temp_dir, "cache")
            output_dir = os.path.join(temp_dir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(input_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(
                os.path.join(input_dir, "masks", "00000000.png")
            )

            writer = _RecordingWriter()
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime={
                    "out_obj_ids": [1],
                    "batch_size": 1,
                    "detection_resolution": [192, 384],
                    "completion_resolution": [256, 512],
                    "save_rendered_frames": False,
                    "save_rendered_frames_individual": False,
                    "save_mesh_4d_individual": False,
                    "save_focal_4d_individual": False,
                },
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
                frame_writer=writer,
            )

            with patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                return_value=([[{"bbox": np.zeros((4,), dtype=np.float32)}]], [[1]], []),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[np.zeros((4, 4, 3), dtype=np.uint8)],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ), patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                run_4d_pipeline_from_context(context)

            self.assertEqual(writer.calls, 1)
            self.assertEqual(writer.finalized, 1)


if __name__ == "__main__":
    unittest.main()
