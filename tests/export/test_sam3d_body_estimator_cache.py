import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") is not None


class Sam3DBodyEstimatorCacheTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch is required for estimator cache test")
    def test_process_frames_reuses_cached_cam_int_for_same_frame_path(self):
        from models.sam_3d_body.sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
        import torch

        estimator = SAM3DBodyEstimator.__new__(SAM3DBodyEstimator)
        estimator.device = "cuda"
        estimator.model = MagicMock()
        estimator.detector = None
        estimator.sam = None
        estimator.fov_estimator = MagicMock()
        estimator.fov_estimator.get_cam_intrinsics.return_value = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        estimator.transform = MagicMock()
        estimator.transform_hand = MagicMock()
        estimator.thresh_wrist_angle = 1.4

        def fake_prepare_batch(*args, **kwargs):
            del args, kwargs
            return {
                "img": torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32),
                "img_ori": [SimpleNamespace(data=np.zeros((4, 4, 3), dtype=np.uint8))],
                "bbox": torch.zeros((1, 1, 4), dtype=torch.float32),
                "cam_int": torch.eye(3, dtype=torch.float32).unsqueeze(0),
                "bbox_format": "xyxy",
            }

        estimator.model.run_inference_batch.return_value = {
            "mhr": {
                "focal_length": torch.ones((2,), dtype=torch.float32),
                "pred_keypoints_3d": torch.zeros((2, 1, 3), dtype=torch.float32),
                "pred_keypoints_2d": torch.zeros((2, 1, 2), dtype=torch.float32),
                "pred_vertices": torch.zeros((2, 3, 3), dtype=torch.float32),
                "pred_cam_t": torch.zeros((2, 3), dtype=torch.float32),
                "pred_pose_raw": torch.zeros((2, 3), dtype=torch.float32),
                "global_rot": torch.zeros((2, 3), dtype=torch.float32),
                "body_pose": torch.zeros((2, 3), dtype=torch.float32),
                "hand": torch.zeros((2, 3), dtype=torch.float32),
                "scale": torch.zeros((2, 1), dtype=torch.float32),
                "shape": torch.zeros((2, 10), dtype=torch.float32),
                "face": torch.zeros((2, 10), dtype=torch.float32),
                "pred_joint_coords": torch.zeros((2, 1, 3), dtype=torch.float32),
                "joint_global_rots": torch.zeros((2, 1, 3, 3), dtype=torch.float32),
                "mhr_model_params": torch.zeros((2, 5), dtype=torch.float32),
            }
        }

        cam_int_cache = {}
        repeated_frame_path = "sample_frame.jpg"

        with patch(
            "models.sam_3d_body.sam_3d_body.sam_3d_body_estimator.load_image",
            return_value=np.zeros((4, 4, 3), dtype=np.uint8),
        ), patch(
            "models.sam_3d_body.sam_3d_body.sam_3d_body_estimator.prepare_batch",
            side_effect=fake_prepare_batch,
        ), patch(
            "models.sam_3d_body.sam_3d_body.sam_3d_body_estimator.recursive_to",
            side_effect=lambda value, *_args, **_kwargs: value,
        ):
            outputs = estimator.process_frames(
                img_list=[repeated_frame_path, repeated_frame_path],
                bboxes=[
                    np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
                    np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
                ],
                masks=[
                    np.ones((1, 4, 4), dtype=np.uint8),
                    np.ones((1, 4, 4), dtype=np.uint8),
                ],
                id_batch=[[1], [1]],
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict={},
                occ_dict=None,
                use_mask=True,
                inference_type="body",
                cam_int_cache=cam_int_cache,
            )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(estimator.fov_estimator.get_cam_intrinsics.call_count, 1)
        self.assertEqual(len(cam_int_cache), 1)


if __name__ == "__main__":
    unittest.main()
