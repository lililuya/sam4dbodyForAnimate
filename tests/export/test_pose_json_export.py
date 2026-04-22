import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

import numpy as np


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


class PoseJsonExportTests(unittest.TestCase):
    def test_write_pose_frame_exports_writes_per_track_openpose_and_smpl_json(self):
        from scripts.pose_json_export import write_pose_frame_exports

        person_outputs = [
            {
                "bbox": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
                "focal_length": np.float32(500.0),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_vertices": np.zeros((6890, 3), dtype=np.float32),
                "pred_cam_t": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "pred_pose_raw": np.zeros((66,), dtype=np.float32),
                "global_rot": np.zeros((3,), dtype=np.float32),
                "body_pose_params": np.zeros((133,), dtype=np.float32),
                "hand_pose_params": np.zeros((90,), dtype=np.float32),
                "scale_params": np.array([1.0], dtype=np.float32),
                "shape_params": np.zeros((10,), dtype=np.float32),
                "expr_params": np.zeros((10,), dtype=np.float32),
                "pred_joint_coords": np.zeros((70, 3), dtype=np.float32),
                "pred_global_rots": np.zeros((55, 3, 3), dtype=np.float32),
                "mhr_model_params": np.zeros((159,), dtype=np.float32),
            }
        ]

        with make_workspace_tempdir() as tmpdir:
            write_pose_frame_exports(
                output_dir=tmpdir,
                frame_stem="00000012",
                person_outputs=person_outputs,
                track_ids=[7],
            )

            openpose_path = os.path.join(
                tmpdir, "openpose_json", "7", "00000012_keypoints.json"
            )
            smpl_path = os.path.join(tmpdir, "smpl_json", "7", "00000012.json")

            with open(openpose_path, "r", encoding="utf-8") as handle:
                openpose_payload = json.load(handle)
            with open(smpl_path, "r", encoding="utf-8") as handle:
                smpl_payload = json.load(handle)

        self.assertEqual(openpose_payload["version"], 1.3)
        self.assertEqual(openpose_payload["person_id"], 7)
        self.assertEqual(openpose_payload["frame_stem"], "00000012")
        self.assertEqual(len(openpose_payload["pose_keypoints_2d"]), 75)
        self.assertEqual(len(openpose_payload["pose_keypoints_3d"]), 100)

        self.assertEqual(smpl_payload["person_id"], 7)
        self.assertEqual(smpl_payload["frame_stem"], "00000012")
        self.assertEqual(smpl_payload["focal_length"], 500.0)
        self.assertEqual(smpl_payload["pred_cam_t"], [1.0, 2.0, 3.0])
        self.assertIn("body_pose_params", smpl_payload)
        self.assertIn("shape_params", smpl_payload)
        self.assertEqual(len(smpl_payload["openpose_pose_keypoints_2d"]), 75)
        self.assertEqual(len(smpl_payload["openpose_pose_keypoints_3d"]), 100)


if __name__ == "__main__":
    unittest.main()
