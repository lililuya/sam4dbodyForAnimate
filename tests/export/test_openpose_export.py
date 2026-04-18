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


class OpenPoseExportTests(unittest.TestCase):
    def test_convert_mhr70_to_openpose_arrays_maps_known_joints_and_midhip(self):
        from scripts.openpose_export import convert_mhr70_to_openpose_arrays

        keypoints_2d = np.zeros((70, 2), dtype=np.float32)
        keypoints_3d = np.zeros((70, 3), dtype=np.float32)

        keypoints_2d[10] = [100.0, 200.0]
        keypoints_2d[9] = [140.0, 240.0]
        keypoints_3d[10] = [1.0, 2.0, 3.0]
        keypoints_3d[9] = [5.0, 6.0, 7.0]

        pose_2d, pose_3d = convert_mhr70_to_openpose_arrays(keypoints_2d, keypoints_3d)

        self.assertEqual(len(pose_2d), 25 * 3)
        self.assertEqual(len(pose_3d), 25 * 4)
        self.assertEqual(pose_2d[9 * 3 : 9 * 3 + 3], [100.0, 200.0, 1.0])
        self.assertEqual(pose_2d[12 * 3 : 12 * 3 + 3], [140.0, 240.0, 1.0])
        self.assertEqual(pose_2d[8 * 3 : 8 * 3 + 3], [120.0, 220.0, 1.0])
        self.assertEqual(pose_3d[8 * 4 : 8 * 4 + 4], [3.0, 4.0, 5.0, 1.0])
        self.assertEqual(pose_2d[21 * 3 : 21 * 3 + 3], [0.0, 0.0, 1.0])
        self.assertEqual(pose_3d[21 * 4 : 21 * 4 + 4], [0.0, 0.0, 0.0, 1.0])

    def test_build_and_write_openpose_frame_payload(self):
        from scripts.openpose_export import build_openpose_people, write_openpose_frame_json

        person_outputs = [
            {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            }
        ]

        people = build_openpose_people(person_outputs, [7])
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_id"], 7)
        self.assertEqual(len(people[0]["pose_keypoints_2d"]), 75)
        self.assertEqual(len(people[0]["pose_keypoints_3d"]), 100)

        with make_workspace_tempdir() as tmpdir:
            json_path = write_openpose_frame_json(tmpdir, "00000012", people)
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["version"], 1.3)
        self.assertEqual(len(payload["people"]), 1)
        self.assertEqual(payload["people"][0]["person_id"], 7)


if __name__ == "__main__":
    unittest.main()
