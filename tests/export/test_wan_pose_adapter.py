import unittest

import numpy as np

from scripts.openpose_export import convert_mhr70_to_openpose_arrays
from scripts.pose_json_export import LEFT_HAND_MHR70, RIGHT_HAND_MHR70


def _build_person_output():
    keypoints_2d = np.stack(
        [np.array([float(index + 1), float(index + 2)], dtype=np.float32) for index in range(70)],
        axis=0,
    )
    keypoints_3d = np.stack(
        [np.array([float(index + 1), float(index + 2), float(index + 3)], dtype=np.float32) for index in range(70)],
        axis=0,
    )
    return {
        "pred_keypoints_2d": keypoints_2d,
        "pred_keypoints_3d": keypoints_3d,
    }


class WanPoseAdapterTests(unittest.TestCase):
    def test_build_wan_pose_meta_normalizes_body_hands_and_face(self):
        from scripts.wan_pose_adapter import build_wan_pose_meta

        person_output = _build_person_output()
        pose_meta = build_wan_pose_meta(
            person_output=person_output,
            track_id=2,
            frame_stem="00000012",
            frame_size=(120, 240),
            face_landmarks=np.array([[-10.0, 150.0], [300.0, -20.0]], dtype=np.float32),
        )

        body = np.asarray(pose_meta["keypoints_body"], dtype=np.float32)
        left_hand = np.asarray(pose_meta["keypoints_left_hand"], dtype=np.float32)
        right_hand = np.asarray(pose_meta["keypoints_right_hand"], dtype=np.float32)
        face = np.asarray(pose_meta["keypoints_face"], dtype=np.float32)

        self.assertEqual(pose_meta["image_id"], "00000012.jpg")
        self.assertEqual(pose_meta["track_id"], 2)
        self.assertEqual(body.shape, (20, 3))
        self.assertEqual(left_hand.shape, (21, 3))
        self.assertEqual(right_hand.shape, (21, 3))
        self.assertEqual(face.shape, (2, 3))
        self.assertTrue(np.all(body[:, :2] >= 0.0))
        self.assertTrue(np.all(body[:, :2] <= 1.0))
        self.assertTrue(np.all(left_hand[:, :2] >= 0.0))
        self.assertTrue(np.all(left_hand[:, :2] <= 1.0))
        self.assertTrue(np.all(right_hand[:, :2] >= 0.0))
        self.assertTrue(np.all(right_hand[:, :2] <= 1.0))
        self.assertTrue(np.all(face[:, :2] >= 0.0))
        self.assertTrue(np.all(face[:, :2] <= 1.0))
        self.assertTrue(np.all(face[:, 2] == 1.0))

        openpose_2d, _ = convert_mhr70_to_openpose_arrays(
            person_output["pred_keypoints_2d"],
            person_output["pred_keypoints_3d"],
        )
        openpose_2d = np.asarray(openpose_2d, dtype=np.float32).reshape(-1, 3)

        np.testing.assert_allclose(
            body[0],
            np.array([openpose_2d[0, 0] / 240.0, openpose_2d[0, 1] / 120.0, openpose_2d[0, 2]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            body[-1],
            np.array([openpose_2d[22, 0] / 240.0, openpose_2d[22, 1] / 120.0, openpose_2d[22, 2]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            left_hand[0],
            np.array(
                [
                    person_output["pred_keypoints_2d"][LEFT_HAND_MHR70[0], 0] / 240.0,
                    person_output["pred_keypoints_2d"][LEFT_HAND_MHR70[0], 1] / 120.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            right_hand[0],
            np.array(
                [
                    person_output["pred_keypoints_2d"][RIGHT_HAND_MHR70[0], 0] / 240.0,
                    person_output["pred_keypoints_2d"][RIGHT_HAND_MHR70[0], 1] / 120.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(face[0], np.array([0.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(face[1], np.array([1.0, 0.0, 1.0], dtype=np.float32))

    def test_render_wan_pose_frame_draws_non_black_pixels(self):
        from scripts.wan_pose_adapter import build_wan_pose_meta, render_wan_pose_frame

        pose_meta = build_wan_pose_meta(
            person_output=_build_person_output(),
            track_id=1,
            frame_stem="00000000",
            frame_size=(120, 240),
            face_landmarks=None,
        )

        pose_frame = render_wan_pose_frame(pose_meta, canvas_shape=(120, 240, 3))

        body = np.asarray(pose_meta["keypoints_body"], dtype=np.float32)
        anchor_x = int(body[0, 0] * 240)
        anchor_y = int(body[0, 1] * 120)
        patch = pose_frame[max(anchor_y - 2, 0) : anchor_y + 3, max(anchor_x - 2, 0) : anchor_x + 3]

        self.assertEqual(pose_frame.shape, (120, 240, 3))
        self.assertGreater(int(pose_frame.sum()), 0)
        self.assertGreater(int(patch.sum()), 0)


if __name__ == "__main__":
    unittest.main()
