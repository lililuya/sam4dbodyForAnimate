import unittest

import numpy as np


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

        pose_meta = build_wan_pose_meta(
            person_output=_build_person_output(),
            track_id=2,
            frame_stem="00000012",
            frame_size=(256, 256),
            face_landmarks=np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float32),
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
        self.assertTrue(np.all(face[:, 2] == 1.0))

    def test_render_wan_pose_frame_draws_non_black_pixels(self):
        from scripts.wan_pose_adapter import build_wan_pose_meta, render_wan_pose_frame

        pose_meta = build_wan_pose_meta(
            person_output=_build_person_output(),
            track_id=1,
            frame_stem="00000000",
            frame_size=(256, 256),
            face_landmarks=None,
        )

        pose_frame = render_wan_pose_frame(pose_meta, canvas_shape=(256, 256, 3))

        self.assertEqual(pose_frame.shape, (256, 256, 3))
        self.assertGreater(int(pose_frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
