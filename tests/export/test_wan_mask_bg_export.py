import unittest

import numpy as np


class WanMaskBgExportTests(unittest.TestCase):
    def test_build_bg_and_mask_frame_only_removes_target_region(self):
        from scripts.wan_mask_bg_export import build_bg_and_mask_frame
        from scripts.wan_sample_types import WanExportConfig

        frame = np.full((8, 8, 3), 100, dtype=np.uint8)
        indexed_mask = np.zeros((8, 8), dtype=np.uint8)
        indexed_mask[1:4, 1:4] = 1
        indexed_mask[4:7, 4:7] = 2

        mask_rgb, bg_rgb, target_mask = build_bg_and_mask_frame(
            frame_rgb=frame,
            indexed_mask=indexed_mask,
            track_id=1,
            config=WanExportConfig(
                enable=True,
                mask_kernel_size=1,
                mask_iterations=1,
                mask_w_len=99,
                mask_h_len=99,
            ),
        )

        self.assertEqual(int(target_mask.sum()), 9)
        self.assertTrue((mask_rgb[2, 2] == np.array([255, 255, 255])).all())
        self.assertTrue((bg_rgb[5, 5] == np.array([100, 100, 100])).all())

    def test_build_bg_and_mask_frame_uses_exported_mask_region_for_background_cutout(self):
        from scripts.wan_mask_bg_export import build_bg_and_mask_frame
        from scripts.wan_sample_types import WanExportConfig

        frame = np.full((7, 7, 3), 100, dtype=np.uint8)
        indexed_mask = np.zeros((7, 7), dtype=np.uint8)
        indexed_mask[3, 3] = 1

        mask_rgb, bg_rgb, target_mask = build_bg_and_mask_frame(
            frame_rgb=frame,
            indexed_mask=indexed_mask,
            track_id=1,
            config=WanExportConfig(
                enable=True,
                mask_kernel_size=3,
                mask_iterations=1,
                mask_w_len=99,
                mask_h_len=99,
            ),
        )

        self.assertEqual(int(target_mask.sum()), 1)
        self.assertTrue((mask_rgb[2, 2] == np.array([255, 255, 255])).all())
        self.assertTrue((mask_rgb[3, 3] == np.array([255, 255, 255])).all())
        self.assertTrue((bg_rgb[2, 2] == np.array([0, 0, 0])).all())
        self.assertTrue((bg_rgb[3, 3] == np.array([0, 0, 0])).all())
        self.assertTrue((bg_rgb[0, 0] == np.array([100, 100, 100])).all())

    def test_score_reference_frame_rewards_face_and_mask_coverage(self):
        from scripts.wan_mask_bg_export import score_reference_frame

        target_mask = np.zeros((8, 8), dtype=np.uint8)
        target_mask[1:6, 1:6] = 1
        pose_meta = {"keypoints_body": [[0.5, 0.5, 1.0]] * 20}

        score = score_reference_frame(target_mask=target_mask, face_detection=True, pose_meta=pose_meta)

        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
