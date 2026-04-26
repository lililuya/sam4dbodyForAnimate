import unittest

import numpy as np


class WanReferenceCompatTests(unittest.TestCase):
    def test_wan_export_config_coerces_runtime_values(self):
        from scripts.wan_sample_types import WanExportConfig

        config = WanExportConfig.from_runtime(
            {
                "enable": True,
                "fps": 25,
                "resolution_area": [512, 768],
                "face_resolution": [512, 512],
                "min_track_frames": 12,
                "output_dir": "./WanExport",
                "metadata_output_dir": "./WanExportMeta",
            }
        )

        self.assertTrue(config.enable)
        self.assertEqual(config.fps, 25)
        self.assertEqual(config.resolution_area, (512, 768))
        self.assertEqual(config.face_resolution, (512, 512))
        self.assertEqual(config.min_track_frames, 12)
        self.assertEqual(config.output_dir, "./WanExport")
        self.assertEqual(config.metadata_output_dir, "./WanExportMeta")
        self.assertFalse(config.save_smpl_sequence_json)

    def test_wan_export_config_coerces_common_string_booleans(self):
        from scripts.wan_sample_types import WanExportConfig

        config = WanExportConfig.from_runtime(
            {
                "enable": "false",
                "save_pose_meta_json": "0",
                "save_smpl_sequence_json": "1",
            }
        )

        self.assertFalse(config.enable)
        self.assertFalse(config.save_pose_meta_json)
        self.assertTrue(config.save_smpl_sequence_json)

    def test_wan_export_config_validates_pair_lengths(self):
        from scripts.wan_sample_types import WanExportConfig

        with self.assertRaises(ValueError):
            WanExportConfig.from_runtime({"resolution_area": [512]})

        with self.assertRaises(ValueError):
            WanExportConfig.from_runtime({"face_resolution": [512, 512, 512]})

    def test_compute_sample_indices_downsamples_to_target_fps(self):
        from scripts.wan_reference_compat import compute_sample_indices

        indices = compute_sample_indices(num_frames=10, source_fps=30.0, target_fps=10.0)
        self.assertEqual(indices, [0, 3, 6])

    def test_compute_sample_indices_preserves_wan_short_clip_behavior(self):
        from scripts.wan_reference_compat import compute_sample_indices

        indices = compute_sample_indices(num_frames=2, source_fps=30.0, target_fps=25.0)
        self.assertEqual(indices, [0])

    def test_resize_frame_by_area_preserves_aspect_and_aligns_to_16(self):
        from scripts.wan_reference_compat import resize_frame_by_area

        frame = np.zeros((777, 333, 3), dtype=np.uint8)
        resized = resize_frame_by_area(frame, resolution_area=(512, 768), align_divisor=16)

        self.assertEqual(resized.shape[0] % 16, 0)
        self.assertEqual(resized.shape[1] % 16, 0)
        self.assertLessEqual(resized.shape[0] * resized.shape[1], 512 * 768)
        original_aspect = frame.shape[1] / frame.shape[0]
        resized_aspect = resized.shape[1] / resized.shape[0]
        self.assertLess(abs(resized_aspect - original_aspect), 0.05)

    def test_dilate_target_mask_is_binary_and_grows_foreground(self):
        from scripts.wan_reference_compat import dilate_target_mask

        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1

        dilated = dilate_target_mask(mask, kernel_size=3, iterations=1)

        self.assertTrue(set(np.unique(dilated).tolist()).issubset({0, 1}))
        self.assertGreater(int(dilated.sum()), int(mask.sum()))

    def test_expand_target_mask_is_binary_and_expands(self):
        from scripts.wan_reference_compat import dilate_target_mask, expand_target_mask

        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2, 2] = 1
        mask[2, 6] = 1
        dilated = dilate_target_mask(mask, kernel_size=1, iterations=1)

        expanded = expand_target_mask(dilated, w_len=2, h_len=2)

        self.assertTrue(set(np.unique(expanded).tolist()).issubset({0, 1}))
        self.assertGreater(int(expanded.sum()), int(dilated.sum()))

    def test_expand_target_mask_does_not_apply_hidden_extra_dilation(self):
        from scripts.wan_reference_compat import dilate_target_mask, expand_target_mask

        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[3:6, 3:6] = 1
        dilated = dilate_target_mask(mask, kernel_size=3, iterations=1)

        expanded = expand_target_mask(dilated, w_len=9, h_len=9)

        np.testing.assert_array_equal(expanded, dilated)


if __name__ == "__main__":
    unittest.main()
