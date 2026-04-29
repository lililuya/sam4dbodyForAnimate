import unittest


class OfflineTrackingCompatTests(unittest.TestCase):
    def test_unpack_propagate_output_accepts_five_value_tracker_output(self):
        from scripts.offline_tracking_compat import unpack_propagate_output

        output = (
            0,
            [1],
            "low-res",
            "video-res",
            "obj-scores",
        )

        frame_idx, obj_ids, low_res_masks, video_res_masks = unpack_propagate_output(output)

        self.assertEqual(frame_idx, 0)
        self.assertEqual(obj_ids, [1])
        self.assertEqual(low_res_masks, "low-res")
        self.assertEqual(video_res_masks, "video-res")

    def test_unpack_propagate_output_accepts_six_value_tracker_output(self):
        from scripts.offline_tracking_compat import unpack_propagate_output

        output = (
            0,
            [1],
            "low-res",
            "video-res",
            "obj-scores",
            "iou-scores",
        )

        frame_idx, obj_ids, low_res_masks, video_res_masks = unpack_propagate_output(output)

        self.assertEqual(frame_idx, 0)
        self.assertEqual(obj_ids, [1])
        self.assertEqual(low_res_masks, "low-res")
        self.assertEqual(video_res_masks, "video-res")

    def test_unpack_propagate_output_rejects_short_tracker_output(self):
        from scripts.offline_tracking_compat import unpack_propagate_output

        with self.assertRaisesRegex(ValueError, "at least 4 values"):
            unpack_propagate_output((0, [1], "low-res"))


if __name__ == "__main__":
    unittest.main()
