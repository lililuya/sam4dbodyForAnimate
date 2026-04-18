import unittest


class MaskRefinementHelperTests(unittest.TestCase):
    def test_cap_consecutive_ones_by_iou_keeps_top_scores(self):
        from scripts.offline_refined_helpers import cap_consecutive_ones_by_iou

        flags = [1, 1, 1, 0, 1, 1]
        ious = [0.1, 0.8, 0.6, 0.0, 0.2, 0.9]
        result = cap_consecutive_ones_by_iou(flags, ious, max_keep=2)
        self.assertEqual(result, [0, 1, 1, 1, 1, 1])

    def test_find_occlusion_window_applies_padding(self):
        from scripts.offline_refined_helpers import find_occlusion_window

        ious = [0.9, 0.4, 0.3, 0.8, 0.95]
        result = find_occlusion_window(ious, threshold=0.55, total_frames=5, pad=1)
        self.assertEqual(result, (0, 4))

    def test_smooth_equal_threshold_hits_zeroes_isolated_middle_hit(self):
        from scripts.offline_refined_helpers import smooth_equal_threshold_hits

        ious = [0.2, 0.5, 0.3]
        result = smooth_equal_threshold_hits(ious, threshold=0.5)
        self.assertEqual(result, [0.2, 0.0, 0.3])

    def test_smooth_equal_threshold_hits_keeps_non_isolated_values(self):
        from scripts.offline_refined_helpers import smooth_equal_threshold_hits

        ious = [0.7, 0.5, 0.8]
        result = smooth_equal_threshold_hits(ious, threshold=0.5)
        self.assertEqual(result, [0.7, 0.5, 0.8])

    def test_smooth_equal_threshold_hits_keeps_boundary_values(self):
        from scripts.offline_refined_helpers import smooth_equal_threshold_hits

        ious = [0.5, 0.2, 0.5]
        result = smooth_equal_threshold_hits(ious, threshold=0.5)
        self.assertEqual(result, [0.5, 0.2, 0.5])


if __name__ == "__main__":
    unittest.main()
