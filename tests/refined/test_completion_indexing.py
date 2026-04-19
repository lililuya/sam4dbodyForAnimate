import unittest


class CompletionIndexingTests(unittest.TestCase):
    def test_build_completion_window_keeps_last_occluded_frame_inside_saved_range(self):
        from scripts.offline_completion_indexing import build_completion_window_from_ious

        occ_flags, window = build_completion_window_from_ious(
            [0.95, 0.92, 0.41],
            padding=2,
            iou_threshold=0.7,
        )

        self.assertEqual([1, 1, 0], occ_flags)
        self.assertEqual((0, 3), window)
        self.assertIn(2, range(*window))

    def test_build_completion_window_covers_every_occluded_index(self):
        from scripts.offline_completion_indexing import build_completion_window_from_ious

        occ_flags, window = build_completion_window_from_ious(
            [0.91, 0.35, 0.88, 0.42, 0.95],
            padding=2,
            iou_threshold=0.7,
        )

        self.assertEqual([1, 0, 1, 0, 1], occ_flags)
        self.assertEqual((0, 5), window)
        saved_indices = set(range(*window))
        occluded_indices = {idx for idx, flag in enumerate(occ_flags) if flag == 0}
        self.assertTrue(occluded_indices.issubset(saved_indices))


if __name__ == "__main__":
    unittest.main()
