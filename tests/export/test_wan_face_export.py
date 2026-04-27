import unittest

import numpy as np


class WanFaceExportTests(unittest.TestCase):
    def test_select_target_face_prefers_overlap_with_target_mask(self):
        from scripts.wan_face_export import WanFaceDetection, select_target_face

        target_mask = np.zeros((64, 64), dtype=np.uint8)
        target_mask[10:34, 12:36] = 1
        body_keypoints = np.zeros((20, 3), dtype=np.float32)
        body_keypoints[0] = [0.35, 0.25, 1.0]
        detections = [
            WanFaceDetection(bbox=(10, 8, 34, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.95),
            WanFaceDetection(bbox=(40, 10, 60, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.99),
        ]

        selected = select_target_face(detections, target_mask, body_keypoints, previous_bbox=None)

        self.assertEqual(selected.bbox, (10, 8, 34, 30))

    def test_select_target_face_rejects_tiny_spurious_box_inside_mask(self):
        from scripts.wan_face_export import WanFaceDetection, select_target_face

        target_mask = np.zeros((64, 64), dtype=np.uint8)
        target_mask[10:34, 12:36] = 1
        body_keypoints = np.zeros((20, 3), dtype=np.float32)
        body_keypoints[0] = [0.37, 0.30, 1.0]
        detections = [
            WanFaceDetection(bbox=(14, 14, 18, 18), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.99),
            WanFaceDetection(bbox=(10, 8, 34, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.90),
        ]

        selected = select_target_face(detections, target_mask, body_keypoints, previous_bbox=None)

        self.assertEqual(selected.bbox, (10, 8, 34, 30))

    def test_select_target_face_handles_missing_body_keypoints(self):
        from scripts.wan_face_export import WanFaceDetection, select_target_face

        target_mask = np.zeros((32, 32), dtype=np.uint8)
        target_mask[8:20, 10:22] = 1
        detections = [
            WanFaceDetection(bbox=(10, 8, 22, 20), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.80),
        ]

        selected = select_target_face(detections, target_mask, body_keypoints=np.zeros((0, 3), dtype=np.float32))

        self.assertEqual(selected.bbox, (10, 8, 22, 20))

    def test_select_target_face_returns_none_when_only_detected_face_belongs_to_other_person(self):
        from scripts.wan_face_export import WanFaceDetection, select_target_face

        target_mask = np.zeros((64, 64), dtype=np.uint8)
        target_mask[10:40, 40:60] = 1
        body_keypoints = np.zeros((20, 3), dtype=np.float32)
        body_keypoints[0] = [0.78, 0.25, 1.0]
        detections = [
            WanFaceDetection(
                bbox=(8, 8, 28, 28),
                landmarks=np.zeros((5, 3), dtype=np.float32),
                score=0.95,
            ),
        ]

        selected = select_target_face(detections, target_mask, body_keypoints, previous_bbox=None)

        self.assertIsNone(selected)

    def test_fill_face_gaps_reuses_previous_box_for_short_missing_span(self):
        from scripts.wan_face_export import WanFaceDetection, fill_face_gaps

        first = WanFaceDetection(bbox=(10, 10, 30, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.9)
        last = WanFaceDetection(bbox=(12, 10, 32, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.9)

        filled = fill_face_gaps([first, None, last], max_gap=2)

        self.assertIsNotNone(filled[1])
        self.assertEqual(filled[1].bbox, first.bbox)

    def test_fill_face_gaps_reuses_previous_box_for_terminal_gap(self):
        from scripts.wan_face_export import WanFaceDetection, fill_face_gaps

        first = WanFaceDetection(bbox=(10, 10, 30, 30), landmarks=np.zeros((5, 3), dtype=np.float32), score=0.9)

        filled = fill_face_gaps([first, None], max_gap=1)

        self.assertIsNotNone(filled[1])
        self.assertEqual(filled[1].bbox, first.bbox)

    def test_crop_face_frame_resizes_to_requested_square(self):
        from scripts.wan_face_export import crop_face_frame

        frame_rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        frame_rgb[8:24, 12:28] = 200

        cropped = crop_face_frame(frame_rgb, bbox=(12, 8, 28, 24), output_size=(32, 32))

        self.assertEqual(cropped.shape, (32, 32, 3))
        self.assertGreater(int(cropped.sum()), 0)


if __name__ == "__main__":
    unittest.main()
