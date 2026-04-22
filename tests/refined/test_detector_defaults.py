import unittest


class DetectorDefaultsTests(unittest.TestCase):
    def test_resolve_detector_runtime_options_uses_official_yolo_defaults(self):
        from scripts.detector_defaults import resolve_detector_runtime_options

        resolved = resolve_detector_runtime_options("yolo")

        self.assertEqual(resolved["bbox_thresh"], 0.25)
        self.assertEqual(resolved["iou_thresh"], 0.7)
        self.assertEqual(resolved["max_det"], 300)

    def test_resolve_detector_runtime_options_uses_official_vitdet_defaults(self):
        from scripts.detector_defaults import resolve_detector_runtime_options

        resolved = resolve_detector_runtime_options("vitdet")

        self.assertEqual(resolved["bbox_thresh"], 0.05)
        self.assertEqual(resolved["iou_thresh"], 0.5)
        self.assertIsNone(resolved["max_det"])

    def test_resolve_detector_runtime_options_preserves_explicit_overrides(self):
        from scripts.detector_defaults import resolve_detector_runtime_options

        resolved = resolve_detector_runtime_options(
            "yolo",
            bbox_thresh=0.6,
            iou_thresh=0.4,
            max_det=5,
        )

        self.assertEqual(resolved["bbox_thresh"], 0.6)
        self.assertEqual(resolved["iou_thresh"], 0.4)
        self.assertEqual(resolved["max_det"], 5)


if __name__ == "__main__":
    unittest.main()
