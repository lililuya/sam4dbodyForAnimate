import unittest
from unittest import mock


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

    def test_run_human_detection_compat_retries_without_unsupported_max_det(self):
        from scripts.detector_defaults import run_human_detection_compat

        detector = mock.MagicMock()

        def _run(_image, **kwargs):
            if "max_det" in kwargs:
                raise TypeError("run_detectron2_vitdet() got an unexpected keyword argument 'max_det'")
            return [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9}]

        detector.run_human_detection.side_effect = _run

        result = run_human_detection_compat(
            detector,
            "image",
            {
                "bbox_thr": 0.05,
                "nms_thr": 0.5,
                "return_scores": True,
                "max_det": 5,
            },
        )

        self.assertEqual(result, [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9}])
        self.assertEqual(detector.run_human_detection.call_count, 2)
        first_kwargs = detector.run_human_detection.call_args_list[0].kwargs
        second_kwargs = detector.run_human_detection.call_args_list[1].kwargs
        self.assertIn("max_det", first_kwargs)
        self.assertNotIn("max_det", second_kwargs)
        self.assertTrue(second_kwargs["return_scores"])

    def test_run_human_detection_compat_retries_without_unsupported_return_scores(self):
        from scripts.detector_defaults import run_human_detection_compat

        detector = mock.MagicMock()

        def _run(_image, **kwargs):
            if "return_scores" in kwargs:
                raise TypeError("run_human_detection() got an unexpected keyword argument 'return_scores'")
            return [[1.0, 2.0, 3.0, 4.0]]

        detector.run_human_detection.side_effect = _run

        result = run_human_detection_compat(
            detector,
            "image",
            {
                "bbox_thr": 0.25,
                "nms_thr": 0.7,
                "return_scores": True,
            },
        )

        self.assertEqual(result, [[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(detector.run_human_detection.call_count, 2)
        first_kwargs = detector.run_human_detection.call_args_list[0].kwargs
        second_kwargs = detector.run_human_detection.call_args_list[1].kwargs
        self.assertIn("return_scores", first_kwargs)
        self.assertNotIn("return_scores", second_kwargs)


if __name__ == "__main__":
    unittest.main()
