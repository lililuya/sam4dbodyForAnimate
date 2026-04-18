import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np


class YoloDetectorTests(unittest.TestCase):
    def test_sort_boxes_xyxy_is_deterministic(self):
        from models.sam_3d_body.tools.build_detector_yolo import sort_boxes_xyxy

        boxes = np.array(
            [
                [20, 10, 40, 30],
                [0, 0, 10, 10],
                [20, 5, 40, 25],
            ],
            dtype=np.float32,
        )
        sorted_boxes = sort_boxes_xyxy(boxes)
        expected = np.array(
            [
                [0, 0, 10, 10],
                [20, 5, 40, 25],
                [20, 10, 40, 30],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(sorted_boxes, expected)

    def test_extract_person_boxes_filters_by_class_and_threshold(self):
        from models.sam_3d_body.tools.build_detector_yolo import extract_person_boxes

        class FakeBoxes:
            def __init__(self):
                self.xyxy = np.array(
                    [[0, 0, 50, 100], [10, 10, 30, 30], [5, 5, 40, 90]],
                    dtype=np.float32,
                )
                self.conf = np.array([0.90, 0.95, 0.20], dtype=np.float32)
                self.cls = np.array([0, 2, 0], dtype=np.float32)

        boxes = extract_person_boxes(FakeBoxes(), bbox_thr=0.30, det_cat_id=0)
        expected = np.array([[0, 0, 50, 100]], dtype=np.float32)
        np.testing.assert_allclose(boxes, expected)

    def test_extract_person_boxes_supports_tensor_like_fields(self):
        from models.sam_3d_body.tools.build_detector_yolo import extract_person_boxes

        class FakeTensor:
            def __init__(self, values):
                self._values = np.array(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        class FakeBoxes:
            def __init__(self):
                self.xyxy = FakeTensor([[0, 0, 50, 100], [20, 10, 40, 30], [20, 5, 40, 25]])
                self.conf = FakeTensor([0.90, 0.95, 0.20])
                self.cls = FakeTensor([0, 0, 0])

        boxes = extract_person_boxes(FakeBoxes(), bbox_thr=0.30, det_cat_id=0)
        expected = np.array([[0, 0, 50, 100], [20, 10, 40, 30]], dtype=np.float32)
        np.testing.assert_allclose(boxes, expected)

    def test_human_detector_yolo_threads_device_to_runner(self):
        import importlib
        import sys

        module_name = "models.sam_3d_body.tools.build_detector"
        sys.modules.pop(module_name, None)
        try:
            with mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
                build_detector = importlib.import_module(module_name)

            with mock.patch(
                "models.sam_3d_body.tools.build_detector_yolo.load_ultralytics_yolo",
                return_value="fake-detector",
            ) as loader_mock:
                runner_mock = mock.Mock(return_value=np.zeros((0, 4), dtype=np.float32))
                with mock.patch(
                    "models.sam_3d_body.tools.build_detector_yolo.run_ultralytics_yolo",
                    runner_mock,
                ):
                    detector = build_detector.HumanDetector(
                        name="yolo",
                        device="cuda:1",
                        weights_path="weights.pt",
                    )
                    image = np.zeros((16, 16, 3), dtype=np.uint8)
                    detector.run_human_detection(image)

            loader_mock.assert_called_once_with(weights_path="weights.pt")
            runner_mock.assert_called_once()
            args, kwargs = runner_mock.call_args
            self.assertEqual(args[0], "fake-detector")
            np.testing.assert_array_equal(args[1], image)
            self.assertEqual(kwargs["device"], "cuda:1")
        finally:
            sys.modules.pop(module_name, None)
        self.assertNotIn(module_name, sys.modules)

    def test_load_ultralytics_yolo_prefers_explicit_weights_path(self):
        from models.sam_3d_body.tools.build_detector_yolo import load_ultralytics_yolo

        yolo_mock = mock.Mock(return_value="fake-detector")
        with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=yolo_mock)}):
            detector = load_ultralytics_yolo(path="weights.pt", weights_path="weights.pt")

        self.assertEqual(detector, "fake-detector")
        yolo_mock.assert_called_once_with("weights.pt")

    def test_load_ultralytics_yolo_defaults_to_auto_downloadable_model(self):
        from models.sam_3d_body.tools.build_detector_yolo import (
            DEFAULT_YOLO_WEIGHTS,
            load_ultralytics_yolo,
        )

        yolo_mock = mock.Mock(return_value="fake-detector")
        with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=yolo_mock)}):
            detector = load_ultralytics_yolo()

        self.assertEqual(detector, "fake-detector")
        yolo_mock.assert_called_once_with(DEFAULT_YOLO_WEIGHTS)

    def test_load_ultralytics_yolo_rejects_conflicting_explicit_paths(self):
        from models.sam_3d_body.tools.build_detector_yolo import load_ultralytics_yolo

        with self.assertRaises(ValueError) as cm:
            load_ultralytics_yolo(path="legacy.pt", weights_path="weights.pt")

        self.assertIn("Conflicting YOLO paths", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
