import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


class VitDetDetectorTests(unittest.TestCase):
    def test_select_detectron2_person_detections_filters_and_applies_nms(self):
        module_name = "models.sam_3d_body.tools.build_detector"
        sys.modules.pop(module_name, None)
        try:
            with mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
                build_detector = importlib.import_module(module_name)

            class FakeTensor:
                def __init__(self, values):
                    self._values = np.array(values, dtype=np.float32)

                def cpu(self):
                    return self

                def numpy(self):
                    return self._values

            instances = SimpleNamespace(
                pred_boxes=SimpleNamespace(
                    tensor=FakeTensor(
                        [
                            [0, 0, 10, 10],
                            [1, 1, 11, 11],
                            [30, 30, 40, 40],
                            [50, 50, 60, 60],
                        ]
                    )
                ),
                scores=FakeTensor([0.95, 0.80, 0.70, 0.99]),
                pred_classes=FakeTensor([0, 0, 0, 2]),
            )

            boxes, scores = build_detector.select_detectron2_person_detections(
                instances,
                det_cat_id=0,
                bbox_thr=0.05,
                nms_thr=0.5,
            )

            np.testing.assert_allclose(
                boxes,
                np.array(
                    [
                        [0, 0, 10, 10],
                        [30, 30, 40, 40],
                    ],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_allclose(scores, np.array([0.95, 0.70], dtype=np.float32))
        finally:
            sys.modules.pop(module_name, None)

    def test_load_detectron2_vitdet_applies_official_score_threshold(self):
        module_name = "models.sam_3d_body.tools.build_detector"
        sys.modules.pop(module_name, None)
        try:
            with mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
                build_detector = importlib.import_module(module_name)

            predictors = [SimpleNamespace(test_score_thresh=None) for _ in range(3)]
            detectron2_cfg = SimpleNamespace(
                train=SimpleNamespace(init_checkpoint=None),
                model=SimpleNamespace(
                    roi_heads=SimpleNamespace(box_predictors=predictors),
                ),
            )
            instantiate_mock = mock.Mock(return_value=mock.MagicMock())
            checkpointer_instance = mock.Mock()
            checkpointer_cls = mock.Mock(return_value=checkpointer_instance)

            with mock.patch.dict(
                sys.modules,
                {
                    "detectron2.checkpoint": SimpleNamespace(DetectionCheckpointer=checkpointer_cls),
                    "detectron2.config": SimpleNamespace(
                        instantiate=instantiate_mock,
                        LazyConfig=SimpleNamespace(load=mock.Mock(return_value=detectron2_cfg)),
                    ),
                },
            ):
                with mock.patch.object(build_detector.Path, "exists", return_value=True):
                    build_detector.load_detectron2_vitdet(path="")

            self.assertEqual(
                [predictor.test_score_thresh for predictor in predictors],
                [0.05, 0.05, 0.05],
            )
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
