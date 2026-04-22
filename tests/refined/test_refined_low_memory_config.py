import os
import unittest

from omegaconf import OmegaConf


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RefinedLowMemoryConfigTests(unittest.TestCase):
    def test_low_memory_config_uses_overlap_friendly_detector_thresholds(self):
        config_path = os.path.join(ROOT, "configs", "body4d_refined_low_memory.yaml")

        cfg = OmegaConf.load(config_path)

        self.assertEqual(str(cfg.detector.backend), "yolo")
        self.assertEqual(float(cfg.detector.bbox_thresh), 0.20)
        self.assertEqual(float(cfg.detector.iou_thresh), 0.70)
        self.assertEqual(int(cfg.detector.max_det), 20)


if __name__ == "__main__":
    unittest.main()
