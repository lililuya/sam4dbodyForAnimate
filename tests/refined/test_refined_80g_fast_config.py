import os
import unittest

from omegaconf import OmegaConf


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Refined80GFastConfigTests(unittest.TestCase):
    def test_80g_fast_config_uses_speed_priority_runtime_settings(self):
        config_path = os.path.join(ROOT, "configs", "body4d_refined_80g_fast.yaml")

        cfg = OmegaConf.load(config_path)

        self.assertEqual(int(cfg.sam_3d_body.batch_size), 32)
        self.assertTrue(bool(cfg.completion.enable))
        self.assertEqual(list(cfg.completion.detection_resolution), [192, 384])
        self.assertEqual(list(cfg.completion.completion_resolution), [256, 512])
        self.assertEqual(int(cfg.completion.batch_size), 4)
        self.assertEqual(int(cfg.completion.decode_chunk_size), 4)
        self.assertEqual(int(cfg.completion.max_occ_len), 6)
        self.assertEqual(str(cfg.detector.backend), "yolo")
        self.assertEqual(float(cfg.detector.bbox_thresh), 0.20)
        self.assertEqual(float(cfg.detector.iou_thresh), 0.70)
        self.assertEqual(int(cfg.detector.max_det), 20)
        self.assertTrue(bool(cfg.refine.enable))
        self.assertEqual(int(cfg.batch.initial_search_frames), 12)
        self.assertEqual(str(cfg.batch.retry_mode), "never")
        self.assertFalse(bool(cfg.reprompt.enable))
        self.assertTrue(bool(cfg.debug.save_metrics))


if __name__ == "__main__":
    unittest.main()
