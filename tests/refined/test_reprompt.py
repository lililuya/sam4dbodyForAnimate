import unittest

from omegaconf import OmegaConf


class RePromptRulesTests(unittest.TestCase):
    def test_should_trigger_reprompt_for_repeated_empty_masks(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_count": 3, "area_ratio": 0.9, "edge_touch_ratio": 0.1, "mask_iou": 0.9}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertTrue(should_trigger_reprompt(metrics, thresholds))

    def test_should_trigger_reprompt_for_low_area_ratio(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_count": 0, "area_ratio": 0.2, "edge_touch_ratio": 0.1, "mask_iou": 0.9}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertTrue(should_trigger_reprompt(metrics, thresholds))

    def test_should_trigger_reprompt_for_edge_touch_ratio(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_count": 0, "area_ratio": 0.9, "edge_touch_ratio": 0.6, "mask_iou": 0.9}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertTrue(should_trigger_reprompt(metrics, thresholds))

    def test_should_trigger_reprompt_for_low_iou(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_count": 0, "area_ratio": 0.9, "edge_touch_ratio": 0.1, "mask_iou": 0.4}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertTrue(should_trigger_reprompt(metrics, thresholds))

    def test_should_not_trigger_reprompt_when_metrics_are_healthy(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_count": 1, "area_ratio": 0.8, "edge_touch_ratio": 0.1, "mask_iou": 0.8}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertFalse(should_trigger_reprompt(metrics, thresholds))

    def test_should_trigger_reprompt_supports_empty_mask_streak_alias(self):
        from scripts.offline_reprompt import should_trigger_reprompt

        metrics = {"empty_mask_streak": 3, "area_ratio": 0.9, "edge_touch_ratio": 0.1, "mask_iou": 0.9}
        thresholds = {
            "empty_mask_patience": 3,
            "area_drop_ratio": 0.35,
            "edge_touch_ratio": 0.4,
            "iou_low_threshold": 0.55,
        }

        self.assertTrue(should_trigger_reprompt(metrics, thresholds))


class DetectionMatchTests(unittest.TestCase):
    def test_match_detection_to_track_picks_highest_overlap_candidate(self):
        from scripts.offline_reprompt import match_detection_to_track

        prev_box = [0.0, 0.0, 10.0, 10.0]
        candidates = [
            {"id": "low", "bbox": [20.0, 20.0, 30.0, 30.0]},
            {"id": "best", "bbox": [1.0, 1.0, 9.0, 9.0]},
            {"id": "mid", "bbox": [2.0, 2.0, 12.0, 12.0]},
        ]

        result = match_detection_to_track(prev_box, candidates)

        self.assertEqual(result, candidates[1])

    def test_match_detection_to_track_returns_none_when_all_ious_are_zero(self):
        from scripts.offline_reprompt import match_detection_to_track

        prev_box = [0.0, 0.0, 10.0, 10.0]
        candidates = [
            {"id": "left", "bbox": [-10.0, 0.0, -1.0, 9.0]},
            {"id": "right", "bbox": [11.0, 0.0, 20.0, 10.0]},
        ]

        result = match_detection_to_track(prev_box, candidates)

        self.assertIsNone(result)

    def test_match_detection_to_track_skips_malformed_candidates(self):
        from scripts.offline_reprompt import match_detection_to_track

        prev_box = [0.0, 0.0, 10.0, 10.0]
        candidates = [
            {"id": "missing_bbox"},
            {"id": "wrong_len", "bbox": [1.0, 1.0, 9.0]},
            {"id": "not_numeric", "bbox": ["x", 1.0, 9.0, 9.0]},
            {"id": "best", "bbox": [1.0, 1.0, 9.0, 9.0]},
        ]

        result = match_detection_to_track(prev_box, candidates)

        self.assertEqual(result, candidates[3])


class RePromptIntegrationSurfaceTests(unittest.TestCase):
    def test_build_reprompt_thresholds_reads_cfg_reprompt(self):
        from scripts.offline_app_refined import build_reprompt_thresholds

        cfg = OmegaConf.create(
            {
                "reprompt": {
                    "empty_mask_patience": 4,
                    "area_drop_ratio": 0.25,
                    "edge_touch_ratio": 0.5,
                    "iou_low_threshold": 0.45,
                }
            }
        )

        result = build_reprompt_thresholds(cfg)

        self.assertEqual(
            result,
            {
                "empty_mask_patience": 4,
                "area_drop_ratio": 0.25,
                "edge_touch_ratio": 0.5,
                "iou_low_threshold": 0.45,
            },
        )


if __name__ == "__main__":
    unittest.main()
