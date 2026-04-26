import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_refined_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class RefinedCliTests(unittest.TestCase):
    def test_parser_accepts_refined_arguments(self):
        from scripts import offline_app_refined

        parser = offline_app_refined.build_parser()
        args = parser.parse_args(
            [
                "--input_video",
                "sample.mp4",
                "--config",
                "configs/body4d_refined.yaml",
                "--detector_backend",
                "yolo",
                "--track_chunk_size",
                "96",
                "--max_targets",
                "3",
                "--disable_mask_refine",
                "--save_debug_metrics",
            ]
        )

        self.assertEqual(args.input_video, "sample.mp4")
        self.assertEqual(args.config, "configs/body4d_refined.yaml")
        self.assertEqual(args.detector_backend, "yolo")
        self.assertEqual(args.track_chunk_size, 96)
        self.assertEqual(args.max_targets, 3)
        self.assertTrue(args.disable_mask_refine)
        self.assertTrue(args.save_debug_metrics)
        expected_default = os.path.abspath(
            os.path.join(
                os.path.dirname(offline_app_refined.__file__),
                "..",
                "configs",
                "body4d_refined.yaml",
            )
        )
        self.assertEqual(parser.get_default("config"), expected_default)

    def test_load_refined_config_reads_detector_backend(self):
        from scripts.offline_app_refined import load_refined_config

        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "body4d_refined.yaml",
            )
        )
        cfg = load_refined_config(config_path)
        self.assertEqual(cfg.detector.backend, "yolo")
        self.assertTrue(hasattr(cfg, "tracking"))
        self.assertTrue(hasattr(cfg, "reprompt"))

    def test_load_low_memory_refined_config_reads_completion_safety_profile(self):
        from scripts.offline_app_refined import load_refined_config

        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "body4d_refined_low_memory.yaml",
            )
        )
        cfg = load_refined_config(config_path)

        self.assertEqual(cfg.detector.backend, "yolo")
        self.assertEqual(cfg.sam_3d_body.batch_size, 16)
        self.assertEqual(cfg.completion.detection_resolution, [192, 384])
        self.assertEqual(cfg.completion.completion_resolution, [256, 512])
        self.assertEqual(cfg.completion.batch_size, 1)
        self.assertEqual(cfg.completion.decode_chunk_size, 1)
        self.assertEqual(cfg.completion.max_occ_len, 8)
        self.assertTrue(cfg.refine.enable)

    def test_main_applies_output_dir_override(self):
        import scripts.offline_app_refined as offline_app_refined

        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir="./custom_out",
            config="configs/body4d_refined.yaml",
            detector_backend="yolo",
            track_chunk_size=None,
            max_targets=None,
            disable_mask_refine=False,
            disable_auto_reprompt=False,
            save_debug_metrics=False,
            skip_existing=False,
        )

        with patch.object(offline_app_refined, "build_parser") as mock_build_parser, patch.object(
            offline_app_refined, "run_refined_pipeline"
        ) as mock_run_refined_pipeline:
            mock_build_parser.return_value.parse_args.return_value = args
            offline_app_refined.main()

        mock_run_refined_pipeline.assert_called_once_with(args)

    def test_main_preserves_config_detector_backend_when_arg_not_supplied(self):
        import scripts.offline_app_refined as offline_app_refined

        args = offline_app_refined.build_parser().parse_args(["--input_video", "sample.mp4"])

        with patch.object(offline_app_refined, "build_parser") as mock_build_parser, patch.object(
            offline_app_refined, "run_refined_pipeline"
        ) as mock_run_refined_pipeline:
            mock_build_parser.return_value.parse_args.return_value = args
            offline_app_refined.main()

        mock_run_refined_pipeline.assert_called_once_with(args)

    def test_run_refined_pipeline_applies_overrides_and_executes_planned_flow(self):
        import scripts.offline_app_refined as offline_app_refined

        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "rtmdet", "max_targets": 0},
                "tracking": {"chunk_size": 180},
                "refine": {"enable": True},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir="./custom_out",
            config="configs/body4d_refined.yaml",
            detector_backend="yolo",
            track_chunk_size=96,
            max_targets=2,
            disable_mask_refine=True,
            disable_auto_reprompt=True,
            save_debug_metrics=True,
            skip_existing=True,
        )
        sample = {"frames": ["f0.png", "f1.png"], "output_dir": "./sample_out"}
        initial_targets = {"obj_ids": [1, 4]}
        chunks = [{"chunk_id": 0}, {"chunk_id": 1}]
        raw_chunks = [{"raw": 0}, {"raw": 1}]
        refined_chunks = [{"refined": 0}, {"refined": 1}]
        final_chunks = [{"final": 0}, {"final": 1}]

        app = unittest.mock.MagicMock()
        app.prepare_input.return_value = sample
        app.detect_initial_targets.return_value = initial_targets
        app.iter_chunks.return_value = chunks
        app.track_chunk.side_effect = raw_chunks
        app.refine_chunk_masks.side_effect = refined_chunks
        app.maybe_reprompt_chunk.side_effect = final_chunks

        with patch.object(offline_app_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_app_refined, "RefinedOfflineApp", return_value=app
        ) as mock_refined_app:
            offline_app_refined.run_refined_pipeline(args)

        self.assertEqual(cfg.runtime.output_dir, "./custom_out")
        self.assertEqual(cfg.detector.backend, "yolo")
        self.assertEqual(cfg.detector.max_targets, 2)
        self.assertEqual(cfg.tracking.chunk_size, 96)
        self.assertFalse(cfg.refine.enable)
        self.assertFalse(cfg.reprompt.enable)
        self.assertTrue(cfg.debug.save_metrics)
        mock_refined_app.assert_called_once_with(args.config, config=cfg)
        self.assertEqual(
            app.reprompt_thresholds,
            {
                "empty_mask_patience": 3,
                "area_drop_ratio": 0.35,
                "edge_touch_ratio": 0.4,
                "iou_low_threshold": 0.55,
            },
        )
        app.prepare_input.assert_called_once_with("sample.mp4", "./custom_out", True)
        app.detect_initial_targets.assert_called_once_with(sample)
        app.prepare_sample_output.assert_called_once_with("./sample_out", [1, 4])
        app.iter_chunks.assert_called_once_with(["f0.png", "f1.png"], 96)
        self.assertEqual(app.track_chunk.call_args_list, [unittest.mock.call(chunks[0], initial_targets), unittest.mock.call(chunks[1], initial_targets)])
        self.assertEqual(app.refine_chunk_masks.call_args_list, [unittest.mock.call(raw_chunks[0]), unittest.mock.call(raw_chunks[1])])
        self.assertEqual(
            app.maybe_reprompt_chunk.call_args_list,
            [unittest.mock.call(chunks[0], refined_chunks[0], initial_targets), unittest.mock.call(chunks[1], refined_chunks[1], initial_targets)],
        )
        self.assertEqual(
            app.write_chunk_outputs.call_args_list,
            [
                unittest.mock.call(chunks[0], raw_chunks[0], final_chunks[0]),
                unittest.mock.call(chunks[1], raw_chunks[1], final_chunks[1]),
            ],
        )
        app.run_refined_4d_generation.assert_called_once_with()
        app.finalize_sample.assert_called_once_with()

    def test_run_refined_pipeline_finalizes_sample_when_chunk_processing_fails(self):
        import scripts.offline_app_refined as offline_app_refined

        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "yolo"},
                "tracking": {"chunk_size": 16},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir=None,
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            max_targets=None,
            disable_mask_refine=False,
            disable_auto_reprompt=False,
            save_debug_metrics=False,
            skip_existing=False,
        )
        app = unittest.mock.MagicMock()
        app.prepare_input.return_value = {"frames": ["f0.png"], "output_dir": "./sample_out"}
        app.detect_initial_targets.return_value = {"obj_ids": [1]}
        app.iter_chunks.return_value = [{"chunk_id": 0}]
        app.track_chunk.side_effect = RuntimeError("chunk failed")

        with patch.object(offline_app_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_app_refined, "RefinedOfflineApp", return_value=app
        ):
            with self.assertRaisesRegex(RuntimeError, "chunk failed"):
                offline_app_refined.run_refined_pipeline(args)

        app.prepare_sample_output.assert_called_once_with("./sample_out", [1])
        app.finalize_sample.assert_called_once_with()
        app.run_refined_4d_generation.assert_not_called()

    def test_run_refined_pipeline_finalizes_sample_when_setup_fails(self):
        import scripts.offline_app_refined as offline_app_refined

        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "yolo"},
                "tracking": {"chunk_size": 16},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": True},
            }
        )
        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir=None,
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            max_targets=None,
            disable_mask_refine=False,
            disable_auto_reprompt=False,
            save_debug_metrics=False,
            skip_existing=False,
        )
        app = unittest.mock.MagicMock()
        app.prepare_input.side_effect = RuntimeError("setup failed")

        with patch.object(offline_app_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_app_refined, "RefinedOfflineApp", return_value=app
        ):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                offline_app_refined.run_refined_pipeline(args)

        app.finalize_sample.assert_called_once_with()
        self.assertEqual(app.sample_summary["status"], "failed")
        app.detect_initial_targets.assert_not_called()
        app.iter_chunks.assert_not_called()

    def test_main_routes_through_run_refined_pipeline(self):
        import scripts.offline_app_refined as offline_app_refined

        args = offline_app_refined.build_parser().parse_args(["--input_video", "sample.mp4"])

        with patch.object(offline_app_refined, "build_parser") as mock_build_parser, patch.object(
            offline_app_refined, "run_refined_pipeline"
        ) as mock_run_refined_pipeline:
            mock_build_parser.return_value.parse_args.return_value = args

            offline_app_refined.main()

        mock_run_refined_pipeline.assert_called_once_with(args)

    def test_prepare_input_collects_frame_stems_for_image_directory(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {"runtime": {"output_dir": "./outputs_refined"}, "debug": {"save_metrics": False}}
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)

        with make_workspace_tempdir() as tmpdir:
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(tmpdir, "frame_a.jpg")
            )
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(tmpdir, "frame_b.png")
            )

            sample = app.prepare_input(tmpdir, None, False)

        self.assertEqual(sample["input_type"], "images")
        self.assertEqual(sample["frame_count"], 2)
        self.assertEqual(sample["frames"], ["00000000", "00000001"])
        self.assertEqual(len(sample["source_frames"]), 2)
        self.assertEqual(os.path.basename(os.path.dirname(sample["output_dir"])), "outputs_refined")
        self.assertNotEqual(os.path.basename(sample["output_dir"]), "outputs_refined")

    def test_run_refined_pipeline_passes_none_output_dir_to_preserve_default_sample_subdir_behavior(self):
        import scripts.offline_app_refined as offline_app_refined

        cfg = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "yolo"},
                "tracking": {"chunk_size": 16},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir=None,
            config="configs/body4d_refined.yaml",
            detector_backend=None,
            track_chunk_size=None,
            max_targets=None,
            disable_mask_refine=False,
            disable_auto_reprompt=False,
            save_debug_metrics=False,
            skip_existing=False,
        )
        app = unittest.mock.MagicMock()
        app.prepare_input.return_value = {"frames": [], "output_dir": "./sample_out"}
        app.detect_initial_targets.return_value = {"obj_ids": [1]}
        app.iter_chunks.return_value = []

        with patch.object(offline_app_refined, "load_refined_config", return_value=cfg), patch.object(
            offline_app_refined, "RefinedOfflineApp", return_value=app
        ):
            offline_app_refined.run_refined_pipeline(args)

        app.prepare_input.assert_called_once_with("sample.mp4", None, False)

    def test_sync_base_app_runtime_propagates_pose_exports_and_storage_flags(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {
                    "output_dir": "./outputs_refined",
                    "pose_exports": ["coco17", "coco_wholebody"],
                    "save_rendered_video": False,
                    "save_rendered_video_direct": True,
                    "save_rendered_frames": False,
                    "save_rendered_frames_individual": False,
                    "save_mesh_4d_individual": True,
                    "save_focal_4d_individual": False,
                },
                "wan_export": {
                    "enable": True,
                    "fps": 25,
                    "resolution_area": [512, 768],
                    "face_resolution": [512, 512],
                },
                "sam_3d_body": {"batch_size": 12},
                "completion": {
                    "detection_resolution": [192, 384],
                    "completion_resolution": [256, 512],
                },
                "detector": {"backend": "yolo"},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        runtime_app = unittest.mock.MagicMock()
        runtime_app.RUNTIME = {}
        runtime_app.OUTPUT_DIR = ""
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.device = "cuda"

        with patch.object(app, "_configure_detector"):
            app._sync_base_app_runtime(runtime_app, output_dir="./sample_out")

        self.assertEqual(runtime_app.RUNTIME["batch_size"], 12)
        self.assertEqual(runtime_app.RUNTIME["detection_resolution"], [192, 384])
        self.assertEqual(runtime_app.RUNTIME["completion_resolution"], [256, 512])
        self.assertEqual(runtime_app.RUNTIME["pose_exports"], ["coco17", "coco_wholebody"])
        self.assertFalse(runtime_app.RUNTIME["save_rendered_video"])
        self.assertTrue(runtime_app.RUNTIME["save_rendered_video_direct"])
        self.assertFalse(runtime_app.RUNTIME["save_rendered_frames"])
        self.assertFalse(runtime_app.RUNTIME["save_rendered_frames_individual"])
        self.assertTrue(runtime_app.RUNTIME["save_mesh_4d_individual"])
        self.assertFalse(runtime_app.RUNTIME["save_focal_4d_individual"])
        self.assertEqual(
            runtime_app.RUNTIME["wan_export"],
            {
                "enable": True,
                "fps": 25,
                "resolution_area": [512, 768],
                "face_resolution": [512, 512],
            },
        )

    def test_iter_chunks_yields_stable_chunk_metadata(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create({"runtime": {"output_dir": "./outputs_refined"}, "debug": {"save_metrics": False}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)

        chunks = list(app.iter_chunks(["00000000", "00000001", "00000002"], 2))

        self.assertEqual(
            chunks,
            [
                {
                    "chunk_id": 0,
                    "start_frame": 0,
                    "end_frame": 1,
                    "frame_indices": [0, 1],
                    "frames": ["00000000", "00000001"],
                },
                {
                    "chunk_id": 1,
                    "start_frame": 2,
                    "end_frame": 2,
                    "frame_indices": [2],
                    "frames": ["00000002"],
                },
            ],
        )

    def test_detect_initial_targets_initializes_tracker_from_best_detected_frame(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"bbox_thresh": 0.35, "iou_thresh": 0.5},
                "batch": {"initial_search_frames": 3},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {
            "input_video": "sample.mp4",
            "input_type": "video",
            "source_frames": None,
            "frames": ["00000000", "00000001", "00000002"],
            "frame_count": 3,
            "output_dir": "./outputs_refined",
        }
        app._load_source_frame = unittest.mock.MagicMock(
            side_effect=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        )

        predictor = unittest.mock.MagicMock()
        predictor.init_state.return_value = {"video_height": 8, "video_width": 8}
        predictor.add_new_points_or_box.return_value = (None, [1], None, None)
        runtime_app = unittest.mock.MagicMock()
        runtime_app.predictor = predictor
        runtime_app.RUNTIME = {}
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.process_one_image.side_effect = [
            [],
            [{"bbox": [1.0, 2.0, 5.0, 6.0]}],
            [],
        ]

        with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
            targets = app.detect_initial_targets(app.sample_state)

        self.assertEqual(targets["obj_ids"], [1])
        self.assertEqual(targets["start_frame_idx"], 1)
        predictor.clear_all_points_in_video.assert_called_once_with(
            {"video_height": 8, "video_width": 8}
        )
        predictor.add_new_points_or_box.assert_called_once()

    def test_detect_initial_targets_selects_frame_with_most_detections(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"bbox_thresh": 0.35, "iou_thresh": 0.5},
                "batch": {"initial_search_frames": 4},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {
            "input_video": "sample.mp4",
            "input_type": "video",
            "source_frames": None,
            "frames": ["00000000", "00000001", "00000002", "00000003"],
            "frame_count": 4,
            "output_dir": "./outputs_refined",
        }
        app._load_source_frame = unittest.mock.MagicMock(
            side_effect=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
        )

        predictor = unittest.mock.MagicMock()
        predictor.init_state.return_value = {"video_height": 8, "video_width": 8}
        predictor.add_new_points_or_box.side_effect = [
            (None, [1], None, None),
            (None, [1, 2], None, None),
            (None, [1, 2, 3], None, None),
        ]
        runtime_app = unittest.mock.MagicMock()
        runtime_app.predictor = predictor
        runtime_app.RUNTIME = {}
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.process_one_image.side_effect = [
            [{"bbox": [0.0, 0.0, 2.0, 2.0]}],
            [{"bbox": [1.0, 1.0, 3.0, 3.0]}],
            [
                {"bbox": [1.0, 2.0, 5.0, 6.0]},
                {"bbox": [2.0, 1.0, 6.0, 5.0]},
                {"bbox": [0.0, 0.0, 4.0, 4.0]},
            ],
            [{"bbox": [0.0, 0.0, 2.0, 2.0]}, {"bbox": [4.0, 4.0, 6.0, 6.0]}],
        ]

        with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
            targets = app.detect_initial_targets(app.sample_state)

        self.assertEqual(targets["obj_ids"], [1, 2, 3])
        self.assertEqual(targets["start_frame_idx"], 2)
        self.assertEqual(
            targets["boxes_xyxy"],
            [
                [1.0, 2.0, 5.0, 6.0],
                [2.0, 1.0, 6.0, 5.0],
                [0.0, 0.0, 4.0, 4.0],
            ],
        )
        self.assertEqual(predictor.add_new_points_or_box.call_count, 3)
        for call in predictor.add_new_points_or_box.call_args_list:
            self.assertEqual(call.kwargs["frame_idx"], 2)

    def test_detect_initial_targets_limits_to_max_targets_by_score(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {
                    "bbox_thresh": 0.35,
                    "iou_thresh": 0.5,
                    "max_targets": 2,
                },
                "batch": {"initial_search_frames": 1},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {
            "input_video": "sample.mp4",
            "input_type": "video",
            "source_frames": None,
            "frames": ["00000000"],
            "frame_count": 1,
            "output_dir": "./outputs_refined",
        }
        app._load_source_frame = unittest.mock.MagicMock(
            return_value=np.zeros((8, 8, 3), dtype=np.uint8)
        )

        predictor = unittest.mock.MagicMock()
        predictor.init_state.return_value = {"video_height": 8, "video_width": 8}
        predictor.add_new_points_or_box.side_effect = [
            (None, [1], None, None),
            (None, [1, 2], None, None),
        ]
        detector = unittest.mock.MagicMock()
        detector.run_human_detection.return_value = [
            {"bbox": [0.0, 0.0, 2.0, 2.0], "score": 0.55},
            {"bbox": [1.0, 1.0, 3.0, 3.0], "score": 0.95},
            {"bbox": [2.0, 2.0, 4.0, 4.0], "score": 0.80},
        ]
        runtime_app = unittest.mock.MagicMock()
        runtime_app.predictor = predictor
        runtime_app.RUNTIME = {}
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.detector = detector

        with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
            targets = app.detect_initial_targets(app.sample_state)

        self.assertEqual(targets["obj_ids"], [1, 2])
        self.assertEqual(
            targets["boxes_xyxy"],
            [
                [1.0, 1.0, 3.0, 3.0],
                [2.0, 2.0, 4.0, 4.0],
            ],
        )
        self.assertEqual(predictor.add_new_points_or_box.call_count, 2)

    def test_detect_initial_targets_limits_to_max_targets_by_original_order_when_scores_missing(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {
                    "bbox_thresh": 0.35,
                    "iou_thresh": 0.5,
                    "max_targets": 2,
                },
                "batch": {"initial_search_frames": 1},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {
            "input_video": "sample.mp4",
            "input_type": "video",
            "source_frames": None,
            "frames": ["00000000"],
            "frame_count": 1,
            "output_dir": "./outputs_refined",
        }
        app._load_source_frame = unittest.mock.MagicMock(
            return_value=np.zeros((8, 8, 3), dtype=np.uint8)
        )

        predictor = unittest.mock.MagicMock()
        predictor.init_state.return_value = {"video_height": 8, "video_width": 8}
        predictor.add_new_points_or_box.side_effect = [
            (None, [1], None, None),
            (None, [1, 2], None, None),
        ]
        detector = unittest.mock.MagicMock()
        detector.run_human_detection.return_value = [
            {"bbox": [0.0, 0.0, 2.0, 2.0]},
            {"bbox": [1.0, 1.0, 3.0, 3.0]},
            {"bbox": [2.0, 2.0, 4.0, 4.0]},
        ]
        runtime_app = unittest.mock.MagicMock()
        runtime_app.predictor = predictor
        runtime_app.RUNTIME = {}
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.detector = detector

        with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
            targets = app.detect_initial_targets(app.sample_state)

        self.assertEqual(targets["obj_ids"], [1, 2])
        self.assertEqual(
            targets["boxes_xyxy"],
            [
                [0.0, 0.0, 2.0, 2.0],
                [1.0, 1.0, 3.0, 3.0],
            ],
        )
        self.assertEqual(predictor.add_new_points_or_box.call_count, 2)

    def test_configure_detector_ignores_yolo_weights_when_backend_is_vitdet(self):
        import sys

        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "detector": {"backend": "vitdet", "weights_path": "yolo11n.pt"},
                "sam_3d_body": {"detector_path": ""},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        runtime_app = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model = unittest.mock.MagicMock()
        runtime_app.sam3_3d_body_model.device = "cuda"
        runtime_app.sam3_3d_body_model.detector = None

        human_detector_mock = unittest.mock.MagicMock()
        fake_build_detector = SimpleNamespace(HumanDetector=human_detector_mock)
        with patch.dict(sys.modules, {"models.sam_3d_body.tools.build_detector": fake_build_detector}):
            app._configure_detector(runtime_app)

        human_detector_mock.assert_called_once()
        _, kwargs = human_detector_mock.call_args
        self.assertEqual(kwargs["name"], "vitdet")
        self.assertEqual(kwargs["device"], "cuda")
        self.assertNotIn("weights_path", kwargs)

    def test_save_indexed_mask_preserves_labels_and_palette(self):
        from scripts.offline_app_refined import save_indexed_mask

        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        with make_workspace_tempdir() as tmpdir:
            mask_path = os.path.join(tmpdir, "mask.png")
            save_indexed_mask(mask, mask_path)

            saved = Image.open(mask_path)

            self.assertEqual(saved.mode, "P")
            self.assertGreater(len(saved.getpalette() or []), 3)
            np.testing.assert_array_equal(np.array(saved), mask)

    def test_write_chunk_outputs_persists_refined_masks_and_chunk_record(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create({"debug": {"save_metrics": True}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)

        with make_workspace_tempdir() as tmpdir:
            app.OUTPUT_DIR = tmpdir
            app.prepare_sample_output(tmpdir, [1])
            chunk = {
                "chunk_id": 0,
                "start_frame": 0,
                "end_frame": 0,
                "frame_indices": [0],
                "frames": ["00000000"],
            }
            final_chunk = {
                "frame_stems": ["00000000"],
                "refined_masks": [np.array([[0, 1], [1, 0]], dtype=np.uint8)],
                "frame_metrics": [{"frame_idx": 0, "triggered_reprompt": False}],
                "reprompt_events": [],
            }

            app.write_chunk_outputs(chunk, {"frame_stems": ["00000000"]}, final_chunk)

            refined_path = os.path.join(tmpdir, "masks_refined", "00000000.png")
            working_path = os.path.join(tmpdir, "masks", "00000000.png")
            self.assertTrue(os.path.isfile(refined_path))
            self.assertTrue(os.path.isfile(working_path))
            self.assertEqual(len(app.chunk_records), 1)
            self.assertEqual(app.chunk_records[0]["chunk_id"], 0)
            self.assertEqual(app.chunk_records[0]["frame_count"], 1)

    def test_refine_chunk_masks_can_be_disabled_to_passthrough_raw_masks(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "refine": {"enable": False},
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.initial_targets = {"obj_ids": [1]}

        raw_mask = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        empty_mask = np.zeros((3, 3), dtype=np.uint8)
        raw_chunk = {
            "frame_indices": [0, 1],
            "frame_stems": ["00000000", "00000001"],
            "raw_masks": [raw_mask, empty_mask],
        }

        refined_chunk = app.refine_chunk_masks(raw_chunk)

        np.testing.assert_array_equal(refined_chunk["refined_masks"][0], raw_mask)
        np.testing.assert_array_equal(refined_chunk["refined_masks"][1], empty_mask)
        self.assertFalse(refined_chunk["frame_metrics"][0]["track_metrics"]["1"]["refined_from_previous"])
        self.assertFalse(refined_chunk["frame_metrics"][1]["track_metrics"]["1"]["refined_from_previous"])
        self.assertEqual(refined_chunk["frame_metrics"][1]["track_metrics"]["1"]["empty_mask_count"], 1)
        self.assertEqual(app._empty_mask_counts[1], 1)

    def test_run_refined_4d_generation_uses_lazy_runtime_app(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create({"debug": {"save_metrics": False}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {"input_video": "sample.mp4"}

        with make_workspace_tempdir() as tmpdir:
            app.OUTPUT_DIR = tmpdir
            os.makedirs(os.path.join(tmpdir, "masks_refined"), exist_ok=True)
            Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
                os.path.join(tmpdir, "masks_refined", "00000000.png")
            )

            runtime_app = unittest.mock.MagicMock()
            runtime_app.OUTPUT_DIR = tmpdir
            runtime_app.on_4d_generation.return_value = os.path.join(tmpdir, "rendered.mp4")

            with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
                out_path = app.run_refined_4d_generation()

        runtime_app.on_4d_generation.assert_called_once_with(video_path="sample.mp4")
        self.assertTrue(out_path.endswith("rendered.mp4"))

    def test_run_refined_4d_generation_disables_autocast(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create({"debug": {"save_metrics": False}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {"input_video": "sample.mp4"}

        state = {"entered": False}

        @contextmanager
        def fake_autocast_disabled():
            state["entered"] = True
            try:
                yield
            finally:
                state["entered"] = False

        with make_workspace_tempdir() as tmpdir:
            app.OUTPUT_DIR = tmpdir
            os.makedirs(os.path.join(tmpdir, "masks_refined"), exist_ok=True)
            Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
                os.path.join(tmpdir, "masks_refined", "00000000.png")
            )

            runtime_app = unittest.mock.MagicMock()
            runtime_app.OUTPUT_DIR = tmpdir

            def assert_autocast_disabled(*, video_path):
                self.assertTrue(state["entered"])
                self.assertEqual(video_path, "sample.mp4")
                return os.path.join(tmpdir, "rendered.mp4")

            runtime_app.on_4d_generation.side_effect = assert_autocast_disabled

            with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True), patch(
                "scripts.offline_app_refined._autocast_disabled",
                side_effect=fake_autocast_disabled,
            ) as autocast_disabled:
                out_path = app.run_refined_4d_generation()

        autocast_disabled.assert_called_once_with()
        self.assertTrue(out_path.endswith("rendered.mp4"))

    def test_run_refined_4d_generation_aligns_completion_pipeline_module_dtypes(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        class FakeModule:
            def __init__(self, dtype):
                self._parameter = SimpleNamespace(dtype=dtype)
                self.to = unittest.mock.MagicMock()

            def parameters(self):
                yield self._parameter

        config = OmegaConf.create({"debug": {"save_metrics": False}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.sample_state = {"input_video": "sample.mp4"}

        pipeline_mask = SimpleNamespace(
            image_encoder=FakeModule("half"),
            unet=FakeModule("float"),
            vae=FakeModule("float"),
        )
        pipeline_rgb = SimpleNamespace(
            image_encoder=FakeModule("float16"),
            unet=FakeModule("float32"),
            vae=FakeModule("float32"),
        )

        with make_workspace_tempdir() as tmpdir:
            app.OUTPUT_DIR = tmpdir
            os.makedirs(os.path.join(tmpdir, "masks_refined"), exist_ok=True)
            Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
                os.path.join(tmpdir, "masks_refined", "00000000.png")
            )

            runtime_app = unittest.mock.MagicMock()
            runtime_app.OUTPUT_DIR = tmpdir
            runtime_app.pipeline_mask = pipeline_mask
            runtime_app.pipeline_rgb = pipeline_rgb
            runtime_app.on_4d_generation.return_value = os.path.join(tmpdir, "rendered.mp4")

            with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True), patch(
                "scripts.offline_app_refined._autocast_disabled"
            ):
                out_path = app.run_refined_4d_generation()

        pipeline_mask.unet.to.assert_called_once_with(dtype="half")
        pipeline_mask.vae.to.assert_called_once_with(dtype="half")
        pipeline_rgb.unet.to.assert_called_once_with(dtype="float16")
        pipeline_rgb.vae.to.assert_called_once_with(dtype="float16")
        self.assertTrue(out_path.endswith("rendered.mp4"))

    def test_reset_sample_state_clears_sample_local_fields(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create({"debug": {"save_metrics": False}})
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.chunk_records = [{"chunk_id": 1}]
        app.output_paths = {"debug_metrics": "./tmp/debug_metrics"}
        app.reprompt_events = [{"chunk_id": 1, "reason": "empty_mask_patience"}]
        app.sample_summary = {"status": "failed", "runtime_profile": {"tracking.chunk_size": 32}}

        app.reset_sample_state()

        self.assertEqual(app.chunk_records, [])
        self.assertEqual(app.output_paths, {})
        self.assertEqual(app.reprompt_events, [])
        self.assertEqual(app.sample_summary, {})

    def test_run_sample_uses_runtime_profile_without_mutating_global_config(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        sample = {"frames": ["f0.png", "f1.png"], "output_dir": "./sample_out"}
        initial_targets = {"obj_ids": [1]}
        chunks = [{"chunk_id": 0}, {"chunk_id": 1}]
        runtime_profile = {
            "tracking.chunk_size": 96,
            "sam_3d_body.batch_size": 2,
            "initial_search_frames": 24,
            "retry_index": 1,
            "reason": "retry_after_empty_masks",
        }

        app.prepare_input = unittest.mock.MagicMock(return_value=sample)
        app.detect_initial_targets = unittest.mock.MagicMock(return_value=initial_targets)
        app.prepare_sample_output = unittest.mock.MagicMock(return_value={"images": "./sample_out/images"})
        app.iter_chunks = unittest.mock.MagicMock(return_value=chunks)
        app.track_chunk = unittest.mock.MagicMock(side_effect=[{"raw": 0}, {"raw": 1}])
        app.refine_chunk_masks = unittest.mock.MagicMock(side_effect=[{"refined": 0}, {"refined": 1}])
        app.maybe_reprompt_chunk = unittest.mock.MagicMock(side_effect=[{"final": 0}, {"final": 1}])
        app.write_chunk_outputs = unittest.mock.MagicMock()
        app.run_refined_4d_generation = unittest.mock.MagicMock()

        app.run_sample("sample.mp4", "./custom_out", skip_existing=True, runtime_profile=runtime_profile)

        app.prepare_input.assert_called_once_with("sample.mp4", "./custom_out", True)
        app.iter_chunks.assert_called_once_with(["f0.png", "f1.png"], 96)
        self.assertEqual(config.tracking.chunk_size, 180)
        self.assertEqual(app.sample_summary["status"], "completed")
        self.assertEqual(app.sample_summary["runtime_profile"], runtime_profile)

    def test_run_sample_writes_video_source_fps_into_runtime_and_wan_summary(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        with make_workspace_tempdir() as tmpdir:
            export_root = os.path.join(tmpdir, "WanExport")
            sample_output_dir = os.path.join(tmpdir, "sample_out")
            os.makedirs(sample_output_dir, exist_ok=True)

            config = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "tracking": {"chunk_size": 180},
                    "wan_export": {"enable": True, "output_dir": export_root, "fps": 25},
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                }
            )
            app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
            sample = {
                "frames": [],
                "output_dir": sample_output_dir,
                "input_video": os.path.join(tmpdir, "sample.mp4"),
                "input_type": "video",
            }

            app.prepare_input = unittest.mock.MagicMock(return_value=sample)
            app.detect_initial_targets = unittest.mock.MagicMock(return_value={"obj_ids": [1]})
            app.prepare_sample_output = unittest.mock.MagicMock(
                return_value={"images": os.path.join(sample_output_dir, "images")}
            )
            app.iter_chunks = unittest.mock.MagicMock(return_value=[])
            app.track_chunk = unittest.mock.MagicMock()
            app.refine_chunk_masks = unittest.mock.MagicMock()
            app.maybe_reprompt_chunk = unittest.mock.MagicMock()
            app.write_chunk_outputs = unittest.mock.MagicMock()
            app.run_refined_4d_generation = unittest.mock.MagicMock()

            with patch("scripts.offline_app_refined.time.perf_counter", side_effect=[10.0, 16.5]), patch(
                "scripts.wan_sample_export.uuid.uuid4",
                return_value=SimpleNamespace(hex="runtimeuuid123456"),
            ), patch(
                "scripts.offline_app_refined.resolve_video_fps",
                return_value=29.97,
            ):
                app.run_sample("sample.mp4", "./custom_out", skip_existing=False, runtime_profile=None)

            runtime_path = os.path.join(sample_output_dir, "sample_runtime.json")
            self.assertTrue(os.path.isfile(runtime_path))
            with open(runtime_path, "r", encoding="utf-8") as handle:
                runtime_payload = json.load(handle)
            self.assertEqual(runtime_payload["status"], "completed")
            self.assertAlmostEqual(runtime_payload["pipeline_seconds"], 6.5)
            self.assertEqual(
                runtime_payload["fps_summary"],
                {
                    "source_fps": 29.97,
                    "source_fps_source": "video_metadata",
                    "rendered_4d_fps": 25.0,
                    "wan_target_fps": 25,
                },
            )

            summary_path = os.path.join(export_root, "runtimeuuid123456_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            with open(summary_path, "r", encoding="utf-8") as handle:
                wan_summary = json.load(handle)
            self.assertEqual(wan_summary["sample_uuid"], "runtimeuuid123456")
            self.assertEqual(wan_summary["fps_summary"], runtime_payload["fps_summary"])
            self.assertEqual(wan_summary["pipeline_runtime"]["status"], "completed")
            self.assertAlmostEqual(wan_summary["pipeline_runtime"]["pipeline_seconds"], 6.5)

    def test_run_sample_marks_image_sequence_source_fps_as_unavailable(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        with make_workspace_tempdir() as tmpdir:
            sample_output_dir = os.path.join(tmpdir, "sample_out")
            os.makedirs(sample_output_dir, exist_ok=True)

            config = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "tracking": {"chunk_size": 180},
                    "wan_export": {"enable": False},
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                }
            )
            app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
            sample = {
                "frames": [],
                "output_dir": sample_output_dir,
                "input_video": os.path.join(tmpdir, "frames_dir"),
                "input_type": "images",
            }

            app.prepare_input = unittest.mock.MagicMock(return_value=sample)
            app.detect_initial_targets = unittest.mock.MagicMock(return_value={"obj_ids": [1]})
            app.prepare_sample_output = unittest.mock.MagicMock(
                return_value={"images": os.path.join(sample_output_dir, "images")}
            )
            app.iter_chunks = unittest.mock.MagicMock(return_value=[])
            app.track_chunk = unittest.mock.MagicMock()
            app.refine_chunk_masks = unittest.mock.MagicMock()
            app.maybe_reprompt_chunk = unittest.mock.MagicMock()
            app.write_chunk_outputs = unittest.mock.MagicMock()
            app.run_refined_4d_generation = unittest.mock.MagicMock()

            with patch("scripts.offline_app_refined.time.perf_counter", side_effect=[20.0, 22.0]):
                app.run_sample("frames_dir", "./custom_out", skip_existing=False, runtime_profile=None)

            runtime_path = os.path.join(sample_output_dir, "sample_runtime.json")
            self.assertTrue(os.path.isfile(runtime_path))
            with open(runtime_path, "r", encoding="utf-8") as handle:
                runtime_payload = json.load(handle)
            self.assertEqual(
                runtime_payload["fps_summary"],
                {
                    "source_fps": None,
                    "source_fps_source": "image_sequence",
                    "rendered_4d_fps": 25.0,
                    "wan_target_fps": None,
                },
            )

    def test_run_sample_finalizes_and_marks_failed_when_setup_fails(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        runtime_profile = {
            "tracking.chunk_size": 96,
            "retry_index": 1,
            "reason": "retry_after_empty_masks",
        }
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        app.prepare_input = unittest.mock.MagicMock(side_effect=RuntimeError("setup failed"))
        app.detect_initial_targets = unittest.mock.MagicMock()
        app.prepare_sample_output = unittest.mock.MagicMock()
        app.iter_chunks = unittest.mock.MagicMock()
        app.finalize_sample = unittest.mock.MagicMock()

        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            app.run_sample("sample.mp4", "./custom_out", skip_existing=True, runtime_profile=runtime_profile)

        app.finalize_sample.assert_called_once_with()
        self.assertEqual(app.sample_summary["status"], "failed")
        self.assertEqual(app.sample_summary["runtime_profile"], runtime_profile)
        app.detect_initial_targets.assert_not_called()
        app.prepare_sample_output.assert_not_called()
        app.iter_chunks.assert_not_called()

    def test_run_sample_applies_profile_fields_to_sample_local_config_and_restores_shared_config(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {"initial_search_frames": 24},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        runtime_profile = {
            "tracking.chunk_size": 96,
            "sam_3d_body.batch_size": 20,
            "initial_search_frames": 48,
            "retry_index": 2,
            "reason": "safer_reconstruction",
        }
        sample = {"frames": [], "output_dir": "./sample_out"}
        initial_targets = {"obj_ids": [1]}
        observed = {}

        app.prepare_input = unittest.mock.MagicMock(return_value=sample)
        app.detect_initial_targets = unittest.mock.MagicMock(return_value=initial_targets)
        app.prepare_sample_output = unittest.mock.MagicMock(return_value={"images": "./sample_out/images"})
        app.iter_chunks = unittest.mock.MagicMock(return_value=[])
        app.track_chunk = unittest.mock.MagicMock()
        app.refine_chunk_masks = unittest.mock.MagicMock()
        app.maybe_reprompt_chunk = unittest.mock.MagicMock()
        app.write_chunk_outputs = unittest.mock.MagicMock()

        def capture_active_config():
            observed["tracking.chunk_size"] = int(app.CONFIG.tracking.chunk_size)
            observed["sam_3d_body.batch_size"] = int(app.CONFIG.sam_3d_body.batch_size)
            observed["batch.initial_search_frames"] = int(app.CONFIG.batch.initial_search_frames)

        app.run_refined_4d_generation = unittest.mock.MagicMock(side_effect=capture_active_config)

        app.run_sample("sample.mp4", "./custom_out", skip_existing=False, runtime_profile=runtime_profile)

        self.assertEqual(observed["tracking.chunk_size"], 96)
        self.assertEqual(observed["sam_3d_body.batch_size"], 20)
        self.assertEqual(observed["batch.initial_search_frames"], 48)
        self.assertEqual(config.tracking.chunk_size, 180)
        self.assertEqual(config.sam_3d_body.batch_size, 32)
        self.assertEqual(config.batch.initial_search_frames, 24)
        self.assertEqual(app.CONFIG, config)

    def test_run_sample_restores_shared_config_even_when_finalize_raises(self):
        from scripts.offline_app_refined import RefinedOfflineApp

        config = OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {"initial_search_frames": 24},
                "reprompt": {
                    "enable": True,
                    "empty_mask_patience": 3,
                    "area_drop_ratio": 0.35,
                    "edge_touch_ratio": 0.4,
                    "iou_low_threshold": 0.55,
                },
                "debug": {"save_metrics": False},
            }
        )
        app = RefinedOfflineApp("configs/body4d_refined.yaml", config=config)
        runtime_profile = {
            "tracking.chunk_size": 96,
            "sam_3d_body.batch_size": 20,
            "initial_search_frames": 48,
            "retry_index": 1,
            "reason": "safer_tracking",
        }

        app.prepare_input = unittest.mock.MagicMock(return_value={"frames": [], "output_dir": "./sample_out"})
        app.detect_initial_targets = unittest.mock.MagicMock(return_value={"obj_ids": [1]})
        app.prepare_sample_output = unittest.mock.MagicMock(return_value={"images": "./sample_out/images"})
        app.iter_chunks = unittest.mock.MagicMock(return_value=[])
        app.track_chunk = unittest.mock.MagicMock()
        app.refine_chunk_masks = unittest.mock.MagicMock()
        app.maybe_reprompt_chunk = unittest.mock.MagicMock()
        app.write_chunk_outputs = unittest.mock.MagicMock()
        app.run_refined_4d_generation = unittest.mock.MagicMock()
        app.finalize_sample = unittest.mock.MagicMock(side_effect=RuntimeError("finalize failed"))

        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            app.run_sample("sample.mp4", "./custom_out", skip_existing=False, runtime_profile=runtime_profile)

        self.assertEqual(int(config.tracking.chunk_size), 180)
        self.assertEqual(int(config.sam_3d_body.batch_size), 32)
        self.assertEqual(int(config.batch.initial_search_frames), 24)
        self.assertIs(app.CONFIG, config)


if __name__ == "__main__":
    unittest.main()
