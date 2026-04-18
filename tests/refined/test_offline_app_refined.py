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
                "--save_debug_metrics",
            ]
        )

        self.assertEqual(args.input_video, "sample.mp4")
        self.assertEqual(args.config, "configs/body4d_refined.yaml")
        self.assertEqual(args.detector_backend, "yolo")
        self.assertEqual(args.track_chunk_size, 96)
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

    def test_main_applies_output_dir_override(self):
        import scripts.offline_app_refined as offline_app_refined

        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir="./custom_out",
            config="configs/body4d_refined.yaml",
            detector_backend="yolo",
            track_chunk_size=None,
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
                "detector": {"backend": "rtmdet"},
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
        args = SimpleNamespace(
            input_video="sample.mp4",
            output_dir="./custom_out",
            config="configs/body4d_refined.yaml",
            detector_backend="yolo",
            track_chunk_size=96,
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
        self.assertEqual(cfg.tracking.chunk_size, 96)
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
        self.assertTrue(sample["output_dir"].endswith("outputs_refined"))

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

    def test_detect_initial_targets_initializes_tracker_from_first_detected_frame(self):
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
        ]

        with patch.object(app, "_ensure_base_app", return_value=runtime_app, create=True):
            targets = app.detect_initial_targets(app.sample_state)

        self.assertEqual(targets["obj_ids"], [1])
        self.assertEqual(targets["start_frame_idx"], 1)
        predictor.clear_all_points_in_video.assert_called_once_with(
            {"video_height": 8, "video_width": 8}
        )
        predictor.add_new_points_or_box.assert_called_once()

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
