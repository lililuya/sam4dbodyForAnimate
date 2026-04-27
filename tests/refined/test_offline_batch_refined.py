import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_batch_refined_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class BatchHelperTests(unittest.TestCase):
    def test_discover_samples_supports_input_root_and_jsonl_manifest(self):
        from scripts.offline_batch_helpers import discover_samples

        with make_workspace_tempdir() as tmpdir:
            input_root = os.path.join(tmpdir, "inputs")
            os.makedirs(input_root, exist_ok=True)

            video_path = os.path.join(input_root, "clip_a.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"mp4")

            frames_dir = os.path.join(input_root, "frames_b")
            os.makedirs(frames_dir, exist_ok=True)
            with open(os.path.join(frames_dir, "00000001.jpg"), "wb") as handle:
                handle.write(b"jpg")

            manifest_path = os.path.join(tmpdir, "inputs.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"input": video_path, "sample_id": "video_a"}) + "\n")
                handle.write(frames_dir + "\n")

            from_root = discover_samples(input_root=input_root)
            from_manifest = discover_samples(input_list=manifest_path)

        self.assertEqual(
            from_root,
            [
                {"input": video_path, "sample_id": "clip_a"},
                {"input": frames_dir, "sample_id": "frames_b"},
            ],
        )
        self.assertEqual(
            from_manifest,
            [
                {"input": video_path, "sample_id": "video_a"},
                {"input": frames_dir, "sample_id": "frames_b"},
            ],
        )

    def test_discover_samples_disambiguates_duplicate_auto_sample_ids(self):
        from scripts.offline_batch_helpers import discover_samples

        with make_workspace_tempdir() as tmpdir:
            input_root = os.path.join(tmpdir, "inputs")
            os.makedirs(input_root, exist_ok=True)

            cam_a = os.path.join(input_root, "cam_a")
            cam_b = os.path.join(input_root, "cam_b")
            os.makedirs(cam_a, exist_ok=True)
            os.makedirs(cam_b, exist_ok=True)

            video_a = os.path.join(cam_a, "clip.mp4")
            video_b = os.path.join(cam_b, "clip.mp4")
            with open(video_a, "wb") as handle:
                handle.write(b"a")
            with open(video_b, "wb") as handle:
                handle.write(b"b")

            manifest_path = os.path.join(tmpdir, "inputs.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(video_a + "\n")
                handle.write(video_b + "\n")
                handle.write(json.dumps({"input": video_b, "sample_id": "clip"}) + "\n")

            samples = discover_samples(input_list=manifest_path)

        self.assertEqual(
            samples,
            [
                {"input": video_a, "sample_id": "clip__2"},
                {"input": video_b, "sample_id": "clip__3"},
                {"input": video_b, "sample_id": "clip"},
            ],
        )

    def test_discover_samples_rejects_duplicate_explicit_sample_ids(self):
        from scripts.offline_batch_helpers import discover_samples

        with make_workspace_tempdir() as tmpdir:
            input_root = os.path.join(tmpdir, "inputs")
            os.makedirs(input_root, exist_ok=True)

            video_a = os.path.join(input_root, "clip_a.mp4")
            video_b = os.path.join(input_root, "clip_b.mp4")
            with open(video_a, "wb") as handle:
                handle.write(b"a")
            with open(video_b, "wb") as handle:
                handle.write(b"b")

            manifest_path = os.path.join(tmpdir, "inputs.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"input": video_a, "sample_id": "same_id"}) + "\n")
                handle.write(json.dumps({"input": video_b, "sample_id": "same_id"}) + "\n")

            with self.assertRaisesRegex(
                ValueError, "Duplicate explicit sample_id values are not allowed: same_id"
            ):
                discover_samples(input_list=manifest_path)

    def test_build_retry_profiles_keeps_quality_safe_defaults(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
            }
        )

        class Args:
            retry_mode = None
            track_chunk_size = None

        profiles = build_retry_profiles(cfg, Args())

        self.assertEqual(
            profiles,
            [
                {
                    "retry_index": 0,
                    "reason": "base",
                    "tracking.chunk_size": 180,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 1,
                    "reason": "safer_tracking",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 2,
                    "reason": "safer_reconstruction",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 24,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 3,
                    "reason": "search_expansion",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 24,
                    "initial_search_frames": 48,
                },
            ],
        )

    def test_build_retry_profiles_supports_never_mode(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
            }
        )

        class Args:
            retry_mode = "never"
            track_chunk_size = None

        profiles = build_retry_profiles(cfg, Args())

        self.assertEqual(
            profiles,
            [
                {
                    "retry_index": 0,
                    "reason": "base",
                    "tracking.chunk_size": 180,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                }
            ],
        )

    def test_build_retry_profiles_supports_aggressive_safe_mode(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
            }
        )

        class Args:
            retry_mode = "aggressive_safe"
            track_chunk_size = None

        profiles = build_retry_profiles(cfg, Args())

        self.assertEqual(
            profiles,
            [
                {
                    "retry_index": 0,
                    "reason": "base",
                    "tracking.chunk_size": 180,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 1,
                    "reason": "safer_tracking",
                    "tracking.chunk_size": 120,
                    "sam_3d_body.batch_size": 32,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 2,
                    "reason": "safer_reconstruction",
                    "tracking.chunk_size": 96,
                    "sam_3d_body.batch_size": 24,
                    "initial_search_frames": 24,
                },
                {
                    "retry_index": 3,
                    "reason": "search_expansion",
                    "tracking.chunk_size": 96,
                    "sam_3d_body.batch_size": 16,
                    "initial_search_frames": 48,
                },
            ],
        )

    def test_build_retry_profiles_rejects_invalid_retry_mode(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
            }
        )

        class Args:
            retry_mode = "mystery_mode"
            track_chunk_size = None

        with self.assertRaisesRegex(
            ValueError, "retry_mode must be one of never, quality_safe, aggressive_safe"
        ):
            build_retry_profiles(cfg, Args())

    def test_build_retry_profiles_rejects_empty_retry_arrays(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        cfg = OmegaConf.create(
            {
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [],
                    "retry_batch_sizes": [],
                },
            }
        )

        class Args:
            retry_mode = None
            track_chunk_size = None

        with self.assertRaisesRegex(
            ValueError, "retry_chunk_sizes and retry_batch_sizes must each contain at least one value"
        ):
            build_retry_profiles(cfg, Args())

    def test_build_retry_profiles_rejects_scalar_retry_values(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        invalid_configs = [
            {
                "retry_chunk_sizes": "120",
                "retry_batch_sizes": [24, 16],
                "message": "retry_chunk_sizes must be a sequence of integers",
            },
            {
                "retry_chunk_sizes": 120,
                "retry_batch_sizes": [24, 16],
                "message": "retry_chunk_sizes must be a sequence of integers",
            },
            {
                "retry_chunk_sizes": [120, 96],
                "retry_batch_sizes": "24",
                "message": "retry_batch_sizes must be a sequence of integers",
            },
            {
                "retry_chunk_sizes": [120, 96],
                "retry_batch_sizes": 24,
                "message": "retry_batch_sizes must be a sequence of integers",
            },
        ]

        class Args:
            retry_mode = None
            track_chunk_size = None

        for case in invalid_configs:
            cfg = OmegaConf.create(
                {
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": case["retry_chunk_sizes"],
                        "retry_batch_sizes": case["retry_batch_sizes"],
                    },
                }
            )

            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, case["message"]):
                    build_retry_profiles(cfg, Args())

    def test_build_retry_profiles_rejects_invalid_retry_elements(self):
        from scripts.offline_batch_helpers import build_retry_profiles

        invalid_configs = [
            {
                "retry_chunk_sizes": [120, None],
                "retry_batch_sizes": [24, 16],
                "message": "retry_chunk_sizes must be a sequence of integers",
            },
            {
                "retry_chunk_sizes": [120, 96],
                "retry_batch_sizes": [24, {}],
                "message": "retry_batch_sizes must be a sequence of integers",
            },
        ]

        class Args:
            retry_mode = None
            track_chunk_size = None

        for case in invalid_configs:
            cfg = OmegaConf.create(
                {
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": case["retry_chunk_sizes"],
                        "retry_batch_sizes": case["retry_batch_sizes"],
                    },
                }
            )

            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, case["message"]):
                    build_retry_profiles(cfg, Args())


class BatchRunnerTests(unittest.TestCase):
    def test_batch_parser_accepts_quality_preserving_arguments(self):
        from scripts.offline_batch_refined import build_batch_parser

        parser = build_batch_parser()
        args = parser.parse_args(
            [
                "--input_root",
                "./inputs",
                "--output_dir",
                "./outputs_batch",
                "--config",
                "configs/body4d_refined.yaml",
                "--retry_mode",
                "quality_safe",
                "--disable_mask_refine",
                "--skip_existing",
            ]
        )

        self.assertEqual(args.input_root, "./inputs")
        self.assertEqual(args.output_dir, "./outputs_batch")
        self.assertEqual(args.config, "configs/body4d_refined.yaml")
        self.assertEqual(args.retry_mode, "quality_safe")
        self.assertTrue(args.disable_mask_refine)
        self.assertTrue(args.skip_existing)

    def test_run_batch_calls_refined_app_once_per_sample(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            args = SimpleNamespace(
                input_root="./inputs",
                input_list="",
                output_dir=batch_output_dir,
                config="configs/body4d_refined.yaml",
                detector_backend=None,
                track_chunk_size=None,
                continue_on_error=False,
                save_debug_metrics=False,
                skip_existing=False,
                max_samples=None,
                retry_mode="never",
            )
            cfg = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "detector": {"backend": "yolo"},
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": [120, 96],
                        "retry_batch_sizes": [24, 16],
                    },
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
            samples = [
                {"input": "./inputs/a.mp4", "sample_id": "a"},
                {"input": "./inputs/b.mp4", "sample_id": "b"},
            ]
            app = MagicMock()
            app.run_sample.side_effect = [
                {"status": "completed"},
                {"status": "completed"},
            ]

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
                results = offline_batch_refined.run_batch(args)

        self.assertEqual(len(results), 2)
        self.assertEqual(app.run_sample.call_count, 2)
        self.assertEqual(
            app.run_sample.call_args_list,
            [
                unittest.mock.call(
                    input_video="./inputs/a.mp4",
                    output_dir=os.path.join(batch_output_dir, "a"),
                    skip_existing=False,
                    runtime_profile={
                        "retry_index": 0,
                        "reason": "base",
                        "tracking.chunk_size": 180,
                        "sam_3d_body.batch_size": 32,
                        "initial_search_frames": 24,
                    },
                ),
                unittest.mock.call(
                    input_video="./inputs/b.mp4",
                    output_dir=os.path.join(batch_output_dir, "b"),
                    skip_existing=False,
                    runtime_profile={
                        "retry_index": 0,
                        "reason": "base",
                        "tracking.chunk_size": 180,
                        "sam_3d_body.batch_size": 32,
                        "initial_search_frames": 24,
                    },
                ),
            ],
        )

    def test_run_batch_rejects_unsafe_sample_ids_before_sample_execution(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            args = SimpleNamespace(
                input_root="./inputs",
                input_list="",
                output_dir=batch_output_dir,
                config="configs/body4d_refined.yaml",
                detector_backend=None,
                track_chunk_size=None,
                continue_on_error=False,
                save_debug_metrics=False,
                skip_existing=False,
                max_samples=None,
                retry_mode="never",
            )
            cfg = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "detector": {"backend": "yolo"},
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": [120, 96],
                        "retry_batch_sizes": [24, 16],
                    },
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
            unsafe_sample_ids = [
                "../escape",
                "nested/subdir",
                "nested\\subdir",
                os.path.join(tmpdir, "outside_root"),
            ]

            for unsafe_sample_id in unsafe_sample_ids:
                with self.subTest(sample_id=unsafe_sample_id):
                    app = MagicMock()
                    app.run_sample.return_value = {"status": "completed"}
                    samples = [{"input": "./inputs/a.mp4", "sample_id": unsafe_sample_id}]

                    with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                        offline_batch_refined, "discover_samples", return_value=samples
                    ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
                        with self.assertRaisesRegex(ValueError, "Unsafe sample_id"):
                            offline_batch_refined.run_batch(args)

                    app.run_sample.assert_not_called()

    def test_run_batch_extracts_face_first_clip_packages_before_refined_processing(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            face_clip_output_dir = os.path.join(tmpdir, "face_clips")
            args = SimpleNamespace(
                input_root="./inputs",
                input_list="",
                output_dir=batch_output_dir,
                config="configs/body4d_refined.yaml",
                detector_backend=None,
                track_chunk_size=None,
                continue_on_error=False,
                save_debug_metrics=False,
                skip_existing=False,
                max_samples=None,
                retry_mode="never",
            )
            cfg = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "detector": {"backend": "yolo"},
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": [120, 96],
                        "retry_batch_sizes": [24, 16],
                    },
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                    "wan_export": {
                        "enable": True,
                        "output_dir": os.path.join(tmpdir, "wan_export"),
                        "face_clip": {
                            "enable": True,
                            "output_dir": face_clip_output_dir,
                            "min_clip_seconds": 5.0,
                        },
                    },
                }
            )
            samples = [{"input": "./inputs/source.mp4", "sample_id": "source"}]
            app = MagicMock()
            app.run_sample.return_value = {"status": "completed"}
            app.prepare_input.return_value = {
                "input_video": "./inputs/source.mp4",
                "input_type": "video",
                "frame_count": 100,
                "frames": [f"{index:08d}" for index in range(100)],
            }
            app._probe_sample_face_presence.return_value = {
                "checked_frame_count": 20,
                "face_detected_frame_count": 5,
                "no_face_frame_count": 15,
                "no_face_ratio": 0.75,
                "face_presence_stride": 5,
                "max_no_face_ratio": 0.80,
                "skip_sample_without_face": True,
                "probe_executed": True,
            }
            app._should_skip_sample_for_face_presence.return_value = False
            app._ensure_face_backend.return_value = object()
            clip_dir = os.path.join(face_clip_output_dir, "clips", "clipuuid_face01_seg001")

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app), patch.object(
                offline_batch_refined,
                "extract_face_clips_from_video",
                return_value=[clip_dir],
            ) as mock_extract_face_clips_from_video:
                results = offline_batch_refined.run_batch(args)

        mock_extract_face_clips_from_video.assert_called_once_with(
            input_video="./inputs/source.mp4",
            output_root=os.path.abspath(face_clip_output_dir),
            min_clip_seconds=5.0,
            face_backend=app._ensure_face_backend.return_value,
            sample_id="source",
        )
        app.run_sample.assert_called_once_with(
            input_video=clip_dir,
            output_dir=os.path.join(batch_output_dir, "clipuuid_face01_seg001"),
            skip_existing=False,
            runtime_profile={
                "retry_index": 0,
                "reason": "base",
                "tracking.chunk_size": 180,
                "sam_3d_body.batch_size": 32,
                "initial_search_frames": 24,
            },
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["stage"], "face_clip_extraction")
        self.assertEqual(results[1]["sample_id"], "clipuuid_face01_seg001")

    def test_run_batch_skips_face_first_sample_before_full_extraction_when_no_face_precheck_fails(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            face_clip_output_dir = os.path.join(tmpdir, "face_clips")
            args = SimpleNamespace(
                input_root="./inputs",
                input_list="",
                output_dir=batch_output_dir,
                config="configs/body4d_refined.yaml",
                detector_backend=None,
                track_chunk_size=None,
                continue_on_error=False,
                save_debug_metrics=False,
                skip_existing=False,
                max_samples=None,
                retry_mode="never",
            )
            cfg = OmegaConf.create(
                {
                    "runtime": {"output_dir": "./outputs_refined"},
                    "detector": {"backend": "yolo"},
                    "tracking": {"chunk_size": 180},
                    "sam_3d_body": {"batch_size": 32},
                    "batch": {
                        "initial_search_frames": 24,
                        "retry_mode": "quality_safe",
                        "retry_chunk_sizes": [120, 96],
                        "retry_batch_sizes": [24, 16],
                    },
                    "reprompt": {
                        "enable": True,
                        "empty_mask_patience": 3,
                        "area_drop_ratio": 0.35,
                        "edge_touch_ratio": 0.4,
                        "iou_low_threshold": 0.55,
                    },
                    "debug": {"save_metrics": False},
                    "wan_export": {
                        "enable": True,
                        "skip_sample_without_face": True,
                        "face_presence_stride": 5,
                        "max_no_face_ratio": 0.80,
                        "output_dir": os.path.join(tmpdir, "wan_export"),
                        "face_clip": {
                            "enable": True,
                            "output_dir": face_clip_output_dir,
                            "min_clip_seconds": 5.0,
                        },
                    },
                }
            )
            samples = [{"input": "./inputs/source.mp4", "sample_id": "source"}]
            app = MagicMock()
            app.prepare_input.return_value = {
                "input_video": "./inputs/source.mp4",
                "input_type": "video",
                "frame_count": 100,
                "frames": [f"{index:08d}" for index in range(100)],
            }
            app._probe_sample_face_presence.return_value = {
                "checked_frame_count": 20,
                "face_detected_frame_count": 0,
                "no_face_frame_count": 20,
                "no_face_ratio": 1.0,
                "face_presence_stride": 5,
                "max_no_face_ratio": 0.80,
                "skip_sample_without_face": True,
                "probe_executed": True,
            }
            app._should_skip_sample_for_face_presence.return_value = True

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app), patch.object(
                offline_batch_refined,
                "extract_face_clips_from_video",
            ) as mock_extract_face_clips_from_video:
                records = offline_batch_refined.run_batch(args)

            summary_path = os.path.join(face_clip_output_dir, "batch_summary.json")
            with open(summary_path, "r", encoding="utf-8") as handle:
                batch_summary = json.load(handle)

        app.prepare_input.assert_called_once_with("./inputs/source.mp4", None, False)
        app._probe_sample_face_presence.assert_called_once()
        app._should_skip_sample_for_face_presence.assert_called_once()
        app.run_sample.assert_not_called()
        mock_extract_face_clips_from_video.assert_not_called()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "skipped_no_face_precheck")
        self.assertEqual(records[0]["kept_clip_count"], 0)
        self.assertEqual(records[0]["face_presence"]["no_face_ratio"], 1.0)
        self.assertEqual(batch_summary["video_count_total"], 1)
        self.assertEqual(batch_summary["clip_count_kept"], 0)
        self.assertEqual(batch_summary["items"][0]["status"], "skipped_no_face_precheck")


class BatchRetryTests(unittest.TestCase):
    def _build_cfg(self):
        return OmegaConf.create(
            {
                "runtime": {"output_dir": "./outputs_refined"},
                "detector": {"backend": "yolo"},
                "tracking": {"chunk_size": 180},
                "sam_3d_body": {"batch_size": 32},
                "batch": {
                    "initial_search_frames": 24,
                    "retry_mode": "quality_safe",
                    "retry_chunk_sizes": [120, 96],
                    "retry_batch_sizes": [24, 16],
                },
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

    def _build_args(self, output_dir, **overrides):
        args = {
            "input_root": "./inputs",
            "input_list": "",
            "output_dir": output_dir,
            "config": "configs/body4d_refined.yaml",
            "detector_backend": None,
            "track_chunk_size": None,
            "continue_on_error": False,
            "save_debug_metrics": False,
            "skip_existing": False,
            "max_samples": None,
            "retry_mode": "quality_safe",
        }
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_run_sample_with_retries_skips_existing_sample_when_summary_is_completed(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            sample = {"input": "./inputs/a.mp4", "sample_id": "a"}
            summary_path = os.path.join(
                batch_output_dir, "a", "debug_metrics", "sample_summary.json"
            )
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump({"status": "completed"}, handle)

            app = MagicMock()
            args = self._build_args(batch_output_dir, skip_existing=True)
            record = offline_batch_refined.run_sample_with_retries(
                app, sample, batch_output_dir, self._build_cfg(), args
            )

        app.run_sample.assert_not_called()
        self.assertEqual(record["status"], "skipped")
        self.assertEqual(record["sample_id"], "a")
        self.assertEqual(record["output_dir"], os.path.join(batch_output_dir, "a"))
        self.assertEqual(record["retry_index"], 0)
        self.assertIsNone(record["runtime_profile"])

    def test_run_sample_with_retries_does_not_skip_when_summary_is_failed(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            sample = {"input": "./inputs/a.mp4", "sample_id": "a"}
            summary_path = os.path.join(
                batch_output_dir, "a", "debug_metrics", "sample_summary.json"
            )
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump({"status": "failed"}, handle)

            app = MagicMock()
            app.run_sample.return_value = {"status": "completed"}
            args = self._build_args(batch_output_dir, skip_existing=True, retry_mode="never")
            record = offline_batch_refined.run_sample_with_retries(
                app, sample, batch_output_dir, self._build_cfg(), args
            )

        app.run_sample.assert_called_once()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["summary_status"], "completed")
        self.assertEqual(record["retry_index"], 0)
        self.assertEqual(record["runtime_profile"]["retry_index"], 0)

    def test_run_sample_with_retries_retries_failed_sample_with_next_profile(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            sample = {"input": "./inputs/a.mp4", "sample_id": "a"}

            app = MagicMock()
            app.run_sample.side_effect = [RuntimeError("unstable"), {"status": "completed"}]
            args = self._build_args(batch_output_dir, skip_existing=False)
            record = offline_batch_refined.run_sample_with_retries(
                app, sample, batch_output_dir, self._build_cfg(), args
            )

        self.assertEqual(app.run_sample.call_count, 2)
        first_profile = app.run_sample.call_args_list[0].kwargs["runtime_profile"]
        second_profile = app.run_sample.call_args_list[1].kwargs["runtime_profile"]
        self.assertEqual(first_profile["retry_index"], 0)
        self.assertEqual(second_profile["retry_index"], 1)
        self.assertEqual(record["status"], "retry_succeeded")
        self.assertEqual(record["summary_status"], "completed")
        self.assertEqual(record["retry_index"], 1)
        self.assertEqual(record["runtime_profile"]["retry_index"], 1)

    def test_run_sample_with_retries_failed_record_preserves_last_profile_and_error(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            sample = {"input": "./inputs/a.mp4", "sample_id": "a"}
            app = MagicMock()
            app.run_sample.side_effect = [
                RuntimeError("boom-0"),
                RuntimeError("boom-1"),
                RuntimeError("boom-2"),
                RuntimeError("boom-final"),
            ]
            args = self._build_args(batch_output_dir, skip_existing=False)
            record = offline_batch_refined.run_sample_with_retries(
                app, sample, batch_output_dir, self._build_cfg(), args
            )

        self.assertEqual(app.run_sample.call_count, 4)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["retry_index"], 3)
        self.assertEqual(record["runtime_profile"]["retry_index"], 3)
        self.assertEqual(record["error"], "boom-final")

    def test_run_batch_raises_on_failed_sample_when_continue_on_error_false(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            args = self._build_args(batch_output_dir, continue_on_error=False)
            cfg = self._build_cfg()
            app = MagicMock()
            app.run_sample.side_effect = RuntimeError("always fails")
            samples = [{"input": "./inputs/a.mp4", "sample_id": "a"}]

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
                with self.assertRaisesRegex(
                    RuntimeError, "Stopping batch on failed sample: a"
                ):
                    offline_batch_refined.run_batch(args)

        self.assertEqual(app.run_sample.call_count, 4)

    def test_run_batch_retry_for_sample_a_does_not_mutate_base_profile_for_sample_b(self):
        import scripts.offline_batch_refined as offline_batch_refined

        with make_workspace_tempdir() as tmpdir:
            batch_output_dir = os.path.join(tmpdir, "outputs_batch")
            args = self._build_args(batch_output_dir)
            cfg = self._build_cfg()
            samples = [
                {"input": "./inputs/a.mp4", "sample_id": "a"},
                {"input": "./inputs/b.mp4", "sample_id": "b"},
            ]
            app = MagicMock()

            def run_sample_side_effect(*, input_video, output_dir, skip_existing, runtime_profile):
                if output_dir.endswith(os.path.join("outputs_batch", "a")) and runtime_profile["retry_index"] == 0:
                    runtime_profile["tracking.chunk_size"] = 999
                    raise RuntimeError("retry sample a")
                return {"status": "completed"}

            app.run_sample.side_effect = run_sample_side_effect

            with patch.object(offline_batch_refined, "load_refined_config", return_value=cfg), patch.object(
                offline_batch_refined, "discover_samples", return_value=samples
            ), patch.object(offline_batch_refined, "RefinedOfflineApp", return_value=app):
                records = offline_batch_refined.run_batch(args)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "retry_succeeded")
        self.assertEqual(records[0]["summary_status"], "completed")
        self.assertEqual(records[1]["status"], "completed")
        self.assertEqual(records[1]["summary_status"], "completed")
        self.assertEqual(app.run_sample.call_count, 3)
        sample_b_profile = app.run_sample.call_args_list[2].kwargs["runtime_profile"]
        self.assertEqual(sample_b_profile["retry_index"], 0)
        self.assertEqual(sample_b_profile["tracking.chunk_size"], 180)


if __name__ == "__main__":
    unittest.main()
