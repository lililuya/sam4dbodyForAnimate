import importlib.util
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


HAS_TORCH = importlib.util.find_spec("torch") is not None


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class App4DPipelineTests(unittest.TestCase):
    def test_run_4d_pipeline_uses_explicit_input_and_output_dirs(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(input_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                os.path.join(input_dir, "masks", "00000000.png")
            )

            runtime = {
                "out_obj_ids": [1],
                "batch_size": 1,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
                "video_fps": 24.0,
            }
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
            )

            with patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                return_value=([], [], [0]),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ), patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ) as mock_video, patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                out_path = run_4d_pipeline_from_context(context)
                mock_video.assert_called_once()
                self.assertEqual(os.path.dirname(out_path), output_dir)
                self.assertTrue(os.path.isdir(os.path.join(output_dir, "rendered_frames")))
                self.assertTrue(os.path.isdir(os.path.join(output_dir, "mesh_4d_individual", "1")))
                self.assertFalse(os.path.isdir(os.path.join(input_dir, "rendered_frames")))

    def test_run_4d_pipeline_uses_completion_batch_size_when_completion_is_enabled(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            for frame_stem in ("00000000", "00000001"):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    os.path.join(input_dir, "images", f"{frame_stem}.jpg")
                )
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                    os.path.join(input_dir, "masks", f"{frame_stem}.png")
                )

            runtime = {
                "out_obj_ids": [1],
                "batch_size": 8,
                "completion_batch_size": 1,
                "completion_decode_chunk_size": 2,
                "max_occ_len": 8,
                "detection_resolution": [192, 384],
                "completion_resolution": [256, 512],
                "smpl_export": False,
                "video_fps": 24.0,
            }

            pipeline_mask = MagicMock(
                side_effect=lambda *args, **kwargs: SimpleNamespace(
                    frames=[
                        [
                            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
                            for _ in range(kwargs["num_frames"])
                        ]
                    ]
                )
            )
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=pipeline_mask,
                pipeline_rgb=MagicMock(),
                depth_model=MagicMock(),
                predictor=MagicMock(),
                generator=None,
            )

            fake_modal_pixels = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
            fake_rgb_pixels = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
            fake_depth_pixels = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
            fake_process_image_with_mask = lambda *args, **kwargs: (
                [[{"bbox": np.zeros((4,), dtype=np.float32)}] for _ in args[1]],
                [[1] for _ in args[1]],
                [],
            )

            with patch(
                "scripts.app_4d_pipeline._load_runtime_utils",
                return_value={
                    "DAVIS_PALETTE": [0] * (256 * 3),
                    "bbox_from_mask": lambda mask: [0.0, 0.0, 1.0, 1.0],
                    "is_skinny_mask": lambda mask: False,
                    "is_super_long_or_wide": lambda mask, obj_id: False,
                    "keep_largest_component": lambda mask: mask,
                    "resize_mask_with_unique_label": lambda mask, h, w, obj_id: mask,
                },
            ), patch(
                "scripts.app_4d_pipeline.load_and_transform_masks",
                return_value=(fake_modal_pixels, (4, 4)),
            ), patch(
                "scripts.app_4d_pipeline.load_and_transform_rgbs",
                return_value=(fake_rgb_pixels, (4, 4), None),
            ), patch(
                "scripts.app_4d_pipeline.rgb_to_depth",
                return_value=fake_depth_pixels,
            ), patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                side_effect=fake_process_image_with_mask,
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ), patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ), patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                run_4d_pipeline_from_context(context)

        self.assertEqual(pipeline_mask.call_count, 2)

    def test_run_4d_pipeline_skips_optional_storage_outputs_when_disabled(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(input_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(
                os.path.join(input_dir, "masks", "00000000.png")
            )

            runtime = {
                "out_obj_ids": [7],
                "batch_size": 1,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
                "video_fps": 24.0,
                "save_rendered_frames": False,
                "save_rendered_frames_individual": False,
                "save_mesh_4d_individual": False,
                "save_focal_4d_individual": False,
            }
            frame_writer = MagicMock(return_value={"coco17": []})
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
                frame_writer=frame_writer,
            )

            with patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                return_value=([[{"bbox": np.zeros((4,), dtype=np.float32)}]], [[7]], []),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[np.zeros((4, 4, 3), dtype=np.uint8)],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ) as mock_save_mesh, patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ) as mock_video, patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ) as mock_imwrite:
                out_path = run_4d_pipeline_from_context(context)

        self.assertIsNone(out_path)
        frame_writer.assert_called_once()
        mock_save_mesh.assert_not_called()
        mock_video.assert_not_called()
        mock_imwrite.assert_not_called()
        self.assertFalse(os.path.isdir(os.path.join(output_dir, "rendered_frames")))
        self.assertFalse(os.path.isdir(os.path.join(output_dir, "rendered_frames_individual")))
        self.assertFalse(os.path.isdir(os.path.join(output_dir, "mesh_4d_individual")))
        self.assertFalse(os.path.isdir(os.path.join(output_dir, "focal_4d_individual")))

    def test_run_4d_pipeline_passes_mesh_and_focal_storage_flags(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                os.path.join(input_dir, "images", "00000000.jpg")
            )
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(
                os.path.join(input_dir, "masks", "00000000.png")
            )

            runtime = {
                "out_obj_ids": [3],
                "batch_size": 1,
                "detection_resolution": [256, 512],
                "completion_resolution": [512, 1024],
                "smpl_export": False,
                "video_fps": 24.0,
                "save_rendered_frames": False,
                "save_rendered_frames_individual": False,
                "save_mesh_4d_individual": False,
                "save_focal_4d_individual": True,
            }
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=None,
                pipeline_rgb=None,
                depth_model=None,
                predictor=MagicMock(),
                generator=None,
            )

            with patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                return_value=([[{"bbox": np.zeros((4,), dtype=np.float32)}]], [[3]], []),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ) as mock_save_mesh, patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ), patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                run_4d_pipeline_from_context(context)

                self.assertFalse(os.path.isdir(os.path.join(output_dir, "mesh_4d_individual")))
                self.assertTrue(os.path.isdir(os.path.join(output_dir, "focal_4d_individual", "3")))

        mock_save_mesh.assert_called_once()
        self.assertEqual(mock_save_mesh.call_args.kwargs["save_mesh"], False)
        self.assertEqual(mock_save_mesh.call_args.kwargs["save_focal"], True)

    @unittest.skipUnless(HAS_TORCH, "torch is required for completion caching test")
    def test_run_4d_pipeline_caches_transforms_and_keeps_completion_artifacts_in_memory(self):
        from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context
        import torch

        with make_workspace_tempdir() as tmpdir:
            input_dir = os.path.join(tmpdir, "cache")
            output_dir = os.path.join(tmpdir, "outputs_4d", "demo")
            os.makedirs(os.path.join(input_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "masks"), exist_ok=True)
            for frame_stem in ("00000000", "00000001", "00000002", "00000003"):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    os.path.join(input_dir, "images", f"{frame_stem}.jpg")
                )
                mask = np.zeros((4, 4), dtype=np.uint8)
                mask[:, :2] = 1
                mask[:, 2:] = 2
                Image.fromarray(mask).save(
                    os.path.join(input_dir, "masks", f"{frame_stem}.png")
                )

            runtime = {
                "out_obj_ids": [1, 2],
                "batch_size": 8,
                "completion_batch_size": 2,
                "completion_decode_chunk_size": 2,
                "max_occ_len": 8,
                "detection_resolution": [192, 384],
                "completion_resolution": [256, 512],
                "smpl_export": False,
                "video_fps": 24.0,
            }

            pipeline_mask = MagicMock(
                side_effect=lambda *args, **kwargs: SimpleNamespace(
                    frames=[
                        [
                            Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))
                            for _ in range(kwargs["num_frames"])
                        ]
                    ]
                )
            )
            pipeline_rgb = MagicMock(
                side_effect=lambda *args, **kwargs: SimpleNamespace(
                    frames=[
                        [
                            Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8))
                            for _ in range(kwargs["num_frames"])
                        ]
                    ]
                )
            )
            context = build_4d_context(
                input_dir=input_dir,
                output_dir=output_dir,
                runtime=runtime,
                sam3_3d_body_model=MagicMock(faces=np.array([[0, 1, 2]], dtype=np.int32)),
                pipeline_mask=pipeline_mask,
                pipeline_rgb=pipeline_rgb,
                depth_model=MagicMock(),
                predictor=MagicMock(),
                generator=None,
            )

            fake_modal_low = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
            fake_modal_high = torch.ones((1, 4, 3, 2, 2), dtype=torch.float32)
            fake_rgb_low = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
            fake_rgb_high = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
            fake_depth_pixels = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
            captured_process_kwargs = {}

            def fake_load_masks(*args, **kwargs):
                resolution = tuple(kwargs["resolution"])
                if resolution == (192, 384):
                    return fake_modal_low, (4, 4)
                return fake_modal_high, (4, 4)

            def fake_load_rgbs(*args, **kwargs):
                resolution = tuple(kwargs["resolution"])
                if resolution == (192, 384):
                    return fake_rgb_low, (4, 4), None
                return fake_rgb_high, (4, 4), None

            def fake_process_image_with_mask(*args, **kwargs):
                captured_process_kwargs["idx_path"] = dict(args[3])
                captured_process_kwargs["completion_cache"] = kwargs["completion_cache"]
                batch_images = args[1]
                return (
                    [[{"bbox": np.zeros((4,), dtype=np.float32)}] for _ in batch_images],
                    [[1] for _ in batch_images],
                    [],
                )

            with patch(
                "scripts.app_4d_pipeline._load_runtime_utils",
                return_value={
                    "DAVIS_PALETTE": [0] * (256 * 3),
                    "bbox_from_mask": lambda mask: [0.0, 0.0, 1.0, 1.0],
                    "is_skinny_mask": lambda mask: False,
                    "is_super_long_or_wide": lambda mask, obj_id: False,
                    "keep_largest_component": lambda mask: mask,
                    "resize_mask_with_unique_label": lambda mask, h, w, obj_id: mask,
                },
            ), patch(
                "scripts.app_4d_pipeline.load_and_transform_masks",
                side_effect=fake_load_masks,
            ) as mock_load_masks, patch(
                "scripts.app_4d_pipeline.load_and_transform_rgbs",
                side_effect=fake_load_rgbs,
            ) as mock_load_rgbs, patch(
                "scripts.app_4d_pipeline.rgb_to_depth",
                return_value=fake_depth_pixels,
            ), patch(
                "scripts.app_4d_pipeline.build_completion_window_from_ious",
                side_effect=lambda ious, padding=0, iou_threshold=0.7: ([0, 0], (0, 2)),
            ), patch(
                "scripts.app_4d_pipeline.process_image_with_mask",
                side_effect=fake_process_image_with_mask,
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample_together",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.visualize_sample",
                return_value=[],
            ), patch(
                "scripts.app_4d_pipeline.save_mesh_results",
            ), patch(
                "scripts.app_4d_pipeline.jpg_folder_to_mp4",
            ), patch(
                "scripts.app_4d_pipeline.cv2.imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch(
                "scripts.app_4d_pipeline.cv2.imwrite",
                return_value=True,
            ):
                run_4d_pipeline_from_context(context)

        self.assertEqual(mock_load_masks.call_count, 4)
        self.assertEqual(mock_load_rgbs.call_count, 2)
        self.assertEqual(captured_process_kwargs["idx_path"], {})
        self.assertIn(1, captured_process_kwargs["completion_cache"])
        self.assertIn("images", captured_process_kwargs["completion_cache"][1])
        self.assertIn("masks", captured_process_kwargs["completion_cache"][1])
        self.assertFalse(os.path.exists(os.path.join(output_dir, "completion")))


if __name__ == "__main__":
    unittest.main()
