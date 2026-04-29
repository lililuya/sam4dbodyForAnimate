import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np


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


class CompareVideoHelpersTests(unittest.TestCase):
    def test_pad_frame_to_canvas_centers_without_resizing(self):
        from verify_output.concat_compare_videos import pad_frame_to_canvas

        frame = np.full((2, 4, 3), 255, dtype=np.uint8)
        padded = pad_frame_to_canvas(frame, target_width=8, target_height=6)

        self.assertEqual(padded.shape, (6, 8, 3))
        self.assertTrue((padded[2:4, 2:6] == 255).all())
        self.assertTrue((padded[:2] == 0).all())
        self.assertTrue((padded[4:] == 0).all())
        self.assertTrue((padded[:, :2] == 0).all())
        self.assertTrue((padded[:, 6:] == 0).all())

    def test_build_overlay_frame_blends_target_and_rendered(self):
        from verify_output.concat_compare_videos import build_overlay_frame

        target = np.zeros((2, 2, 3), dtype=np.uint8)
        rendered = np.full((2, 2, 3), 200, dtype=np.uint8)
        overlay = build_overlay_frame(target, rendered, alpha=0.5)

        self.assertEqual(overlay.shape, (2, 2, 3))
        self.assertTrue((overlay == 100).all())

    def test_compose_grid_frame_uses_uniform_tiles(self):
        from verify_output.concat_compare_videos import compose_grid_frame

        target = np.full((4, 6, 3), 10, dtype=np.uint8)
        rendered = np.full((4, 6, 3), 20, dtype=np.uint8)
        overlay = np.full((4, 6, 3), 30, dtype=np.uint8)
        src_face = np.full((4, 6, 3), 40, dtype=np.uint8)
        src_pose = np.full((4, 6, 3), 50, dtype=np.uint8)
        src_bg = np.full((4, 6, 3), 60, dtype=np.uint8)
        src_mask = np.full((4, 6, 3), 70, dtype=np.uint8)
        src_mask_detail = np.full((4, 6, 3), 80, dtype=np.uint8)
        blank = np.full((4, 6, 3), 90, dtype=np.uint8)

        grid = compose_grid_frame(
            {
                "target": target,
                "4d": rendered,
                "overlay": overlay,
                "src_face": src_face,
                "src_pose": src_pose,
                "src_bg": src_bg,
                "src_mask": src_mask,
                "src_mask_detail": src_mask_detail,
                "blank": blank,
            }
        )

        self.assertEqual(grid.shape, (12, 18, 3))
        self.assertTrue((grid[0:4, 0:6] == 10).all())
        self.assertTrue((grid[0:4, 6:12] == 20).all())
        self.assertTrue((grid[0:4, 12:18] == 30).all())
        self.assertTrue((grid[4:8, 0:6] == 40).all())
        self.assertTrue((grid[8:12, 12:18] == 90).all())

    def test_resolve_input_videos_requires_4d(self):
        from verify_output.concat_compare_videos import resolve_input_videos

        with make_workspace_tempdir() as tmpdir:
            open(os.path.join(tmpdir, "target.mp4"), "wb").close()
            with self.assertRaisesRegex(FileNotFoundError, "4d.mp4"):
                resolve_input_videos(tmpdir)

    def test_resolve_input_videos_collects_optional_panels(self):
        from verify_output.concat_compare_videos import resolve_input_videos

        with make_workspace_tempdir() as tmpdir:
            for name in ("target.mp4", "4d.mp4", "src_face.mp4"):
                open(os.path.join(tmpdir, name), "wb").close()
            resolved = resolve_input_videos(tmpdir)

        self.assertTrue(resolved["target"].endswith("target.mp4"))
        self.assertTrue(resolved["4d"].endswith("4d.mp4"))
        self.assertTrue(resolved["src_face"].endswith("src_face.mp4"))
        self.assertIsNone(resolved["src_pose"])

    def test_validate_required_videos_rejects_frame_count_mismatch(self):
        from verify_output.concat_compare_videos import validate_required_videos

        with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
            validate_required_videos(
                target_info={"width": 8, "height": 8, "fps": 25.0, "frame_count": 10},
                rendered_info={"width": 8, "height": 8, "fps": 25.0, "frame_count": 9},
            )

    def test_build_comparison_video_expands_tile_for_larger_optional_panel(self):
        from verify_output import concat_compare_videos

        with make_workspace_tempdir() as tmpdir:
            sample_dir = os.path.join(tmpdir, "sample_target1")
            os.makedirs(sample_dir, exist_ok=True)
            output_path = os.path.join(tmpdir, "compare.mp4")

            resolved = {
                "target": os.path.join(sample_dir, "target.mp4"),
                "4d": os.path.join(sample_dir, "4d.mp4"),
                "src_face": os.path.join(sample_dir, "src_face.mp4"),
                "src_pose": None,
                "src_bg": None,
                "src_mask": None,
                "src_mask_detail": None,
            }

            capture_by_path = {
                resolved["target"]: MagicMock(),
                resolved["4d"]: MagicMock(),
                resolved["src_face"]: MagicMock(),
            }
            frame_by_path = {
                resolved["target"]: np.full((464, 832, 3), 10, dtype=np.uint8),
                resolved["4d"]: np.full((464, 832, 3), 20, dtype=np.uint8),
                resolved["src_face"]: np.full((512, 512, 3), 30, dtype=np.uint8),
            }

            def fake_load_video_info(video_path):
                if video_path == resolved["target"]:
                    return {"path": video_path, "width": 832, "height": 464, "fps": 25.0, "frame_count": 1}
                if video_path == resolved["4d"]:
                    return {"path": video_path, "width": 832, "height": 464, "fps": 25.0, "frame_count": 1}
                if video_path == resolved["src_face"]:
                    return {"path": video_path, "width": 512, "height": 512, "fps": 25.0, "frame_count": 1}
                raise AssertionError(f"unexpected video path: {video_path}")

            def fake_open_video_capture(video_path):
                if video_path is None:
                    return None
                return capture_by_path[video_path]

            def fake_read_next_frame(capture, *, video_path, required):
                del required
                if capture is None:
                    return None
                return np.array(frame_by_path[video_path], copy=True)

            with patch.object(concat_compare_videos, "resolve_input_videos", return_value=resolved), patch.object(
                concat_compare_videos,
                "load_video_info",
                side_effect=fake_load_video_info,
            ), patch.object(
                concat_compare_videos,
                "_open_video_capture",
                side_effect=fake_open_video_capture,
            ), patch.object(
                concat_compare_videos,
                "_read_next_frame",
                side_effect=fake_read_next_frame,
            ), patch.object(
                concat_compare_videos,
                "encode_frames_to_mp4",
            ) as mock_encode, patch(
                "verify_output.concat_compare_videos.tempfile.TemporaryDirectory",
                side_effect=lambda *args, **kwargs: make_workspace_tempdir(),
            ), patch(
                "verify_output.concat_compare_videos.cv2.imwrite",
                return_value=True,
            ) as mock_imwrite:
                actual_output_path = concat_compare_videos.build_comparison_video(
                    sample_dir=sample_dir,
                    output_path=output_path,
                    overlay_alpha=0.5,
                )

        self.assertEqual(actual_output_path, os.path.abspath(output_path))
        written_frame = mock_imwrite.call_args.args[1]
        self.assertEqual(written_frame.shape[:2], (512 * 3, 832 * 3))
        mock_encode.assert_called_once()

    def test_encode_frames_to_mp4_uses_quality_controlled_h264_defaults(self):
        from verify_output.concat_compare_videos import encode_frames_to_mp4

        with patch("verify_output.concat_compare_videos.subprocess.run") as mock_run:
            encode_frames_to_mp4(
                frames_dir="frames_dir",
                fps=25.0,
                output_path="out.mp4",
            )

        command = mock_run.call_args.args[0]
        self.assertIn("ffmpeg", command[0].lower())
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertIn("20", command)
        self.assertNotIn("0", command)

    def test_encode_frames_to_mp4_accepts_custom_crf(self):
        from verify_output.concat_compare_videos import encode_frames_to_mp4

        with patch("verify_output.concat_compare_videos.subprocess.run") as mock_run:
            encode_frames_to_mp4(
                frames_dir="frames_dir",
                fps=25.0,
                output_path="out.mp4",
                crf=16,
            )

        command = mock_run.call_args.args[0]
        crf_index = command.index("-crf")
        self.assertEqual(command[crf_index + 1], "16")

    def test_main_writes_compare_video_to_verify_output(self):
        from verify_output import concat_compare_videos

        with make_workspace_tempdir() as tmpdir:
            sample_dir = os.path.join(tmpdir, "sample_target1")
            os.makedirs(sample_dir, exist_ok=True)
            for name in ("target.mp4", "4d.mp4"):
                open(os.path.join(sample_dir, name), "wb").close()

            with patch.object(concat_compare_videos, "build_comparison_video") as mock_build:
                concat_compare_videos.main(["--input", sample_dir])

        output_path = mock_build.call_args.kwargs["output_path"]
        self.assertTrue(output_path.endswith("_compare.mp4"))
        self.assertIn("verify_output", output_path)

    def test_main_appends_mp4_extension_for_extensionless_output_path(self):
        from verify_output import concat_compare_videos

        with make_workspace_tempdir() as tmpdir:
            sample_dir = os.path.join(tmpdir, "sample_target1")
            os.makedirs(sample_dir, exist_ok=True)
            for name in ("target.mp4", "4d.mp4"):
                open(os.path.join(sample_dir, name), "wb").close()
            output_stem = os.path.join(tmpdir, "result")

            with patch.object(concat_compare_videos, "build_comparison_video") as mock_build:
                concat_compare_videos.main(["--input", sample_dir, "--output", output_stem])

        output_path = mock_build.call_args.kwargs["output_path"]
        self.assertEqual(output_path, os.path.abspath(output_stem) + ".mp4")

    def test_main_forwards_custom_crf(self):
        from verify_output import concat_compare_videos

        with make_workspace_tempdir() as tmpdir:
            sample_dir = os.path.join(tmpdir, "sample_target1")
            os.makedirs(sample_dir, exist_ok=True)
            for name in ("target.mp4", "4d.mp4"):
                open(os.path.join(sample_dir, name), "wb").close()

            with patch.object(concat_compare_videos, "build_comparison_video") as mock_build:
                concat_compare_videos.main(["--input", sample_dir, "--crf", "16"])

        self.assertEqual(mock_build.call_args.kwargs["crf"], 16)


if __name__ == "__main__":
    unittest.main()
