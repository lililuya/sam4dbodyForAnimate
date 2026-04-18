import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from PIL import Image


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


class MaskVideoExportTests(unittest.TestCase):
    def test_build_binary_mask_frames_returns_strict_zero_or_255(self):
        from scripts.mask_video_export import build_binary_mask_frames

        with make_workspace_tempdir() as tmpdir:
            mask_a = np.array([[0, 1], [2, 0]], dtype=np.uint8)
            mask_b = np.array([[2, 2], [0, 1]], dtype=np.uint8)
            Image.fromarray(mask_a).save(os.path.join(tmpdir, "00000000.png"))
            Image.fromarray(mask_b).save(os.path.join(tmpdir, "00000001.png"))

            frames_all = build_binary_mask_frames(tmpdir)
            frames_person_2 = build_binary_mask_frames(tmpdir, track_id=2)

        self.assertEqual([sorted(np.unique(frame).tolist()) for frame in frames_all], [[0, 255], [0, 255]])
        self.assertTrue(np.array_equal(frames_person_2[0], np.array([[0, 0], [255, 0]], dtype=np.uint8)))
        self.assertTrue(np.array_equal(frames_person_2[1], np.array([[255, 255], [0, 0]], dtype=np.uint8)))

    def test_export_binary_mask_videos_writes_all_and_per_person_outputs(self):
        from scripts.mask_video_export import export_binary_mask_videos

        with make_workspace_tempdir() as tmpdir:
            mask_dir = os.path.join(tmpdir, "masks")
            output_dir = os.path.join(tmpdir, "mask_videos")
            os.makedirs(mask_dir, exist_ok=True)
            Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).save(os.path.join(mask_dir, "00000000.png"))

            with patch("scripts.mask_video_export.images_to_mp4") as mock_images_to_mp4:
                export_binary_mask_videos(mask_dir, output_dir, [2, 1], fps=12)

        written_paths = [call.args[1] for call in mock_images_to_mp4.call_args_list]
        self.assertEqual(
            written_paths,
            [
                os.path.join(output_dir, "mask_binary_all.mp4"),
                os.path.join(output_dir, "mask_binary_person_01.mp4"),
                os.path.join(output_dir, "mask_binary_person_02.mp4"),
            ],
        )
        for call in mock_images_to_mp4.call_args_list:
            frames = call.args[0]
            self.assertTrue(all(set(np.unique(frame).tolist()).issubset({0, 255}) for frame in frames))
            self.assertEqual(call.kwargs["fps"], 12)


if __name__ == "__main__":
    unittest.main()
