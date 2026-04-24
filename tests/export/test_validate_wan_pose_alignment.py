import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from scripts.wan_pose_renderer import draw_pose_frame


def _write_rgb_mp4(path, frames, fps=25):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _make_pose_meta(frame_name: str, width: int, height: int):
    body = np.zeros((20, 3), dtype=np.float32)
    body[0] = [0.50, 0.20, 1.0]
    body[1] = [0.50, 0.30, 1.0]
    body[2] = [0.40, 0.35, 1.0]
    body[5] = [0.60, 0.35, 1.0]
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    return {
        "image_id": frame_name,
        "track_id": 1,
        "width": width,
        "height": height,
        "category_id": 1,
        "keypoints_body": body.tolist(),
        "keypoints_left_hand": left_hand.tolist(),
        "keypoints_right_hand": right_hand.tolist(),
        "keypoints_face": [],
    }


class ValidateWanPoseAlignmentTests(unittest.TestCase):
    def test_cli_help_runs_via_direct_script_path(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script_path = os.path.join(repo_root, "scripts", "validate_wan_pose_alignment.py")

        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--sample_dir", result.stdout)

    def test_validate_wan_sample_alignment_writes_overlay_and_report(self):
        from scripts.validate_wan_pose_alignment import validate_wan_sample_alignment

        temp_dir = tempfile.mkdtemp(prefix="wan_validate_")
        try:
            sample_dir = os.path.join(temp_dir, "demo_track_1")
            pose_meta_dir = os.path.join(sample_dir, "pose_meta_json")
            os.makedirs(pose_meta_dir, exist_ok=True)

            target_frames = [np.full((64, 48, 3), 120, dtype=np.uint8) for _ in range(2)]
            pose_frames = []
            for index in range(2):
                frame_name = f"{index:08d}.jpg"
                pose_meta = _make_pose_meta(frame_name, width=48, height=64)
                with open(os.path.join(pose_meta_dir, f"{index:08d}.json"), "w", encoding="utf-8") as handle:
                    json.dump(pose_meta, handle, indent=2)
                pose_frames.append(draw_pose_frame(np.zeros_like(target_frames[index]), pose_meta))

            _write_rgb_mp4(os.path.join(sample_dir, "target.mp4"), target_frames)
            _write_rgb_mp4(os.path.join(sample_dir, "src_pose.mp4"), pose_frames)

            result = validate_wan_sample_alignment(sample_dir)

            self.assertTrue(os.path.isfile(result["overlay_video_path"]))
            self.assertTrue(os.path.isfile(result["report_path"]))
            self.assertEqual(result["summary"]["target_frame_count"], 2)
            self.assertEqual(result["summary"]["src_pose_frame_count"], 2)
            self.assertEqual(result["summary"]["pose_meta_count"], 2)
            self.assertTrue(result["summary"]["all_counts_match"])
            self.assertEqual(result["summary"]["failed_frames"], 0)
            self.assertEqual(result["summary"]["ok_frames"], 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_validate_wan_sample_alignment_marks_frame_count_mismatch(self):
        from scripts.validate_wan_pose_alignment import validate_wan_sample_alignment

        temp_dir = tempfile.mkdtemp(prefix="wan_validate_")
        try:
            sample_dir = os.path.join(temp_dir, "demo_track_1")
            pose_meta_dir = os.path.join(sample_dir, "pose_meta_json")
            os.makedirs(pose_meta_dir, exist_ok=True)

            target_frames = [np.full((64, 48, 3), 120, dtype=np.uint8) for _ in range(2)]
            pose_meta = _make_pose_meta("00000000.jpg", width=48, height=64)
            with open(os.path.join(pose_meta_dir, "00000000.json"), "w", encoding="utf-8") as handle:
                json.dump(pose_meta, handle, indent=2)
            pose_frames = [draw_pose_frame(np.zeros_like(target_frames[0]), pose_meta)]

            _write_rgb_mp4(os.path.join(sample_dir, "target.mp4"), target_frames)
            _write_rgb_mp4(os.path.join(sample_dir, "src_pose.mp4"), pose_frames)

            result = validate_wan_sample_alignment(sample_dir)

            self.assertFalse(result["summary"]["all_counts_match"])
            self.assertGreaterEqual(result["summary"]["failed_frames"], 1)
            self.assertEqual(result["summary"]["status"], "failed")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
