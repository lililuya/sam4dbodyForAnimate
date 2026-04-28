import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest import mock


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


class ScanBadVideosTests(unittest.TestCase):
    def test_build_parser_accepts_root_and_output_arguments(self):
        from scripts.scan_bad_videos import build_parser

        args = build_parser().parse_args(
            [
                "--input_root",
                "./inputs",
                "--output_json",
                "./bad_videos.json",
                "--ffmpeg_bin",
                "ffmpeg",
                "--max_error_lines",
                "5",
            ]
        )

        self.assertEqual(args.input_root, "./inputs")
        self.assertEqual(args.output_json, "./bad_videos.json")
        self.assertEqual(args.ffmpeg_bin, "ffmpeg")
        self.assertEqual(args.max_error_lines, 5)

    def test_scan_videos_reports_bad_mp4_and_skips_frame_directories(self):
        from scripts.scan_bad_videos import scan_videos

        with make_workspace_tempdir() as temp_dir:
            input_root = os.path.join(temp_dir, "inputs")
            os.makedirs(input_root, exist_ok=True)

            good_video = os.path.join(input_root, "good.mp4")
            bad_video = os.path.join(input_root, "bad.mp4")
            frame_dir = os.path.join(input_root, "frames_a")
            os.makedirs(frame_dir, exist_ok=True)
            with open(good_video, "wb") as handle:
                handle.write(b"good")
            with open(bad_video, "wb") as handle:
                handle.write(b"bad")
            with open(os.path.join(frame_dir, "00000001.jpg"), "wb") as handle:
                handle.write(b"jpg")

            def fake_run(command, capture_output, text, check):
                self.assertTrue(capture_output)
                self.assertTrue(text)
                self.assertFalse(check)
                input_path = command[4]
                if os.path.abspath(input_path) == os.path.abspath(bad_video):
                    return mock.Mock(
                        returncode=0,
                        stdout="",
                        stderr="Invalid NAL unit size\nmissing picture in access unit\n",
                    )
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("scripts.scan_bad_videos.subprocess.run", side_effect=fake_run) as mock_run:
                summary = scan_videos(input_root=input_root, ffmpeg_bin="ffmpeg", max_error_lines=1)

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(summary["sample_count_total"], 3)
        self.assertEqual(summary["checked_video_count"], 2)
        self.assertEqual(summary["bad_video_count"], 1)
        self.assertEqual(summary["skipped_non_video_count"], 1)
        self.assertEqual(summary["bad_videos"][0]["sample_id"], "bad")
        self.assertEqual(summary["bad_videos"][0]["returncode"], 0)
        self.assertEqual(summary["bad_videos"][0]["stderr_preview"], ["Invalid NAL unit size"])

    def test_main_writes_bad_video_json_from_input_list(self):
        from scripts.scan_bad_videos import main

        with make_workspace_tempdir() as temp_dir:
            good_video = os.path.join(temp_dir, "good.mp4")
            bad_video = os.path.join(temp_dir, "bad.mp4")
            frames_dir = os.path.join(temp_dir, "frames_b")
            os.makedirs(frames_dir, exist_ok=True)
            with open(good_video, "wb") as handle:
                handle.write(b"good")
            with open(bad_video, "wb") as handle:
                handle.write(b"bad")
            with open(os.path.join(frames_dir, "00000001.jpg"), "wb") as handle:
                handle.write(b"jpg")

            input_list = os.path.join(temp_dir, "inputs.jsonl")
            with open(input_list, "w", encoding="utf-8") as handle:
                handle.write(good_video + "\n")
                handle.write(json.dumps({"input": bad_video, "sample_id": "broken_sample"}) + "\n")
                handle.write(frames_dir + "\n")

            output_json = os.path.join(temp_dir, "bad_videos.json")

            def fake_run(command, capture_output, text, check):
                input_path = command[4]
                if os.path.abspath(input_path) == os.path.abspath(bad_video):
                    return mock.Mock(returncode=1, stdout="", stderr="Error splitting the input into NAL units.\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("scripts.scan_bad_videos.subprocess.run", side_effect=fake_run):
                exit_code = main(
                    [
                        "--input_list",
                        input_list,
                        "--output_json",
                        output_json,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(output_json))
            with open(output_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["bad_video_count"], 1)
            self.assertEqual(payload["bad_videos"][0]["sample_id"], "broken_sample")
            self.assertEqual(payload["bad_videos"][0]["returncode"], 1)
            self.assertEqual(
                payload["bad_videos"][0]["stderr_preview"],
                ["Error splitting the input into NAL units."],
            )

    def test_scan_videos_discovers_clip_package_directories_under_input_root(self):
        from scripts.scan_bad_videos import scan_videos

        with make_workspace_tempdir() as temp_dir:
            clips_root = os.path.join(temp_dir, "clips")
            clip_dir = os.path.join(clips_root, "uuid_face01_seg001")
            os.makedirs(clip_dir, exist_ok=True)
            clip_video = os.path.join(clip_dir, "clip.mp4")
            with open(clip_video, "wb") as handle:
                handle.write(b"clip")

            def fake_run(command, capture_output, text, check):
                input_path = command[4]
                self.assertEqual(os.path.abspath(input_path), os.path.abspath(clip_video))
                return mock.Mock(
                    returncode=0,
                    stdout="",
                    stderr="Invalid NAL unit size\n",
                )

            with mock.patch("scripts.scan_bad_videos.subprocess.run", side_effect=fake_run) as mock_run:
                summary = scan_videos(input_root=clips_root, ffmpeg_bin="ffmpeg", max_error_lines=5)

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(summary["sample_count_total"], 1)
        self.assertEqual(summary["checked_video_count"], 1)
        self.assertEqual(summary["bad_video_count"], 1)
        self.assertEqual(summary["bad_videos"][0]["sample_id"], "uuid_face01_seg001")
        self.assertEqual(summary["bad_videos"][0]["input"], os.path.abspath(clip_video))


if __name__ == "__main__":
    unittest.main()
