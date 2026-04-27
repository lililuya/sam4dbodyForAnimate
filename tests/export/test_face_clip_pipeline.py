import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

import cv2
import numpy as np

from scripts.wan_face_export import WanFaceDetection


class FaceClipConfigTests(unittest.TestCase):
    def test_wan_export_config_coerces_face_clip_runtime_values(self):
        from scripts.wan_sample_types import WanExportConfig

        config = WanExportConfig.from_runtime(
            {
                "enable": True,
                "output_dir": "./WanExport",
                "face_clip": {
                    "enable": True,
                    "output_dir": "./face_clips",
                    "min_clip_seconds": 5,
                    "debug_save_face_crop_video": False,
                },
            }
        )

        self.assertTrue(config.enable)
        self.assertTrue(config.face_clip_enable)
        self.assertEqual(config.face_clip_output_dir, "./face_clips")
        self.assertEqual(config.face_clip_min_clip_seconds, 5.0)
        self.assertFalse(config.face_clip_debug_save_face_crop_video)


def _det(bbox, score=0.95):
    return WanFaceDetection(
        bbox=tuple(int(value) for value in bbox),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=float(score),
    )


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_face_clip_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class _SequenceFaceBackend:
    def __init__(self, frame_detections):
        self._frame_detections = list(frame_detections)
        self._index = 0

    def detect(self, frame_bgr):
        del frame_bgr
        if self._index >= len(self._frame_detections):
            return []
        detections = self._frame_detections[self._index]
        self._index += 1
        return list(detections)


class FaceClipSegmentationTests(unittest.TestCase):
    def test_extract_face_clips_keeps_only_segments_with_face_on_every_frame(self):
        from scripts.face_clip_pipeline import extract_face_track_segments

        result = extract_face_track_segments(
            [
                [_det((10, 10, 22, 22))],
                [_det((11, 10, 23, 22))],
                [],
                [_det((12, 10, 24, 22))],
            ],
            fps=1.0,
            min_clip_seconds=2.0,
        )

        self.assertEqual(len(result.kept_segments), 1)
        self.assertEqual(result.kept_segments[0].start_frame, 0)
        self.assertEqual(result.kept_segments[0].end_frame, 1)
        self.assertEqual(len(result.kept_segments[0].records), 2)
        self.assertEqual(len(result.dropped_segments), 1)
        self.assertEqual(result.dropped_segments[0]["reason"], "shorter_than_min_duration")

    def test_extract_face_clips_breaks_segment_on_ambiguous_assignment(self):
        from scripts.face_clip_pipeline import extract_face_track_segments

        result = extract_face_track_segments(
            [
                [_det((10, 10, 22, 22))],
                [_det((11, 10, 23, 22)), _det((12, 10, 24, 22))],
            ],
            fps=1.0,
            min_clip_seconds=1.0,
        )

        self.assertEqual(len(result.kept_segments), 1)
        self.assertEqual(result.kept_segments[0].start_frame, 0)
        self.assertEqual(result.kept_segments[0].end_frame, 0)
        self.assertEqual(len(result.dropped_segments), 1)
        self.assertEqual(result.dropped_segments[0]["reason"], "ambiguous_face_assignment")

    def test_build_face_clip_id_uses_deterministic_segment_naming(self):
        from scripts.face_clip_pipeline import build_face_clip_id

        self.assertEqual(
            build_face_clip_id("sampleuuid123", face_track_index=1, segment_index=2),
            "sampleuuid123_face01_seg002",
        )

    def test_extract_face_clips_from_video_writes_clip_package_contract(self):
        from scripts.face_clip_pipeline import extract_face_clips_from_video

        with make_workspace_tempdir() as temp_dir:
            input_video = os.path.join(temp_dir, "input.mp4")
            output_root = os.path.join(temp_dir, "face_clips")

            writer = cv2.VideoWriter(
                input_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                2.0,
                (32, 32),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (40, 60, 80, 100):
                    frame = np.full((32, 32, 3), value, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            clip_dirs = extract_face_clips_from_video(
                input_video=input_video,
                output_root=output_root,
                min_clip_seconds=2.0,
                face_backend=_SequenceFaceBackend(
                    [
                        [_det((8, 8, 20, 20))],
                        [_det((9, 8, 21, 20))],
                        [_det((10, 8, 22, 20))],
                        [_det((11, 8, 23, 20))],
                    ]
                ),
            )

            self.assertEqual(len(clip_dirs), 1)
            clip_dir = clip_dirs[0]
            self.assertTrue(os.path.isfile(os.path.join(clip_dir, "clip.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(clip_dir, "track.json")))
            self.assertTrue(os.path.isfile(os.path.join(clip_dir, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_root, "source_uuid_map.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_root, "batch_summary.json")))

            with open(os.path.join(clip_dir, "meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertIn("sample_uuid", meta)
            self.assertEqual(meta["frame_count"], 4)
            self.assertTrue(str(meta["clip_id"]).endswith("_face01_seg001"))

            with open(os.path.join(clip_dir, "track.json"), "r", encoding="utf-8") as handle:
                track = json.load(handle)
            self.assertEqual(len(track["records"]), 4)

    def test_extract_face_clips_from_video_accumulates_batch_summary_across_sources(self):
        from scripts.face_clip_pipeline import extract_face_clips_from_video

        with make_workspace_tempdir() as temp_dir:
            output_root = os.path.join(temp_dir, "face_clips")
            input_videos = []
            for index, pixel_value in enumerate((40, 90), start=1):
                input_video = os.path.join(temp_dir, f"input_{index}.mp4")
                writer = cv2.VideoWriter(
                    input_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    2.0,
                    (32, 32),
                )
                self.assertTrue(writer.isOpened())
                try:
                    for frame_offset in range(4):
                        frame = np.full((32, 32, 3), pixel_value + frame_offset, dtype=np.uint8)
                        writer.write(frame)
                finally:
                    writer.release()
                input_videos.append(input_video)

            for input_video in input_videos:
                clip_dirs = extract_face_clips_from_video(
                    input_video=input_video,
                    output_root=output_root,
                    min_clip_seconds=2.0,
                    face_backend=_SequenceFaceBackend(
                        [
                            [_det((8, 8, 20, 20))],
                            [_det((9, 8, 21, 20))],
                            [_det((10, 8, 22, 20))],
                            [_det((11, 8, 23, 20))],
                        ]
                    ),
                )
                self.assertEqual(len(clip_dirs), 1)

            with open(os.path.join(output_root, "batch_summary.json"), "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["video_count_total"], 2)
            self.assertEqual(summary["video_count_completed"], 2)
            self.assertEqual(summary["video_count_failed"], 0)
            self.assertEqual(summary["clip_count_kept"], 2)
            self.assertEqual(len(summary["items"]), 2)


if __name__ == "__main__":
    unittest.main()
