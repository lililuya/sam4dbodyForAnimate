import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.wan_face_export import WanFaceDetection


class _FakeFaceBackend:
    def __init__(self, detections_per_call):
        self._detections_per_call = list(detections_per_call)
        self.detect_calls = 0

    def detect(self, frame_bgr):
        del frame_bgr
        call_index = self.detect_calls
        self.detect_calls += 1
        if call_index >= len(self._detections_per_call):
            return []
        return list(self._detections_per_call[call_index])


class InsightFaceVideoDebugScriptTests(unittest.TestCase):
    def test_build_parser_accepts_probe_arguments(self):
        from scripts.debug_insightface_video import build_parser

        args = build_parser().parse_args(
            [
                "--input_video",
                "demo.mp4",
                "--output_video",
                "annotated.mp4",
                "--output_json",
                "summary.json",
                "--stride",
                "3",
                "--ctx_id",
                "1",
                "--provider",
                "buffalo_l",
                "--preload_directory",
                "",
                "--disable_ort_preload",
            ]
        )

        self.assertEqual(args.input_video, "demo.mp4")
        self.assertEqual(args.output_video, "annotated.mp4")
        self.assertEqual(args.output_json, "summary.json")
        self.assertEqual(args.stride, 3)
        self.assertEqual(args.ctx_id, 1)
        self.assertEqual(args.provider, "buffalo_l")
        self.assertEqual(args.preload_directory, "")
        self.assertTrue(args.disable_ort_preload)

    def test_build_output_paths_uses_input_stem_by_default(self):
        from scripts.debug_insightface_video import build_output_paths

        output_video, output_json = build_output_paths("demo/clip.mp4")

        self.assertTrue(output_video.endswith("clip_insightface.mp4"))
        self.assertTrue(output_json.endswith("clip_insightface.json"))

    def test_annotate_frame_draws_face_boxes(self):
        from scripts.debug_insightface_video import annotate_frame

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detections = [
            WanFaceDetection(
                bbox=(4, 4, 20, 20),
                landmarks=np.zeros((5, 2), dtype=np.float32),
                score=0.91,
            )
        ]

        annotated = annotate_frame(frame, detections, frame_index=0, checked=True)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)

    def test_run_probe_writes_video_and_json_summary(self):
        import scripts.debug_insightface_video as debug_insightface_video

        temp_dir = tempfile.mkdtemp(prefix="insightface_probe_")
        try:
            input_video = os.path.join(temp_dir, "clip.mp4")
            output_video = os.path.join(temp_dir, "clip_annotated.mp4")
            output_json = os.path.join(temp_dir, "clip_summary.json")
            with open(input_video, "wb") as handle:
                handle.write(b"")

            args = debug_insightface_video.build_parser().parse_args(
                [
                    "--input_video",
                    input_video,
                    "--output_video",
                    output_video,
                    "--output_json",
                    output_json,
                    "--stride",
                    "2",
                ]
            )

            face_backend = _FakeFaceBackend(
                [
                    [
                        WanFaceDetection(
                            bbox=(2, 2, 10, 10),
                            landmarks=np.zeros((5, 2), dtype=np.float32),
                            score=0.88,
                        )
                    ],
                    [],
                ]
            )

            class FakeCapture:
                def __init__(self):
                    self.frames = [
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        np.zeros((16, 16, 3), dtype=np.uint8),
                    ]
                    self.index = 0

                def isOpened(self):
                    return True

                def read(self):
                    if self.index >= len(self.frames):
                        return False, None
                    frame = self.frames[self.index]
                    self.index += 1
                    return True, frame.copy()

                def get(self, prop_id):
                    if prop_id == debug_insightface_video.cv2.CAP_PROP_FPS:
                        return 25.0
                    if prop_id == debug_insightface_video.cv2.CAP_PROP_FRAME_WIDTH:
                        return 16
                    if prop_id == debug_insightface_video.cv2.CAP_PROP_FRAME_HEIGHT:
                        return 16
                    if prop_id == debug_insightface_video.cv2.CAP_PROP_FRAME_COUNT:
                        return 4
                    return 0

                def release(self):
                    return None

            writer = mock.MagicMock()
            ort_summary = {
                "ort_version": "1.25.0",
                "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "preload_attempted": True,
                "preload_succeeded": True,
            }

            with mock.patch.object(debug_insightface_video.cv2, "VideoCapture", return_value=FakeCapture()):
                with mock.patch.object(debug_insightface_video.cv2, "VideoWriter", return_value=writer):
                    result = debug_insightface_video.run_probe(
                        args,
                        face_backend=face_backend,
                        ort_probe=lambda _args: dict(ort_summary),
                    )

            self.assertEqual(result["frame_count"], 4)
            self.assertEqual(result["checked_frame_count"], 2)
            self.assertEqual(result["face_detected_frame_count"], 1)
            self.assertEqual(result["no_face_frame_count"], 1)
            self.assertEqual(result["total_face_count"], 1)
            self.assertAlmostEqual(result["no_face_ratio"], 0.5)
            self.assertEqual(result["ort"]["available_providers"][0], "CUDAExecutionProvider")
            self.assertEqual(face_backend.detect_calls, 2)
            self.assertEqual(writer.write.call_count, 4)
            self.assertTrue(os.path.isfile(output_json))

            with open(output_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["checked_frame_count"], 2)
            self.assertEqual(len(payload["frame_results"]), 2)
            self.assertEqual(payload["frame_results"][0]["frame_index"], 0)
            self.assertEqual(payload["frame_results"][0]["detection_count"], 1)
            self.assertEqual(payload["frame_results"][1]["frame_index"], 2)
            self.assertEqual(payload["frame_results"][1]["detection_count"], 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
