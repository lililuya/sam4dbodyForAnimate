import unittest
from unittest import mock

import numpy as np


class HumanDetectionDebugScriptTests(unittest.TestCase):
    def test_build_parser_accepts_debug_arguments(self):
        from scripts.debug_human_detection import build_parser

        args = build_parser().parse_args(
            [
                "--input_path",
                "sample.mp4",
                "--detector_backend",
                "vitdet",
                "--output_path",
                "out.mp4",
                "--bbox_thresh",
                "0.6",
                "--max_det",
                "3",
            ]
        )

        self.assertEqual(args.input_path, "sample.mp4")
        self.assertEqual(args.detector_backend, "vitdet")
        self.assertEqual(args.output_path, "out.mp4")
        self.assertEqual(args.bbox_thresh, 0.6)
        self.assertEqual(args.max_det, 3)

    def test_infer_media_type_recognizes_supported_inputs(self):
        from scripts.debug_human_detection import infer_media_type

        self.assertEqual(infer_media_type("frame.png"), "image")
        self.assertEqual(infer_media_type("clip.MP4"), "video")

        with self.assertRaisesRegex(ValueError, "unsupported input file"):
            infer_media_type("notes.txt")

    def test_build_output_path_uses_detected_suffix_by_default(self):
        from scripts.debug_human_detection import build_output_path

        self.assertTrue(build_output_path("demo/frame.png").endswith("frame_detected.png"))
        self.assertTrue(build_output_path("demo/clip.mp4").endswith("clip_detected.mp4"))
        self.assertEqual(build_output_path("demo/frame.png", explicit_output_path="custom.jpg"), "custom.jpg")

    def test_annotate_frame_draws_detection_boxes(self):
        from scripts.debug_human_detection import annotate_frame

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detections = [{"bbox": [4.0, 4.0, 20.0, 20.0], "score": 0.91}]

        annotated = annotate_frame(frame, detections)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)

    def test_run_on_image_writes_annotated_image(self):
        import scripts.debug_human_detection as debug_human_detection

        args = debug_human_detection.build_parser().parse_args(["--input_path", "person.png"])
        detector = mock.MagicMock()
        detector.run_human_detection.return_value = [{"bbox": [1.0, 2.0, 10.0, 12.0], "score": 0.95}]

        with mock.patch.object(debug_human_detection, "create_detector", return_value=detector):
            with mock.patch.object(debug_human_detection.cv2, "imread", return_value=np.zeros((24, 24, 3), dtype=np.uint8)):
                with mock.patch.object(debug_human_detection.cv2, "imwrite", return_value=True) as imwrite_mock:
                    result = debug_human_detection.run_detection(args)

        self.assertEqual(result["media_type"], "image")
        self.assertTrue(result["output_path"].endswith("person_detected.png"))
        imwrite_mock.assert_called_once()
        detector.run_human_detection.assert_called_once()

    def test_run_on_video_writes_annotated_video(self):
        import scripts.debug_human_detection as debug_human_detection

        args = debug_human_detection.build_parser().parse_args(
            ["--input_path", "clip.mp4", "--output_path", "annotated.mp4"]
        )
        detector = mock.MagicMock()
        detector.run_human_detection.return_value = [{"bbox": [2.0, 2.0, 8.0, 8.0], "score": 0.8}]

        class FakeCapture:
            def __init__(self):
                self.frames = [
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
                if prop_id == debug_human_detection.cv2.CAP_PROP_FPS:
                    return 25.0
                if prop_id == debug_human_detection.cv2.CAP_PROP_FRAME_WIDTH:
                    return 16
                if prop_id == debug_human_detection.cv2.CAP_PROP_FRAME_HEIGHT:
                    return 16
                if prop_id == debug_human_detection.cv2.CAP_PROP_FRAME_COUNT:
                    return 2
                return 0

            def release(self):
                return None

        writer = mock.MagicMock()

        with mock.patch.object(debug_human_detection, "create_detector", return_value=detector):
            with mock.patch.object(debug_human_detection.cv2, "VideoCapture", return_value=FakeCapture()):
                with mock.patch.object(debug_human_detection.cv2, "VideoWriter", return_value=writer):
                    result = debug_human_detection.run_detection(args)

        self.assertEqual(result["media_type"], "video")
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["output_path"], "annotated.mp4")
        self.assertEqual(writer.write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
