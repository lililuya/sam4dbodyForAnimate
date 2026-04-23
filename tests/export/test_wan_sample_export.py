import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from scripts.wan_face_export import WanFaceDetection


class _FakeFaceBackend:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame_bgr):
        del frame_bgr
        return list(self._detections)


class WanSampleExportTests(unittest.TestCase):
    def test_finalize_writes_wan_sample_contract(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((64, 48, 3), 180, dtype=np.uint8)
            mask = np.zeros((64, 48), dtype=np.uint8)
            mask[8:40, 10:30] = 1
            for index in range(3):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                cv2.imwrite(os.path.join(mask_dir, f"{frame_stem}.png"), mask)

            exporter = WanSampleExporter(
                sample_id="demo",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path=None,
                config=WanExportConfig(enable=True, min_track_frames=1),
                face_backend=_FakeFaceBackend(
                    [
                        WanFaceDetection(
                            bbox=(12, 10, 28, 28),
                            landmarks=np.zeros((5, 3), dtype=np.float32),
                            score=0.9,
                        )
                    ]
                ),
            )

            person_output = {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            }
            for index in range(3):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            sample_dirs = exporter.finalize()

            self.assertEqual(len(sample_dirs), 1)
            sample_dir = sample_dirs[0]
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "target.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "src_pose.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "src_face.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "src_bg.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "src_mask.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "src_ref.png")))
            with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["track_id"], 1)
            self.assertEqual(meta["fps"], 25)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
