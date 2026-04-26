import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.wan_face_export import WanFaceDetection
from scripts.offline_app_refined import save_indexed_mask


class _FakeFaceBackend:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame_bgr):
        del frame_bgr
        return list(self._detections)


class WanSampleExportTests(unittest.TestCase):
    def test_finalize_exports_to_external_uuid_target_directory_and_writes_mapping(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_external_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((64, 48, 3), 180, dtype=np.uint8)
            mask = np.zeros((64, 48), dtype=np.uint8)
            mask[8:40, 10:30] = 1
            for index in range(3):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                save_indexed_mask(mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo.mp4",
                config=WanExportConfig(enable=True, min_track_frames=1, output_dir=export_root),
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

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="abc123def4567890")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [os.path.join(export_root, "abc123def4567890_target1")])
            sample_dir = sample_dirs[0]
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "target.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(export_root, "source_uuid_map.json")))
            self.assertTrue(os.path.isfile(os.path.join(export_root, "abc123def4567890_summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(export_root, "abc123def4567890_skipped.json")))

            with open(os.path.join(export_root, "source_uuid_map.json"), "r", encoding="utf-8") as handle:
                mapping = json.load(handle)
            self.assertEqual(mapping["items"][0]["sample_uuid"], "abc123def4567890")
            self.assertEqual(mapping["items"][0]["source_path"], os.path.abspath("/dataset/source/demo.mp4"))

            with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["sample_uuid"], "abc123def4567890")
            self.assertEqual(meta["track_id"], 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_writes_skipped_target_report(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_skipped_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((48, 48, 3), 180, dtype=np.uint8)
            indexed_mask = np.zeros((48, 48), dtype=np.uint8)
            indexed_mask[8:32, 12:28] = 2
            for index in range(2):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                save_indexed_mask(indexed_mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_skipped",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_skipped.mp4",
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    min_valid_face_ratio=0.6,
                    output_dir=export_root,
                ),
                face_backend=_FakeFaceBackend([]),
            )

            person_output = {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            }
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [2])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="feedfacecafebeef")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            skipped_path = os.path.join(export_root, "feedfacecafebeef_skipped.json")
            self.assertTrue(os.path.isfile(skipped_path))
            with open(skipped_path, "r", encoding="utf-8") as handle:
                skipped = json.load(handle)
            self.assertEqual(len(skipped["skipped_targets"]), 1)
            self.assertEqual(skipped["skipped_targets"][0]["track_id"], 2)
            self.assertEqual(skipped["skipped_targets"][0]["reason"], "valid_face_ratio_below_threshold")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_appends_wan_target_skip_to_issue_ledger(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_skipped_ledger_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((48, 48, 3), 180, dtype=np.uint8)
            indexed_mask = np.zeros((48, 48), dtype=np.uint8)
            indexed_mask[8:32, 12:28] = 2
            for index in range(2):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                save_indexed_mask(indexed_mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_skipped",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_skipped.mp4",
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    min_valid_face_ratio=0.6,
                    output_dir=export_root,
                ),
                face_backend=_FakeFaceBackend([]),
            )

            person_output = {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            }
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [2])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="feedfacecafebeef")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            ledger_path = os.path.join(export_root, "sample_issue_ledger.json")
            self.assertTrue(os.path.isfile(ledger_path))
            with open(ledger_path, "r", encoding="utf-8") as handle:
                ledger = json.load(handle)

            self.assertEqual(len(ledger["items"]), 1)
            self.assertEqual(ledger["items"][0]["event_type"], "wan_target_skipped")
            self.assertEqual(ledger["items"][0]["reason"], "valid_face_ratio_below_threshold")
            self.assertEqual(ledger["items"][0]["sample_uuid"], "feedfacecafebeef")
            self.assertEqual(ledger["items"][0]["details"]["track_id"], 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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

    def test_finalize_writes_bundled_pose_and_smpl_sequences(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_bundled_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((64, 48, 3), 180, dtype=np.uint8)
            mask = np.zeros((64, 48), dtype=np.uint8)
            mask[8:40, 10:30] = 1
            for index in range(3):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                save_indexed_mask(mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_bundled",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_bundled.mp4",
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    output_dir=export_root,
                    save_pose_meta_json=False,
                ),
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
                "bbox": np.array([10.0, 8.0, 30.0, 40.0], dtype=np.float32),
            }
            for index in range(3):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="bundleduuid123456")):
                sample_dirs = exporter.finalize()

            self.assertEqual(len(sample_dirs), 1)
            sample_dir = sample_dirs[0]

            pose_meta_sequence_path = os.path.join(sample_dir, "pose_meta_sequence.json")
            smpl_sequence_path = os.path.join(sample_dir, "smpl_sequence.json")
            self.assertTrue(os.path.isfile(pose_meta_sequence_path))
            self.assertTrue(os.path.isfile(smpl_sequence_path))
            self.assertFalse(os.path.isdir(os.path.join(sample_dir, "pose_meta_json")))

            with open(pose_meta_sequence_path, "r", encoding="utf-8") as handle:
                pose_meta_sequence = json.load(handle)
            self.assertEqual(pose_meta_sequence["sample_id"], "demo_bundled")
            self.assertEqual(pose_meta_sequence["sample_uuid"], "bundleduuid123456")
            self.assertEqual(pose_meta_sequence["track_id"], 1)
            self.assertEqual(pose_meta_sequence["frame_count"], 3)
            self.assertEqual(len(pose_meta_sequence["records"]), 3)

            with open(smpl_sequence_path, "r", encoding="utf-8") as handle:
                smpl_sequence = json.load(handle)
            self.assertEqual(smpl_sequence["sample_id"], "demo_bundled")
            self.assertEqual(smpl_sequence["sample_uuid"], "bundleduuid123456")
            self.assertEqual(smpl_sequence["track_id"], 1)
            self.assertEqual(smpl_sequence["frame_count"], 3)
            self.assertEqual(len(smpl_sequence["records"]), 3)
            self.assertIsInstance(smpl_sequence["records"][0]["person_output"]["pred_keypoints_2d"], list)
            self.assertIsInstance(smpl_sequence["records"][0]["person_output"]["bbox"], list)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_preserves_target_labels_from_paletted_png_masks(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_palette_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((48, 48, 3), 180, dtype=np.uint8)
            indexed_mask = np.zeros((48, 48), dtype=np.uint8)
            indexed_mask[8:32, 12:28] = 1
            for index in range(2):
                frame_stem = f"{index:08d}"
                cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
                save_indexed_mask(indexed_mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_palette",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path=None,
                config=WanExportConfig(enable=True, min_track_frames=1, min_valid_face_ratio=0.0),
                face_backend=_FakeFaceBackend([]),
            )

            person_output = {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            }
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            sample_dirs = exporter.finalize()

            self.assertEqual(len(sample_dirs), 1)
            sample_dir = sample_dirs[0]
            capture = cv2.VideoCapture(os.path.join(sample_dir, "src_mask.mp4"))
            try:
                ok, frame_bgr = capture.read()
            finally:
                capture.release()
            self.assertTrue(ok)
            self.assertGreater(int(frame_bgr.sum()), 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
