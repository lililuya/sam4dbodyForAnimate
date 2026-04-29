import json
import os
import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from scripts.wan_face_export import WanFaceDetection
from scripts.offline_app_refined import save_indexed_mask


@contextmanager
def make_workspace_tempdir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(repo_root, ".tmp_wan_export_tests")
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = os.path.join(base_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class _FakeFaceBackend:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame_bgr):
        del frame_bgr
        return list(self._detections)


DEFAULT_MESH_FACES = np.array([[0, 1, 2]], dtype=np.int32)


def _build_person_output(**overrides):
    person_output = {
        "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
        "pred_vertices": np.array(
            [
                [-0.30, -0.30, 1.00],
                [0.30, -0.30, 1.00],
                [-0.30, 0.30, 1.00],
            ],
            dtype=np.float32,
        ),
        "pred_cam_t": np.zeros((3,), dtype=np.float32),
        "focal_length": np.float32(24.0),
    }
    person_output.update(overrides)
    return person_output


class WanSampleExportTests(unittest.TestCase):
    def test_finalize_reuses_pre_resolved_clip_identity_for_target_directory(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        with make_workspace_tempdir() as temp_dir:
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
                sample_id="demo_clip",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_clip.mp4",
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
                sample_uuid="sampleuuid123",
                clip_id="sampleuuid123_face01_seg001",
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
            for index in range(3):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [os.path.join(export_root, "sampleuuid123_face01_seg001_target1")])
            self.assertFalse(os.path.isfile(os.path.join(export_root, "source_uuid_map.json")))
            with open(os.path.join(sample_dirs[0], "meta.json"), "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["sample_uuid"], "sampleuuid123")
            self.assertEqual(meta["clip_id"], "sampleuuid123_face01_seg001")

    def test_finalize_exports_to_external_uuid_target_directory_and_writes_mapping(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_external_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            metadata_root = os.path.join(temp_dir, "WanExportMeta")
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
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    output_dir=export_root,
                    metadata_output_dir=metadata_root,
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
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
            for index in range(3):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="abc123def4567890")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [os.path.join(export_root, "abc123def4567890_target1")])
            sample_dir = sample_dirs[0]
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "target.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(metadata_root, "source_uuid_map.json")))
            self.assertTrue(os.path.isfile(os.path.join(metadata_root, "abc123def4567890_summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(metadata_root, "abc123def4567890_skipped.json")))
            self.assertFalse(os.path.isfile(os.path.join(export_root, "source_uuid_map.json")))

            with open(os.path.join(metadata_root, "source_uuid_map.json"), "r", encoding="utf-8") as handle:
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
            metadata_root = os.path.join(temp_dir, "WanExportMeta")
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
                    metadata_output_dir=metadata_root,
                ),
                face_backend=_FakeFaceBackend([]),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [2])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="feedfacecafebeef")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            skipped_path = os.path.join(metadata_root, "feedfacecafebeef_skipped.json")
            self.assertTrue(os.path.isfile(skipped_path))
            with open(skipped_path, "r", encoding="utf-8") as handle:
                skipped = json.load(handle)
            self.assertEqual(len(skipped["skipped_targets"]), 1)
            self.assertEqual(skipped["skipped_targets"][0]["track_id"], 2)
            self.assertEqual(skipped["skipped_targets"][0]["reason"], "no_valid_face_detected")
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
            metadata_root = os.path.join(temp_dir, "WanExportMeta")
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
                    metadata_output_dir=metadata_root,
                ),
                face_backend=_FakeFaceBackend([]),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [2])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="feedfacecafebeef")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            ledger_path = os.path.join(metadata_root, "sample_issue_ledger.json")
            self.assertTrue(os.path.isfile(ledger_path))
            with open(ledger_path, "r", encoding="utf-8") as handle:
                ledger = json.load(handle)

            self.assertEqual(len(ledger["items"]), 1)
            self.assertEqual(ledger["items"][0]["event_type"], "wan_target_skipped")
            self.assertEqual(ledger["items"][0]["reason"], "no_valid_face_detected")
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
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
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

    def test_finalize_does_not_write_smpl_sequence_by_default(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_pose_only_")
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
                sample_id="demo_pose_only",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_pose_only.mp4",
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
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output(
                bbox=np.array([10.0, 8.0, 30.0, 40.0], dtype=np.float32),
                mask=np.zeros((64, 48, 1), dtype=np.uint8),
            )
            for index in range(3):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="poseonlyuuid1234")):
                sample_dirs = exporter.finalize()

            self.assertEqual(len(sample_dirs), 1)
            sample_dir = sample_dirs[0]
            self.assertTrue(os.path.isfile(os.path.join(sample_dir, "pose_meta_sequence.json")))
            self.assertFalse(os.path.isfile(os.path.join(sample_dir, "smpl_sequence.json")))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_skips_target_when_smpl_projection_mask_is_unavailable(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_projection_skip_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((48, 48, 3), 180, dtype=np.uint8)
            indexed_mask = np.zeros((48, 48), dtype=np.uint8)
            indexed_mask[30:40, 30:40] = 1
            raw_mask = np.zeros((48, 48), dtype=np.uint8)
            raw_mask[8:20, 8:20] = 255
            frame_stem = "00000000"
            cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
            save_indexed_mask(indexed_mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_projection_skip",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_projection_skip.mp4",
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    min_valid_face_ratio=0.0,
                    output_dir=export_root,
                    save_pose_meta_json=False,
                ),
                face_backend=_FakeFaceBackend(
                    [
                        WanFaceDetection(
                            bbox=(8, 8, 22, 22),
                            landmarks=np.zeros((5, 3), dtype=np.float32),
                            score=0.9,
                        )
                    ]
                ),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = {
                "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
                "mask": raw_mask,
            }
            exporter(os.path.join(image_dir, f"{frame_stem}.jpg"), [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="projectionskipuuid")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            skipped_path = os.path.join(export_root, "projectionskipuuid_skipped.json")
            self.assertTrue(os.path.isfile(skipped_path))
            with open(skipped_path, "r", encoding="utf-8") as handle:
                skipped = json.load(handle)
            self.assertEqual(len(skipped["skipped_targets"]), 1)
            self.assertEqual(skipped["skipped_targets"][0]["track_id"], 1)
            self.assertEqual(skipped["skipped_targets"][0]["reason"], "smpl_projection_mask_unavailable")
            self.assertEqual(skipped["skipped_targets"][0]["frame_stem"], frame_stem)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_skips_target_when_no_valid_face_is_matched_even_with_zero_threshold(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_no_face_skip_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            export_root = os.path.join(temp_dir, "WanExport")
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
                sample_id="demo_no_face_skip",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path="/dataset/source/demo_no_face_skip.mp4",
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    min_valid_face_ratio=0.0,
                    output_dir=export_root,
                    save_pose_meta_json=False,
                ),
                face_backend=_FakeFaceBackend([]),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
            for index in range(2):
                image_path = os.path.join(image_dir, f"{index:08d}.jpg")
                exporter(image_path, [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="nofaceuuid123456")):
                sample_dirs = exporter.finalize()

            self.assertEqual(sample_dirs, [])
            skipped_path = os.path.join(export_root, "nofaceuuid123456_skipped.json")
            self.assertTrue(os.path.isfile(skipped_path))
            with open(skipped_path, "r", encoding="utf-8") as handle:
                skipped = json.load(handle)
            self.assertEqual(len(skipped["skipped_targets"]), 1)
            self.assertEqual(skipped["skipped_targets"][0]["track_id"], 1)
            self.assertEqual(skipped["skipped_targets"][0]["reason"], "no_valid_face_detected")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_writes_bundled_pose_and_optional_smpl_sequences(self):
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
                    save_smpl_sequence_json=True,
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
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output(
                bbox=np.array([10.0, 8.0, 30.0, 40.0], dtype=np.float32),
                mask=np.zeros((64, 48, 1), dtype=np.uint8),
            )
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
            self.assertIsInstance(smpl_sequence["records"][0]["person_output"]["mask"], list)
            self.assertEqual(smpl_sequence["records"][0]["person_output"]["mask_source"], "smpl_projection")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_finalize_writes_smpl_sequence_without_indent_and_preserves_full_float_precision(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_smpl_rounding_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((64, 48, 3), 120, dtype=np.uint8)
            mask = np.zeros((64, 48), dtype=np.uint8)
            mask[8:40, 10:30] = 1
            frame_stem = "00000000"
            Image.fromarray(frame).save(os.path.join(image_dir, f"{frame_stem}.jpg"))
            save_indexed_mask(mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_precision",
                output_dir=os.path.join(temp_dir, "working"),
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path=None,
                config=WanExportConfig(
                    enable=True,
                    output_dir=os.path.join(temp_dir, "wan_export"),
                    fps=25,
                    resolution_area=[48, 64],
                    face_resolution=[32, 32],
                    min_track_frames=1,
                    min_valid_face_ratio=0.0,
                    save_pose_meta_json=False,
                    save_smpl_sequence_json=True,
                ),
                face_backend=_FakeFaceBackend(
                    [
                        WanFaceDetection(
                            bbox=(8, 8, 20, 20),
                            landmarks=np.zeros((5, 2), dtype=np.float32),
                            score=0.99,
                        )
                    ]
                ),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output(
                pred_keypoints_2d=np.array([[0.123456789, 1.987654321]], dtype=np.float64),
                pred_keypoints_3d=np.array(
                    [[0.2565191686153412, -1.5157936811447144, -0.092339888215065]],
                    dtype=np.float64,
                ),
                bbox=np.array([10.123456789, 8.987654321, 30.111111111, 40.999999999], dtype=np.float64),
            )
            exporter(os.path.join(image_dir, f"{frame_stem}.jpg"), [person_output], [1])

            with patch("scripts.wan_sample_export.uuid.uuid4", return_value=SimpleNamespace(hex="rounduuid123456")):
                sample_dirs = exporter.finalize()

            sample_dir = sample_dirs[0]
            smpl_sequence_path = os.path.join(sample_dir, "smpl_sequence.json")
            with open(smpl_sequence_path, "r", encoding="utf-8") as handle:
                raw_text = handle.read()
            with open(smpl_sequence_path, "r", encoding="utf-8") as handle:
                smpl_sequence = json.load(handle)

            self.assertNotIn("\n  ", raw_text)
            self.assertIn('"frame_count": 1', raw_text)
            person_output_payload = smpl_sequence["records"][0]["person_output"]
            self.assertNotEqual(person_output_payload["pred_keypoints_2d"][0][0], 0.123457)
            self.assertAlmostEqual(person_output_payload["pred_keypoints_2d"][0][0], 0.123456789, places=15)
            self.assertAlmostEqual(person_output_payload["pred_keypoints_2d"][0][1], 1.987654321, places=15)
            self.assertAlmostEqual(person_output_payload["pred_keypoints_3d"][0][0], 0.2565191686153412, places=15)
            self.assertAlmostEqual(person_output_payload["pred_keypoints_3d"][0][1], -1.5157936811447144, places=15)
            self.assertAlmostEqual(person_output_payload["pred_keypoints_3d"][0][2], -0.092339888215065, places=15)
            self.assertAlmostEqual(person_output_payload["bbox"][0], 10.123456789, places=15)
            self.assertAlmostEqual(person_output_payload["bbox"][1], 8.987654321, places=15)
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
                face_backend=_FakeFaceBackend(
                    [
                        WanFaceDetection(
                            bbox=(12, 10, 28, 28),
                            landmarks=np.zeros((5, 3), dtype=np.float32),
                            score=0.9,
                        )
                    ]
                ),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output()
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

    def test_finalize_uses_smpl_projection_mask_for_src_mask_detail_and_background_cutout(self):
        from scripts.wan_sample_export import WanSampleExporter
        from scripts.wan_sample_types import WanExportConfig

        temp_dir = tempfile.mkdtemp(prefix="wan_export_mask_detail_")
        try:
            image_dir = os.path.join(temp_dir, "images")
            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            frame = np.full((48, 48, 3), 160, dtype=np.uint8)
            indexed_mask = np.zeros((48, 48), dtype=np.uint8)
            indexed_mask[36:40, 36:40] = 1
            raw_mask = np.zeros((48, 48), dtype=np.uint8)
            raw_mask[8:16, 8:16] = 255
            frame_stem = "00000000"
            cv2.imwrite(os.path.join(image_dir, f"{frame_stem}.jpg"), frame)
            save_indexed_mask(indexed_mask, os.path.join(mask_dir, f"{frame_stem}.png"))

            exporter = WanSampleExporter(
                sample_id="demo_mask_detail",
                output_dir=temp_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
                source_video_path=None,
                config=WanExportConfig(
                    enable=True,
                    min_track_frames=1,
                    min_valid_face_ratio=0.0,
                    resolution_area=(48, 48),
                    face_resolution=(32, 32),
                    mask_kernel_size=3,
                    mask_iterations=1,
                    mask_w_len=99,
                    mask_h_len=99,
                ),
                face_backend=_FakeFaceBackend(
                    [
                        WanFaceDetection(
                            bbox=(18, 18, 30, 30),
                            landmarks=np.zeros((5, 3), dtype=np.float32),
                            score=0.9,
                        )
                    ]
                ),
                mesh_faces=DEFAULT_MESH_FACES,
            )

            person_output = _build_person_output(
                mask=raw_mask,
                pred_vertices=np.array(
                    [
                        [-0.15, -0.15, 1.00],
                        [0.45, -0.15, 1.00],
                        [-0.15, 0.45, 1.00],
                    ],
                    dtype=np.float32,
                ),
                focal_length=np.float32(16.0),
            )
            exporter(os.path.join(image_dir, f"{frame_stem}.jpg"), [person_output], [1])

            sample_dirs = exporter.finalize()

            self.assertEqual(len(sample_dirs), 1)
            sample_dir = sample_dirs[0]
            src_mask_detail_path = os.path.join(sample_dir, "src_mask_detail.mp4")
            src_mask_path = os.path.join(sample_dir, "src_mask.mp4")
            self.assertTrue(os.path.isfile(src_mask_path))
            self.assertTrue(os.path.isfile(src_mask_detail_path))

            mask_capture = cv2.VideoCapture(src_mask_path)
            try:
                ok_mask, mask_frame_bgr = mask_capture.read()
            finally:
                mask_capture.release()

            detail_capture = cv2.VideoCapture(src_mask_detail_path)
            try:
                ok_detail, detail_frame_bgr = detail_capture.read()
            finally:
                detail_capture.release()

            bg_capture = cv2.VideoCapture(os.path.join(sample_dir, "src_bg.mp4"))
            try:
                ok_bg, bg_frame_bgr = bg_capture.read()
            finally:
                bg_capture.release()

            self.assertTrue(ok_mask)
            self.assertTrue(ok_detail)
            self.assertTrue(ok_bg)
            self.assertLess(int(mask_frame_bgr[10:14, 10:14].mean()), 50)
            self.assertLess(int(mask_frame_bgr[36:40, 36:40].mean()), 50)
            self.assertGreater(int(mask_frame_bgr[22:26, 22:26].mean()), 200)
            self.assertLess(int(detail_frame_bgr[10:14, 10:14].mean()), 50)
            self.assertLess(int(detail_frame_bgr[36:40, 36:40].mean()), 50)
            self.assertGreater(int(detail_frame_bgr[22:26, 22:26].mean()), 200)
            self.assertGreater(int(bg_frame_bgr[10:14, 10:14].mean()), 120)
            self.assertGreater(int(bg_frame_bgr[36:40, 36:40].mean()), 120)
            self.assertLess(int(bg_frame_bgr[22:26, 22:26].mean()), 10)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
