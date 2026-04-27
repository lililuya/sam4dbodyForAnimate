from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Iterable
import uuid

import cv2
import numpy as np
from PIL import Image

from scripts.wan_face_export import InsightFaceBackend, crop_face_frame, fill_face_gaps, select_target_face
from scripts.wan_mask_bg_export import build_bg_and_mask_frame, score_reference_frame
from scripts.wan_pose_adapter import build_wan_pose_meta, render_wan_pose_frame
from scripts.wan_reference_compat import compute_sample_indices, resize_frame_by_area
from scripts.wan_sample_types import WanExportConfig


SOURCE_UUID_MAP_FILENAME = "source_uuid_map.json"
ISSUE_LEDGER_FILENAME = "sample_issue_ledger.json"


class CompositeFrameWriter:
    def __init__(self, writers: Iterable[object]):
        self._writers = [writer for writer in writers if writer is not None]

    def __call__(self, image_path, mask_output, id_current):
        results = []
        for writer in self._writers:
            results.append(writer(image_path, mask_output, id_current))
        return results

    def finalize(self) -> list[str]:
        outputs: list[str] = []
        for writer in self._writers:
            finalize = getattr(writer, "finalize", None)
            if not callable(finalize):
                continue
            result = finalize()
            if result:
                outputs.extend(result)
        return outputs


def _coerce_config(config) -> WanExportConfig:
    if isinstance(config, WanExportConfig):
        return config
    return WanExportConfig.from_runtime(config)


def _read_json_dict(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _to_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def _merge_dicts(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
            continue
        merged[key] = value
    return merged


def resolve_wan_source_path(source_video_path: str | None, images_dir: str | None) -> str:
    if source_video_path:
        return os.path.abspath(source_video_path)
    if images_dir:
        return os.path.abspath(images_dir)
    return os.path.abspath(".")


def get_wan_source_uuid_map_path(export_root: str) -> str:
    return os.path.join(os.path.abspath(export_root), SOURCE_UUID_MAP_FILENAME)


def get_wan_summary_path(export_root: str, sample_uuid: str) -> str:
    return os.path.join(os.path.abspath(export_root), f"{sample_uuid}_summary.json")


def get_wan_skipped_path(export_root: str, sample_uuid: str) -> str:
    return os.path.join(os.path.abspath(export_root), f"{sample_uuid}_skipped.json")


def get_wan_issue_ledger_path(root_dir: str) -> str:
    return os.path.join(os.path.abspath(root_dir), ISSUE_LEDGER_FILENAME)


def resolve_or_create_wan_sample_uuid(
    export_root: str,
    source_path: str,
    sample_id: str | None = None,
    working_output_dir: str | None = None,
) -> str:
    export_root_abs = os.path.abspath(export_root)
    source_path_abs = os.path.abspath(source_path)
    mapping_path = get_wan_source_uuid_map_path(export_root_abs)
    mapping = _read_json_dict(mapping_path)
    items = mapping.get("items")
    if not isinstance(items, list):
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        if os.path.abspath(str(item.get("source_path", ""))) != source_path_abs:
            continue
        sample_uuid = str(item.get("sample_uuid", "")).strip()
        if sample_uuid:
            return sample_uuid

    sample_uuid = uuid.uuid4().hex
    items.append(
        {
            "sample_uuid": sample_uuid,
            "sample_id": None if sample_id is None else str(sample_id),
            "source_path": source_path_abs,
            "working_output_dir": None if not working_output_dir else os.path.abspath(working_output_dir),
        }
    )
    mapping["items"] = items
    _write_json(mapping_path, mapping)
    return sample_uuid


def update_wan_sample_summary(export_root: str, sample_uuid: str, updates: dict) -> dict:
    summary_path = get_wan_summary_path(export_root, sample_uuid)
    current = _read_json_dict(summary_path)
    merged = _merge_dicts(current, {"sample_uuid": str(sample_uuid), **dict(updates or {})})
    _write_json(summary_path, merged)
    return merged


def write_wan_skipped_report(export_root: str, sample_uuid: str, payload: dict) -> dict:
    report = {"sample_uuid": str(sample_uuid), **dict(payload or {})}
    _write_json(get_wan_skipped_path(export_root, sample_uuid), report)
    return report


def append_wan_issue_records(root_dir: str, records: list[dict]) -> dict:
    ledger_path = get_wan_issue_ledger_path(root_dir)
    ledger = _read_json_dict(ledger_path)
    items = ledger.get("items")
    if not isinstance(items, list):
        items = []

    for record in records:
        if not isinstance(record, dict):
            continue
        items.append(dict(record))

    ledger["items"] = items
    _write_json(ledger_path, ledger)
    return ledger


def _resolve_source_fps(source_video_path: str | None, default_fps: float) -> float:
    if source_video_path and os.path.isfile(source_video_path):
        capture = cv2.VideoCapture(source_video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if fps > 0:
            return fps
    return float(default_fps)


def _resize_indexed_mask(indexed_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    return cv2.resize(
        np.asarray(indexed_mask, dtype=np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _load_indexed_mask(mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as mask_image:
        return np.asarray(mask_image.convert("P"), dtype=np.uint8)


def _scale_person_output(person_output: dict, scale_x: float, scale_y: float) -> dict:
    scaled = dict(person_output)
    keypoints_2d = np.asarray(person_output["pred_keypoints_2d"], dtype=np.float32).copy()
    keypoints_2d[:, 0] *= float(scale_x)
    keypoints_2d[:, 1] *= float(scale_y)
    scaled["pred_keypoints_2d"] = keypoints_2d

    if "bbox" in person_output:
        bbox = np.asarray(person_output["bbox"], dtype=np.float32).copy()
        if bbox.shape[-1] >= 4:
            bbox[..., 0] *= float(scale_x)
            bbox[..., 2] *= float(scale_x)
            bbox[..., 1] *= float(scale_y)
            bbox[..., 3] *= float(scale_y)
        scaled["bbox"] = bbox
    return scaled


def _expand_face_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_shape: tuple[int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    expanded_area = width * height * max(float(scale), 1.0)
    new_width = math.sqrt(expanded_area * (width / height))
    new_height = math.sqrt(expanded_area * (height / width))
    delta_width = (new_width - width) / 2.0
    delta_height = (new_height - height) / 4.0

    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    expanded_x1 = max(int(round(x1 - delta_width)), 0)
    expanded_x2 = min(int(round(x2 + delta_width)), frame_width)
    expanded_y1 = max(int(round(y1 - 3.0 * delta_height)), 0)
    expanded_y2 = min(int(round(y2 + delta_height)), frame_height)
    return expanded_x1, expanded_y1, expanded_x2, expanded_y2


class WanSampleExporter:
    def __init__(
        self,
        *,
        sample_id: str,
        output_dir: str,
        images_dir: str,
        masks_dir: str,
        source_video_path: str | None,
        config,
        face_backend=None,
        sample_uuid: str | None = None,
        clip_id: str | None = None,
        source_reference: str | None = None,
    ):
        self.sample_id = str(sample_id)
        self.output_dir = output_dir
        self.working_output_dir = os.path.abspath(output_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.source_video_path = source_video_path
        self.config = _coerce_config(config)
        self.export_root = os.path.abspath(self.config.output_dir) if self.config.output_dir else None
        self.metadata_root = (
            os.path.abspath(self.config.metadata_output_dir)
            if self.config.metadata_output_dir
            else self.export_root
        )
        self.source_reference = (
            os.path.abspath(str(source_reference))
            if source_reference not in {None, ""}
            else resolve_wan_source_path(source_video_path, images_dir)
        )
        self.sample_uuid = None if sample_uuid in {None, ""} else str(sample_uuid)
        self.clip_id = None if clip_id in {None, ""} else str(clip_id)
        self.face_backend = face_backend or InsightFaceBackend()
        self._records: list[dict] = []
        self.finalized_targets: list[dict] = []

    def __call__(self, image_path, mask_output, id_current):
        frame_stem = os.path.splitext(os.path.basename(image_path))[0]
        self._records.append(
            {
                "frame_stem": frame_stem,
                "image_path": image_path,
                "mask_path": os.path.join(self.masks_dir, f"{frame_stem}.png"),
                "person_outputs": {
                    int(track_id): person_output
                    for track_id, person_output in zip(id_current or [], mask_output or [])
                },
            }
        )
        return None

    def _write_mp4(self, frames: list[np.ndarray], path: str, fps: int) -> None:
        if not frames:
            return
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
        try:
            for frame_rgb in frames:
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def _resolve_sample_dir(self, track_id: int) -> str:
        sample_identity = self._summary_identity()
        if self.export_root and sample_identity:
            return os.path.join(self.export_root, f"{sample_identity}_target{int(track_id)}")
        return os.path.join(self.working_output_dir, "wan_export", f"{self.sample_id}_track_{track_id}")

    def _ensure_sample_uuid(self) -> str | None:
        if self.sample_uuid:
            return self.sample_uuid
        if not self.metadata_root:
            return None
        self.sample_uuid = resolve_or_create_wan_sample_uuid(
            self.metadata_root,
            self.source_reference,
            sample_id=self.sample_id,
            working_output_dir=self.working_output_dir,
        )
        return self.sample_uuid

    def _summary_identity(self) -> str | None:
        if self.clip_id:
            return self.clip_id
        return self._ensure_sample_uuid()

    def finalize(self) -> list[str]:
        if not self._records:
            return []

        sampled_records = [
            self._records[index]
            for index in compute_sample_indices(
                num_frames=len(self._records),
                source_fps=_resolve_source_fps(self.source_video_path, self.config.fps),
                target_fps=self.config.fps,
            )
        ]
        track_ids = sorted(
            {
                int(track_id)
                for record in sampled_records
                for track_id in record["person_outputs"].keys()
            }
        )

        written: list[str] = []
        written_targets: list[dict] = []
        self.finalized_targets = []
        skipped_targets: list[dict] = []
        for track_id in track_ids:
            track_records = [record for record in sampled_records if track_id in record["person_outputs"]]
            if len(track_records) < int(self.config.min_track_frames):
                skipped_targets.append(
                    {
                        "track_id": int(track_id),
                        "reason": "track_frames_below_threshold",
                        "track_frame_count": len(track_records),
                        "min_track_frames": int(self.config.min_track_frames),
                    }
                )
                continue

            face_sequence = []
            frame_payloads = []
            previous_bbox = None
            for record in track_records:
                frame_bgr = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
                indexed_mask = _load_indexed_mask(record["mask_path"])
                if frame_bgr is None or indexed_mask is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                resized_rgb = resize_frame_by_area(frame_rgb, self.config.resolution_area)
                resized_mask = _resize_indexed_mask(indexed_mask, resized_rgb.shape[:2])
                scale_x = float(resized_rgb.shape[1]) / float(max(frame_rgb.shape[1], 1))
                scale_y = float(resized_rgb.shape[0]) / float(max(frame_rgb.shape[0], 1))
                scaled_person_output = _scale_person_output(record["person_outputs"][track_id], scale_x, scale_y)

                mask_rgb, bg_rgb, target_mask = build_bg_and_mask_frame(
                    frame_rgb=resized_rgb,
                    indexed_mask=resized_mask,
                    track_id=track_id,
                    config=self.config,
                )
                probe_pose_meta = build_wan_pose_meta(
                    person_output=scaled_person_output,
                    track_id=track_id,
                    frame_stem=record["frame_stem"],
                    frame_size=resized_rgb.shape[:2],
                    face_landmarks=None,
                )
                detections = self.face_backend.detect(cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
                selected_face = select_target_face(
                    detections,
                    target_mask,
                    np.asarray(probe_pose_meta["keypoints_body"], dtype=np.float32),
                    previous_bbox=previous_bbox,
                )
                face_sequence.append(selected_face)
                previous_bbox = selected_face.bbox if selected_face is not None else previous_bbox
                frame_payloads.append(
                    {
                        "frame_stem": record["frame_stem"],
                        "frame_rgb": resized_rgb,
                        "mask_rgb": mask_rgb,
                        "bg_rgb": bg_rgb,
                        "target_mask": target_mask,
                        "person_output": scaled_person_output,
                        "raw_person_output": record["person_outputs"][track_id],
                    }
                )

            if len(frame_payloads) < int(self.config.min_track_frames):
                skipped_targets.append(
                    {
                        "track_id": int(track_id),
                        "reason": "readable_frames_below_threshold",
                        "readable_frame_count": len(frame_payloads),
                        "min_track_frames": int(self.config.min_track_frames),
                    }
                )
                continue

            face_sequence = fill_face_gaps(face_sequence, self.config.face_gap)
            target_frames = []
            pose_frames = []
            face_frames = []
            bg_frames = []
            mask_frames = []
            pose_meta_records = []
            valid_face_count = 0
            best_ref_frame = None
            best_ref_score = None

            for payload, face_detection in zip(frame_payloads, face_sequence):
                pose_meta = build_wan_pose_meta(
                    person_output=payload["person_output"],
                    track_id=track_id,
                    frame_stem=payload["frame_stem"],
                    frame_size=payload["frame_rgb"].shape[:2],
                    face_landmarks=None if face_detection is None else face_detection.landmarks,
                )
                pose_meta_records.append((payload["frame_stem"], pose_meta))

                target_frame = payload["frame_rgb"]
                target_frames.append(target_frame)
                pose_frames.append(render_wan_pose_frame(pose_meta, target_frame.shape))
                bg_frames.append(payload["bg_rgb"])
                mask_frames.append(payload["mask_rgb"])

                if face_detection is not None:
                    valid_face_count += 1
                    expanded_bbox = _expand_face_bbox(
                        face_detection.bbox,
                        frame_shape=target_frame.shape,
                        scale=self.config.face_expand,
                    )
                    face_frames.append(crop_face_frame(target_frame, expanded_bbox, self.config.face_resolution))
                else:
                    face_frames.append(
                        np.zeros(
                            (int(self.config.face_resolution[1]), int(self.config.face_resolution[0]), 3),
                            dtype=np.uint8,
                        )
                    )

                ref_score = score_reference_frame(
                    target_mask=payload["target_mask"],
                    face_detection=face_detection is not None,
                    pose_meta=pose_meta,
                )
                if best_ref_frame is None or best_ref_score is None or ref_score > best_ref_score:
                    best_ref_frame = target_frame
                    best_ref_score = ref_score

            valid_face_ratio = float(valid_face_count) / float(max(len(target_frames), 1))
            if valid_face_count <= 0:
                skipped_targets.append(
                    {
                        "track_id": int(track_id),
                        "reason": "no_valid_face_detected",
                        "valid_face_ratio": valid_face_ratio,
                        "frame_count": len(target_frames),
                    }
                )
                continue
            if valid_face_ratio < float(self.config.min_valid_face_ratio):
                skipped_targets.append(
                    {
                        "track_id": int(track_id),
                        "reason": "valid_face_ratio_below_threshold",
                        "valid_face_ratio": valid_face_ratio,
                        "min_valid_face_ratio": float(self.config.min_valid_face_ratio),
                        "frame_count": len(target_frames),
                    }
                )
                continue

            sample_dir = self._resolve_sample_dir(track_id)
            os.makedirs(sample_dir, exist_ok=True)
            sample_uuid = self._ensure_sample_uuid()
            if self.config.save_pose_meta_json:
                pose_meta_json_dir = os.path.join(sample_dir, "pose_meta_json")
                os.makedirs(pose_meta_json_dir, exist_ok=True)
                for frame_stem, pose_meta in pose_meta_records:
                    with open(os.path.join(pose_meta_json_dir, f"{frame_stem}.json"), "w", encoding="utf-8") as handle:
                        json.dump(pose_meta, handle, indent=2)

            _write_json(
                os.path.join(sample_dir, "pose_meta_sequence.json"),
                {
                    "sample_id": self.sample_id,
                    "sample_uuid": sample_uuid,
                    "track_id": int(track_id),
                    "source_path": self.source_reference,
                    "frame_count": len(pose_meta_records),
                    "records": [pose_meta for _, pose_meta in pose_meta_records],
                },
            )
            if bool(self.config.save_smpl_sequence_json):
                _write_json(
                    os.path.join(sample_dir, "smpl_sequence.json"),
                    {
                        "sample_id": self.sample_id,
                        "sample_uuid": sample_uuid,
                        "track_id": int(track_id),
                        "source_path": self.source_reference,
                        "frame_count": len(frame_payloads),
                        "records": [
                            {
                                "frame_stem": payload["frame_stem"],
                                "person_output": _to_json_safe(payload["raw_person_output"]),
                            }
                            for payload in frame_payloads
                        ],
                    },
                )

            self._write_mp4(target_frames, os.path.join(sample_dir, "target.mp4"), self.config.fps)
            self._write_mp4(pose_frames, os.path.join(sample_dir, "src_pose.mp4"), self.config.fps)
            self._write_mp4(face_frames, os.path.join(sample_dir, "src_face.mp4"), self.config.fps)
            self._write_mp4(bg_frames, os.path.join(sample_dir, "src_bg.mp4"), self.config.fps)
            self._write_mp4(mask_frames, os.path.join(sample_dir, "src_mask.mp4"), self.config.fps)
            if best_ref_frame is not None:
                Image.fromarray(best_ref_frame).save(os.path.join(sample_dir, "src_ref.png"))

            with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "sample_id": self.sample_id,
                        "sample_uuid": self.sample_uuid,
                        "clip_id": self.clip_id,
                        "track_id": int(track_id),
                        "source_path": self.source_reference,
                        "fps": int(self.config.fps),
                        "resolution_area": list(self.config.resolution_area),
                        "face_resolution": list(self.config.face_resolution),
                        "frame_count": len(target_frames),
                        "valid_face_ratio": valid_face_ratio,
                    },
                    handle,
                    indent=2,
                )
            written.append(sample_dir)
            written_targets.append(
                {
                    "track_id": int(track_id),
                    "sample_dir": sample_dir,
                    "frame_count": len(target_frames),
                    "valid_face_ratio": valid_face_ratio,
                }
            )
            self.finalized_targets.append(dict(written_targets[-1]))

        sample_uuid = self._ensure_sample_uuid()
        summary_identity = self._summary_identity()
        if self.metadata_root and summary_identity:
            update_wan_sample_summary(
                self.metadata_root,
                summary_identity,
                {
                    "sample_id": self.sample_id,
                    "sample_uuid": sample_uuid,
                    "clip_id": self.clip_id,
                    "source_path": self.source_reference,
                    "working_output_dir": self.working_output_dir,
                    "export_root": self.export_root,
                    "metadata_root": self.metadata_root,
                    "exported_target_count": len(written_targets),
                    "exported_targets": written_targets,
                    "skipped_target_count": len(skipped_targets),
                },
            )
            write_wan_skipped_report(
                self.metadata_root,
                summary_identity,
                {
                    "sample_id": self.sample_id,
                    "sample_uuid": sample_uuid,
                    "clip_id": self.clip_id,
                    "source_path": self.source_reference,
                    "skipped_targets": skipped_targets,
                },
            )
            if skipped_targets:
                append_wan_issue_records(
                    self.metadata_root,
                    [
                        {
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "event_type": "wan_target_skipped",
                            "status": "skipped",
                            "reason": str(skipped_target.get("reason", "unknown")),
                            "source_path": self.source_reference,
                            "sample_id": self.sample_id,
                            "sample_uuid": sample_uuid,
                            "clip_id": self.clip_id,
                            "working_output_dir": self.working_output_dir,
                            "runtime_profile": None,
                            "details": dict(skipped_target),
                        }
                        for skipped_target in skipped_targets
                    ],
                )

        return written
