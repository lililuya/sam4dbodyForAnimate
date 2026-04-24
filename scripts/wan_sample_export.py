from __future__ import annotations

import json
import math
import os
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from scripts.wan_face_export import InsightFaceBackend, crop_face_frame, fill_face_gaps, select_target_face
from scripts.wan_mask_bg_export import build_bg_and_mask_frame, score_reference_frame
from scripts.wan_pose_adapter import build_wan_pose_meta, render_wan_pose_frame
from scripts.wan_reference_compat import compute_sample_indices, resize_frame_by_area
from scripts.wan_sample_types import WanExportConfig


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
    ):
        self.sample_id = str(sample_id)
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.source_video_path = source_video_path
        self.config = _coerce_config(config)
        self.face_backend = face_backend or InsightFaceBackend()
        self._records: list[dict] = []

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
        for track_id in track_ids:
            track_records = [record for record in sampled_records if track_id in record["person_outputs"]]
            if len(track_records) < int(self.config.min_track_frames):
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
                    }
                )

            if len(frame_payloads) < int(self.config.min_track_frames):
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
            if valid_face_ratio < float(self.config.min_valid_face_ratio):
                continue

            sample_dir = os.path.join(self.output_dir, "wan_export", f"{self.sample_id}_track_{track_id}")
            os.makedirs(sample_dir, exist_ok=True)
            if self.config.save_pose_meta_json:
                pose_meta_json_dir = os.path.join(sample_dir, "pose_meta_json")
                os.makedirs(pose_meta_json_dir, exist_ok=True)
                for frame_stem, pose_meta in pose_meta_records:
                    with open(os.path.join(pose_meta_json_dir, f"{frame_stem}.json"), "w", encoding="utf-8") as handle:
                        json.dump(pose_meta, handle, indent=2)

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
                        "track_id": int(track_id),
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

        return written
