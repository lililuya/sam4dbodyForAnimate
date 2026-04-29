from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Iterable

import cv2
import numpy as np

from scripts.wan_face_export import WanFaceDetection, _bbox_iou
from scripts.wan_reference_compat import compute_sample_indices
from scripts.wan_sample_export import resolve_or_create_wan_sample_uuid


@dataclass(frozen=True)
class FaceClipRecord:
    frame_index_in_source: int
    bbox_xyxy: tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float


@dataclass(frozen=True)
class FaceTrackSegment:
    face_track_index: int
    segment_index: int
    start_frame: int
    end_frame: int
    records: list[FaceClipRecord]


@dataclass(frozen=True)
class FaceClipExtractionResult:
    kept_segments: list[FaceTrackSegment]
    dropped_segments: list[dict]


@dataclass
class _LiveTrack:
    face_track_index: int
    records: list[FaceClipRecord]


def _read_json_dict(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str, payload) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_mp4(path: str, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("cannot write mp4 with no frames")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    if hasattr(writer, "isOpened") and not writer.isOpened():
        raise RuntimeError(f"failed to open clip writer: {path}")
    try:
        for frame in frames:
            writer.write(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.release()


def build_face_clip_id(sample_uuid: str, *, face_track_index: int, segment_index: int) -> str:
    return f"{str(sample_uuid)}_face{int(face_track_index):02d}_seg{int(segment_index):03d}"


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    return float(max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1])))


def _is_plausible_successor(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    *,
    min_iou: float = 0.10,
    max_center_distance_ratio: float = 0.35,
    max_area_ratio: float = 1.8,
) -> bool:
    iou_value = _bbox_iou(previous_bbox, current_bbox)
    if iou_value >= float(min_iou):
        return True

    prev_center = _bbox_center(previous_bbox)
    curr_center = _bbox_center(current_bbox)
    prev_width = max(1.0, float(previous_bbox[2] - previous_bbox[0]))
    prev_height = max(1.0, float(previous_bbox[3] - previous_bbox[1]))
    norm = max(prev_width, prev_height)
    center_distance_ratio = float(np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])) / norm
    if center_distance_ratio > float(max_center_distance_ratio):
        return False

    prev_area = max(1.0, _bbox_area(previous_bbox))
    curr_area = max(1.0, _bbox_area(current_bbox))
    area_ratio = max(curr_area / prev_area, prev_area / curr_area)
    return area_ratio <= float(max_area_ratio)


def _append_drop_event(dropped_segments: list[dict], *, reason: str, frame_index: int, face_track_index: int) -> None:
    dropped_segments.append(
        {
            "reason": str(reason),
            "frame_index": int(frame_index),
            "face_track_index": int(face_track_index),
        }
    )


def _finalize_track(track: _LiveTrack) -> FaceTrackSegment:
    return FaceTrackSegment(
        face_track_index=int(track.face_track_index),
        segment_index=1,
        start_frame=int(track.records[0].frame_index_in_source),
        end_frame=int(track.records[-1].frame_index_in_source),
        records=list(track.records),
    )


def _record_from_detection(frame_index: int, detection: WanFaceDetection) -> FaceClipRecord:
    return FaceClipRecord(
        frame_index_in_source=int(frame_index),
        bbox_xyxy=tuple(int(value) for value in detection.bbox),
        landmarks=np.asarray(detection.landmarks, dtype=np.float32).copy(),
        score=float(detection.score),
    )


def _filter_segments_by_duration(
    segments: Iterable[FaceTrackSegment],
    *,
    fps: float,
    min_clip_seconds: float,
    dropped_segments: list[dict],
) -> FaceClipExtractionResult:
    kept_segments: list[FaceTrackSegment] = []
    safe_fps = max(float(fps), 1e-6)
    for segment in segments:
        duration_seconds = float(len(segment.records)) / safe_fps
        if duration_seconds >= float(min_clip_seconds):
            kept_segments.append(segment)
            continue
        dropped_segments.append(
            {
                "reason": "shorter_than_min_duration",
                "face_track_index": int(segment.face_track_index),
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                "frame_count": len(segment.records),
                "duration_seconds": duration_seconds,
            }
        )
    return FaceClipExtractionResult(kept_segments=kept_segments, dropped_segments=dropped_segments)


def extract_face_track_segments(
    frame_detections: list[list[WanFaceDetection]],
    *,
    fps: float,
    min_clip_seconds: float,
) -> FaceClipExtractionResult:
    live_tracks: list[_LiveTrack] = []
    completed_segments: list[FaceTrackSegment] = []
    dropped_segments: list[dict] = []
    next_track_index = 1

    for frame_index, detections in enumerate(frame_detections):
        if not live_tracks:
            for detection in detections:
                live_tracks.append(_LiveTrack(next_track_index, [_record_from_detection(frame_index, detection)]))
                next_track_index += 1
            continue

        candidate_map: dict[int, list[int]] = {}
        detection_claims: dict[int, list[int]] = {}
        for track in live_tracks:
            previous_bbox = track.records[-1].bbox_xyxy
            candidate_indices: list[int] = []
            for detection_index, detection in enumerate(detections):
                if _is_plausible_successor(previous_bbox, detection.bbox):
                    candidate_indices.append(detection_index)
            candidate_map[track.face_track_index] = candidate_indices
            for candidate_index in candidate_indices:
                detection_claims.setdefault(candidate_index, []).append(track.face_track_index)

        finalizing_track_ids: set[int] = set()
        blocked_detection_indices: set[int] = set()
        continuing_assignments: dict[int, int] = {}

        for track in live_tracks:
            candidate_indices = candidate_map.get(track.face_track_index, [])
            if len(candidate_indices) == 0:
                finalizing_track_ids.add(track.face_track_index)
                continue
            if len(candidate_indices) > 1:
                finalizing_track_ids.add(track.face_track_index)
                blocked_detection_indices.update(candidate_indices)
                _append_drop_event(
                    dropped_segments,
                    reason="ambiguous_face_assignment",
                    frame_index=frame_index,
                    face_track_index=track.face_track_index,
                )
                continue
            continuing_assignments[track.face_track_index] = candidate_indices[0]

        for detection_index, claiming_track_ids in detection_claims.items():
            if len(claiming_track_ids) <= 1:
                continue
            blocked_detection_indices.add(detection_index)
            for face_track_index in claiming_track_ids:
                finalizing_track_ids.add(face_track_index)
                if continuing_assignments.get(face_track_index) == detection_index:
                    continuing_assignments.pop(face_track_index, None)
                _append_drop_event(
                    dropped_segments,
                    reason="track_conflict",
                    frame_index=frame_index,
                    face_track_index=face_track_index,
                )

        new_live_tracks: list[_LiveTrack] = []
        used_detection_indices: set[int] = set()
        for track in live_tracks:
            if track.face_track_index in finalizing_track_ids:
                completed_segments.append(_finalize_track(track))
                continue

            detection_index = continuing_assignments.get(track.face_track_index)
            if detection_index is None or detection_index in blocked_detection_indices:
                completed_segments.append(_finalize_track(track))
                continue

            track.records.append(_record_from_detection(frame_index, detections[detection_index]))
            used_detection_indices.add(detection_index)
            new_live_tracks.append(track)

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detection_indices or detection_index in blocked_detection_indices:
                continue
            new_live_tracks.append(_LiveTrack(next_track_index, [_record_from_detection(frame_index, detection)]))
            next_track_index += 1

        live_tracks = new_live_tracks

    for track in live_tracks:
        completed_segments.append(_finalize_track(track))

    return _filter_segments_by_duration(
        completed_segments,
        fps=float(fps),
        min_clip_seconds=float(min_clip_seconds),
        dropped_segments=dropped_segments,
    )


def _serialize_record(record: FaceClipRecord, *, clip_frame_index: int, fps: float) -> dict:
    return {
        "frame_index_in_source": int(record.frame_index_in_source),
        "frame_index_in_clip": int(clip_frame_index),
        "timestamp_seconds": float(clip_frame_index) / float(max(fps, 1e-6)),
        "bbox_xyxy": [int(value) for value in record.bbox_xyxy],
        "landmarks": np.asarray(record.landmarks, dtype=np.float32).tolist(),
        "score": float(record.score),
    }


def _resample_segment_records(
    records: list[FaceClipRecord],
    *,
    source_fps: float,
    target_fps: float | None,
) -> list[FaceClipRecord]:
    if not records:
        return []
    if target_fps is None or float(target_fps) <= 0 or abs(float(target_fps) - float(source_fps)) < 1e-6:
        return list(records)
    sampled_indices = compute_sample_indices(
        num_frames=len(records),
        source_fps=float(source_fps),
        target_fps=float(target_fps),
    )
    if not sampled_indices:
        return [records[0]]
    return [records[int(index)] for index in sampled_indices]


def _resample_segment_frames(
    frames_bgr: list[np.ndarray],
    *,
    source_fps: float,
    target_fps: float | None,
) -> list[np.ndarray]:
    if not frames_bgr:
        return []
    if target_fps is None or float(target_fps) <= 0 or abs(float(target_fps) - float(source_fps)) < 1e-6:
        return list(frames_bgr)
    sampled_indices = compute_sample_indices(
        num_frames=len(frames_bgr),
        source_fps=float(source_fps),
        target_fps=float(target_fps),
    )
    if not sampled_indices:
        return [frames_bgr[0]]
    return [frames_bgr[int(index)] for index in sampled_indices]


def update_face_clip_batch_summary(output_root: str, *, input_video: str, item: dict) -> None:
    summary_path = os.path.join(output_root, "batch_summary.json")
    current = _read_json_dict(summary_path)
    items = [existing for existing in list(current.get("items") or []) if isinstance(existing, dict)]
    source_path = os.path.abspath(str(item.get("source_path") or input_video))
    sample_uuid = str(item.get("sample_uuid") or "").strip()

    updated_items: list[dict] = []
    replaced = False
    for existing in items:
        existing_source_path = os.path.abspath(str(existing.get("source_path") or ""))
        existing_sample_uuid = str(existing.get("sample_uuid") or "").strip()
        if existing_source_path == source_path or (sample_uuid and existing_sample_uuid == sample_uuid):
            updated_items.append(dict(item))
            replaced = True
            continue
        updated_items.append(existing)

    if not replaced:
        updated_items.append(dict(item))

    source_dirs = [
        os.path.dirname(os.path.abspath(str(existing.get("source_path") or "")))
        for existing in updated_items
        if str(existing.get("source_path") or "").strip()
    ]
    if source_dirs:
        try:
            input_dir = os.path.commonpath(source_dirs)
        except ValueError:
            input_dir = source_dirs[0]
    else:
        input_dir = os.path.dirname(os.path.abspath(input_video))

    video_count_failed = sum(
        1 for existing in updated_items if str(existing.get("status") or "").strip() == "failed"
    )
    video_count_completed = sum(
        1
        for existing in updated_items
        if str(existing.get("status") or "").strip() in {"completed", "completed_no_clips"}
    )
    payload = {
        "input_dir": input_dir,
        "output_dir": os.path.abspath(output_root),
        "video_count_total": len(updated_items),
        "video_count_completed": video_count_completed,
        "video_count_failed": video_count_failed,
        "clip_count_kept": sum(int(existing.get("kept_clip_count", 0) or 0) for existing in updated_items),
        "clip_count_dropped": sum(int(existing.get("dropped_segment_count", 0) or 0) for existing in updated_items),
        "items": updated_items,
    }
    _write_json(summary_path, payload)


def extract_face_clips_from_video(
    input_video: str,
    output_root: str,
    *,
    min_clip_seconds: float,
    face_backend,
    sample_id: str | None = None,
    target_fps: float | None = None,
) -> list[str]:
    input_video = os.path.abspath(str(input_video))
    output_root = os.path.abspath(str(output_root))
    sample_id = str(sample_id or os.path.splitext(os.path.basename(input_video))[0])

    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    frames_bgr: list[np.ndarray] = []
    frame_detections: list[list[WanFaceDetection]] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames_bgr.append(np.asarray(frame_bgr, dtype=np.uint8).copy())
            frame_detections.append(list(face_backend.detect(frame_bgr)))
    finally:
        capture.release()

    sample_uuid = resolve_or_create_wan_sample_uuid(output_root, input_video, sample_id=sample_id, working_output_dir=None)
    extraction = extract_face_track_segments(
        frame_detections,
        fps=float(fps),
        min_clip_seconds=float(min_clip_seconds),
    )

    clip_dirs: list[str] = []
    for segment in extraction.kept_segments:
        clip_id = build_face_clip_id(
            sample_uuid,
            face_track_index=int(segment.face_track_index),
            segment_index=int(segment.segment_index),
        )
        clip_dir = os.path.join(output_root, "clips", clip_id)
        source_segment_frames = frames_bgr[int(segment.start_frame) : int(segment.end_frame) + 1]
        clip_records = _resample_segment_records(
            segment.records,
            source_fps=float(fps),
            target_fps=target_fps,
        )
        clip_frames = _resample_segment_frames(
            source_segment_frames,
            source_fps=float(fps),
            target_fps=target_fps,
        )
        clip_fps = float(target_fps) if target_fps is not None and float(target_fps) > 0 else float(fps)
        track_records = [
            _serialize_record(record, clip_frame_index=idx, fps=clip_fps)
            for idx, record in enumerate(clip_records)
        ]
        source_start_frame = int(segment.records[0].frame_index_in_source)
        source_end_frame = int(segment.records[-1].frame_index_in_source)
        clip_start_frame = int(clip_records[0].frame_index_in_source) if clip_records else source_start_frame
        clip_end_frame = int(clip_records[-1].frame_index_in_source) if clip_records else source_end_frame
        meta = {
            "clip_id": clip_id,
            "sample_uuid": sample_uuid,
            "source_path": input_video,
            "source_start_frame": source_start_frame,
            "source_end_frame": source_end_frame,
            "source_fps": float(fps),
            "source_frame_count": len(segment.records),
            "start_frame": clip_start_frame,
            "end_frame": clip_end_frame,
            "fps": clip_fps,
            "frame_count": len(clip_records),
            "duration_seconds": float(len(clip_records)) / float(max(clip_fps, 1e-6)),
            "clip_path": os.path.join(clip_dir, "clip.mp4"),
            "track_json_path": os.path.join(clip_dir, "track.json"),
        }

        _write_mp4(os.path.join(clip_dir, "clip.mp4"), clip_frames, fps=clip_fps)
        _write_json(
            os.path.join(clip_dir, "track.json"),
            {
                "clip_id": clip_id,
                "sample_uuid": sample_uuid,
                "source_path": input_video,
                "source_fps": float(fps),
                "source_frame_count": len(segment.records),
                "fps": clip_fps,
                "records": track_records,
            },
        )
        _write_json(os.path.join(clip_dir, "meta.json"), meta)
        clip_dirs.append(clip_dir)

    update_face_clip_batch_summary(
        output_root,
        input_video=input_video,
        item={
            "sample_uuid": sample_uuid,
            "source_path": input_video,
            "status": "completed" if clip_dirs else "completed_no_clips",
            "kept_clip_count": len(extraction.kept_segments),
            "clip_ids": [os.path.basename(os.path.abspath(path)) for path in clip_dirs],
            "dropped_segment_count": len(extraction.dropped_segments),
            "drop_reasons": [str(item.get("reason", "unknown")) for item in extraction.dropped_segments],
            "dropped_segments": list(extraction.dropped_segments),
        },
    )
    return clip_dirs
