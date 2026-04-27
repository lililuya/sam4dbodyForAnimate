from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_FACE_CENTER_DISTANCE = 0.30
MIN_FACE_CONTINUITY_IOU = 0.05


@dataclass(frozen=True)
class WanFaceDetection:
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float


class InsightFaceBackend:
    def __init__(self, provider: str = "buffalo_l", ctx_id: int = 0):
        self.provider = provider
        self.ctx_id = int(ctx_id)
        self._app = None

    def _ensure_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name=self.provider)
            self._app.prepare(ctx_id=self.ctx_id)
        return self._app

    def detect(self, frame_bgr: np.ndarray) -> list[WanFaceDetection]:
        detections = []
        for face in self._ensure_app().get(frame_bgr):
            detections.append(
                WanFaceDetection(
                    bbox=tuple(int(value) for value in face.bbox.tolist()),
                    landmarks=np.asarray(face.kps, dtype=np.float32),
                    score=float(getattr(face, "det_score", 1.0)),
                )
            )
        return detections


def _bbox_iou(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
    left = max(lhs[0], rhs[0])
    top = max(lhs[1], rhs[1])
    right = min(lhs[2], rhs[2])
    bottom = min(lhs[3], rhs[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    lhs_area = float(max(lhs[2] - lhs[0], 0) * max(lhs[3] - lhs[1], 0))
    rhs_area = float(max(rhs[2] - rhs[0], 0) * max(rhs[3] - rhs[1], 0))
    union = lhs_area + rhs_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _mask_overlap(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), mask.shape[1])
    y2 = min(int(y2), mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = mask[y1:y2, x1:x2]
    total_mask_area = float((mask > 0).sum())
    if total_mask_area <= 0:
        return 0.0
    return float((crop > 0).sum()) / total_mask_area


def _resolve_head_anchor(target_mask: np.ndarray, body_keypoints: np.ndarray) -> np.ndarray:
    if body_keypoints.ndim == 2 and len(body_keypoints) > 0 and body_keypoints.shape[1] >= 2:
        head_anchor = body_keypoints[0, :2].astype(np.float32).copy()
        if head_anchor.max(initial=0.0) <= 1.0:
            head_anchor[0] *= float(target_mask.shape[1])
            head_anchor[1] *= float(target_mask.shape[0])
        return head_anchor

    ys, xs = np.nonzero(target_mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
    return np.array([target_mask.shape[1] / 2.0, target_mask.shape[0] / 2.0], dtype=np.float32)


def _is_plausible_target_face(*, overlap_score: float, continuity_score: float, distance_score: float) -> bool:
    if overlap_score > 0.0:
        return True
    if continuity_score >= MIN_FACE_CONTINUITY_IOU:
        return True
    return distance_score <= MAX_FACE_CENTER_DISTANCE


def select_target_face(detections, target_mask: np.ndarray, body_keypoints: np.ndarray, previous_bbox=None):
    if not detections:
        return None

    body_keypoints = np.asarray(body_keypoints, dtype=np.float32)
    head_anchor = _resolve_head_anchor(target_mask, body_keypoints)

    best_detection = None
    best_score = None
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        overlap_score = _mask_overlap(target_mask, detection.bbox)
        continuity_score = 0.0 if previous_bbox is None else _bbox_iou(detection.bbox, previous_bbox)
        distance_score = float(np.linalg.norm(center - head_anchor)) / float(max(target_mask.shape))
        if not _is_plausible_target_face(
            overlap_score=overlap_score,
            continuity_score=continuity_score,
            distance_score=distance_score,
        ):
            continue
        score = overlap_score * 5.0 + continuity_score * 2.0 + float(detection.score) * 0.1 - distance_score
        if best_detection is None or score > best_score:
            best_detection = detection
            best_score = score
    return best_detection


def fill_face_gaps(sequence, max_gap: int):
    filled = list(sequence)
    gap_start = None
    for index in range(len(filled)):
        current = filled[index]
        if current is None:
            if gap_start is None:
                gap_start = index
            continue
        if gap_start is None:
            continue

        previous = filled[gap_start - 1] if gap_start > 0 else None
        gap_length = index - gap_start
        if previous is not None and gap_length <= int(max_gap):
            for fill_index in range(gap_start, index):
                filled[fill_index] = previous
        gap_start = None

    if gap_start is not None:
        previous = filled[gap_start - 1] if gap_start > 0 else None
        gap_length = len(filled) - gap_start
        if previous is not None and gap_length <= int(max_gap):
            for fill_index in range(gap_start, len(filled)):
                filled[fill_index] = previous
    return filled


def crop_face_frame(frame_rgb: np.ndarray, bbox: tuple[int, int, int, int], output_size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), frame_rgb.shape[1])
    y2 = min(int(y2), frame_rgb.shape[0])

    if x2 <= x1 or y2 <= y1:
        return np.zeros((int(output_size[1]), int(output_size[0]), 3), dtype=np.uint8)

    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, tuple(int(value) for value in output_size), interpolation=cv2.INTER_LINEAR)
