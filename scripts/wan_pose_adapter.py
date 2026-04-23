from __future__ import annotations

import numpy as np

from scripts.openpose_export import convert_mhr70_to_openpose_arrays
from scripts.pose_json_export import LEFT_HAND_MHR70, RIGHT_HAND_MHR70
from scripts.wan_pose_renderer import draw_pose_frame


WAN_BODY_FROM_OPENPOSE25 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22]


def _normalize_xyc(points: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = np.zeros((points.shape[0], 3), dtype=np.float32)
    normalized[:, 0] = points[:, 0] / float(max(width, 1))
    normalized[:, 1] = points[:, 1] / float(max(height, 1))
    normalized[:, 2] = points[:, 2]
    return normalized


def _extract_hand(keypoints_2d, indices, width: int, height: int) -> np.ndarray:
    hand = np.zeros((len(indices), 3), dtype=np.float32)
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    for output_index, source_index in enumerate(indices):
        if source_index >= len(keypoints_2d):
            continue
        hand[output_index, 0] = keypoints_2d[source_index, 0] / float(max(width, 1))
        hand[output_index, 1] = keypoints_2d[source_index, 1] / float(max(height, 1))
        hand[output_index, 2] = 1.0
    return hand


def _normalize_face_landmarks(face_landmarks, width: int, height: int) -> np.ndarray:
    if face_landmarks is None:
        return np.zeros((0, 3), dtype=np.float32)

    landmarks = np.asarray(face_landmarks, dtype=np.float32)
    if landmarks.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if landmarks.ndim != 2 or landmarks.shape[1] not in {2, 3}:
        raise ValueError(f"face_landmarks must have shape [N, 2] or [N, 3], got {landmarks.shape}")

    if landmarks.shape[1] == 2:
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1), dtype=np.float32)], axis=1)

    return _normalize_xyc(landmarks, width, height)


def build_wan_pose_meta(*, person_output, track_id: int, frame_stem: str, frame_size, face_landmarks=None) -> dict:
    height, width = int(frame_size[0]), int(frame_size[1])
    openpose_2d, _ = convert_mhr70_to_openpose_arrays(
        person_output["pred_keypoints_2d"],
        person_output["pred_keypoints_3d"],
    )
    openpose_2d = np.asarray(openpose_2d, dtype=np.float32).reshape(-1, 3)

    body = _normalize_xyc(openpose_2d[WAN_BODY_FROM_OPENPOSE25], width, height)
    left_hand = _extract_hand(person_output["pred_keypoints_2d"], LEFT_HAND_MHR70, width, height)
    right_hand = _extract_hand(person_output["pred_keypoints_2d"], RIGHT_HAND_MHR70, width, height)
    face = _normalize_face_landmarks(face_landmarks, width, height)

    return {
        "image_id": f"{frame_stem}.jpg",
        "track_id": int(track_id),
        "width": width,
        "height": height,
        "category_id": 1,
        "keypoints_body": body.tolist(),
        "keypoints_left_hand": left_hand.tolist(),
        "keypoints_right_hand": right_hand.tolist(),
        "keypoints_face": face.tolist(),
    }


def render_wan_pose_frame(pose_meta: dict, canvas_shape) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    return draw_pose_frame(canvas, pose_meta)
