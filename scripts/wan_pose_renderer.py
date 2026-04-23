from __future__ import annotations

import cv2
import numpy as np


BODY_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (13, 18),
    (10, 19),
]

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pixels = points.copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    return pixels


def _draw_edges(canvas: np.ndarray, points: np.ndarray, edges, color: tuple[int, int, int], thickness: int) -> None:
    for start_idx, end_idx in edges:
        if points[start_idx, 2] <= 0 or points[end_idx, 2] <= 0:
            continue
        start_point = tuple(int(value) for value in points[start_idx, :2])
        end_point = tuple(int(value) for value in points[end_idx, :2])
        cv2.line(canvas, start_point, end_point, color, thickness)


def _draw_points(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    for x_coord, y_coord, confidence in points:
        if confidence <= 0:
            continue
        cv2.circle(canvas, (int(x_coord), int(y_coord)), radius, color, -1)


def draw_pose_frame(canvas: np.ndarray, pose_meta: dict) -> np.ndarray:
    output = np.asarray(canvas, dtype=np.uint8).copy()
    height, width = output.shape[:2]

    body = _to_pixels(np.asarray(pose_meta["keypoints_body"], dtype=np.float32), width, height)
    left_hand = _to_pixels(np.asarray(pose_meta["keypoints_left_hand"], dtype=np.float32), width, height)
    right_hand = _to_pixels(np.asarray(pose_meta["keypoints_right_hand"], dtype=np.float32), width, height)

    _draw_edges(output, body, BODY_EDGES, (255, 255, 255), 2)
    _draw_edges(output, left_hand, HAND_EDGES, (0, 255, 255), 1)
    _draw_edges(output, right_hand, HAND_EDGES, (0, 255, 255), 1)
    _draw_points(output, body, (0, 200, 255), 3)
    _draw_points(output, left_hand, (255, 128, 0), 2)
    _draw_points(output, right_hand, (255, 128, 0), 2)
    return output
