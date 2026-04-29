from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np


BODY_LIMB_SEQ = [
    (2, 3),
    (2, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (2, 12),
    (12, 13),
    (13, 14),
    (2, 1),
    (1, 15),
    (15, 17),
    (1, 16),
    (16, 18),
    (14, 19),
    (11, 20),
]

BODY_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
    (200, 200, 0),
    (100, 100, 0),
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
    pixels = np.asarray(points, dtype=np.float32).copy()
    if pixels.size == 0:
        return pixels.reshape(0, 3)
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    return pixels


def _resolve_body_stickwidth(height: int, width: int, stickwidth_type: str = "v2") -> int:
    min_side = min(int(height), int(width))
    if stickwidth_type == "v1":
        return max(int(min_side / 200), 1)
    if stickwidth_type == "v2":
        return max(int(min_side / 200) - 1, 1)
    raise ValueError(f"unsupported stickwidth_type: {stickwidth_type}")


def _resolve_hand_stickwidth(height: int, width: int, stickwidth_type: str = "v2") -> int:
    min_side = min(int(height), int(width))
    if stickwidth_type == "v1":
        return max(int(min_side / 200), 1)
    if stickwidth_type == "v2":
        return max(max(int(min_side / 200) - 1, 1) // 2, 1)
    raise ValueError(f"unsupported stickwidth_type: {stickwidth_type}")


def _draw_body_pose(
    canvas: np.ndarray,
    body_points: np.ndarray,
    *,
    threshold: float = 0.5,
    stickwidth_type: str = "v2",
    draw_head: bool = True,
) -> np.ndarray:
    output = np.asarray(canvas, dtype=np.uint8).copy()
    body_points = np.asarray(body_points, dtype=np.float32).copy()
    if body_points.size == 0:
        return output

    if not draw_head and len(body_points) >= 18:
        body_points[[0, 14, 15, 16, 17], 2] = 0.0

    height, width = output.shape[:2]
    stickwidth = _resolve_body_stickwidth(height, width, stickwidth_type=stickwidth_type)

    for (start_index, end_index), color in zip(BODY_LIMB_SEQ, BODY_COLORS):
        keypoint1 = body_points[start_index - 1]
        keypoint2 = body_points[end_index - 1]
        if keypoint1[2] < float(threshold) or keypoint2[2] < float(threshold):
            continue

        y_coords = np.array([keypoint1[0], keypoint2[0]], dtype=np.float32)
        x_coords = np.array([keypoint1[1], keypoint2[1]], dtype=np.float32)
        center_x = float(np.mean(x_coords))
        center_y = float(np.mean(y_coords))
        length = float(np.hypot(x_coords[0] - x_coords[1], y_coords[0] - y_coords[1]))
        angle = math.degrees(math.atan2(x_coords[0] - x_coords[1], y_coords[0] - y_coords[1]))
        polygon = cv2.ellipse2Poly(
            (int(center_y), int(center_x)),
            (int(length / 2.0), int(stickwidth)),
            int(angle),
            0,
            360,
            1,
        )
        cv2.fillConvexPoly(output, polygon, [int(float(channel) * 0.6) for channel in color])

    for keypoint, color in zip(body_points, BODY_COLORS):
        if keypoint[2] < float(threshold):
            continue
        cv2.circle(
            output,
            (int(keypoint[0]), int(keypoint[1])),
            int(stickwidth),
            color,
            thickness=-1,
        )
    return output


def _draw_hand_pose(
    canvas: np.ndarray,
    hand_points: np.ndarray,
    *,
    threshold: float = 0.5,
    stickwidth_type: str = "v2",
) -> np.ndarray:
    output = np.asarray(canvas, dtype=np.uint8).copy()
    hand_points = np.asarray(hand_points, dtype=np.float32)
    if hand_points.size == 0:
        return output

    eps = 0.01
    height, width = output.shape[:2]
    stickwidth = _resolve_hand_stickwidth(height, width, stickwidth_type=stickwidth_type)

    for edge_index, (start_index, end_index) in enumerate(HAND_EDGES):
        keypoint1 = hand_points[start_index]
        keypoint2 = hand_points[end_index]
        if keypoint1[2] < float(threshold) or keypoint2[2] < float(threshold):
            continue

        x1 = int(keypoint1[0])
        y1 = int(keypoint1[1])
        x2 = int(keypoint2[0])
        y2 = int(keypoint2[1])
        if x1 <= eps or y1 <= eps or x2 <= eps or y2 <= eps:
            continue

        line_color = tuple(
            int(float(channel) * 255.0)
            for channel in colorsys.hsv_to_rgb(float(edge_index) / float(len(HAND_EDGES)), 1.0, 1.0)
        )
        cv2.line(output, (x1, y1), (x2, y2), line_color, thickness=int(stickwidth))

    for keypoint in hand_points:
        if keypoint[2] < float(threshold):
            continue
        x_coord = int(keypoint[0])
        y_coord = int(keypoint[1])
        if x_coord <= eps or y_coord <= eps:
            continue
        cv2.circle(output, (x_coord, y_coord), int(stickwidth), (0, 0, 255), thickness=-1)
    return output


def draw_pose_frame(
    canvas: np.ndarray,
    pose_meta: dict,
    *,
    threshold: float = 0.5,
    stickwidth_type: str = "v2",
    draw_hand: bool = True,
    draw_head: bool = True,
) -> np.ndarray:
    output = np.asarray(canvas, dtype=np.uint8).copy()
    height, width = output.shape[:2]

    body = _to_pixels(np.asarray(pose_meta.get("keypoints_body", []), dtype=np.float32), width, height)
    left_hand = _to_pixels(np.asarray(pose_meta.get("keypoints_left_hand", []), dtype=np.float32), width, height)
    right_hand = _to_pixels(np.asarray(pose_meta.get("keypoints_right_hand", []), dtype=np.float32), width, height)

    output = _draw_body_pose(
        output,
        body,
        threshold=float(threshold),
        stickwidth_type=str(stickwidth_type),
        draw_head=bool(draw_head),
    )
    if draw_hand:
        output = _draw_hand_pose(
            output,
            left_hand,
            threshold=float(threshold),
            stickwidth_type=str(stickwidth_type),
        )
        output = _draw_hand_pose(
            output,
            right_hand,
            threshold=float(threshold),
            stickwidth_type=str(stickwidth_type),
        )
    return output
