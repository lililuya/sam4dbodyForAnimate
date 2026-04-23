from __future__ import annotations

import math

import cv2
import numpy as np


def compute_sample_indices(num_frames: int, source_fps: float, target_fps: float) -> list[int]:
    if num_frames <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        return list(range(int(num_frames)))

    # Match WanAnimate's current target_num/timestamp rounding behavior.
    target_num = int(float(num_frames) / float(source_fps) * float(target_fps))
    if target_num <= 0:
        return [0]

    times = np.arange(0, target_num, dtype=np.float32) / float(target_fps)
    indices = np.round(times * float(source_fps)).astype(np.int32)
    indices = np.clip(indices, 0, int(num_frames) - 1)
    return indices.tolist()


def _resolve_target_area(resolution_area: int | tuple[int, int] | list[int]) -> int:
    if isinstance(resolution_area, (tuple, list)):
        if len(resolution_area) != 2:
            raise ValueError("resolution_area must contain exactly two values")
        return int(resolution_area[0]) * int(resolution_area[1])
    return int(resolution_area)


def resize_frame_by_area(
    frame: np.ndarray,
    resolution_area: int | tuple[int, int] | list[int],
    align_divisor: int = 16,
) -> np.ndarray:
    height, width = frame.shape[:2]
    target_area = _resolve_target_area(resolution_area)
    aspect_ratio = float(width) / float(max(height, 1))
    aligned_height_float = math.sqrt(float(target_area) / max(aspect_ratio, 1e-6))
    aligned_width_float = float(target_area) / max(aligned_height_float, 1e-6)
    aligned_height = max(
        int(align_divisor),
        int((aligned_height_float // float(align_divisor)) * int(align_divisor)),
    )
    aligned_width = max(
        int(align_divisor),
        int((aligned_width_float // float(align_divisor)) * int(align_divisor)),
    )
    interpolation = cv2.INTER_AREA if aligned_width * aligned_height <= height * width else cv2.INTER_LINEAR
    return cv2.resize(frame, (aligned_width, aligned_height), interpolation=interpolation)


def dilate_target_mask(mask: np.ndarray, kernel_size: int = 7, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return (dilated > 0).astype(np.uint8)


def expand_target_mask(
    mask: np.ndarray,
    *,
    w_len: int = 10,
    h_len: int = 20,
) -> np.ndarray:
    expanded = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8).copy()
    ys, xs = np.nonzero(expanded)
    if len(xs) == 0 or len(ys) == 0:
        return expanded

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w_slice = max(1, int((x_max - x_min + 1) / max(int(w_len), 1)))
    h_slice = max(1, int((y_max - y_min + 1) / max(int(h_len), 1)))

    for x_start in range(x_min, x_max + 1, w_slice):
        x_end = min(x_start + w_slice, x_max + 1)
        for y_start in range(y_min, y_max + 1, h_slice):
            y_end = min(y_start + h_slice, y_max + 1)
            if expanded[y_start:y_end, x_start:x_end].sum() > 0:
                expanded[y_start:y_end, x_start:x_end] = 1
    return expanded.astype(np.uint8)
