from __future__ import annotations

import math

import cv2
import numpy as np


def compute_sample_indices(num_frames: int, source_fps: float, target_fps: float) -> list[int]:
    if num_frames <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0 or target_fps >= source_fps:
        return list(range(int(num_frames)))

    step = float(source_fps) / float(target_fps)
    indices = np.arange(0.0, float(num_frames), step, dtype=np.float32)
    sampled = np.rint(indices).astype(np.int32)
    sampled = np.clip(sampled, 0, int(num_frames) - 1)
    return sampled.tolist()


def resize_frame_by_area(frame: np.ndarray, target_area: int, align_divisor: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    current_area = max(height * width, 1)
    scale = math.sqrt(float(target_area) / float(current_area))
    scaled_width = max(int(width * scale), 1)
    scaled_height = max(int(height * scale), 1)

    aligned_width = max(align_divisor, int(round(scaled_width / align_divisor)) * align_divisor)
    aligned_height = max(align_divisor, int(round(scaled_height / align_divisor)) * align_divisor)

    interpolation = cv2.INTER_AREA if aligned_width * aligned_height <= current_area else cv2.INTER_LINEAR
    return cv2.resize(frame, (aligned_width, aligned_height), interpolation=interpolation)


def dilate_target_mask(mask: np.ndarray, kernel_size: int = 7, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return (dilated > 0).astype(np.uint8)


def expand_target_mask(
    mask: np.ndarray,
    dilation_iters: int = 1,
    expansion_iters: int = 1,
    *,
    w_len: int | None = None,
    h_len: int | None = None,
) -> np.ndarray:
    expanded = dilate_target_mask(mask, kernel_size=3, iterations=max(int(dilation_iters), 0))

    if w_len is not None and h_len is not None:
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

    if int(expansion_iters) > 0:
        expanded = dilate_target_mask(expanded, kernel_size=3, iterations=int(expansion_iters))
    return expanded.astype(np.uint8)
