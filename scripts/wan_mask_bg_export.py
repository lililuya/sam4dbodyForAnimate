from __future__ import annotations

import numpy as np

from scripts.wan_reference_compat import dilate_target_mask, expand_target_mask


def build_target_binary_mask(indexed_mask: np.ndarray, track_id: int) -> np.ndarray:
    indexed_mask = np.asarray(indexed_mask, dtype=np.uint8)
    return (indexed_mask == int(track_id)).astype(np.uint8)


def build_bg_and_mask_frame(
    *,
    frame_rgb: np.ndarray,
    indexed_mask: np.ndarray,
    track_id: int,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mask = build_target_binary_mask(indexed_mask, track_id)
    aug_mask = dilate_target_mask(
        target_mask,
        kernel_size=int(config.mask_kernel_size),
        iterations=int(config.mask_iterations),
    )
    aug_mask = expand_target_mask(
        aug_mask,
        w_len=int(config.mask_w_len),
        h_len=int(config.mask_h_len),
    ).astype(np.uint8)

    # Keep src_bg aligned with the exported src_mask so masked regions are removed consistently.
    bg_rgb = np.asarray(frame_rgb, dtype=np.uint8) * (1 - aug_mask[:, :, None])
    mask_rgb = np.stack([aug_mask * 255, aug_mask * 255, aug_mask * 255], axis=2).astype(np.uint8)
    return mask_rgb, bg_rgb.astype(np.uint8), target_mask


def score_reference_frame(*, target_mask: np.ndarray, face_detection: bool, pose_meta: dict) -> float:
    body_keypoints = np.asarray(pose_meta.get("keypoints_body", []), dtype=np.float32).reshape(-1, 3)
    visible_body = int((body_keypoints[:, 2] > 0).sum()) if len(body_keypoints) > 0 else 0
    mask_score = float(np.asarray(target_mask, dtype=np.uint8).sum()) / max(int(target_mask.size), 1)
    face_score = 1.0 if face_detection else 0.0
    return mask_score * 3.0 + face_score * 2.0 + float(visible_body) / 20.0
