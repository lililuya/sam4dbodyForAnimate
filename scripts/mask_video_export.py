import glob
import os

import numpy as np
from PIL import Image


def images_to_mp4(*args, **kwargs):
    from utils.image2video import images_to_mp4 as _images_to_mp4

    return _images_to_mp4(*args, **kwargs)


def list_mask_paths(mask_dir):
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        raise ValueError(f"No mask PNG files found in: {mask_dir}")
    return mask_paths


def build_binary_mask_frames(mask_dir, track_id=None):
    frames = []
    for mask_path in list_mask_paths(mask_dir):
        mask = np.array(Image.open(mask_path).convert("P"), dtype=np.uint8)
        if track_id is None:
            binary = (mask > 0).astype(np.uint8) * 255
        else:
            binary = (mask == int(track_id)).astype(np.uint8) * 255
        frames.append(binary)
    return frames


def export_binary_mask_videos(mask_dir, output_dir, track_ids, fps=25):
    os.makedirs(output_dir, exist_ok=True)

    images_to_mp4(
        build_binary_mask_frames(mask_dir),
        os.path.join(output_dir, "mask_binary_all.mp4"),
        fps=int(fps),
    )

    for track_id in sorted({int(track_id) for track_id in track_ids}):
        images_to_mp4(
            build_binary_mask_frames(mask_dir, track_id=track_id),
            os.path.join(output_dir, f"mask_binary_person_{track_id:02d}.mp4"),
            fps=int(fps),
        )
