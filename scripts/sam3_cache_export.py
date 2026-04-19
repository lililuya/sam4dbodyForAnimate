import json
import os
import shutil
from glob import glob

from PIL import Image

from scripts.sam3_cache_contract import build_cache_meta, validate_cache_dir


def _frame_stems_from_dir(path, ext):
    pattern = os.path.join(path, f"*{ext}")
    return sorted(
        os.path.splitext(os.path.basename(file_path))[0]
        for file_path in glob(pattern)
    )


def export_sam3_cache(
    *,
    working_dir,
    cache_root,
    sample_id,
    source_video,
    runtime,
    config_path,
):
    cache_dir = os.path.join(cache_root, sample_id)
    cache_images_dir = os.path.join(cache_dir, "images")
    cache_masks_dir = os.path.join(cache_dir, "masks")
    os.makedirs(cache_images_dir, exist_ok=True)
    os.makedirs(cache_masks_dir, exist_ok=True)

    for image_path in sorted(glob(os.path.join(working_dir, "images", "*.jpg"))):
        shutil.copy2(image_path, os.path.join(cache_images_dir, os.path.basename(image_path)))
    for mask_path in sorted(glob(os.path.join(working_dir, "masks", "*.png"))):
        shutil.copy2(mask_path, os.path.join(cache_masks_dir, os.path.basename(mask_path)))

    frame_stems = _frame_stems_from_dir(cache_images_dir, ".jpg")
    if not frame_stems:
        raise ValueError(f"no exported images found under {cache_images_dir}")

    first_image_path = os.path.join(cache_images_dir, f"{frame_stems[0]}.jpg")
    with Image.open(first_image_path) as first_image:
        width, height = first_image.size

    meta = build_cache_meta(
        sample_id=sample_id,
        source_video=source_video,
        frame_stems=frame_stems,
        image_size={"width": width, "height": height},
        obj_ids=runtime["out_obj_ids"],
        runtime_profile={
            "batch_size": runtime["batch_size"],
            "detection_resolution": list(runtime["detection_resolution"]),
            "completion_resolution": list(runtime["completion_resolution"]),
            "smpl_export": bool(runtime.get("smpl_export", False)),
            "fps": float(runtime.get("video_fps", 0.0)),
        },
        config_path=config_path,
    )

    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(cache_dir, "prompts.json"), "w", encoding="utf-8") as handle:
        json.dump({"targets": runtime.get("prompt_log", {})}, handle, indent=2)
    with open(os.path.join(cache_dir, "frame_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime.get("frame_metrics", []), handle, indent=2)
    with open(os.path.join(cache_dir, "events.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime.get("events", []), handle, indent=2)

    ok, errors = validate_cache_dir(cache_dir)
    if not ok:
        raise ValueError(f"exported cache is invalid: {errors}")
    return cache_dir
