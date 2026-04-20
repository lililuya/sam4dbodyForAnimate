import copy
import json
import os
import shutil
import uuid
from glob import glob

import numpy as np
from PIL import Image

from scripts.sam3_cache_contract import build_cache_meta, validate_cache_dir


def _bbox_xyxy_from_mask(binary_mask):
    coords = np.argwhere(binary_mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def start_sam3_export_session(runtime, *, output_root, source_video, id_factory=None):
    if id_factory is None:
        id_factory = lambda: uuid.uuid4().hex

    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    output_dir = os.path.join(output_root, id_factory())
    os.makedirs(output_dir, exist_ok=False)

    runtime["prompt_log"] = {}
    runtime["frame_metrics"] = []
    runtime["events"] = [{"type": "video_loaded", "source_video": source_video}]
    runtime["session_video_path"] = source_video
    runtime["session_output_dir"] = output_dir
    runtime["mask_generation_completed"] = False
    return output_dir


def ensure_sam3_export_ready(*, runtime, video_path, output_dir):
    output_dir = os.path.abspath(output_dir)
    if video_path is None:
        raise ValueError("No video loaded.")
    if runtime.get("session_video_path") != video_path:
        raise ValueError("Export is not ready for the current video session.")
    if os.path.abspath(runtime.get("session_output_dir", "")) != output_dir:
        raise ValueError("Export is not ready for the current video session.")
    if not runtime.get("mask_generation_completed", False):
        raise ValueError("Run Mask Generation before exporting SAM3 cache.")

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        raise ValueError("Run Mask Generation before exporting SAM3 cache.")
    return images_dir, masks_dir


def record_prompt_update(runtime, *, obj_id, frame_idx, input_point, input_label):
    target_entry = runtime.setdefault("prompt_log", {}).setdefault(
        str(obj_id),
        {"name": f"Target {obj_id}", "frames": {}},
    )
    target_entry["frames"][str(int(frame_idx))] = {
        "points": np.asarray(input_point).tolist(),
        "labels": np.asarray(input_label).tolist(),
    }
    runtime.setdefault("events", []).append(
        {"type": "prompt_updated", "obj_id": int(obj_id), "frame_idx": int(frame_idx)}
    )
    runtime["frame_metrics"] = []
    runtime["mask_generation_completed"] = False


def build_frame_metrics_from_video_segments(video_segments):
    frame_metrics = []
    for frame_idx in sorted(video_segments):
        track_metrics = {}
        for obj_id, out_mask in sorted(video_segments[frame_idx].items()):
            mask_array = np.asarray(out_mask)
            if mask_array.ndim >= 3:
                mask_array = mask_array[0]
            binary_mask = (mask_array > 0).astype(np.uint8)
            track_metrics[str(int(obj_id))] = {
                "mask_area": int(binary_mask.sum()),
                "bbox_xyxy": _bbox_xyxy_from_mask(binary_mask),
            }
        frame_metrics.append(
            {
                "frame_idx": int(frame_idx),
                "frame_stem": f"{int(frame_idx):08d}",
                "track_metrics": track_metrics,
            }
        )
    return frame_metrics


def build_runtime_export_state(runtime):
    return {
        "out_obj_ids": list(runtime.get("out_obj_ids", [])),
        "runtime_profile": {
            "batch_size": int(runtime.get("batch_size", 1)),
            "detection_resolution": list(runtime.get("detection_resolution", [])),
            "completion_resolution": list(runtime.get("completion_resolution", [])),
            "smpl_export": bool(runtime.get("smpl_export", False)),
            "fps": float(runtime.get("video_fps", 0.0)),
        },
        "prompt_log": copy.deepcopy(runtime.get("prompt_log", {})),
        "frame_metrics": copy.deepcopy(runtime.get("frame_metrics", [])),
        "events": copy.deepcopy(runtime.get("events", [])),
    }


def _resolve_safe_cache_dir(cache_root, sample_id):
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("unsafe sample_id")

    invalid_separators = {os.sep}
    if os.altsep:
        invalid_separators.add(os.altsep)
    if sample_id in {".", ".."} or any(sep in sample_id for sep in invalid_separators):
        raise ValueError("unsafe sample_id")

    cache_root_abs = os.path.abspath(cache_root)
    cache_dir = os.path.abspath(os.path.join(cache_root_abs, sample_id))
    if os.path.dirname(cache_dir) != cache_root_abs:
        raise ValueError("unsafe sample_id")
    return cache_root_abs, cache_dir


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
    runtime_export = build_runtime_export_state(runtime)
    cache_root, cache_dir = _resolve_safe_cache_dir(cache_root, sample_id)
    os.makedirs(cache_root, exist_ok=True)
    staging_dir = os.path.join(cache_root, f"sam3_cache_{sample_id}_{uuid.uuid4().hex}")
    os.makedirs(staging_dir, exist_ok=False)
    cache_images_dir = os.path.join(staging_dir, "images")
    cache_masks_dir = os.path.join(staging_dir, "masks")
    os.makedirs(cache_images_dir, exist_ok=True)
    os.makedirs(cache_masks_dir, exist_ok=True)

    try:
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
            obj_ids=runtime_export["out_obj_ids"],
            runtime_profile=runtime_export["runtime_profile"],
            config_path=config_path,
        )

        with open(os.path.join(staging_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        with open(os.path.join(staging_dir, "prompts.json"), "w", encoding="utf-8") as handle:
            json.dump({"targets": runtime_export["prompt_log"]}, handle, indent=2)
        with open(os.path.join(staging_dir, "frame_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(runtime_export["frame_metrics"], handle, indent=2)
        with open(os.path.join(staging_dir, "events.json"), "w", encoding="utf-8") as handle:
            json.dump(runtime_export["events"], handle, indent=2)

        ok, errors = validate_cache_dir(staging_dir)
        if not ok:
            raise ValueError(f"exported cache is invalid: {errors}")

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        shutil.move(staging_dir, cache_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return cache_dir


def export_session_cache(*, runtime, video_path, output_dir, output_root, config_path):
    ensure_sam3_export_ready(runtime=runtime, video_path=video_path, output_dir=output_dir)
    cache_root = os.path.join(output_root, "sam3_cache")
    sample_id = os.path.basename(os.path.normpath(output_dir))
    return export_sam3_cache(
        working_dir=output_dir,
        cache_root=cache_root,
        sample_id=sample_id,
        source_video=video_path,
        runtime=runtime,
        config_path=config_path,
    )
