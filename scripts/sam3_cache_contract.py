import copy
import json
import os
from datetime import datetime, timezone
from typing import Any


CACHE_VERSION = 1
TRACEABILITY_FILE_TYPES = {
    "prompts.json": dict,
    "frame_metrics.json": list,
    "events.json": list,
}


def build_cache_meta(
    *,
    sample_id: str,
    source_video: str,
    frame_stems: list[str],
    image_size: dict[str, int],
    obj_ids: list[int],
    runtime_profile: dict[str, Any],
    config_path: str,
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "sample_id": sample_id,
        "source_video": source_video,
        "frame_count": len(frame_stems),
        "frame_stems": list(frame_stems),
        "image_size": dict(image_size),
        "image_ext": image_ext,
        "mask_ext": mask_ext,
        "obj_ids": list(obj_ids),
        "runtime_profile": copy.deepcopy(runtime_profile),
        "config_path": config_path,
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if "fps" in runtime_profile:
        meta["fps"] = runtime_profile["fps"]
    return meta


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_cache_dir(cache_dir: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return False, ["Missing required file: meta.json"]

    try:
        meta = load_json(meta_path)
    except json.JSONDecodeError as exc:
        return False, [f"Invalid JSON in meta.json: {exc}"]
    if not isinstance(meta, dict):
        return False, ["meta.json top-level value must be an object"]

    cache_version = meta.get("cache_version")
    if cache_version != CACHE_VERSION:
        errors.append(
            f"meta.cache_version must be {CACHE_VERSION}, got {cache_version!r}"
        )

    frame_stems = meta.get("frame_stems", [])
    image_ext = meta.get("image_ext", ".jpg")
    mask_ext = meta.get("mask_ext", ".png")
    frame_count = meta.get("frame_count")

    if not isinstance(frame_stems, list):
        errors.append("meta.frame_stems must be a list")
        frame_stems = []

    if frame_count != len(frame_stems):
        errors.append("meta.frame_count does not match len(frame_stems)")

    for filename, expected_type in TRACEABILITY_FILE_TYPES.items():
        payload_path = os.path.join(cache_dir, filename)
        if not os.path.isfile(payload_path):
            errors.append(f"Missing required file: {filename}")
            continue
        try:
            payload = load_json(payload_path)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in {filename}: {exc}")
            continue
        if not isinstance(payload, expected_type):
            errors.append(
                f"{filename} top-level value must be a {expected_type.__name__}"
            )

    for stem in frame_stems:
        image_rel = f"images/{stem}{image_ext}"
        mask_rel = f"masks/{stem}{mask_ext}"
        image_path = os.path.join(cache_dir, "images", f"{stem}{image_ext}")
        mask_path = os.path.join(cache_dir, "masks", f"{stem}{mask_ext}")
        if not os.path.isfile(image_path):
            errors.append(f"Missing required image: {image_rel}")
        if not os.path.isfile(mask_path):
            errors.append(f"Missing required mask: {mask_rel}")

    return len(errors) == 0, errors
