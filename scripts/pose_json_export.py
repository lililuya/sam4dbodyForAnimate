import json
import os
from typing import Any

import numpy as np

from scripts.openpose_export import convert_mhr70_to_openpose_arrays


SMPL_JSON_FIELDS = (
    "bbox",
    "focal_length",
    "pred_cam_t",
    "pred_pose_raw",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "expr_params",
    "pred_keypoints_2d",
    "pred_keypoints_3d",
    "pred_joint_coords",
    "pred_global_rots",
    "mhr_model_params",
)


def _to_json_ready(value: Any):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    return value


def build_openpose_person_payload(*, person_output, track_id: int, frame_stem: str) -> dict:
    pose_2d, pose_3d = convert_mhr70_to_openpose_arrays(
        person_output["pred_keypoints_2d"],
        person_output["pred_keypoints_3d"],
    )
    return {
        "version": 1.3,
        "person_id": int(track_id),
        "frame_stem": str(frame_stem),
        "pose_keypoints_2d": pose_2d,
        "pose_keypoints_3d": pose_3d,
    }


def build_smpl_person_payload(*, person_output, track_id: int, frame_stem: str) -> dict:
    openpose_payload = build_openpose_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    payload = {
        "person_id": int(track_id),
        "frame_stem": str(frame_stem),
        "openpose_pose_keypoints_2d": openpose_payload["pose_keypoints_2d"],
        "openpose_pose_keypoints_3d": openpose_payload["pose_keypoints_3d"],
    }
    for field_name in SMPL_JSON_FIELDS:
        if field_name not in person_output:
            continue
        payload[field_name] = _to_json_ready(person_output[field_name])
    return payload


def write_openpose_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_openpose_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    target_dir = os.path.join(output_dir, "openpose_json", str(int(track_id)))
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"{frame_stem}_keypoints.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path


def write_smpl_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_smpl_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    target_dir = os.path.join(output_dir, "smpl_json", str(int(track_id)))
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"{frame_stem}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path


def write_pose_frame_exports(*, output_dir: str, frame_stem: str, person_outputs, track_ids) -> dict[str, list[str]]:
    if not person_outputs or not track_ids:
        return {"openpose": [], "smpl": []}
    if len(person_outputs) != len(track_ids):
        raise ValueError("person_outputs and track_ids must have matching lengths")

    openpose_paths = []
    smpl_paths = []
    for person_output, track_id in zip(person_outputs, track_ids):
        openpose_paths.append(
            write_openpose_person_json(
                output_dir=output_dir,
                frame_stem=frame_stem,
                person_output=person_output,
                track_id=track_id,
            )
        )
        smpl_paths.append(
            write_smpl_person_json(
                output_dir=output_dir,
                frame_stem=frame_stem,
                person_output=person_output,
                track_id=track_id,
            )
        )
    return {"openpose": openpose_paths, "smpl": smpl_paths}
