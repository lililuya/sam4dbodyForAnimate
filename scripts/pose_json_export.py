import json
import os
import runpy
from typing import Any

import numpy as np

from scripts.openpose_export import convert_mhr70_to_openpose_arrays


SUPPORTED_POSE_EXPORTS = ("openpose", "smpl", "coco17", "coco_wholebody")
DEFAULT_POSE_EXPORTS = ("openpose", "smpl")

COCO17_JOINTS = 17
COCO17_KEYPOINT_DIM_2D = 3
COCO17_KEYPOINT_DIM_3D = 4

COCO_WHOLEBODY_BODY_JOINTS = 17
COCO_WHOLEBODY_FOOT_JOINTS = 6
COCO_WHOLEBODY_FACE_JOINTS = 68
COCO_WHOLEBODY_HAND_JOINTS = 21

LEFT_FOOT_MHR70 = [15, 16, 17]
RIGHT_FOOT_MHR70 = [18, 19, 20]
RIGHT_HAND_MHR70 = [
    41, 24, 23, 22, 21,
    28, 27, 26, 25,
    32, 31, 30, 29,
    36, 35, 34, 33,
    40, 39, 38, 37,
]
LEFT_HAND_MHR70 = [
    62, 45, 44, 43, 42,
    49, 48, 47, 46,
    53, 52, 51, 50,
    57, 56, 55, 54,
    61, 60, 59, 58,
]
WHOLEBODY_FACE_SOURCE = "zero_fill_missing_from_mhr70"

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


def _load_pose_metadata():
    metadata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "sam_3d_body",
        "sam_3d_body",
        "metadata",
        "__init__.py",
    )
    return runpy.run_path(metadata_path)


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


def _normalize_export_formats(export_formats):
    if export_formats is None:
        formats = list(DEFAULT_POSE_EXPORTS)
    elif isinstance(export_formats, str):
        formats = [export_formats]
    else:
        formats = list(export_formats)

    normalized = []
    for export_name in formats:
        name = str(export_name).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_POSE_EXPORTS:
            raise ValueError(f"unsupported pose export format: {export_name}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def _count_visible_keypoints(flat_keypoints, dims):
    array = np.asarray(flat_keypoints, dtype=np.float32).reshape(-1, dims)
    confidence_idx = dims - 1
    return int((array[:, confidence_idx] > 0).sum())


def _convert_mhr70_subset(keypoints_2d, keypoints_3d, mhr70_indices):
    keypoints_2d = _to_json_ready(_to_numpy_array(keypoints_2d, dims=2))
    keypoints_3d = _to_json_ready(_to_numpy_array(keypoints_3d, dims=3))
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)

    pose_2d = np.zeros((len(mhr70_indices), COCO17_KEYPOINT_DIM_2D), dtype=np.float32)
    pose_3d = np.zeros((len(mhr70_indices), COCO17_KEYPOINT_DIM_3D), dtype=np.float32)

    for output_idx, mhr_idx in enumerate(mhr70_indices):
        if mhr_idx is None:
            continue
        if mhr_idx < len(keypoints_2d):
            pose_2d[output_idx, :2] = keypoints_2d[mhr_idx, :2]
            pose_2d[output_idx, 2] = 1.0
        if mhr_idx < len(keypoints_3d):
            pose_3d[output_idx, :3] = keypoints_3d[mhr_idx, :3]
            pose_3d[output_idx, 3] = 1.0
    return pose_2d, pose_3d


def _to_numpy_array(value, dims):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < dims:
        raise ValueError(f"expected keypoints with shape [N, >={dims}], got {array.shape}")
    return array


def convert_mhr70_to_coco17_arrays(keypoints_2d, keypoints_3d):
    metadata = _load_pose_metadata()
    openpose_to_coco = list(metadata["OPENPOSE_TO_COCO"])

    openpose_2d, openpose_3d = convert_mhr70_to_openpose_arrays(keypoints_2d, keypoints_3d)
    openpose_2d = np.asarray(openpose_2d, dtype=np.float32).reshape(-1, COCO17_KEYPOINT_DIM_2D)
    openpose_3d = np.asarray(openpose_3d, dtype=np.float32).reshape(-1, COCO17_KEYPOINT_DIM_3D)

    coco_2d = openpose_2d[openpose_to_coco]
    coco_3d = openpose_3d[openpose_to_coco]
    return coco_2d.reshape(-1).tolist(), coco_3d.reshape(-1).tolist()


def convert_mhr70_to_coco_wholebody_arrays(keypoints_2d, keypoints_3d):
    body_2d_flat, body_3d_flat = convert_mhr70_to_coco17_arrays(keypoints_2d, keypoints_3d)
    body_2d = np.asarray(body_2d_flat, dtype=np.float32).reshape(COCO_WHOLEBODY_BODY_JOINTS, COCO17_KEYPOINT_DIM_2D)
    body_3d = np.asarray(body_3d_flat, dtype=np.float32).reshape(COCO_WHOLEBODY_BODY_JOINTS, COCO17_KEYPOINT_DIM_3D)

    foot_2d_left, foot_3d_left = _convert_mhr70_subset(keypoints_2d, keypoints_3d, LEFT_FOOT_MHR70)
    foot_2d_right, foot_3d_right = _convert_mhr70_subset(keypoints_2d, keypoints_3d, RIGHT_FOOT_MHR70)
    foot_2d = np.concatenate([foot_2d_left, foot_2d_right], axis=0)
    foot_3d = np.concatenate([foot_3d_left, foot_3d_right], axis=0)

    face_2d = np.zeros((COCO_WHOLEBODY_FACE_JOINTS, COCO17_KEYPOINT_DIM_2D), dtype=np.float32)
    face_3d = np.zeros((COCO_WHOLEBODY_FACE_JOINTS, COCO17_KEYPOINT_DIM_3D), dtype=np.float32)

    left_hand_2d, left_hand_3d = _convert_mhr70_subset(keypoints_2d, keypoints_3d, LEFT_HAND_MHR70)
    right_hand_2d, right_hand_3d = _convert_mhr70_subset(keypoints_2d, keypoints_3d, RIGHT_HAND_MHR70)

    wholebody_2d = np.concatenate([body_2d, foot_2d, face_2d, left_hand_2d, right_hand_2d], axis=0)
    wholebody_3d = np.concatenate([body_3d, foot_3d, face_3d, left_hand_3d, right_hand_3d], axis=0)
    return {
        "body_2d": body_2d.reshape(-1).tolist(),
        "body_3d": body_3d.reshape(-1).tolist(),
        "foot_2d": foot_2d.reshape(-1).tolist(),
        "foot_3d": foot_3d.reshape(-1).tolist(),
        "face_2d": face_2d.reshape(-1).tolist(),
        "face_3d": face_3d.reshape(-1).tolist(),
        "left_hand_2d": left_hand_2d.reshape(-1).tolist(),
        "left_hand_3d": left_hand_3d.reshape(-1).tolist(),
        "right_hand_2d": right_hand_2d.reshape(-1).tolist(),
        "right_hand_3d": right_hand_3d.reshape(-1).tolist(),
        "wholebody_2d": wholebody_2d.reshape(-1).tolist(),
        "wholebody_3d": wholebody_3d.reshape(-1).tolist(),
    }


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


def build_coco17_person_payload(*, person_output, track_id: int, frame_stem: str) -> dict:
    pose_2d, pose_3d = convert_mhr70_to_coco17_arrays(
        person_output["pred_keypoints_2d"],
        person_output["pred_keypoints_3d"],
    )
    return {
        "format": "coco17",
        "version": 1.0,
        "person_id": int(track_id),
        "frame_stem": str(frame_stem),
        "num_keypoints": _count_visible_keypoints(pose_2d, COCO17_KEYPOINT_DIM_2D),
        "keypoints_2d": pose_2d,
        "keypoints_3d": pose_3d,
    }


def build_coco_wholebody_person_payload(*, person_output, track_id: int, frame_stem: str) -> dict:
    wholebody = convert_mhr70_to_coco_wholebody_arrays(
        person_output["pred_keypoints_2d"],
        person_output["pred_keypoints_3d"],
    )
    return {
        "format": "coco_wholebody",
        "version": 1.0,
        "person_id": int(track_id),
        "frame_stem": str(frame_stem),
        "num_keypoints": _count_visible_keypoints(wholebody["wholebody_2d"], COCO17_KEYPOINT_DIM_2D),
        "keypoints_2d": wholebody["wholebody_2d"],
        "keypoints_3d": wholebody["wholebody_3d"],
        "body_keypoints_2d": wholebody["body_2d"],
        "body_keypoints_3d": wholebody["body_3d"],
        "foot_keypoints_2d": wholebody["foot_2d"],
        "foot_keypoints_3d": wholebody["foot_3d"],
        "face_keypoints_2d": wholebody["face_2d"],
        "face_keypoints_3d": wholebody["face_3d"],
        "left_hand_keypoints_2d": wholebody["left_hand_2d"],
        "left_hand_keypoints_3d": wholebody["left_hand_3d"],
        "right_hand_keypoints_2d": wholebody["right_hand_2d"],
        "right_hand_keypoints_3d": wholebody["right_hand_3d"],
        "face_keypoints_source": WHOLEBODY_FACE_SOURCE,
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


def _write_pose_payload_json(*, output_dir: str, subdir: str, track_id: int, filename: str, payload: dict) -> str:
    target_dir = os.path.join(output_dir, subdir, str(int(track_id)))
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, filename)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path


def write_openpose_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_openpose_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    return _write_pose_payload_json(
        output_dir=output_dir,
        subdir="openpose_json",
        track_id=track_id,
        filename=f"{frame_stem}_keypoints.json",
        payload=payload,
    )


def write_smpl_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_smpl_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    return _write_pose_payload_json(
        output_dir=output_dir,
        subdir="smpl_json",
        track_id=track_id,
        filename=f"{frame_stem}.json",
        payload=payload,
    )


def write_coco17_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_coco17_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    return _write_pose_payload_json(
        output_dir=output_dir,
        subdir="coco17_json",
        track_id=track_id,
        filename=f"{frame_stem}.json",
        payload=payload,
    )


def write_coco_wholebody_person_json(*, output_dir: str, frame_stem: str, person_output, track_id: int) -> str:
    payload = build_coco_wholebody_person_payload(
        person_output=person_output,
        track_id=track_id,
        frame_stem=frame_stem,
    )
    return _write_pose_payload_json(
        output_dir=output_dir,
        subdir="coco_wholebody_json",
        track_id=track_id,
        filename=f"{frame_stem}.json",
        payload=payload,
    )


def build_pose_frame_writer(*, output_dir: str, export_formats=None):
    normalized_formats = _normalize_export_formats(export_formats)
    if not normalized_formats:
        return None

    def frame_writer(image_path, mask_output, id_current):
        if not mask_output or not id_current:
            return {format_name: [] for format_name in normalized_formats}
        frame_stem = os.path.splitext(os.path.basename(image_path))[0]
        return write_pose_frame_exports(
            output_dir=output_dir,
            frame_stem=frame_stem,
            person_outputs=mask_output,
            track_ids=id_current,
            export_formats=normalized_formats,
        )

    return frame_writer


def write_pose_frame_exports(*, output_dir: str, frame_stem: str, person_outputs, track_ids, export_formats=None) -> dict[str, list[str]]:
    if not person_outputs or not track_ids:
        return {format_name: [] for format_name in _normalize_export_formats(export_formats)}
    if len(person_outputs) != len(track_ids):
        raise ValueError("person_outputs and track_ids must have matching lengths")

    normalized_formats = _normalize_export_formats(export_formats)
    format_writers = {
        "openpose": write_openpose_person_json,
        "smpl": write_smpl_person_json,
        "coco17": write_coco17_person_json,
        "coco_wholebody": write_coco_wholebody_person_json,
    }
    results = {format_name: [] for format_name in normalized_formats}
    for person_output, track_id in zip(person_outputs, track_ids):
        for format_name in normalized_formats:
            results[format_name].append(
                format_writers[format_name](
                output_dir=output_dir,
                frame_stem=frame_stem,
                person_output=person_output,
                track_id=track_id,
            )
            )
    return results
