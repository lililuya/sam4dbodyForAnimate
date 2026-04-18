import json
import os
import runpy

import numpy as np


OPENPOSE_BODY_25_JOINTS = 25
OPENPOSE_MIDHIP_INDEX = 8
OPENPOSE_RIGHT_HIP_INDEX = 9
OPENPOSE_LEFT_HIP_INDEX = 12


def _load_mhr70_to_openpose():
    metadata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "sam_3d_body",
        "sam_3d_body",
        "metadata",
        "__init__.py",
    )
    metadata_vars = runpy.run_path(metadata_path)
    return metadata_vars["MHR70_TO_OPENPOSE"]


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def convert_mhr70_to_openpose_arrays(keypoints_2d, keypoints_3d):
    keypoints_2d = _to_numpy(keypoints_2d)
    keypoints_3d = _to_numpy(keypoints_3d)
    mhr70_to_openpose = _load_mhr70_to_openpose()

    pose_2d = np.zeros((OPENPOSE_BODY_25_JOINTS, 3), dtype=np.float32)
    pose_3d = np.zeros((OPENPOSE_BODY_25_JOINTS, 4), dtype=np.float32)

    for openpose_idx, mhr_idx in mhr70_to_openpose.items():
        if mhr_idx >= len(keypoints_2d) or mhr_idx >= len(keypoints_3d):
            continue
        pose_2d[openpose_idx, :2] = keypoints_2d[mhr_idx, :2]
        pose_2d[openpose_idx, 2] = 1.0
        pose_3d[openpose_idx, :3] = keypoints_3d[mhr_idx, :3]
        pose_3d[openpose_idx, 3] = 1.0

    if pose_2d[OPENPOSE_RIGHT_HIP_INDEX, 2] > 0 and pose_2d[OPENPOSE_LEFT_HIP_INDEX, 2] > 0:
        pose_2d[OPENPOSE_MIDHIP_INDEX, :2] = (
            pose_2d[OPENPOSE_RIGHT_HIP_INDEX, :2] + pose_2d[OPENPOSE_LEFT_HIP_INDEX, :2]
        ) / 2.0
        pose_2d[OPENPOSE_MIDHIP_INDEX, 2] = 1.0

    if pose_3d[OPENPOSE_RIGHT_HIP_INDEX, 3] > 0 and pose_3d[OPENPOSE_LEFT_HIP_INDEX, 3] > 0:
        pose_3d[OPENPOSE_MIDHIP_INDEX, :3] = (
            pose_3d[OPENPOSE_RIGHT_HIP_INDEX, :3] + pose_3d[OPENPOSE_LEFT_HIP_INDEX, :3]
        ) / 2.0
        pose_3d[OPENPOSE_MIDHIP_INDEX, 3] = 1.0

    return pose_2d.reshape(-1).tolist(), pose_3d.reshape(-1).tolist()


def build_openpose_people(person_outputs, track_ids):
    people = []
    for person_output, track_id in zip(person_outputs, track_ids):
        pose_2d, pose_3d = convert_mhr70_to_openpose_arrays(
            person_output["pred_keypoints_2d"],
            person_output["pred_keypoints_3d"],
        )
        people.append(
            {
                "person_id": int(track_id),
                "pose_keypoints_2d": pose_2d,
                "pose_keypoints_3d": pose_3d,
            }
        )
    return people


def write_openpose_frame_json(output_dir, frame_stem, people):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{frame_stem}_keypoints.json")
    payload = {"version": 1.3, "people": people}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path
