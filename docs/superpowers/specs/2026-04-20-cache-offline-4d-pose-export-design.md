# Cache Offline 4D Pose Export

## Summary

This design extends the cache-based offline 4D runner so it exports per-track, per-frame pose artifacts in addition to the existing mesh and focal outputs.

The new behavior adds two traceability-oriented output trees under `outputs_4d/<sample_id>/`:

- `openpose_json/<track_id>/<frame>_keypoints.json`
- `smpl_json/<track_id>/<frame>.json`

The OpenPose payload is derived from the same per-person 4D outputs already produced during inference. The SMPL JSON payload preserves the per-person model parameters and related pose tensors that are needed for downstream inspection and conversion.

## Goals

- Keep `scripts/run_4d_from_cache.py` as the supported offline cache runner.
- Export OpenPose-format pose data during cache-based offline 4D execution.
- Export per-track, per-frame SMPL/MHR-related parameters as JSON.
- Preserve the existing mesh, focal, and rendered video outputs.
- Reuse a shared export helper instead of duplicating JSON-writing logic across entrypoints.

## Non-Goals

- Changing the cache contract itself.
- Replacing the existing `scripts/openpose_export.py` frame-level helper.
- Changing the numerical meaning of model outputs.
- Introducing new batch orchestration behavior.

## Output Structure

For each offline cache run:

```text
outputs_4d/<sample_id>/
  openpose_json/
    1/
      00000000_keypoints.json
    2/
      00000000_keypoints.json
  smpl_json/
    1/
      00000000.json
    2/
      00000000.json
```

## Data Model

### OpenPose JSON

Each file stores a single person for a single frame with:

- `version`
- `person_id`
- `frame_stem`
- `pose_keypoints_2d`
- `pose_keypoints_3d`

The keypoints come from the existing MHR 70-joint outputs via the current OpenPose conversion mapping.

### SMPL JSON

Each file stores a single person for a single frame with:

- `person_id`
- `frame_stem`
- `bbox`
- `focal_length`
- `pred_cam_t`
- `pred_pose_raw`
- `global_rot`
- `body_pose_params`
- `hand_pose_params`
- `scale_params`
- `shape_params`
- `expr_params`
- `pred_keypoints_2d`
- `pred_keypoints_3d`
- `pred_joint_coords`
- `pred_global_rots`
- `mhr_model_params`
- `openpose_pose_keypoints_2d`
- `openpose_pose_keypoints_3d`

All tensor-like values are serialized as JSON-compatible lists or scalars.

## Architecture

Add a new shared helper module responsible for:

- converting per-person outputs into JSON-safe payloads
- writing per-track OpenPose files
- writing per-track SMPL JSON files

`scripts/run_4d_from_cache.py` wires this helper into the shared 4D pipeline through the existing `frame_writer` hook. The shared pipeline remains responsible for inference and calls the hook once per frame after per-person outputs are available.

## Testing Strategy

- add helper tests for OpenPose and SMPL payload writing
- add cache-runner wiring tests to prove the offline runner supplies a working `frame_writer`
- keep the shared pipeline contract unchanged except for using the existing hook

## Success Criteria

This change is successful only if:

- cache-based offline 4D still runs successfully
- `outputs_4d/<sample_id>/openpose_json/<track_id>/` is written
- `outputs_4d/<sample_id>/smpl_json/<track_id>/` is written
- each exported SMPL JSON includes OpenPose-converted pose arrays
- existing mesh and focal exports continue to work
