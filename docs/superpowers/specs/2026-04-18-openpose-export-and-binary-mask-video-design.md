# OpenPose Export And Binary Mask Video Design

## Overview

This design adds export-focused offline outputs without overwriting the existing offline entrypoint. The new work must preserve the current inference and reconstruction behavior while saving two additional result families:

- OpenPose-style per-frame JSON files with both 2D and 3D pose data
- binary mask videos generated from the existing per-frame mask outputs

The current offline pipeline already produces:

- per-frame RGB images in the sample output directory
- per-frame palette masks in `masks`
- rendered 4D visualization frames and videos
- per-person mesh `.ply` files and focal/camera JSON

The missing piece is a stable export path for downstream pose consumers and video-based mask consumers.

## Goals

- Add a new offline export entrypoint without replacing `scripts/offline_app.py`.
- Save OpenPose-style per-frame JSON files containing both `pose_keypoints_2d` and `pose_keypoints_3d`.
- Save binary mask videos from the already generated mask frames.
- Reuse existing in-memory predictions and on-disk mask frames rather than adding a second inference pass.
- Keep output structure deterministic so batch or offline post-processing is easy.

## Non-Goals

- No overwrite of `scripts/offline_app.py`.
- No change to detector thresholds, tracking behavior, occlusion logic, or reconstruction quality.
- No requirement to recover a full SMPL parameter set before exporting pose JSON.
- No attempt to export face or hand OpenPose tracks in this phase.
- No dependency on a separate post-processing rerun after the main pipeline completes.

## Recommended Approach

Use a new export-oriented offline script that reuses the existing offline pipeline flow and adds focused export helpers:

- `scripts/offline_app_export.py`
- `scripts/openpose_export.py`
- `scripts/mask_video_export.py`

Why this is the recommended approach:

- It keeps the original script untouched.
- It avoids duplicating core inference logic.
- It exports from the same prediction objects already produced by the mesh stage.
- It keeps result quality aligned with the current working offline path.

## Data Sources Already Available

The current SAM-3D-Body outputs already expose, per detected person and frame:

- `pred_keypoints_2d`
- `pred_keypoints_3d`
- `pred_vertices`
- `pred_cam_t`
- `body_pose_params`
- `hand_pose_params`

This means the export path does not need to reconstruct keypoints from saved mesh files. It can write OpenPose JSON directly from the current prediction dictionaries produced during 4D generation.

## Output Layout

Within the sample output directory, add:

- `openpose_json/`
- `mask_videos/`

Recommended contents:

- `openpose_json/00000000_keypoints.json`
- `openpose_json/00000001_keypoints.json`
- `mask_videos/mask_binary_all.mp4`
- `mask_videos/mask_binary_person_01.mp4`
- `mask_videos/mask_binary_person_02.mp4`

The file names should stay zero-padded and frame-aligned with the existing `images` and `masks` directories.

## OpenPose JSON Format

Each frame should be written as one JSON file with an OpenPose-like top-level layout:

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": 1,
      "pose_keypoints_2d": [...],
      "pose_keypoints_3d": [...]
    }
  ]
}
```

Implementation details:

- `pose_keypoints_2d` should flatten as `(x, y, conf)` for 25 OpenPose body joints.
- `pose_keypoints_3d` should flatten as `(x, y, z, conf)` for the same 25 joints.
- `person_id` should use the tracked object id already propagated through the offline pipeline.

The OpenPose compatibility priority is on body pose ordering and JSON shape. Extra fields should stay minimal.

## Joint Mapping Strategy

The repository already contains `MHR70_TO_OPENPOSE` in `models/sam_3d_body/sam_3d_body/metadata/__init__.py`.

The export helper should:

1. Start from zero-filled OpenPose arrays for 25 joints.
2. Copy all directly mapped MHR70 joints into their OpenPose slots.
3. Derive OpenPose `MidHip` from the average of left hip and right hip when both are available.
4. Leave unsupported joints as zeros with confidence `0.0`.

This preserves compatibility while avoiding fabricated joints outside the known mapping.

## Confidence Policy

The current prediction tensors expose coordinates but not an explicit OpenPose-style confidence score.

The export policy should therefore be:

- mapped or derived joints get confidence `1.0`
- missing or unsupported joints get confidence `0.0`

This keeps the JSON explicit and deterministic instead of inventing pseudo-confidence values.

## 2D And 3D Coordinate Policy

For this phase:

- 2D output should use the predicted image-space coordinates from `pred_keypoints_2d`
- 3D output should use the predicted camera-space coordinates from `pred_keypoints_3d`

The export should not rescale or renormalize these values beyond the required OpenPose flattening and mapping.

## Binary Mask Video Strategy

The current offline pipeline already writes palette mask PNG files in `masks/`.

The new mask-video helper should derive binary videos directly from those files:

### Combined Foreground Video

- `mask_binary_all.mp4`
- foreground pixel is `255` if any person label is present
- background is `0`

### Per-Person Binary Videos

- `mask_binary_person_{track_id}.mp4`
- foreground pixel is `255` only where the given object id is present
- background is `0`

This avoids introducing a second mask source and guarantees exact agreement with the masks already used for reconstruction.

## Video Encoding Requirements

Binary mask videos should satisfy:

- single-channel logical content represented as 0/255 pixels
- frame size equal to the original saved mask resolution
- frame order aligned to the existing mask PNG naming

The container may still be encoded through OpenCV's normal video writer path, but the frame content itself must remain binary.

## Integration Points

### New Export Script

`scripts/offline_app_export.py` should mirror the existing offline flow:

1. initialize the existing offline app stack
2. run mask generation
3. run 4D generation
4. export OpenPose JSON from per-frame prediction outputs
5. export binary mask videos from the saved mask PNG sequence

### OpenPose Export Hook

The OpenPose export should happen during or immediately after the stage where `process_image_with_mask(...)` returns frame-aligned pose outputs, before those outputs are discarded.

### Mask Video Export Hook

The mask video export should run after the `masks/` directory has been written and before the script exits.

## Testing Strategy

Add focused tests for the export helpers:

- mapping test from MHR70 joints into 25-joint OpenPose order
- `MidHip` derivation test
- JSON schema and flattened-length test for both 2D and 3D arrays
- binary mask video frame-content test using synthetic mask PNGs
- smoke test that the export script writes the new output directories without changing the original output layout

These tests should stay lightweight and avoid requiring full model inference.

## Risks And Mitigations

Risk: exporting from saved mesh files would drift from the actual online prediction state.
Mitigation: export from the existing prediction dictionaries while they are still in memory.

Risk: OpenPose joint order could be wrong or incomplete.
Mitigation: use the repository's existing `MHR70_TO_OPENPOSE` mapping and add tests around derived joints.

Risk: mask videos might become visually binary but numerically non-binary after encoding.
Mitigation: build tests around the pre-encoding frame generation path and keep encoded frames sourced from strict 0/255 arrays.

Risk: new outputs could clutter the original script flow.
Mitigation: place the functionality in a new export-specific script instead of modifying the existing entrypoint.

## Rollout Plan

Implementation should proceed in this order:

1. add OpenPose export helper functions and mapping tests
2. add binary mask video export helper and tests
3. add the new export entrypoint script that reuses the current offline flow
4. add README usage notes for the new exported artifacts
