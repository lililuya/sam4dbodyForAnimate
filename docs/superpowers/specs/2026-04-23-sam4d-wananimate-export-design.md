# SAM4D To WanAnimate Export Design

## Goal

Build an offline export path that uses `sam4dbody` tracking, masks, and pose outputs to generate WanAnimate-compatible training samples for single-person and two-person videos.

The exported samples should preserve WanAnimate's preprocessing conventions where they matter for downstream training:

- `target.mp4`
- `src_ref.png`
- `src_pose.mp4`
- `src_face.mp4`
- `src_bg.mp4`
- `src_mask.mp4`

The main reason for this export path is that WanAnimate's original pose detector is not reliable enough on the target data, especially under overlap, occlusion, and multi-person scenes. The new path should replace WanAnimate pose estimation with `sam4dbody` pose/tracking while keeping the final exported data contract close to WanAnimate's expected inputs.

## Scope

This design covers:

- exporting WanAnimate-style sample folders from the refined offline SAM4D pipeline
- single-person and two-person input videos
- one exported training sample per target `track_id`
- per-frame pose conversion from `sam4dbody` outputs into WanAnimate-style pose metadata
- per-frame face crop generation using `InsightFace`
- per-frame mask and background video generation using WanAnimate-style mask augmentation
- sample-level filtering using face-quality statistics

This design does not cover:

- changes to WanAnimate model training code
- changes to the split SAM3 cache / `run_4d_from_cache.py` path
- replacement of WanAnimate's face encoder
- introducing a new face landmark format into WanAnimate itself

## Existing Behavior

### SAM4D side

The current refined offline path eventually enters [`scripts/app_4d_pipeline.py`](E:\Project\sam-body4d-master\scripts\app_4d_pipeline.py), where it:

- reads exported `images/` and `masks/`
- runs 4D inference per batch
- optionally writes mesh renders, individual renders, mesh/focal outputs
- calls `frame_writer(image_path, mask_output, id_current)` once per processed frame

Current frame export behavior is JSON-oriented. It already supports:

- OpenPose-like JSON
- SMPL JSON
- COCO17 JSON
- COCO WholeBody JSON

The current export path does not generate WanAnimate-style:

- pose condition videos
- face crop videos
- background videos
- augmented mask videos

### WanAnimate side

The reference preprocessing behavior lives in:

- [`process_pipepline.py`](G:\Project\WanAnimateDataProcess\wan\modules\animate\preprocess\process_pipepline.py)
- [`pose2d_utils.py`](G:\Project\WanAnimateDataProcess\wan\modules\animate\preprocess\pose2d_utils.py)
- [`human_visualization.py`](G:\Project\WanAnimateDataProcess\wan\modules\animate\preprocess\human_visualization.py)
- [`utils.py`](G:\Project\WanAnimateDataProcess\wan\modules\animate\preprocess\utils.py)

Important observations:

1. WanAnimate expects `src_pose.mp4`, `src_face.mp4`, `src_ref.png`, and `target.mp4` to exist for a usable sample.
2. WanAnimate's original face path is frame-by-frame, not fixed-box. It recomputes a face crop each frame from per-frame face keypoints.
3. WanAnimate does not guarantee every frame has a high-quality face crop. It simply consumes the available `src_face.mp4`.
4. In replacement mode it also exports `src_bg.mp4` and `src_mask.mp4`.
5. WanAnimate's resolution handling is area-based rather than a hard resize to a fixed output width and height.

## Required Output Contract

For each selected target person, export one sample directory:

```text
<sample_id>_track_<track_id>/
  target.mp4
  src_ref.png
  src_pose.mp4
  src_face.mp4
  src_bg.mp4
  src_mask.mp4
  meta.json
  pose_meta_json/
    00000000.json
    00000001.json
    ...
```

The output contract should satisfy the following rules:

- all frame-based assets must use the same sampled frame indices
- `target.mp4`, `src_pose.mp4`, `src_bg.mp4`, and `src_mask.mp4` must share the same frame count and working resolution
- `src_face.mp4` must share the same frame count, but it is always `512x512`
- `fps` is fixed to `25`
- the main video stream uses WanAnimate-style area-based resizing with `resolution_area = [512, 768]`
- one source video with two tracked persons produces two sample directories

## Resolution And FPS Policy

### FPS

All exported videos are written at `25fps` regardless of source FPS.

Sampling behavior:

- determine source FPS
- compute target frame count for `25fps`
- compute sampled frame indices
- use those exact indices for all derived assets

This keeps all outputs frame-aligned and avoids hidden drift across condition streams.

### Resolution

Main streams use WanAnimate-style area control, not a strict fixed-size resize.

Configuration:

- `resolution_area = [512, 768]`
- target area = `512 * 768`
- preserve aspect ratio
- align dimensions to multiples of `16`

This produces a per-sample `working_resolution = [height, width]` that is:

- portrait-style when the source is portrait-like
- close to the target area
- consistent with WanAnimate preprocessing assumptions

The following files use `working_resolution`:

- `target.mp4`
- `src_pose.mp4`
- `src_bg.mp4`
- `src_mask.mp4`
- `src_ref.png`

`src_face.mp4` always uses `512x512`.

## Sample Construction Model

### Target-centric export

Each export sample corresponds to exactly one target track.

Example:

- source video contains track `1` and track `2`
- export sample A for `track 1`
- export sample B for `track 2`

For a sample targeting `track 1`:

- pose stream describes only `track 1`
- face stream crops only `track 1`
- mask stream masks only `track 1`
- bg stream removes only `track 1`
- the other person remains part of the background/target scene

This matches the stated data goal: choose one person as the target and treat everyone else as context/background.

## Architecture

The export path should be implemented as a new Wan-specific export layer on top of the refined offline pipeline, not as a replacement for existing pose JSON export.

Recommended decomposition:

1. `wan_sample_export.py`
2. `wan_pose_adapter.py`
3. `wan_face_export.py`
4. `wan_mask_bg_export.py`
5. `wan_sample_types.py` or equivalent helper structures if needed

### Why a new export layer

The existing export path already has stable responsibilities:

- run refined offline tracking and mask generation
- run 4D generation
- optionally export JSON pose formats

WanAnimate export is a different output contract with different artifact types. Reusing the current `frame_writer` hook is useful, but overloading the existing JSON exporter with video rendering responsibilities would mix unrelated concerns and make the pipeline harder to reason about.

## Data Flow

### Stage 1: Source frame preparation

Input:

- source video path

Output:

- sampled frame indices at `25fps`
- resized working frames in `working_resolution`
- per-frame frame stems

Responsibilities:

- compute frame indices once
- resize frames once
- reuse the same resized frames for target, pose, bg, mask, and face association

### Stage 2: SAM4D inference

Input:

- sampled frames

Output:

- target track IDs
- per-frame target masks
- per-frame target `sam4dbody` outputs, including 2D keypoints

The existing refined pipeline remains responsible for:

- detection
- tracking
- mask propagation
- optional mask refinement
- 4D pose/mesh estimation

### Stage 3: Track-specific sample splitting

Input:

- all tracked outputs for one source video

Output:

- one export job per target track

Responsibilities:

- group per-frame outputs by `track_id`
- skip tracks with insufficient temporal coverage
- generate a stable sample ID for each target track

### Stage 4: Wan pose adaptation

Input:

- `sam4dbody` person output for one frame
- optional target face landmarks for the same frame
- frame width/height

Output:

- Wan-style pose meta JSON for that frame

Expected JSON shape:

```json
{
  "image_id": "frame_00000001.jpg",
  "width": 496,
  "height": 736,
  "category_id": 1,
  "keypoints_body": [...],
  "keypoints_left_hand": [...],
  "keypoints_right_hand": [...],
  "keypoints_face": [...]
}
```

Coordinate convention:

- all exported keypoints are normalized to `[0, 1]`
- confidence is stored as the third value per keypoint

### Stage 5: Pose condition rendering

Input:

- Wan-style pose meta

Output:

- one RGB pose condition frame

Rendering behavior:

- mimic WanAnimate `draw_aapose_by_meta_new` style
- black background
- body and hand skeletons
- no mesh rendering

### Stage 6: Face crop generation

Input:

- resized working frame
- target mask / target head region
- `InsightFace` detections and landmarks

Output:

- one `512x512` face frame per sampled frame

Rules:

1. run `InsightFace` every frame
2. if multiple faces exist, choose the face belonging to the target person
3. crop a square face region with configurable expansion
4. resize to `512x512`
5. if no face is detected for a short gap, use temporal fallback
6. record quality statistics

### Stage 7: Mask and background generation

Input:

- working frame
- target binary mask

Output:

- augmented mask frame
- background frame with target removed

Behavior:

- use SAM4D target mask directly as the body mask source
- apply Wan-like augmentation logic equivalent to `get_mask_body_img` and `get_aug_mask`
- `src_mask.mp4` is a 3-channel white mask over black background
- `src_bg.mp4` preserves all non-target content

### Stage 8: Sample packaging

Input:

- aligned frame sequences

Output:

- final sample directory

Responsibilities:

- write all mp4 files
- choose and write `src_ref.png`
- write `pose_meta_json`
- write `meta.json`

## Pose Mapping Design

The pose adapter must produce Wan-compatible body and hand structures from `sam4dbody` outputs.

Recommended strategy:

1. reuse the existing `MHR70 -> OpenPose25` mapping already present in this repository
2. derive Wan body ordering from the OpenPose-style intermediate representation
3. map left/right hands directly from the MHR70 hand regions already used by current JSON exporters
4. populate `keypoints_face` from `InsightFace` landmarks instead of trying to synthesize them from `sam4dbody`

Rationale:

- existing repository mappings are already tested and reduce risk
- Wan-style skeleton rendering mainly depends on body and hands
- face landmarks are more reliable when sourced from `InsightFace`

### Face keypoints format

`keypoints_face` should contain the available `InsightFace` landmarks normalized into `[0, 1]` coordinates with confidence values.

This is enough for:

- debugging
- sample quality measurement
- optional future face-box recomputation

The design does not require a perfect match to Wan's original dense face keypoint set as long as the face export module itself owns the crop generation logic.

## Face Association Design

The face exporter must solve a multi-person association problem.

Per frame:

1. detect all faces with `InsightFace`
2. compute the target head region
3. choose the best-matching face

Recommended matching score:

- primary: overlap between face box and target head/mask region
- secondary: distance from face center to target head anchor points
- tertiary: temporal continuity with previous frame face box

This keeps identity stable when two people are close together or overlap.

### Face fallback policy

WanAnimate requires a face video stream, but not provably a perfect face in every frame.

Fallback policy:

- short gaps: reuse previous valid face box or interpolate between neighboring valid boxes
- long gaps: continue producing fallback crops but record the failure in metadata

Sample quality statistic:

- `valid_face_ratio = valid_detection_frames / total_frames`

Suggested default filter:

- keep sample when `valid_face_ratio >= 0.60`
- otherwise mark or discard

## Reference Frame Selection

`src_ref.png` should be chosen from the target track, not arbitrarily from the full video.

Recommended scoring features:

- target mask area is stable and not too small
- target face is valid and centered
- pose is not heavily occluded
- track is clearly present

Use the highest-scoring frame as the reference frame and write it at `working_resolution`.

## Metadata Design

Each exported sample should include `meta.json` with at least:

```json
{
  "source_video": "...",
  "sample_id": "...",
  "track_id": 1,
  "fps": 25,
  "resolution_area": [512, 768],
  "working_resolution": [736, 496],
  "face_resolution": [512, 512],
  "frame_count": 120,
  "sample_indices": [0, 1, 2],
  "valid_face_ratio": 0.88,
  "face_fallback_count": 9,
  "mask_refine_enabled": true
}
```

Optional extra fields:

- source frame count
- source FPS
- detector backend
- refine configuration
- sample rejection reason if the export is skipped

## Error Handling

### Export should fail hard when

- source video cannot be read
- sampled frames are empty
- no valid target track exists
- video streams cannot be written

### Export should continue with recorded degradation when

- some frames lack a valid face detection
- target mask is temporarily unstable but recoverable
- a reference-frame candidate is weak but still usable

### Export should skip a track when

- track coverage is too short
- target is absent in most frames
- face quality is below threshold

Skipped samples should be logged with explicit reasons.

## Testing Strategy

Implementation should be testable without running full heavyweight models in unit tests.

### Unit tests

1. pose mapping tests
   - verify `sam4dbody` outputs convert into Wan-style body and hand arrays
   - verify normalization and ordering

2. face association tests
   - two faces in one frame, ensure correct target face is selected
   - missing detections trigger fallback logic

3. mask/bg generation tests
   - verify augmented mask expansion
   - verify background frame removes only target region

4. packaging tests
   - verify output directory layout
   - verify `meta.json` fields
   - verify aligned frame counts across exported videos

### Integration tests

1. single-person synthetic sample
2. two-person synthetic sample with overlapping masks
3. sample rejection on low face coverage

## Rollout Plan

Recommended implementation order:

1. Wan pose adapter and per-frame pose meta JSON
2. pose video rendering
3. face exporter with association and fallback
4. mask and background exporter
5. sample packager and metadata
6. integration into the refined offline batch path

This order gives an early milestone where pose export is already verifiable before face and background streams are added.

## Open Decisions Resolved By This Design

- use `sam4dbody` pose instead of WanAnimate `Pose2d`: yes
- use `InsightFace` for face detection and cropping: yes
- one sample per target track: yes
- replacement-style `src_bg.mp4` and `src_mask.mp4`: yes
- fixed output FPS: `25`
- Wan-style area-based portrait resolution: yes
- resolution area: `[512, 768]`
- face stream strict size: `512x512`
- face requirement policy: every frame must produce a crop, but not every frame must be a successful fresh detection

## Risks

1. Face association may still swap identities during severe overlap.
2. Some `sam4dbody` tracks may not provide sufficiently stable target presence for usable sample export.
3. The Wan-style pose rendering may look slightly different if the pose ordering or confidence handling differs from Wan's original detector output.
4. Hardcoded assumptions about face landmark layout should stay isolated inside the face export module so they can be replaced later.

## Recommendation

Implement this as a new Wan-specific export path attached to the refined offline batch pipeline. Keep the existing JSON export path intact. Treat Wan export as a distinct output mode with its own sample contract, quality checks, and tests.
