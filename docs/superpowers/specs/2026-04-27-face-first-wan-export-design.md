# Face-First Wan Export Design

## Goal

Guarantee that the final Wan export clips come from stable face-consistent video segments where the target face is present on every frame and identity swaps are aggressively avoided.

The design adds a face-first preprocessing stage before `sam4d`. This stage is not a generic face-mining system. Its only purpose is to feed `sam4d -> wan export` with safer clip inputs so the final Wan export artifacts are more reliable in multi-person videos.

The final deliverable remains the Wan export output written to a user-specified `wan_export.output_dir`.

## Scope

This design must:

- add a first-stage stable face clip extractor that scans raw input videos
- detect faces on every frame for multi-person videos
- build conservative per-face temporal segments
- keep only segments where the target face exists on every frame
- drop segments shorter than `5` seconds
- write deterministic `sample_uuid` and `clip_id` metadata for kept clips
- make `sam4d` consume stable clip packages instead of raw source videos
- bind one body target to one face-guided clip inside a short initialization window
- make Wan export reuse the first-stage IDs instead of generating a new runtime UUID
- preserve `wan_export.output_dir` as the final export destination

This design must not:

- turn the first stage into a standalone face dataset generator
- optimize for maximum face-track recall at the cost of identity safety
- keep ambiguous or partially missing tracks alive through gap filling
- preserve the current "raw multi-person video in, detect all targets inside `sam4d`" behavior for the new path

## Current State

The refined offline path currently starts from a raw source video, detects people, creates target prompts, runs mask generation and 4D inference, and only later attempts to validate or crop faces during Wan export.

This causes two structural problems for the user's end goal:

1. Face validity is checked too late.
2. `sam4d` still sees the full multi-person source and may initialize targets that are not safe for Wan export.

There is already reusable face-detection logic in [`scripts/wan_face_export.py`](E:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\wan_face_export.py), and there are already reusable helpers for stable source-to-UUID mapping and per-sample summaries in [`scripts/wan_sample_export.py`](E:\Project\sam-body4d-master\.worktrees\sam4d-wananimate-export\scripts\wan_sample_export.py).

The current pipeline does not yet have a stage whose job is:

- scan raw videos first
- cut stable face-consistent subclips first
- then run `sam4d` only on those subclips

## Design Summary

The new path is explicitly two-stage.

### Stage 1: Stable Face Clip Extraction

Input:

- raw source video directory

Output:

- deterministic face-stable clip packages

Responsibilities:

- run per-frame face detection
- associate faces across adjacent frames conservatively
- cut segments immediately on missing face, ambiguity, or unsafe reassignment
- keep only segments with face presence on every frame and duration `>= 5s`
- write `clip.mp4`, `track.json`, and `meta.json`

### Stage 2: Face-Guided SAM4D To Wan Export

Input:

- one stable clip package from Stage 1

Output:

- final Wan export sample under the configured `wan_export.output_dir`

Responsibilities:

- treat `clip.mp4` as the video input instead of the original raw video
- bind one body target to the clip's face track using a short front-window check
- create exactly one `sam4d` target
- run the existing refined path in single-target clip mode
- write Wan export outputs using the first-stage `clip_id`

## Stage 1 Design

### Detection Policy

Stage 1 runs face detection on every frame of every input video.

This is intentionally dense rather than sampled because the user's requirement is not "some face evidence exists." The user's requirement is "the kept segment must contain the target face on every frame."

The default backend remains `InsightFaceBackend`.

### Association Policy

Association must be conservative and identity-safe.

Rules:

- only associate between adjacent frames
- do not reconnect across long gaps
- do not interpolate missing detections
- do not preserve a track through a face-missing frame
- do not guess through crowded ambiguity

For each live face track, the next frame must offer exactly one sufficiently plausible candidate. Plausibility is based on geometric continuity only, such as:

- center displacement
- box scale change
- IoU continuity
- landmark geometry consistency

If the next frame has:

- no plausible candidate: end the segment
- more than one plausible candidate: end the segment
- a candidate claimed by multiple tracks: end the affected segments

The design deliberately prefers over-segmentation to identity switches.

### Segment Rule

A kept segment must satisfy all of the following:

- it belongs to one face track identity
- every frame in the segment has one face detection assigned to that identity
- the segment length is at least `5` seconds

A segment ends immediately before any of these events:

- missing face detection
- ambiguous assignment
- track conflict
- unreadable frame

No gap-filling policy is allowed in Stage 1 for kept segments.

### Stage 1 Output Contract

Stage 1 produces a minimal clip package per kept segment:

```text
<clip_output_root>/
  source_uuid_map.json
  batch_summary.json
  clips/
    <clip_id>/
      clip.mp4
      track.json
      meta.json
```

Definitions:

- `sample_uuid`: one stable ID per raw source video
- `clip_id`: deterministic per kept face segment, for example `<sample_uuid>_face01_seg001`

#### `clip.mp4`

- original video content clipped to the kept time range
- no face crop
- this becomes the direct Stage 2 `sam4d` input

#### `track.json`

- per-frame face records for the kept segment only
- enough information to verify that the segment really contains one stable target face on every frame

Minimum fields:

- `clip_id`
- `sample_uuid`
- `source_path`
- `fps`
- `records`

Each record contains:

- `frame_index_in_source`
- `frame_index_in_clip`
- `timestamp_seconds`
- `bbox_xyxy`
- `landmarks`
- `score`

#### `meta.json`

Minimum fields:

- `clip_id`
- `sample_uuid`
- `source_path`
- `start_frame`
- `end_frame`
- `fps`
- `frame_count`
- `duration_seconds`
- `clip_path`
- `track_json_path`

#### `source_uuid_map.json`

Stable mapping between original source videos and `sample_uuid`.

Minimum fields per item:

- `source_path`
- `sample_uuid`
- `first_processed_at`
- `last_processed_at`

#### `batch_summary.json`

Run-level summary for the clip extraction stage.

Minimum fields:

- `input_dir`
- `output_dir`
- `started_at`
- `finished_at`
- `video_count_total`
- `video_count_completed`
- `video_count_failed`
- `clip_count_kept`
- `clip_count_dropped`
- `items`

Each item contains:

- `sample_uuid`
- `source_path`
- `status`
- `kept_clip_count`
- `dropped_segment_count`
- `drop_reasons`

## Stage 2 Design

### Input Contract

The `sam4d` path no longer starts from a raw multi-person video for this mode.

Instead, it starts from one Stage 1 clip package:

```text
<clip_output_root>/clips/<clip_id>/
  clip.mp4
  track.json
  meta.json
```

The refined path must be able to accept a clip-package input mode where:

- `clip.mp4` is the actual video source
- `meta.json` defines the clip identity and provenance
- `track.json` provides the target face track

### Body Binding Policy

Stage 2 still needs a body target for `sam4d`, but this target must now be face-guided and single-target.

Binding policy:

- inspect the first `8` to `16` frames of `clip.mp4`
- run body detection on those frames
- read the corresponding face boxes from `track.json`
- select the one body candidate that most consistently matches the face track across the window

Allowed cues:

- face center lies in the upper body region
- face box placement is geometrically reasonable relative to body box
- the same body candidate remains preferred across the window

If the body binding window cannot produce exactly one stable body candidate, the clip is skipped.

This keeps the design aligned with the user's requirement to avoid face swaps even if it reduces yield.

### SAM4D Target Policy

For this new mode:

- create exactly one target
- do not initialize multiple people
- do not search for the frame with the largest people count
- do not preserve the existing raw-video multi-target initialization behavior

The new mode is effectively a face-guided single-target `sam4d` run.

### ID Policy

Stage 2 must not generate a new runtime UUID for this mode.

Instead:

- reuse `sample_uuid` from Stage 1
- treat `clip_id` as the clip-level primary key

This means:

- output directory naming is driven by `clip_id`
- sample summaries reference `clip_id`
- Wan export metadata references `clip_id`
- no second source-to-UUID map is introduced inside the `sam4d` stage

## Wan Export Policy

The final user-facing artifact remains the Wan export output.

The design keeps `wan_export.output_dir` as the final destination for exported training samples.

Stage 1 clip packages and Stage 2 `sam4d` intermediates are supporting artifacts. They exist to make the final Wan export safer, not to replace it.

For this mode:

- Wan export directory selection remains explicit and configurable
- exported sample folders should be keyed by `clip_id`
- Wan export metadata should record both `clip_id` and `sample_uuid`
- Wan export should not allocate a new sample UUID separate from the Stage 1 identity

Recommended final contract:

```text
<wan_export.output_dir>/
  <clip_id>/
    target.mp4
    src_ref.png
    src_pose.mp4
    src_face.mp4
    src_bg.mp4
    src_mask.mp4
    meta.json
    pose_meta_sequence.json
```

The exact per-sample artifact set can continue following the existing Wan export contract, but the directory identity should come from Stage 1.

## Failure Handling

### Stage 1 drop reasons

Suggested normalized reasons:

- `shorter_than_min_duration`
- `missing_face_detection`
- `ambiguous_face_assignment`
- `track_conflict`
- `decode_error`

### Stage 2 skip reasons

Suggested normalized reasons:

- `body_binding_failed`
- `body_binding_ambiguous`
- `sam4d_tracking_failed`
- `wan_export_failed`

The important rule is that both stages must remain traceable through `sample_uuid` and `clip_id`.

## Configuration

Stage 1 should introduce its own small configuration surface rather than overloading unrelated `wan_export` fields.

Recommended fields:

- `face_clip.enable`
- `face_clip.input_dir`
- `face_clip.output_dir`
- `face_clip.min_clip_seconds`
- `face_clip.association_window_frames`
- `face_clip.debug_save_face_crop_video`

Recommended defaults:

- `min_clip_seconds = 5.0`
- `association_window_frames = 1`
- `debug_save_face_crop_video = false`

Stage 2 continues using `wan_export.output_dir` for final outputs.

## Testing Strategy

### Unit tests

1. adjacent-frame face association
   - stable motion keeps one track
   - ambiguous next-frame match breaks the segment

2. segment extraction
   - missing face ends the segment
   - segments shorter than `5s` are dropped
   - kept segments contain per-frame records without gaps

3. identity propagation
   - `sample_uuid` and `clip_id` are deterministic and reused

4. body binding
   - face-guided body selection resolves one candidate
   - ambiguous body candidates skip the clip

### Integration tests

1. one-person video produces one or more valid clip packages
2. two-person video with only one stable visible face produces only that person's clip
3. two close faces crossing each other causes conservative segment splitting rather than merged identity
4. Stage 2 consumes a clip package and writes Wan export outputs under the configured final directory

## Risks

1. Conservative segmentation will reduce yield on crowded or partially occluded videos.
2. Very short face losses will break segments instead of being repaired.
3. If body detection is unstable even on the stabilized clip, some otherwise valid face clips will still be skipped before `sam4d`.

These risks are acceptable because the user's stated priority is final Wan export reliability, not maximum recall.

## Recommendation

Implement a new face-first clip extraction stage and make the refined `sam4d` path consume those clip packages in a face-guided single-target mode.

This is the most direct way to align the pipeline with the actual product requirement:

- the final Wan export clips should have the target face on every frame
- the final Wan export clips should avoid face identity swaps
- the final artifact should still land under the configured Wan export directory
