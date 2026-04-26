# Sample FPS Summary Design

## Goal

Add sample-level FPS reporting so each refined run records the source video FPS, the effective 4D render FPS, and the Wan export target FPS without changing any tracking, completion, or export behavior.

## Scope

This change is metadata-only.

It must not:
- resample input videos
- modify frame ordering
- change 4D generation behavior
- change Wan export sampling behavior
- introduce new required config fields

It must:
- record source FPS when the input is a video file
- record that FPS is unavailable when the input is an image sequence or unreadable
- record the effective rendered 4D FPS used by `app_4d_pipeline.py`
- record the configured Wan export target FPS when Wan export is enabled

## Current State

The refined offline path currently counts frames for `.mp4` inputs but does not persist the source FPS in sample summaries.

The final 4D writer in `app_4d_pipeline.py` uses `context.runtime.get("video_fps", 25)`, so the effective exported 4D FPS may fall back to `25` even when the input video has a different FPS.

Wan export already reads source FPS from `cv2.CAP_PROP_FPS` and uses it to compute sampled frame indices before writing fixed-FPS Wan outputs.

## Design

### Sample Metadata

Add a small `fps_summary` object to sample-level summaries and runtime outputs with these fields:

- `source_fps`
  - `float | null`
  - The FPS read from the input video metadata.
  - `null` for image-sequence inputs or unreadable video FPS metadata.

- `source_fps_source`
  - `str`
  - One of:
    - `video_metadata`
    - `image_sequence`
    - `unavailable`

- `rendered_4d_fps`
  - `float`
  - The effective FPS used for the final `4d_*.mp4` write path.
  - This is the actual value that the current runtime would use, not a newly introduced override.

- `wan_target_fps`
  - `int | null`
  - The configured `wan_export.fps` value when Wan export is enabled.
  - `null` when Wan export is disabled.

### Output Locations

Persist the same `fps_summary` payload in:

- `sample_runtime.json`
- `debug_metrics/sample_summary.json`
- `<sample_uuid>_summary.json` under external Wan export root when enabled

This keeps the data visible both in local sample working directories and in the external Wan export bookkeeping.

### Runtime Rules

For video inputs:
- attempt to read FPS from `cv2.CAP_PROP_FPS`
- accept only values greater than `0`
- otherwise treat the source FPS as unavailable

For image-sequence inputs:
- do not invent a synthetic source FPS
- store `source_fps = null`
- store `source_fps_source = "image_sequence"`

For rendered 4D FPS:
- report the value the current runtime will actually use
- if `runtime["video_fps"]` is absent, report the current fallback value `25`

For Wan target FPS:
- report `wan_export.fps` only when `wan_export.enable` is true
- otherwise report `null`

## Code Touch Points

### `scripts/offline_app_refined.py`

Add a small helper to resolve input source FPS from the sample input path.

Attach an `fps_summary` object to `self.sample_summary` early in `run_sample()` after `prepare_input()`.

Ensure `_persist_sample_runtime()` includes `fps_summary` in the written JSON payload.

Ensure Wan sample summary updates also receive the same `fps_summary`.

### `scripts/wan_sample_export.py`

No behavior change is required for Wan sampling logic.

This module already resolves source FPS for frame resampling. The design only relies on its existing behavior remaining unchanged.

### Tests

Extend refined runtime tests to verify:
- video input writes `source_fps` when readable
- image-sequence input reports `source_fps = null` and `source_fps_source = "image_sequence"`
- `sample_runtime.json` contains `fps_summary`
- external Wan summary contains the same `fps_summary`

## Risks

The main risk is accidental confusion between:
- source video FPS
- rendered 4D output FPS
- Wan export target FPS

This design avoids ambiguity by storing all three as separate named fields in one object.

## Non-Goals

This design does not:
- normalize all inputs to `25 fps`
- add CLI FPS overrides
- change completion model conditioning FPS
- change Wan export sampling logic
- fix the broader issue of refined runtime not always propagating original video FPS into `runtime["video_fps"]`

That larger runtime propagation change can be handled separately if needed.
