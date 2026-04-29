# Face Presence Sample Skipping Design

## Goal

Add a sample-level face-presence precheck before refined tracking starts so samples with too little face visibility can be skipped early, avoid leaving per-sample output directories, and record all skipped or failed cases in a single append-only ledger for later tracing.

## Scope

This change must:

- add configurable face-presence sampling controls under `wan_export`
- check face presence before refined tracking, mask generation, 4D generation, and Wan export
- skip samples when the sampled no-face ratio reaches a configurable threshold
- avoid creating the per-sample output directory for samples skipped by this precheck
- write sample-level skipped and failed cases into one append-only JSON ledger
- write Wan target-level skipped cases into the same ledger

This change must not:

- change the existing human detection or tracker initialization logic
- change Wan target filtering semantics such as `min_track_frames` or `min_valid_face_ratio`
- resample input videos or change frame ordering

## Current State

The refined offline path currently creates the sample output directory inside `prepare_input()` before any sample-quality gate runs.

Wan export performs face detection later per target inside `WanSampleExporter.finalize()`, but there is no sample-level gate before the expensive tracking and 4D stages.

Skipped Wan targets are already written to `<sample_uuid>_skipped.json`, but there is no single ledger that collects sample failures, sample skips, and target-level skips across runs.

## Design

### New Wan Export Config Fields

Extend `wan_export` with these fields:

- `skip_sample_without_face`
  - `bool`
  - default `True`
  - enables the sample-level face-presence gate

- `face_presence_stride`
  - `int`
  - default `5`
  - sample every Nth frame during the face-presence precheck

- `max_no_face_ratio`
  - `float`
  - default `0.80`
  - skip the sample when `no_face_frame_count / checked_frame_count >= max_no_face_ratio`

These fields live under `wan_export` because the precheck exists to protect Wan-exportable training samples, not the core tracker itself.

### Sample Precheck

Run a face-presence precheck immediately after input metadata is prepared and before:

- sample output directory creation
- initial target detection
- tracking
- refined mask writing
- 4D generation
- Wan export summary allocation

The precheck will:

1. read every `face_presence_stride` frame from the source
2. run `InsightFaceBackend.detect()` on each sampled frame
3. count:
   - `checked_frame_count`
   - `face_detected_frame_count`
   - `no_face_frame_count`
   - `no_face_ratio`
4. compare `no_face_ratio` against `max_no_face_ratio`

If `skip_sample_without_face` is disabled, the precheck reports metrics but never skips.

If `checked_frame_count` is `0`, treat the sample as failed input handling rather than a skip condition.

### Skip Rule

Skip the sample when all of the following are true:

- `wan_export.enable` is `True`
- `skip_sample_without_face` is `True`
- `checked_frame_count > 0`
- `no_face_ratio >= max_no_face_ratio`

For skipped samples:

- set sample status to `skipped`
- set skip reason to `face_presence_below_threshold`
- do not create the per-sample working directory
- do not allocate Wan summary UUIDs or write sample-level runtime JSON under a sample directory
- do not execute tracking, mask generation, 4D generation, or Wan export

### Output Directory Lifecycle

`prepare_input()` will stop creating the output directory eagerly. It will only resolve and attach the planned `output_dir` path to the sample metadata.

The directory becomes real only when the refined pipeline decides the sample should continue and `prepare_sample_output()` runs.

This preserves the existing downstream directory layout for valid samples while preventing empty directories for skipped samples.

### Unified Issue Ledger

Add a single append-only JSON ledger:

- filename: `sample_issue_ledger.json`

Ledger root resolution:

1. use `wan_export.output_dir` when configured and non-empty
2. otherwise use the refined runtime output root

The file stores a dictionary with an `items` list. Each append adds a new record without deduplication.

Each record contains:

- `recorded_at`
- `event_type`
- `status`
- `reason`
- `source_path`
- `sample_id`
- `sample_uuid`
- `working_output_dir`
- `runtime_profile`
- `details`

### Ledger Event Types

#### Sample skipped by face gate

- `event_type`: `sample_skipped_no_face`
- `status`: `skipped`
- `reason`: `face_presence_below_threshold`

`details` must include:

- `checked_frame_count`
- `face_detected_frame_count`
- `no_face_frame_count`
- `no_face_ratio`
- `face_presence_stride`
- `max_no_face_ratio`

#### Sample failed during pipeline

- `event_type`: `sample_failed`
- `status`: `failed`
- `reason`: `pipeline_exception`

`details` must include:

- `error_type`
- `error_message`

#### Wan target skipped

- `event_type`: `wan_target_skipped`
- `status`: `skipped`
- `reason`: existing Wan target skip reason such as:
  - `track_frames_below_threshold`
  - `readable_frames_below_threshold`
  - `valid_face_ratio_below_threshold`

`details` must include the original target-skip payload, including `track_id`.

## Code Touch Points

### `scripts/offline_app_refined.py`

- delay sample output directory creation until after the face-presence gate passes
- add a face-presence probe helper using `InsightFaceBackend`
- add a skip-decision helper driven by `wan_export` config
- add ledger-root resolution and sample-level ledger append helpers
- record sample-level `face_presence` metrics into `sample_summary`
- write sample failure ledger entries on exceptions
- route `run_refined_pipeline()` through the same `run_sample()` behavior to avoid divergent control flow

### `scripts/wan_sample_export.py`

- add shared ledger helpers for append-only JSON writes
- append one ledger event per skipped Wan target while preserving the existing `<sample_uuid>_skipped.json` report

### `scripts/wan_sample_types.py`

- add typed config fields for the new face-presence options

### Config files

- add the new `wan_export` keys to refined config variants so the feature is visible and configurable

## Tests

Add tests for:

- sample skip when sampled no-face ratio reaches the threshold
- no per-sample directory created for a precheck-skipped sample
- skipped sample ledger append payload
- failed sample ledger append payload
- Wan target skipped ledger append payload
- existing Wan skipped JSON still written

## Risks

The main risk is applying the sample skip gate too broadly and dropping useful clips. This is mitigated by:

- stride-based sampling instead of dense per-frame scanning
- a configurable threshold
- a persistent ledger with explicit metrics for every skipped sample

## Non-Goals

This design does not:

- add a new human detector for faces
- replace Wan target-level face filtering
- backfill or mutate old ledger entries
- guarantee deduplication in the append-only ledger
