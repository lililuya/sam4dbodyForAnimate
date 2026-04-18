# Quality-Preserving Refined Batch Runner Design

## Overview

This design extends the existing refined offline pipeline with a new batch runner whose first priority is preserving per-sample output quality. The batch feature must not trade away mask accuracy, detector stability, or occlusion recovery quality merely to improve throughput.

The existing refined work already introduced:

- a separate refined entrypoint in `scripts/offline_app_refined.py`
- YOLO-based automatic prompting
- stronger occlusion-aware mask refinement helpers
- re-prompt trigger and matching helpers
- refined output layout and debug manifests

What is still missing is a true batch-safe execution path that can run many videos automatically while keeping the quality behavior of each single-sample run intact.

## Goals

- Add a new batch entrypoint that runs many samples fully automatically.
- Preserve the same per-video quality policy as the refined single-sample path.
- Reuse one model initialization across many samples when safe, to reduce overhead without cross-sample interference.
- Add adaptive fallback behavior for difficult videos, but only through quality-preserving adjustments.
- Persist enough diagnostics to identify when a sample was retried, downgraded for stability, or flagged for manual inspection.
- Make the batch runner resumable so interrupted runs do not force recomputation of already completed samples.

## Non-Goals

- No cross-video inference batching that mixes frames from different videos into one tracking or refinement pass.
- No silent threshold relaxation that would make masks easier to produce but worse in quality.
- No changes to the behavior of `scripts/offline_app.py`.
- No first-pass optimization for maximum throughput at the expense of output consistency.
- No distributed scheduler or multi-machine orchestration in this phase.

## Recommended Approach

Use a quality-preserving sequential batch runner:

1. Initialize the refined model stack once.
2. Enumerate input samples from a manifest, directory, or globbed list.
3. Process exactly one sample at a time through the refined pipeline.
4. Allow only intra-sample fallback, such as smaller chunk windows or smaller reconstruction batches, when stability issues are detected.
5. Save the exact runtime profile used for each sample.

Why this is the recommended approach:

- It reuses expensive model initialization and avoids startup overhead.
- It does not mix identities or temporal states across videos.
- It keeps the quality characteristics close to the single-sample pipeline.
- It allows targeted retries for only the problematic sample instead of weakening the whole batch.

## New User-Facing Entry Point

Add a new script:

- `scripts/offline_batch_refined.py`

This script should sit beside, not replace:

- `scripts/offline_app.py`
- `scripts/offline_app_refined.py`

Recommended CLI shape:

- `--input_root`: root directory containing videos or frame folders
- `--input_list`: optional text or JSONL manifest of sample paths
- `--output_dir`: batch output root
- `--config`: refined config path
- `--detector_backend`: optional override
- `--track_chunk_size`: optional default override
- `--skip_existing`: skip samples already marked complete
- `--continue_on_error`: continue after sample failures
- `--save_debug_metrics`: force debug outputs on
- `--max_samples`: optional cap for debugging
- `--retry_mode`: one of `never`, `quality_safe`, `aggressive_safe`

The batch runner should treat `offline_app_refined.py` as the per-sample engine rather than reimplementing detector, tracker, or reconstruction logic separately.

## Core Quality Principle

The batch runner must separate throughput controls from quality controls.

Allowed automatic adjustments:

- reduce chunk length for a difficult sample
- reduce reconstruction batch size for memory stability
- expand the initial person-search window across more early frames when no human is found immediately
- rerun the same sample with stricter isolation after a recoverable failure

Disallowed automatic adjustments unless explicitly configured:

- lowering detector confidence thresholds just to produce more boxes
- disabling occlusion refinement to make the run faster
- disabling re-prompt checks to avoid resets
- shrinking refinement resolution as a silent fallback
- reusing boxes or masks from a different sample

In short: the batch runner may spend more time on a hard sample, but it must not silently make the result cheaper and worse.

## Architecture

### Batch-Orchestrator Layer

`scripts/offline_batch_refined.py` should own:

- sample discovery
- resume / skip logic
- per-sample runtime profile selection
- failure handling and retries
- writing a batch summary manifest

It should not own low-level mask generation or 4D reconstruction math.

### Per-Sample Engine Layer

`scripts/offline_app_refined.py` should remain the per-sample engine and should be extended so the batch runner can call it programmatically.

Recommended additions:

- `RefinedOfflineApp.run_sample(...)`
- `RefinedOfflineApp.reset_sample_state()`
- small helper methods for:
  - initial-frame search
  - sample-local runtime overrides
  - per-sample summary writing

### Shared Runtime Profile

The batch runner should construct a sample execution profile before each run. This profile can override selected runtime settings for one sample only.

Suggested profile fields:

- `tracking.chunk_size`
- `sam_3d_body.batch_size`
- `completion.max_occ_len`
- initial detection search frame window
- retry count
- reason for any fallback

The profile must be written into each sample's debug outputs so downstream inspection can see exactly how the sample was processed.

## Sample Discovery and Resume

The batch runner should support two input modes:

### Directory Walk

- discover `*.mp4` files
- discover frame directories containing supported image extensions

### Explicit Manifest

Support a file listing one sample per line or JSONL entries such as:

```json
{"input": "/path/to/video_a.mp4", "sample_id": "video_a"}
```

Batch-level bookkeeping should write:

- `batch_manifest.json`
- `batch_results.jsonl`

Each sample record should include:

- sample path
- sample id
- output directory
- final status: `completed`, `failed`, `skipped`, `retry_succeeded`
- runtime profile used
- retry count
- error summary if failed

## Quality-Preserving Fallback Strategy

The batch runner should retry only with adjustments that preserve the refined logic.

### Retry Level 0: Normal

- use config defaults
- use refined occlusion path
- use re-prompt heuristics

### Retry Level 1: Safer Tracking

- reduce `tracking.chunk_size`
- keep detection thresholds unchanged
- keep refinement enabled

### Retry Level 2: Safer Reconstruction

- reduce `sam_3d_body.batch_size`
- keep chunk size at the safer value
- keep refinement and re-prompt enabled

### Retry Level 3: Conservative Search Expansion

- expand initial human-search window across more early frames
- keep detector thresholds unchanged

If a sample still fails after the configured retry ladder, mark it failed and preserve diagnostics. The runner should not silently drop to a low-quality path.

## Initial Detection Strategy for Batch Robustness

A common batch failure case is "person not visible in frame 0." The batch runner should therefore decouple sample start detection from tracking start index.

Recommended behavior:

1. Scan the first configurable `N` frames for valid person detections.
2. Choose the first frame whose detections pass the standard detector criteria.
3. Start the track from that frame.
4. Record the selected start frame in debug outputs.

This improves robustness without weakening detector quality.

## State Isolation Between Samples

To avoid hidden quality regressions across a long batch:

- clear predictor state between samples
- clear sample-specific runtime dictionaries
- clear chunk records, re-prompt events, and temporary path references
- ensure no cached object ids, bboxes, or masks leak across samples

Model weights may stay loaded, but sample state must be freshly initialized every time.

## Diagnostics

Each sample should write:

- `debug_metrics/chunk_manifest.json`
- `debug_metrics/sample_summary.json`
- `debug_metrics/reprompt_events.json`
- `debug_metrics/runtime_profile.json`

The sample summary should include:

- input path
- frame count
- selected start frame
- object count
- retry count
- chunk count
- whether refinement ran
- whether re-prompt ran
- failure reason if any

The batch root should also write aggregate results so low-quality outliers are easy to spot without opening every sample directory.

## Error Handling

Batch mode should isolate errors per sample.

Recommended behavior:

- if one sample fails, preserve all diagnostics for that sample
- continue to the next sample only when `continue_on_error` is enabled
- if `continue_on_error` is disabled, stop immediately after writing the failure record
- never erase partial outputs from a failed sample unless explicitly asked

Importantly, `finalize_sample()` must still run for samples that fail after output preparation begins so debug manifests are not lost.

## Testing Strategy

Add lightweight tests for the new batch layer without requiring heavy model inference.

### Unit Tests

- sample discovery from directory and manifest
- skip-existing behavior
- retry ladder chooses the expected next profile
- failed sample still writes status and summary records
- profile isolation between samples

### Integration-Style Smoke Tests

- batch runner calls the refined per-sample engine once per sample
- failure in sample A does not contaminate sample B
- sample-local fallback does not mutate the global default config for later samples

## Rollout Plan

Implementation should proceed in this order:

1. Add a batch-runner design-compatible CLI and batch manifest helpers.
2. Extend `RefinedOfflineApp` with sample reset and sample execution hooks.
3. Add sample discovery, skip-existing logic, and per-sample summaries.
4. Add retry ladder and runtime profile isolation.
5. Add batch smoke tests and documentation.

## Risks and Mitigations

Risk: batch mode accidentally changes the single-sample quality profile.
Mitigation: batch mode must call the same refined sample engine and only vary sample-local runtime profile values.

Risk: retries hide systematic quality problems.
Mitigation: record every retry and fallback reason in per-sample debug JSON and batch summary outputs.

Risk: long runs leak state across samples.
Mitigation: add explicit sample-state reset hooks and tests for profile/state isolation.

Risk: throughput pressure encourages silent low-quality shortcuts later.
Mitigation: make disallowed fallbacks explicit in code comments, spec, and tests.

## Open Decisions Resolved

These design choices are fixed for implementation:

- Batch mode will be added as a separate new script.
- The runner will reuse one loaded model stack but process one sample at a time.
- Stability fallbacks may reduce chunk size or reconstruction batch size, but may not silently weaken detector or refinement quality.
- The refined single-sample path remains the source of truth for quality behavior.
