# Offline Refined Mask Pipeline Design

## Overview

This design adds a new refined offline pipeline for SAM-Body4D that improves automatic mask quality under heavy occlusion while preserving the existing offline entrypoint. The new work must not overwrite or change the behavior of the current `scripts/offline_app.py`.

The refined pipeline targets these constraints:

- Fully automatic batch processing with no manual prompts.
- Better mask quality in occlusion-heavy sequences.
- Higher robustness when tracks drift or reappear after occlusion.
- A new YOLO-based bbox detector option for automatic prompting.
- Clear debug artifacts so bad cases can be diagnosed offline.

The implementation will create a new script and supporting modules rather than replacing the current offline flow.

## Goals

- Add a new offline entrypoint that preserves the current script as a baseline.
- Improve amodal mask quality by reusing the stronger two-stage occlusion refinement logic from `app.py`.
- Add automatic re-prompting based on drift and occlusion heuristics.
- Add a YOLO detector backend that can be selected for initial prompts and re-prompts.
- Keep the pipeline compatible with batch runs over many videos or frame folders.
- Save enough intermediate outputs and metrics to debug failure modes at scale.

## Non-Goals

- No changes to the behavior of the existing `scripts/offline_app.py`.
- No manual annotation, no interactive prompts, and no GUI-only workflow.
- No training or finetuning of new segmentation models.
- No repository-wide detector refactor beyond the minimum needed for the new path.

## User-Facing Entry Points

### New Script

Add a new script:

- `scripts/offline_app_refined.py`

This script will be the refined offline pipeline and will coexist with the current:

- `scripts/offline_app.py`

### CLI Shape

The new script should keep the familiar arguments and add a small set of new controls.

Baseline-compatible arguments:

- `--input_video`
- `--output_dir`

New arguments:

- `--config`: path to the refined config file
- `--detector_backend`: detector choice, defaulting to YOLO for the refined path
- `--track_chunk_size`: maximum frames tracked per chunk before re-initialization
- `--disable_auto_reprompt`: disable automatic re-prompting for debugging
- `--save_debug_metrics`: save JSON diagnostics and intermediate masks
- `--skip_existing`: skip samples whose output directory already contains a completion marker

If the input is a single `mp4` or a single frame directory, the script behaves like a refined single-sample runner. The internal design should also make it easy to add a dataset-level wrapper later without reworking the core logic.

## Architecture

### High-Level Flow

The refined script will follow this sequence:

1. Load config and initialize models once.
2. Build SAM3 tracker and SAM-3D-Body estimator.
3. Build the selected detector backend, with YOLO supported as a first-class option.
4. Detect initial humans on the first valid frame.
5. Track masks in chunks rather than forcing a single unbounded tracking run.
6. Save raw masks and images for each tracked chunk.
7. Run two-stage occlusion refinement using the stronger logic from `app.py`.
8. Trigger automatic re-prompting when drift signals exceed thresholds.
9. Run 4D reconstruction using refined masks.
10. Save outputs, debug masks, per-object metrics, and re-prompt events.

### Main Components

The refined pipeline will be split into small units with clear boundaries:

- `scripts/offline_app_refined.py`
  - CLI parsing
  - high-level orchestration
  - batch-safe sample loop
- `models/sam_3d_body/tools/build_detector_yolo.py`
  - YOLO model loading
  - bbox inference
  - bbox filtering and sorting
- `utils` or a new helper module under `scripts/`
  - drift metrics
  - auto re-prompt decision rules
  - per-chunk bookkeeping
- refined mask helpers
  - low-resolution occlusion screening
  - high-resolution amodal mask refinement
  - IoU-based temporal filtering

## Detector Design

### Why YOLO

The current detector path is pluggable but only implements ViTDet in `models/sam_3d_body/tools/build_detector.py`. A YOLO backend is a good fit for the refined offline flow because:

- it is commonly used for robust person detection,
- it is easy to run on a single frame or sparse keyframes,
- it is suitable for automatic re-prompting after drift,
- it can be used fully offline with local weights.

### Detector Scope

YOLO support should be added without changing the existing offline baseline behavior.

Implementation shape:

- Add a new module `models/sam_3d_body/tools/build_detector_yolo.py`.
- Extend `HumanDetector` to support a new backend name such as `yolo` or `yolo11`.
- Keep `vitdet` as the default in existing code paths.
- Set the refined script default detector to YOLO.

### Detector Interface

The YOLO backend must match the existing detector contract:

- input: image array in OpenCV BGR format
- output: `numpy.ndarray` of boxes shaped `(N, 4)` in `xyxy`

The backend should:

- filter to the `person` class only,
- apply configurable confidence thresholding,
- apply configurable NMS if needed,
- sort boxes deterministically for stable prompt ordering,
- optionally fall back to full-image behavior only when explicitly requested by the caller.

### Dependency Handling

The repository currently does not list a YOLO package in `pyproject.toml`. The refined design should therefore:

- add the new dependency explicitly,
- support local model weights from config,
- avoid any hidden runtime download requirement.

If the runtime package is `ultralytics`, the config must allow:

- model file path
- confidence threshold
- IoU threshold
- max detections
- device selection

## Refined Mask Design

### Current Limitation

The existing `scripts/offline_app.py` uses a simplified occlusion flow:

- one low-resolution amodal mask pass,
- thresholding and simple heuristics,
- direct use of the resulting masks for later stages.

This is weaker than the richer logic already present in `app.py`.

### Refined Strategy

The new pipeline will reuse the stronger two-stage mask logic already proven in `app.py`.

Stage 1:

- Run low-resolution amodal mask inference over the batch window.
- Compute modal-versus-amodal IoU per object and per frame.
- Detect candidate occlusion windows.
- Filter obvious false completions using area and shape heuristics.

Stage 2:

- For the candidate occlusion windows only, rerun amodal mask inference at higher resolution.
- Use the IoU-aware temporal filter from `cap_consecutive_ones_by_iou`.
- Save only trusted refined masks for the heavily occluded spans.

Why this helps:

- low-resolution inference remains cheap enough for broad screening,
- high-resolution inference is focused only where needed,
- long noisy occlusion spans are constrained by temporal ranking instead of blindly trusting every frame.

### Expected Code Reuse

The refined script should reuse or extract equivalent logic from these `app.py` parts:

- `cap_consecutive_ones_by_iou`
- `mask_completion_and_iou_init`
- `mask_completion_and_iou_final`
- the refined `on_4d_generation` occlusion branch

The goal is behavior parity with the stronger path, not copy-paste drift between two unrelated implementations.

## Automatic Re-Prompt Design

### Purpose

The current offline path provides one initial box prompt and then relies on propagation. This is fragile for long sequences and strong occlusions. The refined path should automatically recover when the track drifts.

### Trigger Signals

Automatic re-prompting should be driven by conservative rules combining multiple signals:

- repeated empty masks
- sudden area collapse relative to recent history
- long stretches where the object mask hugs image borders
- persistent modal-versus-amodal IoU anomalies
- confidence drop or implausible bbox change after occlusion recovery

The default behavior should avoid excessive resets. Re-prompting is a recovery mechanism, not the steady-state path.

### Re-Prompt Mechanism

When a chunk enters a drift-risk state:

1. Pick a keyframe near the failure point.
2. Re-run the chosen detector on that frame.
3. Match detected boxes to existing tracked identities using simple spatial overlap and proximity heuristics.
4. Inject a new box prompt for the affected identity.
5. Continue propagation from that frame onward.

### Chunking Strategy

Tracking should be performed in bounded chunks rather than a single unbounded run. Chunking reduces accumulated drift and also creates natural places to evaluate recovery logic.

Default behavior:

- track up to `track_chunk_size` frames,
- persist chunk outputs to shared sample directories,
- run recovery checks at chunk boundaries and at local failure points,
- allow chunk overlap if needed for future tuning, but keep the first version simple.

## Output Layout

The refined script should keep outputs separated from the baseline and save more diagnostics.

Suggested layout inside one sample output directory:

- `images/`
- `masks_raw/`
- `masks_refined/`
- `completion_refined/images/`
- `completion_refined/masks/`
- `rendered_frames/`
- `rendered_frames_individual/`
- `mesh_4d_individual/`
- `focal_4d_individual/`
- `debug_metrics/summary.json`
- `debug_metrics/per_object_iou.json`
- `debug_metrics/reprompt_events.json`
- `debug_metrics/chunk_manifest.json`

This layout is meant to support large-scale offline diagnosis without needing to rerun the full job just to inspect a bad result.

## Configuration

Add a new config file:

- `configs/body4d_refined.yaml`

This keeps the refined pipeline isolated from the baseline configuration.

Suggested sections:

- `paths`
- `sam3`
- `sam_3d_body`
- `runtime`
- `completion`
- `detector`
- `tracking`
- `reprompt`
- `debug`

Important refined-only keys:

- detector backend and weights path
- detector thresholds
- chunk size
- re-prompt enable flag
- re-prompt trigger thresholds
- debug artifact flags

## Error Handling

The refined script should fail clearly and early for:

- missing detector weights
- unsupported detector backend names
- empty input directories
- invalid config paths
- sample outputs with mismatched image and mask counts

Recovery-oriented behavior:

- if one chunk fails, write an error record for that chunk and continue to the next sample only when `runtime.continue_on_error` is enabled,
- save enough context in debug JSON so failed samples can be resumed or inspected,
- never silently fall back from refined logic to the baseline path.

## Testing Strategy

The first implementation should prioritize small deterministic tests for the added control logic plus one smoke path for the new script.

Unit-level tests:

- YOLO detector adapter returns sorted `xyxy` boxes.
- detector filtering keeps only persons.
- temporal keep logic selects the correct frames.
- drift signal logic triggers and does not trigger on representative sequences.
- re-prompt matcher selects the correct detection for a tracked identity.

Integration-level smoke tests:

- refined script CLI parses correctly,
- refined script can initialize with YOLO config,
- chunk bookkeeping writes expected manifests,
- refined path preserves outputs in separate directories from the baseline.

Because model-heavy inference is expensive, full end-to-end tests can stay lightweight and focus on orchestration behavior with stubs or tiny fixtures where possible.

## Rollout Plan

Implementation should proceed in this order:

1. Add the refined config and script shell.
2. Add the YOLO detector backend.
3. Port the stronger two-stage mask refinement logic.
4. Add chunked tracking and re-prompt metrics.
5. Add automatic re-prompting.
6. Add debug artifacts and smoke tests.

This order keeps the pipeline usable early and reduces the risk of debugging too many moving parts at once.

## Risks and Mitigations

Risk: YOLO improves recall but changes box ordering or duplicate behavior.
Mitigation: deterministic sorting and explicit filtering.

Risk: Auto re-prompting causes identity switches.
Mitigation: conservative trigger thresholds and overlap-based matching.

Risk: High-resolution occlusion refinement is too slow.
Mitigation: run it only on candidate occlusion windows rather than every frame.

Risk: The refined path drifts away from the stronger online implementation again.
Mitigation: extract shared helpers where practical instead of duplicating logic.

## Open Decisions Resolved

These design choices are fixed for implementation:

- We will create a new script rather than overwrite the old one.
- We will follow the design path that keeps the baseline untouched.
- We will add YOLO detection as the refined default detector option.
- We will preserve a fully automatic workflow with no manual prompts.
- We will prioritize robustness under occlusion over minimal code churn.
