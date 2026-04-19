# SAM3 Cache Contract For Offline 4D

## Summary

This design splits the current `app.py` workflow into two independent stages:

1. `SAM3 interactive annotation and mask propagation`
2. `Offline 4D batch inference from exported cache`

The exported cache is not a UI session restore format. It is a stable, file-based contract that preserves enough data for:

- direct offline 4D execution without WebUI runtime state
- post-hoc debugging and quality tracing
- future continuation of annotation work if needed

The design favors stable disk artifacts over fragile runtime internals. The exported cache must be sufficient for 4D to run successfully even when the WebUI process is not present.

## Goals

- Make `app.py` responsible only for manual SAM3-assisted annotation, target management, propagation, and cache export.
- Make offline 4D runnable from exported cache directories alone.
- Preserve enough metadata and diagnostics to explain why a sample later succeeds or fails.
- Keep the cache format explicit, versioned, and batch-friendly.
- Avoid dependence on WebUI-only in-memory state such as Gradio state, live predictor objects, or transient GPU tensors.

## Non-Goals

- Restoring the entire interactive WebUI process state bit-for-bit.
- Persisting raw GPU runtime state, `predictor` internals, or `inference_state` as a required offline contract.
- Making the first version of the cache contract support every future debug artifact by default.
- Replacing the refined offline pipeline immediately. The first step is to define a stable export and consumption boundary.

## Current State

The current `app.py` flow already has a natural split:

- interactive clicks call `predictor.add_new_points_or_box(...)`
- mask propagation writes `images/` and `masks/`
- 4D reads those disk outputs from `OUTPUT_DIR`

The problem is that the contract is implicit. Offline 4D still relies on process-local state such as `RUNTIME['out_obj_ids']`, `RUNTIME['video_fps']`, and other configuration values that are not formalized as export artifacts. That makes it awkward to use the manual SAM3 stage as a standalone upstream batch-preparation step.

## Proposed Architecture

### Stage 1: Interactive SAM3 Export

`app.py` remains the human-in-the-loop tool. Its responsibilities are:

- load a video
- create and edit targets
- collect positive and negative prompts
- propagate SAM3 masks through the sequence
- validate and optionally refine mask quality
- export a cache directory that satisfies the 4D contract

At the end of this stage, the operator should have a deterministic on-disk sample package.

### Stage 2: Offline 4D From Cache

A separate offline entry point reads the exported cache directory and performs the current 4D work:

- optional completion and occlusion handling
- body estimation
- rendering
- mesh and focal export
- final 4D video rendering

This stage must not depend on WebUI runtime globals.

## Cache Contract

Each sample is exported as:

```text
sam3_cache/
  <sample_id>/
    images/
      00000000.jpg
      ...
    masks/
      00000000.png
      ...
    meta.json
    prompts.json
    frame_metrics.json
    events.json
```

### Required For 4D Execution

These are mandatory. If any of them are missing or inconsistent, offline 4D must refuse to run.

#### `images/`

- consecutive frame filenames
- one image per frame
- stable extension recorded in `meta.json`
- spatial size must match mask size

#### `masks/`

- consecutive frame filenames aligned with `images/`
- indexed masks with stable object ids across the sequence
- palette-preserving export for inspection, but the contract is the integer labels

#### `meta.json`

This is the authoritative runtime contract. Minimum fields:

```json
{
  "cache_version": 1,
  "sample_id": "example_sample",
  "source_video": "/abs/path/to/video.mp4",
  "frame_count": 165,
  "fps": 24.0,
  "image_ext": ".jpg",
  "mask_ext": ".png",
  "frame_stems": ["00000000", "00000001"],
  "image_size": {"width": 1280, "height": 720},
  "obj_ids": [1, 2, 3],
  "runtime_profile": {
    "batch_size": 32,
    "detection_resolution": [256, 512],
    "completion_resolution": [512, 1024],
    "smpl_export": false
  },
  "config": {
    "config_path": "configs/body4d.yaml"
  },
  "exported_at": "2026-04-19T16:30:00+08:00"
}
```

Offline 4D should restore all required runtime knobs from this file instead of relying on WebUI globals.

### Enhanced Traceability Artifacts

These are not strictly required to execute 4D, but they are part of the chosen "enhanced traceability" design.

#### `prompts.json`

Purpose:

- capture how the final masks were produced
- support future annotation continuation or audit

Recommended structure:

```json
{
  "targets": {
    "1": {
      "name": "Target 1",
      "frames": {
        "12": {
          "points": [[341, 280], [365, 412]],
          "labels": [1, 0]
        }
      }
    }
  }
}
```

#### `frame_metrics.json`

Purpose:

- diagnose whether failures originate in SAM3 segmentation or later 4D stages
- support later quality filtering or selective re-annotation

Recommended per-frame, per-object fields:

- `bbox_xyxy`
- `mask_area`
- `empty_mask_count`
- `edge_touch_ratio`
- `mask_iou`
- `refined_from_previous`
- `source`
  - for example: `interactive_prompt`, `sam3_propagation`, `manual_override`

#### `events.json`

Purpose:

- maintain a compact human-readable event history
- explain how the sample evolved during annotation

Recommended event types:

- `video_loaded`
- `target_added`
- `prompt_updated`
- `mask_generation_started`
- `mask_generation_completed`
- `quality_warning`
- `cache_export_completed`

## Data Flow

### Export Path

1. Operator loads a video in `app.py`.
2. Operator adds targets and clicks positive or negative prompts.
3. `predictor.add_new_points_or_box(...)` updates the segmentation state.
4. Mask propagation runs across the video.
5. `images/` and `masks/` are written.
6. The export layer writes `meta.json`, `prompts.json`, `frame_metrics.json`, and `events.json`.
7. The sample is marked as 4D-runnable only if validation passes.

### Offline 4D Path

1. Batch runner discovers cache directories.
2. Runner validates the cache contract before inference.
3. Runner reconstructs the required runtime profile from `meta.json`.
4. Runner reads `images/` and `masks/`.
5. Runner executes the existing 4D path against those disk inputs.
6. 4D outputs are written into a separate result directory, not back into the source cache directory unless explicitly configured.

## Validation Rules

Before a cache is accepted as 4D-runnable, validate:

- `images/` and `masks/` both exist
- `frame_count` matches actual file count
- frame stems are continuous and aligned
- every image has a matching mask
- image and mask sizes match
- all `obj_ids` listed in `meta.json` appear in at least one mask
- runtime profile fields required by 4D are present

If validation fails, export should either:

- refuse to mark the cache as complete, or
- write a structured validation failure into `events.json`

## Error Handling

### Export Errors

If SAM3 export is incomplete:

- do not produce a falsely "complete" cache
- record the last successful export step
- emit a validation failure entry

### Offline 4D Errors

If 4D cannot run from a cache:

- fail fast during cache validation when possible
- report which required file or field is missing
- preserve a structured error summary for batch reporting

## Storage Decisions

### Persist By Default

- final `images/`
- final `masks/`
- `meta.json`
- `prompts.json`
- `frame_metrics.json`
- `events.json`

### Do Not Depend On For Contract Validity

- `predictor`
- `inference_state`
- GPU tensors
- low-level SAM3 runtime embeddings or logits

These may be optionally dumped under a separate debug path in the future, but they must not be required to run offline 4D.

## Testing Strategy

### Export-Side Tests

- export creates a valid cache directory with all required files
- `meta.json` reflects actual file counts and runtime profile
- prompt logs preserve positive and negative point data
- frame metrics align with exported frame stems

### Validation Tests

- offline validator rejects missing `meta.json`
- offline validator rejects mismatched image and mask counts
- offline validator rejects gaps in frame numbering
- offline validator rejects missing runtime profile fields

### 4D Consumption Tests

- offline 4D runner reconstructs runtime settings from `meta.json`
- offline 4D runner reads masks without WebUI globals
- batch execution can process multiple cache directories in sequence

## Implementation Boundaries

### `app.py`

Additions:

- explicit export step for SAM3 cache
- cache validation before marking export complete
- metadata and event serialization

No longer required for downstream 4D:

- Gradio state
- live WebUI callbacks
- in-memory click buffers after export is complete

### New Offline Entry Point

Introduce a separate script, for example:

`scripts/run_4d_from_cache.py`

Responsibilities:

- discover caches
- validate cache contract
- load runtime profile from `meta.json`
- invoke the existing 4D logic against exported `images/` and `masks/`
- summarize failures per sample

## Rollout Plan

1. Define and validate the cache contract in code.
2. Add export support in `app.py`.
3. Add offline cache validator.
4. Add `run_4d_from_cache.py`.
5. Add batch runner support over cache directories.
6. Only then consider optional richer debug dumps.

## Decision Summary

Chosen direction:

- use a stable on-disk cache contract, not full WebUI session restore
- make the cache directly sufficient for offline 4D
- include enhanced traceability artifacts by default
- avoid depending on fragile runtime internals

This gives a practical boundary: manual SAM3 work happens once, then offline 4D can be rerun in batch from exported cache directories as many times as needed.
