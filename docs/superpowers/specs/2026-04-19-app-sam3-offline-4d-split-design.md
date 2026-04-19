# Split App SAM3 From Offline 4D

## Summary

This design separates the current `app.py` workflow into two explicit stages with a stable disk boundary between them:

1. `WebUI SAM3 annotation and mask propagation`
2. `Offline 4D execution from exported SAM3 cache`

The goal is to keep the interactive UI focused on human-in-the-loop segmentation work while moving 4D inference into a separate offline runner that can be repeated in batch without reopening the WebUI.

## Goals

- Make `app.py` responsible only for manual SAM3 target creation, prompt updates, mask propagation, and cache export.
- Move 4D execution behind a dedicated offline entry point that consumes exported cache directories.
- Preserve the SAM3 cache as a stable read-only contract between annotation and downstream 4D execution.
- Write 4D outputs into a separate `outputs_4d/<sample_id>/` directory tree so repeated runs do not mutate cache contents.
- Reuse as much of the existing 4D implementation as possible by extracting a shared pipeline core instead of duplicating logic.

## Non-Goals

- Rebuilding the full interactive WebUI session from disk.
- Persisting or restoring SAM3 tracker internals, `inference_state`, or other GPU runtime objects as part of the 4D contract.
- Redesigning the refined offline mask pipeline in this step.
- Introducing broad parameter-sweep or distributed batch orchestration in the first offline 4D runner version.

## Current State

The repository already has a usable SAM3 cache export contract:

- `scripts/sam3_cache_contract.py` validates cache structure and metadata.
- `scripts/sam3_cache_export.py` exports `images/`, `masks/`, `meta.json`, `prompts.json`, `frame_metrics.json`, and `events.json`.
- `app.py` now tracks enough export state to invalidate stale mask exports after prompt changes.

However, `app.py` still contains the full 4D execution body inside `on_4d_generation()`, and that implementation is still anchored to WebUI-centric globals such as `OUTPUT_DIR` and `RUNTIME`. The disk boundary exists for export, but the downstream 4D side has not yet been fully split away from the UI.

## Proposed Architecture

### Stage 1: WebUI SAM3 Export

`app.py` remains the interactive tool for:

- loading a video
- adding targets
- recording positive and negative prompts
- running mask propagation
- exporting a validated SAM3 cache

After a successful export, the output under `sam3_cache/<sample_id>/` becomes an immutable upstream artifact for downstream 4D execution.

### Stage 2: Offline 4D From Cache

A new offline script, `scripts/run_4d_from_cache.py`, becomes the supported way to run 4D from an exported cache. It will:

- validate the cache directory
- reconstruct the minimal 4D runtime profile from `meta.json`
- initialize the runtime app and model stack from the recorded config
- call a shared reusable 4D pipeline core
- write results into `outputs_4d/<sample_id>/`

### Shared 4D Pipeline Core

A new module, `scripts/app_4d_pipeline.py`, will own the shared 4D implementation. Its job is to:

- accept explicit input and output directories
- accept an explicit runtime profile and model context
- execute the current 4D logic without relying on WebUI globals

This shared core should be extracted from the existing offline-oriented implementation path, prioritizing the structure already present in `scripts/offline_app.py:on_4d_generation()` rather than treating `app.py` as the long-term source of truth.

## Responsibility Boundaries

### `app.py`

Owns:

- Gradio UI wiring
- video load flow
- click handling and target management
- SAM3 propagation
- SAM3 cache export

Does not own after this split:

- the primary 4D execution path
- batch 4D execution
- downstream 4D output management

### `scripts/sam3_cache_contract.py`

Owns:

- cache schema validation
- cache metadata loading
- path and discovery helpers as needed for offline consumption

### `scripts/sam3_cache_export.py`

Owns:

- export session lifecycle
- runtime export payload construction
- prompt, event, and frame metric serialization
- final cache bundle validation

### `scripts/app_4d_pipeline.py`

Owns:

- the reusable 4D pipeline implementation
- explicit `input_dir` and `output_dir` handling
- writing completion and rendering artifacts into the requested output tree

### `scripts/run_4d_from_cache.py`

Owns:

- cache validation before execution
- runtime profile reconstruction from `meta.json`
- output directory planning
- overwrite policy
- per-sample run summaries
- future batch discovery and iteration

## Data Flow

### WebUI To Cache

1. User loads a video in `app.py`.
2. User creates targets and updates prompts.
3. User runs mask generation.
4. WebUI writes `images/` and `masks/` into the current working directory.
5. User exports a SAM3 cache.
6. Export helper validates the cache contract and writes:
   - `images/`
   - `masks/`
   - `meta.json`
   - `prompts.json`
   - `frame_metrics.json`
   - `events.json`

### Cache To Offline 4D Result

1. User runs `scripts/run_4d_from_cache.py --cache_dir <cache>`.
2. Runner validates the cache directory before loading models.
3. Runner loads `meta.json` and reconstructs the minimal runtime profile needed for 4D.
4. Runner initializes the runtime app and model stack from the recorded config.
5. Runner calls the shared 4D pipeline with:
   - `input_dir = <cache_dir>`
   - `output_dir = <output_root>/<sample_id>`
6. Shared 4D pipeline reads `images/` and `masks/` from the cache and writes all downstream artifacts into the independent output directory.

## Input And Output Directory Rules

### Cache Root

Exported caches remain under:

```text
<CONFIG.runtime.output_dir>/sam3_cache/<sample_id>/
```

This directory is read-only from the perspective of offline 4D execution.

### 4D Output Root

Offline 4D results are written under:

```text
<CONFIG.runtime.output_dir>/outputs_4d/<sample_id>/
```

This keeps annotation artifacts and 4D artifacts separate while still making output discovery predictable.

### Hard Rules

- Cache directories are treated as immutable input once exported successfully.
- `sample_id` for offline 4D output is derived from the cache directory name.
- Offline 4D must never silently write results back into the cache root.
- Existing `outputs_4d/<sample_id>/` directories are not overwritten unless the user explicitly opts in.

## Minimal Runtime Reconstruction

The offline runner must reconstruct only the state required to execute 4D successfully. The required runtime values come from `meta.json`:

- `obj_ids`
- `batch_size`
- `detection_resolution`
- `completion_resolution`
- `smpl_export`
- `fps`
- `config_path`

The runner then rebuilds the model context from the recorded config and passes the following into the shared 4D core:

- loaded model objects
- runtime profile
- cache input directory
- output directory

### State The Runner Must Not Reconstruct

These values can remain as traceability artifacts but must not be required for execution:

- `prompt_log`
- `events`
- `frame_metrics`
- `clicks`
- `inference_state`
- live predictor tracking state
- Gradio component state

This boundary is intentional. The offline runner should consume a stable disk package, not simulate a WebUI restore.

## Offline Runner Interface

The first supported CLI shape should be intentionally narrow:

```bash
python scripts/run_4d_from_cache.py --cache_dir <sam3_cache/sample_id>
```

Supported options in the first version:

- `--cache_dir`
  Required. Single-sample cache directory to execute.
- `--output_root`
  Optional. Defaults to `<CONFIG.runtime.output_dir>/outputs_4d`.
- `--config`
  Optional. Overrides the config path recorded in `meta.json`.
- `--overwrite`
  Optional. Allows replacing an existing `outputs_4d/<sample_id>/`.

The internal implementation should still define reusable functions that make later batch support straightforward:

- `run_cache_sample(...)`
- `run_cache_batch(...)`

The public CLI should stay single-sample first until the shared core is stable.

## Error Handling

### Preflight Failures

If cache validation fails before execution, the runner must fail fast with specific, file- or field-level messages. Examples:

- missing `meta.json`
- missing required runtime profile fields
- mismatched `images/` and `masks/`
- missing frame stems
- invalid or unreadable config path
- empty or invalid `obj_ids`

These errors should happen before expensive model initialization whenever possible.

### Execution Failures

If 4D execution fails after validation succeeds:

- keep any generated output artifacts for debugging
- write a structured summary into the per-sample output directory
- do not mutate the source cache

Execution failures are expected to be debugged from the output directory, not by rewriting upstream cache artifacts.

## Overwrite Policy

Default behavior:

- if `outputs_4d/<sample_id>/` already exists, abort with a clear error

Explicit overwrite behavior:

- only remove and rebuild the output directory when `--overwrite` is provided

This default protects previously generated 4D results from accidental replacement during repeated batch runs.

## Run Summary

Each offline 4D execution should emit:

```text
outputs_4d/<sample_id>/run_summary.json
```

Minimum fields:

- `sample_id`
- `cache_dir`
- `status`
- `config_path`
- `started_at`
- `finished_at`
- `output_video`
- `error`

This summary belongs in the 4D output tree, not in the cache.

## UI Transition

The end-state boundary should be visible in the UI:

- `app.py` should no longer be a supported entry point for 4D execution

Recommended rollout:

1. First transition step:
   - keep the button temporarily if needed
   - replace direct execution with a clear message pointing users to `scripts/run_4d_from_cache.py`
2. Final step:
   - remove the `4D Generation` button from the WebUI entirely

The long-term goal is to make the product boundary obvious: WebUI prepares SAM3 cache, offline scripts run 4D.

## Testing Strategy

### Contract Layer Tests

Extend the current cache tests to verify:

- cache validation still passes for exported SAM3 bundles
- `sample_id` maps predictably to `outputs_4d/<sample_id>`
- overwrite and non-overwrite behaviors are enforced correctly

### Shared 4D Core Tests

Add focused tests for the shared pipeline module that verify:

- it reads `images/` and `masks/` from an explicit input directory
- it writes outputs only into the explicit output directory
- it does not require `app.py` globals such as `RUNTIME` or `OUTPUT_DIR`
- missing input artifacts fail with clear errors

### Offline Runner Tests

Add runner tests that verify:

- runtime profile reconstruction from `meta.json`
- default output path behavior
- refusal to overwrite by default
- successful execution under explicit overwrite
- per-sample summary writing

### Existing Regression Coverage

After the split, re-run the existing refined and export suites to confirm that:

- SAM3 cache export remains stable
- refined offline pipelines still work
- export-side tooling still behaves correctly

## Rollout Sequence

1. Extract the shared 4D pipeline core from the existing offline-oriented implementation path.
2. Add a single-sample offline cache runner that validates cache, rebuilds runtime state, and writes results into `outputs_4d/<sample_id>/`.
3. Transition `app.py` away from owning live 4D execution.
4. Add overwrite policy, run summaries, and focused unit tests.
5. Re-run export, refined, and offline regressions.

## Success Criteria

This split is successful only if all of the following are true:

- `app.py` still supports manual SAM3 annotation, mask generation, and cache export.
- offline 4D can execute from a single exported cache without a running WebUI process.
- 4D outputs land in `outputs_4d/<sample_id>/`.
- the original cache remains unchanged after offline 4D execution.
- the shared 4D logic no longer depends on `app.py` global state.
- existing export and refined regression tests continue to pass.

## Decision Summary

Chosen decisions:

- `app.py` keeps only the SAM3 front half
- offline 4D becomes a separate script
- cache is the stable read-only boundary
- 4D results are written into `outputs_4d/<sample_id>/`
- the reusable 4D core is extracted from the offline-oriented implementation path
- overwrite is opt-in only
- long-term UI direction is to remove the WebUI 4D button entirely
