# Refined Max Target Selection Design

## Summary

Add a refined-only limit on automatically selected initial targets in `scripts/offline_app_refined.py`.
The refined runner should support a config default plus a CLI override:

- `detector.max_targets` in `configs/body4d_refined.yaml`
- `--max_targets` in `scripts/offline_app_refined.py`

The runner should keep at most the top `max_targets` detections ranked by detector confidence.
`0` or a negative value should mean "no limit".

This change must not alter:

- `app.py`
- `scripts/offline_app.py`
- `scripts/offline_app_export.py`
- `scripts/run_4d_from_cache.py`

## Goals

- Let refined offline runs cap the number of tracked people when crowded frames would otherwise create too many targets.
- Rank candidates by detector probability rather than bbox area or discovery order.
- Preserve the current refined workflow when no limit is configured.

## Non-Goals

- No changes to manual target selection in the UI.
- No changes to the baseline offline runner.
- No changes to cache-based offline 4D.
- No changes to the refined mask refinement algorithm itself.

## Current Behavior

`RefinedOfflineApp.detect_initial_targets()` scans the early search window, chooses the frame with the largest detection count, and then adds every detected box as a target.

Today it gets detections from `runtime_app.sam3_3d_body_model.process_one_image(...)`.
That path currently exposes boxes in the final output but does not preserve detector confidence in the returned person records.
As a result, refined code cannot currently do a true top-k by detector probability.

## Proposed Design

### 1. Config and CLI surface

Add a refined-only config key:

```yaml
detector:
  max_targets: 0
```

Interpretation:

- `max_targets <= 0`: keep all detections
- `max_targets > 0`: keep the top `max_targets` detections by score

Add a CLI override:

```bash
python scripts/offline_app_refined.py --max_targets 3
```

CLI must override config in `apply_runtime_overrides(...)`.

### 2. Refined-only selection point

Apply the limit in `RefinedOfflineApp.detect_initial_targets()`, after the search frame has been chosen and before prompts are added to SAM3 tracking.

This preserves the current search logic:

- still scan the first `batch.initial_search_frames`
- still choose the frame with the largest detection count
- then rank detections from that chosen frame by detector confidence
- then keep only the first `max_targets`

This is the smallest behavior change that matches the request.

### 3. Detector confidence propagation

To support real probability-based ranking, extend the detector plumbing so refined code can access detection scores without changing the default behavior for existing callers.

Recommended shape:

- add an optional scored-detection mode to detector wrappers
- normalize each detection as:

```python
{
    "bbox": [x1, y1, x2, y2],
    "score": 0.93,
}
```

Implementation boundary:

- detector wrappers may need small internal changes to surface scores
- only `offline_app_refined.py` will consume the new score-ranked list for target limiting
- existing callers that only need boxes should continue to work unchanged

For YOLO:

- use `results[0].boxes.conf`

For ViTDet:

- use `det_instances.scores`

### 4. Fallback behavior

If scored detections are unavailable for any reason:

- do not fail the refined run
- fall back to the current detection order
- still apply the top-k limit to that order

This keeps the feature robust even if a backend does not provide confidence exactly as expected.

## Data Flow

1. Load refined config.
2. Apply CLI overrides, including `--max_targets`.
3. In `detect_initial_targets()`:
   - gather detections for each candidate search frame
   - choose the frame with the highest detection count
   - sort detections from that frame by descending `score`
   - trim to `max_targets` when `max_targets > 0`
4. Add the selected boxes to SAM3 tracking as `obj_id = 1..N`.
5. Continue with refined mask generation and 4D as before.

## Error Handling

- Reject non-integer CLI values through argparse typing.
- Treat `max_targets <= 0` as unlimited instead of erroring.
- If score extraction fails, continue with stable fallback ordering rather than crashing.

## Testing

Add focused tests for:

- config default is accepted
- CLI override updates `cfg.detector.max_targets`
- refined detection selection keeps all detections when `max_targets <= 0`
- refined detection selection keeps only top-k detections when scores are present
- refined detection selection falls back safely when scores are missing

## Files Expected to Change

- `configs/body4d_refined.yaml`
- `scripts/offline_app_refined.py`
- possibly detector helper files if score propagation needs a small compatibility layer:
  - `models/sam_3d_body/tools/build_detector.py`
  - `models/sam_3d_body/tools/build_detector_yolo.py`
- refined tests covering target selection behavior

## Why This Design

This design keeps the feature where it belongs: the refined automatic target bootstrap path.
It avoids spreading target-limit behavior into unrelated entrypoints while still using real detector confidence when available.
It also preserves the current user mental model:

- config gives a default policy
- CLI can override it for one run
- no limit means current behavior
