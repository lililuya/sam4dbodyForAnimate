# Verify Output Video Comparison Design

## Summary

Build a lightweight verification script under `verify_output/` that takes a single exported target directory, reads its generated videos, and produces one combined comparison video for manual inspection.

The first version is intentionally narrow:

- Input is a single target result directory such as `5a179af6a1994ab3b78d694b24b136cf_face01_seg001_target1`
- Output is one stitched comparison `mp4`
- The comparison is designed for fast human review of alignment between `target.mp4` and `4d.mp4`
- Auxiliary videos are included when available to provide more context during review

This version does **not** implement JSON auditing or batch validation. Those can be added later if this manual verification loop proves useful.

## Goals

- Make it easy to visually inspect whether exported `4d.mp4` is aligned with `target.mp4`
- Preserve the original target resolution as the tile reference for all panels
- Avoid degrading the source panels via forced rescaling
- Keep the tool standalone and decoupled from the main export pipeline
- Support manual spot-checking of sampled outputs rather than mandatory full-dataset validation

## Non-Goals

- No automatic semantic correctness scoring
- No JSON correctness audit in the first version
- No batch directory scanning in the first version
- No hidden frame interpolation or silent frame count repair
- No support for videos that exceed the target tile size in the first version

## Inputs

The script accepts one required argument:

- `--input`: absolute or relative path to a single exported target directory

Expected files inside the directory:

- Required:
  - `target.mp4`
  - `4d.mp4`
- Optional:
  - `src_face.mp4`
  - `src_pose.mp4`
  - `src_bg.mp4`
  - `src_mask.mp4`
  - `src_mask_detail.mp4`

## Output

The script writes one output video to a new directory:

- `verify_output/<sample_dir_name>_compare.mp4`

The script may also print a short terminal summary with:

- target resolution
- target fps
- frame counts for each discovered video
- any missing optional panels

## Layout

The stitched video uses a fixed 3x3 grid:

- Row 1:
  - `target`
  - `4d`
  - `overlay(target + 4d)`
- Row 2:
  - `src_face`
  - `src_pose`
  - `src_bg`
- Row 3:
  - `src_mask`
  - `src_mask_detail`
  - blank panel

This layout keeps the primary alignment signal in the most visible row while preserving supporting context below.

## Resolution Policy

`target.mp4` defines the canonical tile size.

For every other panel:

- If the panel resolution is smaller than the target tile, preserve original pixels and center-pad it on a black canvas
- If the panel resolution exactly matches the target tile, use it directly
- If the panel resolution exceeds the target tile in either dimension, fail fast with a clear error

This preserves the target resolution standard without introducing hidden downscaling.

## Timebase And Frame Policy

`target.mp4` is the master timeline.

Rules:

- `target.mp4` and `4d.mp4` must have the same frame count
- `target.mp4` and `4d.mp4` should have matching fps; if they differ beyond a tiny tolerance, fail
- Optional panels may be shorter or missing
- Missing optional panels are represented as black frames with their label still visible
- No interpolation, duplication, or silent padding is applied to required panels

This keeps the first version honest: if the core pair is not aligned structurally, the script should make that obvious immediately.

## Encoding Strategy

The output format is `mp4`, as requested.

To avoid poor visual quality from OpenCV's default `mp4v`, the script should prefer `ffmpeg` with `libx264` and high-quality settings. The implementation may:

- render combined frames into a temporary directory or temporary lossless stream
- invoke `ffmpeg` to encode the final `mp4`

If `ffmpeg` is unavailable, the script should fail clearly rather than silently falling back to low-quality encoding.

## Panel Labels

Each panel should include a small text label in the upper-left corner:

- `target`
- `4d`
- `overlay`
- `src_face`
- `src_pose`
- `src_bg`
- `src_mask`
- `src_mask_detail`

The full composite should also include a thin top header with:

- sample directory name
- current frame index
- target fps
- target resolution

## Overlay Panel

The overlay panel blends `target` and `4d` at a fixed alpha, defaulting to roughly `0.5`.

This panel exists to quickly reveal:

- frame offset issues
- body drift
- large silhouette mismatches
- obvious temporal misalignment

## Error Handling

The script should fail with explicit messages for:

- missing `target.mp4`
- missing `4d.mp4`
- unreadable video files
- required panel frame count mismatch
- required panel fps mismatch
- any panel larger than the target tile size
- failure to invoke `ffmpeg`

Optional panels should not fail the run if absent.

## File Placement

New files for the first version:

- `verify_output/concat_compare_videos.py`
- optional `verify_output/README.md`

No existing export code needs to be changed for this first version.

## Testing Strategy

The first implementation should include targeted tests for:

- layout assembly with padding
- required video discovery
- optional video fallback to blank panels
- rejection of oversized panels
- rejection of target/4d frame count mismatch
- output path resolution

Tests should avoid heavy video encoding where possible by isolating frame assembly helpers from the final `ffmpeg` invocation.

## Future Extensions

If this first version works well, later iterations can add:

- batch directory scanning
- key-frame snapshot export
- JSON consistency checks
- simple numeric drift summaries
- side-by-side montage generation across multiple target directories
