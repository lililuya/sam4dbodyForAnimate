# Verify Output

## Compare One Exported Target Directory

Run:

```bash
python verify_output/concat_compare_videos.py --input path/to/sample_target1
```

This writes:

- `verify_output/<sample_dir_name>_compare.mp4`

## Expected Inputs

Required files inside the sample target directory:

- `target.mp4`
- `4d.mp4`

Optional context panels:

- `src_face.mp4`
- `src_pose.mp4`
- `src_bg.mp4`
- `src_mask.mp4`
- `src_mask_detail.mp4`

## Resolution Policy

- `target.mp4` defines the tile size
- smaller panels are center-padded on a black canvas
- panels larger than `target.mp4` cause the script to fail fast
