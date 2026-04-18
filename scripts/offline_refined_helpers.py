from typing import List, Optional, Sequence, Tuple


def cap_consecutive_ones_by_iou(
    flag: Sequence[int],
    iou: Sequence[float],
    max_keep: int = 3,
) -> List[int]:
    """Flip each contiguous run of ones to keep only the top-IoU indices.

    Output semantics are intentionally asymmetric:
    - input `flag == 0` becomes output `1`
    - input `flag == 1` becomes output `1` only for selected run members
    """
    n = len(flag)
    if len(iou) != n:
        raise ValueError(f"len(flag)={n} != len(iou)={len(iou)}")

    out = [1 if flag[i] == 0 else 0 for i in range(n)]
    i = 0
    while i < n:
        if flag[i] != 1:
            i += 1
            continue
        j = i
        while j < n and flag[j] == 1:
            j += 1
        run_idx = list(range(i, j))
        if len(run_idx) <= max_keep:
            for k in run_idx:
                out[k] = 1
        else:
            top = sorted(run_idx, key=lambda k: (-float(iou[k]), k))[:max_keep]
            for k in top:
                out[k] = 1
        i = j
    return out


def smooth_equal_threshold_hits(iou_values: Sequence[float], threshold: float) -> List[float]:
    """Suppress isolated exact-threshold hits that are between two low values."""
    arr = [float(x) for x in iou_values]
    for idx in range(1, len(arr) - 1):
        if arr[idx] == threshold and arr[idx - 1] < threshold and arr[idx + 1] < threshold:
            arr[idx] = 0.0
    return arr


def find_occlusion_window(
    iou_values: Sequence[float],
    threshold: float,
    total_frames: int,
    pad: int = 2,
) -> Optional[Tuple[int, int]]:
    """Return `(start, end)` where `end` is exclusive for Python slicing."""
    low = [idx for idx, value in enumerate(iou_values) if float(value) < float(threshold)]
    if not low:
        return None
    start = max(0, low[0] - pad)
    end = min(total_frames, low[-1] + pad + 1)
    return start, end
