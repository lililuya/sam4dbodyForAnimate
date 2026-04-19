from typing import Iterable, List, Optional, Tuple


def build_completion_window_from_ious(
    ious: Iterable[float],
    padding: int = 2,
    iou_threshold: float = 0.7,
) -> Tuple[List[int], Optional[Tuple[int, int]]]:
    values = [float(iou) for iou in ious]
    padding = max(0, int(padding))

    occ_flags = [0 if value < float(iou_threshold) else 1 for value in values]
    occluded_indices = [idx for idx, flag in enumerate(occ_flags) if flag == 0]
    if not occluded_indices:
        return occ_flags, None

    start = max(0, occluded_indices[0] - padding)
    end = min(len(values), occluded_indices[-1] + padding + 1)
    return occ_flags, (start, end)
