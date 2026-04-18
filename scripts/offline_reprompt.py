from typing import Any, Mapping, Optional, Sequence


def iou_xyxy(a, b) -> float:
    """Compute IoU for two boxes in ``[x1, y1, x2, y2]`` format."""
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def should_trigger_reprompt(metrics, thresholds) -> bool:
    """Return whether current tracking metrics indicate prompt drift."""
    empty_mask_count = _first_number(
        metrics,
        "empty_mask_count",
        "empty_mask_streak",
        "consecutive_empty_masks",
        default=0,
    )
    area_ratio = _first_number(metrics, "area_ratio", "mask_area_ratio", default=1.0)
    edge_touch_ratio = _first_number(metrics, "edge_touch_ratio", default=0.0)
    mask_iou = _first_number(metrics, "mask_iou", "iou", "box_iou", default=1.0)

    empty_mask_patience = int(_first_number(thresholds, "empty_mask_patience", default=0))
    area_drop_ratio = _first_number(thresholds, "area_drop_ratio", default=0.0)
    edge_touch_limit = _first_number(thresholds, "edge_touch_ratio", default=1.0)
    iou_low_threshold = _first_number(thresholds, "iou_low_threshold", default=0.0)

    return any(
        [
            empty_mask_patience > 0 and empty_mask_count >= empty_mask_patience,
            area_ratio <= area_drop_ratio,
            edge_touch_ratio >= edge_touch_limit,
            mask_iou <= iou_low_threshold,
        ]
    )


def match_detection_to_track(prev_box, candidates):
    """Return the candidate with the highest overlap against ``prev_box``."""
    best_candidate = None
    best_iou = 0.0
    for candidate in candidates:
        candidate_box = _extract_box(candidate)
        if candidate_box is None:
            continue
        overlap = iou_xyxy(prev_box, candidate_box)
        if overlap > best_iou:
            best_iou = overlap
            best_candidate = candidate
    return best_candidate


def _first_number(source: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in source and source[key] is not None:
            return float(source[key])
    return float(default)


def _extract_box(candidate: Any) -> Optional[Sequence[float]]:
    if isinstance(candidate, Mapping):
        for key in ("bbox", "box", "xyxy"):
            if key in candidate and candidate[key] is not None:
                return _normalize_box(candidate[key])
        return None
    return _normalize_box(candidate)


def _normalize_box(box: Any) -> Optional[Sequence[float]]:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
        return None
    if len(box) != 4:
        return None
    try:
        return [float(value) for value in box]
    except (TypeError, ValueError):
        return None
