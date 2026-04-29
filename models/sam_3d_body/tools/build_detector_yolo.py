import numpy as np

DEFAULT_YOLO_WEIGHTS = "yolo11n.pt"
DEFAULT_YOLO_BBOX_THRESH = 0.25
DEFAULT_YOLO_NMS_THRESH = 0.70
DEFAULT_YOLO_MAX_DET = 300


def _to_numpy_float32(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach()
    if hasattr(values, "cpu") and callable(values.cpu):
        values = values.cpu()
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def sort_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    sorted_indices = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[sorted_indices].astype(np.float32)


def _sort_boxes_and_scores_xyxy(boxes: np.ndarray, scores: np.ndarray):
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32), scores.reshape(0).astype(np.float32)
    sorted_indices = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[sorted_indices].astype(np.float32), scores[sorted_indices].astype(np.float32)


def extract_person_boxes(boxes, bbox_thr: float, det_cat_id: int = 0, return_scores: bool = False):
    xyxy = _to_numpy_float32(boxes.xyxy)
    conf = _to_numpy_float32(boxes.conf)
    cls = _to_numpy_float32(boxes.cls)
    keep = (cls == float(det_cat_id)) & (conf >= float(bbox_thr))
    filtered_boxes = xyxy[keep]
    filtered_scores = conf[keep]
    filtered_boxes, filtered_scores = _sort_boxes_and_scores_xyxy(filtered_boxes, filtered_scores)
    if return_scores:
        return [
            {"bbox": box.astype(np.float32).tolist(), "score": float(score)}
            for box, score in zip(filtered_boxes, filtered_scores)
        ]
    return filtered_boxes


def resolve_yolo_weights_path(
    path: str = "",
    weights_path: str = "",
    default_weights: str = DEFAULT_YOLO_WEIGHTS,
) -> str:
    if path and weights_path and path != weights_path:
        raise ValueError("Conflicting YOLO paths: provide either path or weights_path")
    selected_path = weights_path or path or default_weights
    if not selected_path:
        raise FileNotFoundError(
            "YOLO path, weights_path, or default_weights must be set for offline refined runs"
        )
    return selected_path


def load_ultralytics_yolo(
    path: str = "",
    weights_path: str = "",
    default_weights: str = DEFAULT_YOLO_WEIGHTS,
):
    selected_path = resolve_yolo_weights_path(
        path=path,
        weights_path=weights_path,
        default_weights=default_weights,
    )
    from ultralytics import YOLO

    print(f"########### Loading YOLO weights: {selected_path}")
    return YOLO(selected_path)


def run_ultralytics_yolo(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = DEFAULT_YOLO_BBOX_THRESH,
    nms_thr: float = DEFAULT_YOLO_NMS_THRESH,
    default_to_full_image: bool = False,
    device=None,
    max_det: int = DEFAULT_YOLO_MAX_DET,
    return_scores: bool = False,
):
    height, width = img.shape[:2]
    predict_kwargs = {
        "source": img,
        "conf": bbox_thr,
        "iou": nms_thr,
        "max_det": max_det,
        "verbose": False,
    }
    if device is not None:
        predict_kwargs["device"] = device
    results = detector.predict(
        **predict_kwargs,
    )
    boxes = extract_person_boxes(
        results[0].boxes,
        bbox_thr=bbox_thr,
        det_cat_id=det_cat_id,
        return_scores=return_scores,
    )
    if return_scores:
        if not boxes and default_to_full_image:
            return [{"bbox": [0.0, 0.0, float(width), float(height)], "score": None}]
        return boxes
    if boxes.size == 0 and default_to_full_image:
        return np.array([[0, 0, width, height]], dtype=np.float32)
    return boxes
