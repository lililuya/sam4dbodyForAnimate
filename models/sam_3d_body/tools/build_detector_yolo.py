import numpy as np


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


def extract_person_boxes(boxes, bbox_thr: float, det_cat_id: int = 0) -> np.ndarray:
    xyxy = _to_numpy_float32(boxes.xyxy)
    conf = _to_numpy_float32(boxes.conf)
    cls = _to_numpy_float32(boxes.cls)
    keep = (cls == float(det_cat_id)) & (conf >= float(bbox_thr))
    filtered = xyxy[keep]
    return sort_boxes_xyxy(filtered)


def load_ultralytics_yolo(path: str = "", weights_path: str = ""):
    if path and weights_path and path != weights_path:
        raise ValueError("Conflicting YOLO paths: provide either path or weights_path")
    selected_path = weights_path or path
    if not selected_path:
        raise FileNotFoundError(
            "YOLO path or weights_path must be set for offline refined runs"
        )
    from ultralytics import YOLO

    return YOLO(selected_path)


def run_ultralytics_yolo(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.35,
    nms_thr: float = 0.50,
    default_to_full_image: bool = False,
    device=None,
    max_det: int = 20,
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
    boxes = extract_person_boxes(results[0].boxes, bbox_thr=bbox_thr, det_cat_id=det_cat_id)
    if boxes.size == 0 and default_to_full_image:
        return np.array([[0, 0, width, height]], dtype=np.float32)
    return boxes
