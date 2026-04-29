OFFICIAL_DETECTOR_DEFAULTS = {
    "yolo": {
        "bbox_thresh": 0.25,
        "iou_thresh": 0.70,
        "max_det": 300,
    },
    "yolo11": {
        "bbox_thresh": 0.25,
        "iou_thresh": 0.70,
        "max_det": 300,
    },
    "vitdet": {
        "bbox_thresh": 0.05,
        "iou_thresh": 0.50,
        "max_det": None,
    },
}


def normalize_detector_backend(detector_backend: str) -> str:
    backend = str(detector_backend or "vitdet").strip().lower()
    if backend not in OFFICIAL_DETECTOR_DEFAULTS:
        raise ValueError(f"unsupported detector backend: {detector_backend}")
    return backend


def resolve_detector_runtime_options(
    detector_backend: str,
    bbox_thresh=None,
    iou_thresh=None,
    max_det=None,
):
    backend = normalize_detector_backend(detector_backend)
    defaults = OFFICIAL_DETECTOR_DEFAULTS[backend]
    return {
        "bbox_thresh": float(defaults["bbox_thresh"] if bbox_thresh is None else bbox_thresh),
        "iou_thresh": float(defaults["iou_thresh"] if iou_thresh is None else iou_thresh),
        "max_det": defaults["max_det"] if max_det is None else int(max_det),
    }


def run_human_detection_compat(detector, image, detector_kwargs: dict):
    kwargs = dict(detector_kwargs or {})

    while True:
        try:
            return detector.run_human_detection(image, **kwargs)
        except TypeError as exc:
            message = str(exc)
            removed = False
            for optional_key in ("max_det", "return_scores"):
                token = f"unexpected keyword argument '{optional_key}'"
                if optional_key in kwargs and token in message:
                    kwargs.pop(optional_key, None)
                    removed = True
                    break
            if removed:
                continue
            raise
