# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_VITDET_SCORE_THRESH = 0.05
DEFAULT_VITDET_NMS_THRESH = 0.50


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


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[current] + areas[remaining] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = remaining[iou <= float(iou_threshold)]

    return np.asarray(keep, dtype=np.int64)


def select_detectron2_person_detections(
    instances,
    det_cat_id: int = 0,
    bbox_thr: float = DEFAULT_VITDET_SCORE_THRESH,
    nms_thr: float = DEFAULT_VITDET_NMS_THRESH,
):
    boxes = _to_numpy_float32(instances.pred_boxes.tensor).reshape(-1, 4)
    scores = _to_numpy_float32(instances.scores).reshape(-1)
    classes = _to_numpy_float32(instances.pred_classes).reshape(-1)

    valid_idx = (classes == float(det_cat_id)) & (scores > float(bbox_thr))
    boxes = boxes[valid_idx]
    scores = scores[valid_idx]
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32), scores.reshape(0).astype(np.float32)

    keep = nms_xyxy(boxes, scores, float(nms_thr))
    boxes = boxes[keep]
    scores = scores[keep]

    sorted_indices = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[sorted_indices].astype(np.float32), scores[sorted_indices].astype(np.float32)


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device
        self.detector_kwargs = {}

        if name == "vitdet":
            print("########### Using human detector: ViTDet")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        elif name in {"yolo", "yolo11"}:
            print("########### Using human detector: YOLO")
            from .build_detector_yolo import load_ultralytics_yolo, run_ultralytics_yolo

            self.detector = load_ultralytics_yolo(**kwargs)
            self.detector_func = run_ultralytics_yolo
            self.detector_kwargs["device"] = self.device
        else:
            raise NotImplementedError

    def run_human_detection(self, img, **kwargs):
        run_kwargs = {**self.detector_kwargs, **kwargs}
        return self.detector_func(self.detector, img, **run_kwargs)


def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    # Get config file from tools directory (same folder as this file)
    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = DEFAULT_VITDET_SCORE_THRESH
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = DEFAULT_VITDET_SCORE_THRESH,
    nms_thr: float = DEFAULT_VITDET_NMS_THRESH,
    default_to_full_image: bool = True,
    return_scores: bool = False,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    boxes, scores = select_detectron2_person_detections(
        det_instances,
        det_cat_id=det_cat_id,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
    )
    if boxes.size == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4).astype(np.float32)
        scores = np.array([np.nan], dtype=np.float32)
    if return_scores:
        return [
            {
                "bbox": box.astype(np.float32).tolist(),
                "score": None if np.isnan(score) else float(score),
            }
            for box, score in zip(boxes, scores)
        ]
    return boxes
