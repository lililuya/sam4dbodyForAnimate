import argparse
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.detector_defaults import resolve_detector_runtime_options, run_human_detection_compat

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug human detection on a single image or video by drawing detector boxes."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--detector_backend", type=str, default="yolo", choices=["yolo", "yolo11", "vitdet"])
    parser.add_argument("--detector_path", type=str, default="")
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bbox_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=None)
    parser.add_argument("--max_det", type=int, default=None)
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--font_scale", type=float, default=0.6)
    return parser


def infer_media_type(input_path: str) -> str:
    extension = os.path.splitext(str(input_path))[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"unsupported input file: {input_path}")


def _format_output_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return format(float(value), "g")
    return str(value)


def resolve_detection_runtime_options(args) -> Dict[str, object]:
    return resolve_detector_runtime_options(
        args.detector_backend,
        bbox_thresh=args.bbox_thresh,
        iou_thresh=args.iou_thresh,
        max_det=args.max_det,
    )


def build_output_path(
    input_path: str,
    explicit_output_path: str = "",
    *,
    output_dir: str = "",
    detector_backend: str = "yolo",
    bbox_thresh=None,
    iou_thresh=None,
    max_det=None,
) -> str:
    if explicit_output_path:
        return explicit_output_path

    stem, extension = os.path.splitext(os.path.basename(input_path))
    filename = (
        f"{stem}_{str(detector_backend).lower()}_"
        f"bbox{_format_output_value(bbox_thresh)}_"
        f"iou{_format_output_value(iou_thresh)}_"
        f"maxdet{_format_output_value(max_det)}{extension}"
    )
    target_dir = str(output_dir).strip() or os.path.dirname(input_path)
    return os.path.join(target_dir, filename)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_detection_outputs(outputs) -> List[Dict[str, Optional[float]]]:
    if outputs is None:
        return []

    if isinstance(outputs, np.ndarray):
        array = np.asarray(outputs, dtype=np.float32)
        if array.size == 0:
            return []
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] < 4:
            raise ValueError("detector outputs must provide at least 4 bbox values")
        normalized = []
        for row in array:
            score = float(row[4]) if row.shape[0] > 4 else None
            normalized.append({"bbox": [float(value) for value in row[:4]], "score": score})
        return normalized

    if not isinstance(outputs, (list, tuple)):
        raise ValueError("detector outputs must be an array, list, or tuple")

    normalized = []
    for output in outputs:
        if isinstance(output, dict):
            bbox = output.get("bbox")
            score = output.get("score")
        else:
            bbox = output
            score = None

        if bbox is None:
            continue

        bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_array.size < 4:
            continue

        normalized.append(
            {
                "bbox": [float(value) for value in bbox_array[:4]],
                "score": None if score is None else float(score),
            }
        )
    return normalized


def create_detector(args):
    from models.sam_3d_body.tools.build_detector import HumanDetector

    detector_kwargs = {}
    backend = str(args.detector_backend).lower()
    if backend in {"yolo", "yolo11"}:
        if args.weights_path:
            detector_kwargs["weights_path"] = args.weights_path
        elif args.detector_path:
            detector_kwargs["path"] = args.detector_path
    elif args.detector_path:
        detector_kwargs["path"] = args.detector_path

    return HumanDetector(name=args.detector_backend, device=args.device, **detector_kwargs)


def detect_people(detector, frame_bgr: np.ndarray, resolved_options: Dict[str, object]) -> List[Dict[str, Optional[float]]]:
    detector_kwargs = {
        "det_cat_id": 0,
        "bbox_thr": resolved_options["bbox_thresh"],
        "nms_thr": resolved_options["iou_thresh"],
        "default_to_full_image": False,
        "return_scores": True,
    }
    if resolved_options["max_det"] is not None:
        detector_kwargs["max_det"] = int(resolved_options["max_det"])

    outputs = run_human_detection_compat(detector, frame_bgr, detector_kwargs)
    return normalize_detection_outputs(outputs)


def annotate_frame(frame_bgr: np.ndarray, detections: List[Dict[str, Optional[float]]], line_thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
    annotated = frame_bgr.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox"]
        pt1 = (max(0, int(round(x1))), max(0, int(round(y1))))
        pt2 = (max(0, int(round(x2))), max(0, int(round(y2))))
        cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), int(line_thickness))

        score = detection.get("score")
        if score is None:
            label = f"person {index}"
        else:
            label = f"person {index}: {float(score):.2f}"
        text_origin = (pt1[0], max(18, pt1[1] - 8))
        cv2.putText(
            annotated,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            (0, 255, 0),
            max(1, int(line_thickness)),
            cv2.LINE_AA,
        )
    return annotated


def run_on_image(args, detector) -> Dict[str, object]:
    input_path = str(args.input_path)
    resolved_options = resolve_detection_runtime_options(args)
    output_path = build_output_path(
        input_path,
        args.output_path,
        output_dir=args.output_dir,
        detector_backend=args.detector_backend,
        bbox_thresh=resolved_options["bbox_thresh"],
        iou_thresh=resolved_options["iou_thresh"],
        max_det=resolved_options["max_det"],
    )
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {input_path}")

    detections = detect_people(detector, image, resolved_options)
    annotated = annotate_frame(
        image,
        detections,
        line_thickness=args.line_thickness,
        font_scale=args.font_scale,
    )
    ensure_parent_dir(output_path)
    if not cv2.imwrite(output_path, annotated):
        raise RuntimeError(f"failed to write annotated image: {output_path}")

    return {
        "media_type": "image",
        "output_path": output_path,
        "frame_count": 1,
        "detection_count": len(detections),
    }


def _build_video_writer(output_path: str, width: int, height: int, fps: float):
    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (int(width), int(height)))
    if hasattr(writer, "isOpened") and not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")
    return writer


def run_on_video(args, detector) -> Dict[str, object]:
    input_path = str(args.input_path)
    resolved_options = resolve_detection_runtime_options(args)
    output_path = build_output_path(
        input_path,
        args.output_path,
        output_dir=args.output_dir,
        detector_backend=args.detector_backend,
        bbox_thresh=resolved_options["bbox_thresh"],
        iou_thresh=resolved_options["iou_thresh"],
        max_det=resolved_options["max_det"],
    )

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to read video dimensions: {input_path}")

    writer = _build_video_writer(output_path, width, height, fps)
    frame_count = 0
    total_detections = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detect_people(detector, frame, resolved_options)
            annotated = annotate_frame(
                frame,
                detections,
                line_thickness=args.line_thickness,
                font_scale=args.font_scale,
            )
            writer.write(annotated)
            frame_count += 1
            total_detections += len(detections)
    finally:
        capture.release()
        writer.release()

    return {
        "media_type": "video",
        "output_path": output_path,
        "frame_count": frame_count,
        "detection_count": total_detections,
        "fps": fps,
    }


def run_detection(args) -> Dict[str, object]:
    detector = create_detector(args)
    media_type = infer_media_type(args.input_path)
    if media_type == "image":
        return run_on_image(args, detector)
    return run_on_video(args, detector)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_detection(args)
    print(
        f"Saved annotated {result['media_type']} to {result['output_path']} "
        f"(frames={result['frame_count']}, detections={result['detection_count']})"
    )


if __name__ == "__main__":
    main()
