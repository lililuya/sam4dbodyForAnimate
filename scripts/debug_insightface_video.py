import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.wan_face_export import InsightFaceBackend, WanFaceDetection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run InsightFace on a video and write an annotated verification video plus JSON summary."
    )
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--provider", type=str, default="buffalo_l")
    parser.add_argument("--ctx_id", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--font_scale", type=float, default=0.6)
    parser.add_argument(
        "--preload_directory",
        type=str,
        default="",
        help="Passed through to onnxruntime.preload_dlls(). Empty string uses NVIDIA site-packages lookup.",
    )
    parser.add_argument("--disable_ort_preload", action="store_true")
    parser.add_argument("--import_torch_first", action="store_true")
    return parser


def build_output_paths(
    input_video: str,
    explicit_output_video: str = "",
    explicit_output_json: str = "",
) -> Tuple[str, str]:
    input_dir = os.path.dirname(os.path.abspath(input_video))
    stem, _ = os.path.splitext(os.path.basename(input_video))
    output_video = explicit_output_video or os.path.join(input_dir, f"{stem}_insightface.mp4")
    output_json = explicit_output_json or os.path.join(input_dir, f"{stem}_insightface.json")
    return output_video, output_json


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def probe_onnxruntime_environment(args) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "ort_version": None,
        "available_providers": [],
        "preload_attempted": not bool(args.disable_ort_preload),
        "preload_directory": str(args.preload_directory),
        "preload_succeeded": False,
        "preload_error": None,
        "torch_imported": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "import_error": None,
    }

    if bool(getattr(args, "import_torch_first", False)):
        try:
            import torch

            summary["torch_imported"] = True
            summary["torch_version"] = str(getattr(torch, "__version__", ""))
            summary["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            summary["torch_imported"] = False
            summary["torch_import_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        summary["import_error"] = f"{type(exc).__name__}: {exc}"
        return summary

    summary["ort_version"] = str(getattr(ort, "__version__", ""))

    if not bool(args.disable_ort_preload):
        try:
            ort.preload_dlls(directory=str(args.preload_directory))
            summary["preload_succeeded"] = True
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            summary["preload_error"] = f"{type(exc).__name__}: {exc}"

    try:
        summary["available_providers"] = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        summary["available_providers_error"] = f"{type(exc).__name__}: {exc}"

    return summary


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: List[WanFaceDetection],
    *,
    frame_index: int,
    checked: bool,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    color = (0, 255, 0) if checked else (0, 215, 255)

    for face_index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.bbox
        pt1 = (max(0, int(round(x1))), max(0, int(round(y1))))
        pt2 = (max(0, int(round(x2))), max(0, int(round(y2))))
        cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), int(line_thickness))
        label = f"face {face_index}: {float(detection.score):.2f}"
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

    status = "checked" if checked else "stride-skip"
    header = f"frame={int(frame_index)} status={status} faces={len(detections)}"
    cv2.putText(
        annotated,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        max(1, int(line_thickness)),
        cv2.LINE_AA,
    )
    return annotated


def _serialize_detection(detection: WanFaceDetection) -> Dict[str, object]:
    return {
        "bbox": [int(value) for value in detection.bbox],
        "score": float(detection.score),
        "landmarks": np.asarray(detection.landmarks, dtype=np.float32).tolist(),
    }


def _build_video_writer(output_path: str, width: int, height: int, fps: float):
    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (int(width), int(height)))
    if hasattr(writer, "isOpened") and not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")
    return writer


def run_probe(
    args,
    *,
    face_backend: Optional[InsightFaceBackend] = None,
    ort_probe: Optional[Callable[[object], Dict[str, object]]] = None,
) -> Dict[str, object]:
    input_video = os.path.abspath(str(args.input_video))
    output_video, output_json = build_output_paths(input_video, args.output_video, args.output_json)
    stride = max(1, int(args.stride))
    max_frames = max(0, int(args.max_frames))

    ort_summary = dict((ort_probe or probe_onnxruntime_environment)(args))
    backend = face_backend or InsightFaceBackend(provider=str(args.provider), ctx_id=int(args.ctx_id))

    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    frame_count = 0
    checked_frame_count = 0
    face_detected_frame_count = 0
    total_face_count = 0
    frame_results: List[Dict[str, object]] = []
    writer = None

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if max_frames > 0 and frame_count >= max_frames:
                break

            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = _build_video_writer(output_video, width, height, fps)

            checked = (frame_count % stride) == 0
            detections: List[WanFaceDetection] = []
            if checked:
                detections = list(backend.detect(frame_bgr))
                checked_frame_count += 1
                total_face_count += len(detections)
                if detections:
                    face_detected_frame_count += 1
                frame_results.append(
                    {
                        "frame_index": int(frame_count),
                        "detection_count": len(detections),
                        "detections": [_serialize_detection(detection) for detection in detections],
                    }
                )

            annotated = annotate_frame(
                frame_bgr,
                detections,
                frame_index=frame_count,
                checked=checked,
                line_thickness=args.line_thickness,
                font_scale=args.font_scale,
            )
            writer.write(annotated)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    no_face_frame_count = max(0, checked_frame_count - face_detected_frame_count)
    no_face_ratio = float(no_face_frame_count) / float(max(checked_frame_count, 1))
    summary = {
        "input_video": input_video,
        "output_video": os.path.abspath(output_video),
        "output_json": os.path.abspath(output_json),
        "provider": str(args.provider),
        "ctx_id": int(args.ctx_id),
        "stride": stride,
        "fps": fps,
        "frame_count": frame_count,
        "checked_frame_count": checked_frame_count,
        "face_detected_frame_count": face_detected_frame_count,
        "no_face_frame_count": no_face_frame_count,
        "no_face_ratio": no_face_ratio,
        "total_face_count": total_face_count,
        "ort": ort_summary,
        "frame_results": frame_results,
    }

    ensure_parent_dir(output_json)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_probe(args)
    print(
        f"Saved annotated video to {summary['output_video']} "
        f"(frames={summary['frame_count']}, checked={summary['checked_frame_count']}, "
        f"face_frames={summary['face_detected_frame_count']}, total_faces={summary['total_face_count']})"
    )
    print(f"Saved summary to {summary['output_json']}")
    print(f"ORT providers: {summary['ort'].get('available_providers', [])}")
    if summary["ort"].get("preload_error"):
        print(f"ORT preload error: {summary['ort']['preload_error']}")
    if summary["ort"].get("import_error"):
        print(f"ORT import error: {summary['ort']['import_error']}")


if __name__ == "__main__":
    main()
