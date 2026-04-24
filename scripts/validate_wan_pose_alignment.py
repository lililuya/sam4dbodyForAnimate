from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.wan_pose_renderer import draw_pose_frame


def _load_video_frames_rgb(video_path: str) -> list[np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _write_rgb_mp4(path: str, frames: list[np.ndarray], fps: int = 25) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _load_pose_meta_jsons(pose_meta_dir: str) -> list[dict]:
    metas: list[dict] = []
    if not os.path.isdir(pose_meta_dir):
        return metas
    for name in sorted(os.listdir(pose_meta_dir)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(pose_meta_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_filename"] = name
        metas.append(payload)
    return metas


def _count_visible(points: Iterable[Iterable[float]]) -> int:
    array = np.asarray(list(points), dtype=np.float32)
    if array.size == 0:
        return 0
    array = array.reshape(-1, 3)
    return int((array[:, 2] > 0).sum())


def _count_out_of_bounds(points: Iterable[Iterable[float]]) -> int:
    array = np.asarray(list(points), dtype=np.float32)
    if array.size == 0:
        return 0
    array = array.reshape(-1, 3)
    visible = array[:, 2] > 0
    if not np.any(visible):
        return 0
    coords = array[visible, :2]
    out_of_bounds = (coords[:, 0] < 0.0) | (coords[:, 0] > 1.0) | (coords[:, 1] < 0.0) | (coords[:, 1] > 1.0)
    return int(out_of_bounds.sum())


def _render_pose_meta_frame(pose_meta: dict, canvas_shape: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    return draw_pose_frame(canvas, pose_meta)


def _blend_overlay(target_frame: np.ndarray, pose_frame: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(target_frame, 1.0 - alpha, pose_frame, alpha, 0.0)


def validate_wan_sample_alignment(
    sample_dir: str,
    *,
    output_dir: str | None = None,
    alpha: float = 0.65,
    overlay_fps: int = 25,
) -> dict:
    sample_dir = os.path.abspath(sample_dir)
    output_dir = os.path.abspath(output_dir or os.path.join(sample_dir, "validation"))
    os.makedirs(output_dir, exist_ok=True)

    target_path = os.path.join(sample_dir, "target.mp4")
    src_pose_path = os.path.join(sample_dir, "src_pose.mp4")
    pose_meta_dir = os.path.join(sample_dir, "pose_meta_json")

    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"missing target video: {target_path}")
    if not os.path.isfile(src_pose_path):
        raise FileNotFoundError(f"missing src_pose video: {src_pose_path}")

    target_frames = _load_video_frames_rgb(target_path)
    src_pose_frames = _load_video_frames_rgb(src_pose_path)
    pose_metas = _load_pose_meta_jsons(pose_meta_dir)

    total_frames = max(len(target_frames), len(src_pose_frames), len(pose_metas))
    overlay_frames: list[np.ndarray] = []
    frame_reports: list[dict] = []

    for frame_index in range(total_frames):
        target_frame = target_frames[frame_index] if frame_index < len(target_frames) else None
        src_pose_frame = src_pose_frames[frame_index] if frame_index < len(src_pose_frames) else None
        pose_meta = pose_metas[frame_index] if frame_index < len(pose_metas) else None

        reasons: list[str] = []
        if target_frame is None:
            reasons.append("missing_target_frame")
        if src_pose_frame is None:
            reasons.append("missing_src_pose_frame")
        if pose_meta is None:
            reasons.append("missing_pose_meta")

        frame_name = f"{frame_index:08d}.jpg"
        target_frame_size = None
        pose_meta_width = None
        pose_meta_height = None
        body_visible_count = 0
        left_hand_visible_count = 0
        right_hand_visible_count = 0
        face_visible_count = 0
        out_of_bounds_points = 0
        rendered_pose_frame = None

        if target_frame is not None:
            target_frame_size = [int(target_frame.shape[1]), int(target_frame.shape[0])]

        if pose_meta is not None:
            frame_name = str(pose_meta.get("image_id", frame_name))
            pose_meta_width = int(pose_meta.get("width", 0))
            pose_meta_height = int(pose_meta.get("height", 0))
            body_visible_count = _count_visible(pose_meta.get("keypoints_body", []))
            left_hand_visible_count = _count_visible(pose_meta.get("keypoints_left_hand", []))
            right_hand_visible_count = _count_visible(pose_meta.get("keypoints_right_hand", []))
            face_visible_count = _count_visible(pose_meta.get("keypoints_face", []))
            out_of_bounds_points = (
                _count_out_of_bounds(pose_meta.get("keypoints_body", []))
                + _count_out_of_bounds(pose_meta.get("keypoints_left_hand", []))
                + _count_out_of_bounds(pose_meta.get("keypoints_right_hand", []))
                + _count_out_of_bounds(pose_meta.get("keypoints_face", []))
            )
            if out_of_bounds_points > 0:
                reasons.append("out_of_bounds_pose_points")
            if target_frame is not None and (
                pose_meta_width != int(target_frame.shape[1]) or pose_meta_height != int(target_frame.shape[0])
            ):
                reasons.append("pose_meta_size_mismatch")
            if target_frame is not None:
                rendered_pose_frame = _render_pose_meta_frame(pose_meta, target_frame.shape)

        visible_pose_count = body_visible_count + left_hand_visible_count + right_hand_visible_count + face_visible_count
        is_empty_pose = visible_pose_count <= 0
        if pose_meta is not None and is_empty_pose:
            reasons.append("empty_pose")

        src_pose_frame_nonzero = int(np.count_nonzero(src_pose_frame)) if src_pose_frame is not None else 0
        rendered_pose_frame_nonzero = int(np.count_nonzero(rendered_pose_frame)) if rendered_pose_frame is not None else 0

        if pose_meta is not None and visible_pose_count > 0 and src_pose_frame_nonzero <= 0:
            reasons.append("src_pose_empty")
        if pose_meta is not None and visible_pose_count > 0 and rendered_pose_frame_nonzero <= 0:
            reasons.append("rendered_pose_empty")

        status = "ok" if not reasons else "failed"
        frame_reports.append(
            {
                "frame_index": int(frame_index),
                "frame_name": frame_name,
                "target_frame_size": target_frame_size,
                "pose_meta_width": pose_meta_width,
                "pose_meta_height": pose_meta_height,
                "body_visible_count": int(body_visible_count),
                "left_hand_visible_count": int(left_hand_visible_count),
                "right_hand_visible_count": int(right_hand_visible_count),
                "face_visible_count": int(face_visible_count),
                "out_of_bounds_points": int(out_of_bounds_points),
                "is_empty_pose": bool(is_empty_pose),
                "src_pose_frame_nonzero": int(src_pose_frame_nonzero),
                "rendered_pose_frame_nonzero": int(rendered_pose_frame_nonzero),
                "status": status,
                "reasons": reasons,
            }
        )

        if target_frame is not None:
            overlay = target_frame.copy()
            if rendered_pose_frame is not None:
                overlay = _blend_overlay(target_frame, rendered_pose_frame, alpha=alpha)
            overlay_frames.append(overlay.astype(np.uint8))

    summary = {
        "target_frame_count": int(len(target_frames)),
        "src_pose_frame_count": int(len(src_pose_frames)),
        "pose_meta_count": int(len(pose_metas)),
        "all_counts_match": bool(len(target_frames) == len(src_pose_frames) == len(pose_metas)),
        "empty_pose_frames": int(sum(1 for frame in frame_reports if frame["is_empty_pose"])),
        "frames_with_out_of_bounds_points": int(sum(1 for frame in frame_reports if frame["out_of_bounds_points"] > 0)),
        "ok_frames": int(sum(1 for frame in frame_reports if frame["status"] == "ok")),
        "failed_frames": int(sum(1 for frame in frame_reports if frame["status"] != "ok")),
    }
    summary["status"] = "ok" if summary["all_counts_match"] and summary["failed_frames"] == 0 else "failed"

    overlay_video_path = os.path.join(output_dir, "pose_overlay.mp4")
    report_path = os.path.join(output_dir, "alignment_report.json")
    _write_rgb_mp4(overlay_video_path, overlay_frames, fps=overlay_fps)
    report = {
        "sample_dir": sample_dir,
        "target_video_path": target_path,
        "src_pose_video_path": src_pose_path,
        "pose_meta_dir": pose_meta_dir,
        "overlay_video_path": overlay_video_path,
        "summary": summary,
        "frames": frame_reports,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return {
        "overlay_video_path": overlay_video_path,
        "report_path": report_path,
        "summary": summary,
        "frames": frame_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Wan-exported target/pose alignment frame by frame.")
    parser.add_argument("--sample_dir", required=True, type=str, help="Wan sample directory containing target.mp4.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for validation artifacts.")
    parser.add_argument("--alpha", type=float, default=0.65, help="Overlay alpha for rendered pose.")
    parser.add_argument("--overlay_fps", type=int, default=25, help="FPS for the overlay video.")
    parser.add_argument(
        "--fail_on_mismatch",
        action="store_true",
        help="Return a non-zero exit code when the validation summary status is failed.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = validate_wan_sample_alignment(
        args.sample_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        overlay_fps=args.overlay_fps,
    )
    print(f"validation status: {result['summary']['status']}")
    print(f"overlay video: {result['overlay_video_path']}")
    print(f"report json: {result['report_path']}")
    if args.fail_on_mismatch and result["summary"]["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
