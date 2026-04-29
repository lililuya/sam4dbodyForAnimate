from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OPTIONAL_PANEL_NAMES = [
    "src_face",
    "src_pose",
    "src_bg",
    "src_mask",
    "src_mask_detail",
]
PANEL_ORDER = [
    "target",
    "4d",
    "overlay",
    "src_face",
    "src_pose",
    "src_bg",
    "src_mask",
    "src_mask_detail",
    "blank",
]
FPS_TOLERANCE = 1e-3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stitched comparison mp4 for one exported target directory")
    parser.add_argument("--input", type=str, required=True, help="Path to one exported target directory")
    parser.add_argument("--output", type=str, default=None, help="Optional output mp4 path")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay blend alpha for 4d over target")
    return parser


def resolve_input_videos(sample_dir: str) -> dict[str, str | None]:
    sample_dir_abs = os.path.abspath(sample_dir)
    if not os.path.isdir(sample_dir_abs):
        raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")

    resolved: dict[str, str | None] = {}
    for panel_name in ("target", "4d"):
        path = os.path.join(sample_dir_abs, f"{panel_name}.mp4")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"required video is missing: {path}")
        resolved[panel_name] = path

    for panel_name in OPTIONAL_PANEL_NAMES:
        path = os.path.join(sample_dir_abs, f"{panel_name}.mp4")
        resolved[panel_name] = path if os.path.isfile(path) else None
    return resolved


def build_default_output_path(sample_dir: str) -> str:
    sample_name = os.path.basename(os.path.abspath(sample_dir).rstrip("\\/"))
    if not sample_name:
        raise ValueError(f"unable to derive sample directory name from input: {sample_dir}")
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    return os.path.join(SCRIPT_DIR, f"{sample_name}_compare.mp4")


def resolve_output_path(sample_dir: str, output_path: str | None) -> str:
    if not output_path:
        return build_default_output_path(sample_dir)

    resolved_output_path = os.path.abspath(output_path)
    if os.path.isdir(resolved_output_path):
        sample_name = os.path.basename(os.path.abspath(sample_dir).rstrip("\\/"))
        return os.path.join(resolved_output_path, f"{sample_name}_compare.mp4")

    _, extension = os.path.splitext(resolved_output_path)
    if not extension:
        return f"{resolved_output_path}.mp4"
    return resolved_output_path


def load_video_info(video_path: str) -> dict:
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video resolution for {video_path}")
    if fps <= 0:
        raise RuntimeError(f"invalid video fps for {video_path}")
    if frame_count <= 0:
        raise RuntimeError(f"invalid video frame count for {video_path}")
    return {
        "path": os.path.abspath(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def validate_required_videos(target_info: dict, rendered_info: dict) -> None:
    if int(target_info["frame_count"]) != int(rendered_info["frame_count"]):
        raise RuntimeError(
            f"target/4d frame count mismatch: {target_info['frame_count']} vs {rendered_info['frame_count']}"
        )
    if abs(float(target_info["fps"]) - float(rendered_info["fps"])) > FPS_TOLERANCE:
        raise RuntimeError(
            f"target/4d fps mismatch: {target_info['fps']} vs {rendered_info['fps']}"
        )


def resolve_tile_size(panel_infos: dict[str, dict]) -> tuple[int, int]:
    max_width = 0
    max_height = 0
    for panel_info in panel_infos.values():
        max_width = max(max_width, int(panel_info["width"]))
        max_height = max(max_height, int(panel_info["height"]))
    if max_width <= 0 or max_height <= 0:
        raise RuntimeError("unable to resolve a valid tile size from panel videos")
    return max_width, max_height


def ensure_color_frame(frame: np.ndarray) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.ndim == 2:
        return np.stack([frame_array] * 3, axis=2)
    if frame_array.ndim == 3 and frame_array.shape[2] == 1:
        return np.repeat(frame_array, 3, axis=2)
    if frame_array.ndim == 3 and frame_array.shape[2] >= 3:
        return frame_array[:, :, :3]
    raise ValueError(f"unsupported frame shape: {frame_array.shape}")


def pad_frame_to_canvas(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    frame_rgb = ensure_color_frame(frame)
    frame_height, frame_width = frame_rgb.shape[:2]
    if frame_width > int(target_width) or frame_height > int(target_height):
        raise RuntimeError(
            f"frame size {frame_width}x{frame_height} exceeds target canvas {target_width}x{target_height}"
        )
    canvas = np.zeros((int(target_height), int(target_width), 3), dtype=frame_rgb.dtype)
    offset_x = (int(target_width) - int(frame_width)) // 2
    offset_y = (int(target_height) - int(frame_height)) // 2
    canvas[offset_y : offset_y + frame_height, offset_x : offset_x + frame_width] = frame_rgb
    return canvas


def build_overlay_frame(target_frame: np.ndarray, rendered_frame: np.ndarray, alpha: float) -> np.ndarray:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    target_float = np.asarray(target_frame, dtype=np.float32)
    rendered_float = np.asarray(rendered_frame, dtype=np.float32)
    blended = target_float * (1.0 - alpha_clamped) + rendered_float * alpha_clamped
    return np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)


def annotate_frame(frame: np.ndarray, label: str, frame_index: int | None = None) -> np.ndarray:
    annotated = np.array(frame, copy=True)
    overlay_text = str(label)
    if frame_index is not None:
        overlay_text = f"{overlay_text} | f={int(frame_index):05d}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        overlay_text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def compose_grid_frame(named_tiles: dict[str, np.ndarray]) -> np.ndarray:
    ordered_tiles = [np.asarray(named_tiles[name]) for name in PANEL_ORDER]
    rows = []
    for row_start in range(0, len(ordered_tiles), 3):
        rows.append(np.concatenate(ordered_tiles[row_start : row_start + 3], axis=1))
    return np.concatenate(rows, axis=0)


def encode_frames_to_mp4(frames_dir: str, fps: float, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    input_pattern = os.path.join(os.path.abspath(frames_dir), "frame_%08d.png")
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(float(fps)),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "0",
        os.path.abspath(output_path),
    ]
    subprocess.run(command, check=True)


def _open_video_capture(video_path: str | None):
    if not video_path:
        return None
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video: {video_path}")
    return capture


def _read_next_frame(capture, *, video_path: str, required: bool) -> np.ndarray | None:
    if capture is None:
        return None
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        if required:
            raise RuntimeError(f"failed to read required frame from {video_path}")
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _normalize_panel_frame(
    frame: np.ndarray | None,
    *,
    panel_name: str,
    target_width: int,
    target_height: int,
    frame_index: int,
) -> np.ndarray:
    if frame is None:
        blank = np.zeros((int(target_height), int(target_width), 3), dtype=np.uint8)
        return annotate_frame(blank, f"{panel_name} | missing", frame_index)
    padded = pad_frame_to_canvas(frame, target_width=target_width, target_height=target_height)
    return annotate_frame(padded, panel_name, frame_index)


def build_comparison_video(sample_dir: str, output_path: str, overlay_alpha: float = 0.5) -> str:
    resolved = resolve_input_videos(sample_dir)
    panel_infos: dict[str, dict] = {
        "target": load_video_info(str(resolved["target"])),
        "4d": load_video_info(str(resolved["4d"])),
    }
    target_info = panel_infos["target"]
    rendered_info = panel_infos["4d"]
    validate_required_videos(target_info, rendered_info)

    for panel_name in OPTIONAL_PANEL_NAMES:
        panel_path = resolved.get(panel_name)
        if not panel_path:
            continue
        panel_infos[panel_name] = load_video_info(str(panel_path))

    target_width, target_height = resolve_tile_size(panel_infos)

    captures = {}
    try:
        for panel_name, panel_path in resolved.items():
            captures[panel_name] = _open_video_capture(panel_path)

        with tempfile.TemporaryDirectory(prefix="verify_output_compare_") as temp_dir:
            for frame_index in range(int(target_info["frame_count"])):
                target_frame = _read_next_frame(
                    captures["target"],
                    video_path=str(resolved["target"]),
                    required=True,
                )
                rendered_frame = _read_next_frame(
                    captures["4d"],
                    video_path=str(resolved["4d"]),
                    required=True,
                )
                target_base = pad_frame_to_canvas(
                    target_frame,
                    target_width=target_width,
                    target_height=target_height,
                )
                rendered_base = pad_frame_to_canvas(
                    rendered_frame,
                    target_width=target_width,
                    target_height=target_height,
                )
                target_tile = annotate_frame(target_base, "target", frame_index)
                rendered_tile = annotate_frame(rendered_base, "4d", frame_index)
                overlay_tile = annotate_frame(
                    build_overlay_frame(target_base, rendered_base, alpha=overlay_alpha),
                    "overlay",
                    frame_index,
                )

                named_tiles: dict[str, np.ndarray] = {
                    "target": target_tile,
                    "4d": rendered_tile,
                    "overlay": overlay_tile,
                }
                for panel_name in OPTIONAL_PANEL_NAMES:
                    panel_frame = _read_next_frame(
                        captures.get(panel_name),
                        video_path=str(resolved.get(panel_name) or panel_name),
                        required=False,
                    )
                    named_tiles[panel_name] = _normalize_panel_frame(
                        panel_frame,
                        panel_name=panel_name,
                        target_width=target_width,
                        target_height=target_height,
                        frame_index=frame_index,
                    )

                named_tiles["blank"] = annotate_frame(
                    np.zeros((target_height, target_width, 3), dtype=np.uint8),
                    os.path.basename(os.path.abspath(sample_dir)),
                    frame_index,
                )
                grid_frame = compose_grid_frame(named_tiles)
                frame_path = os.path.join(temp_dir, f"frame_{frame_index:08d}.png")
                if not cv2.imwrite(frame_path, cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"failed to write temporary comparison frame: {frame_path}")

            encode_frames_to_mp4(temp_dir, fps=float(target_info["fps"]), output_path=output_path)
    finally:
        for capture in captures.values():
            if capture is not None:
                capture.release()
    return os.path.abspath(output_path)


def main(argv: list[str] | None = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)
    sample_dir = os.path.abspath(args.input)
    resolve_input_videos(sample_dir)
    output_path = resolve_output_path(sample_dir, args.output)
    return build_comparison_video(
        sample_dir=sample_dir,
        output_path=output_path,
        overlay_alpha=float(args.overlay_alpha),
    )


if __name__ == "__main__":
    main()
