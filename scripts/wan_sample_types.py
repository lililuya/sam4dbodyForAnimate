from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _coerce_pair(value: Any, default: tuple[int, int], field_name: str) -> tuple[int, int]:
    if value is None:
        return default

    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return pair


@dataclass(frozen=True)
class WanExportConfig:
    enable: bool = False
    output_dir: str | None = None
    metadata_output_dir: str | None = None
    fps: int = 25
    resolution_area: tuple[int, int] = (512, 768)
    face_resolution: tuple[int, int] = (512, 512)
    min_track_frames: int = 16
    min_valid_face_ratio: float = 0.60
    face_expand: float = 1.30
    face_gap: int = 8
    skip_sample_without_face: bool = True
    face_presence_stride: int = 5
    max_no_face_ratio: float = 0.80
    copy_rendered_4d_to_targets: bool = True
    cleanup_sample_workdir_after_export: bool = True
    mask_kernel_size: int = 7
    mask_iterations: int = 3
    mask_w_len: int = 10
    mask_h_len: int = 20
    save_pose_meta_json: bool = True
    save_smpl_sequence_json: bool = False

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, Any] | None) -> "WanExportConfig":
        payload = dict(runtime or {})
        return cls(
            enable=_coerce_bool(payload.get("enable", False)),
            output_dir=None if payload.get("output_dir") in {None, ""} else str(payload.get("output_dir")),
            metadata_output_dir=(
                None if payload.get("metadata_output_dir") in {None, ""} else str(payload.get("metadata_output_dir"))
            ),
            fps=int(payload.get("fps", 25)),
            resolution_area=_coerce_pair(payload.get("resolution_area"), (512, 768), "resolution_area"),
            face_resolution=_coerce_pair(payload.get("face_resolution"), (512, 512), "face_resolution"),
            min_track_frames=int(payload.get("min_track_frames", 16)),
            min_valid_face_ratio=float(payload.get("min_valid_face_ratio", 0.60)),
            face_expand=float(payload.get("face_expand", 1.30)),
            face_gap=int(payload.get("face_gap", 8)),
            skip_sample_without_face=_coerce_bool(payload.get("skip_sample_without_face", True)),
            face_presence_stride=int(payload.get("face_presence_stride", 5)),
            max_no_face_ratio=float(payload.get("max_no_face_ratio", 0.80)),
            copy_rendered_4d_to_targets=_coerce_bool(payload.get("copy_rendered_4d_to_targets", True)),
            cleanup_sample_workdir_after_export=_coerce_bool(
                payload.get("cleanup_sample_workdir_after_export", True)
            ),
            mask_kernel_size=int(payload.get("mask_kernel_size", 7)),
            mask_iterations=int(payload.get("mask_iterations", 3)),
            mask_w_len=int(payload.get("mask_w_len", 10)),
            mask_h_len=int(payload.get("mask_h_len", 20)),
            save_pose_meta_json=_coerce_bool(payload.get("save_pose_meta_json", True)),
            save_smpl_sequence_json=_coerce_bool(payload.get("save_smpl_sequence_json", False)),
        )
