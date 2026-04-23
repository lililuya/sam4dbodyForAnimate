from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class WanExportConfig:
    enable: bool = False
    fps: int = 25
    resolution_area: tuple[int, int] = (512, 768)
    face_resolution: tuple[int, int] = (512, 512)
    min_track_frames: int = 16
    min_valid_face_ratio: float = 0.60
    face_expand: float = 1.30
    face_gap: int = 8
    mask_kernel_size: int = 7
    mask_iterations: int = 3
    mask_w_len: int = 10
    mask_h_len: int = 20
    save_pose_meta_json: bool = True

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, Any] | None) -> "WanExportConfig":
        payload = dict(runtime or {})
        return cls(
            enable=bool(payload.get("enable", False)),
            fps=int(payload.get("fps", 25)),
            resolution_area=tuple(int(value) for value in payload.get("resolution_area", (512, 768))),
            face_resolution=tuple(int(value) for value in payload.get("face_resolution", (512, 512))),
            min_track_frames=int(payload.get("min_track_frames", 16)),
            min_valid_face_ratio=float(payload.get("min_valid_face_ratio", 0.60)),
            face_expand=float(payload.get("face_expand", 1.30)),
            face_gap=int(payload.get("face_gap", 8)),
            mask_kernel_size=int(payload.get("mask_kernel_size", 7)),
            mask_iterations=int(payload.get("mask_iterations", 3)),
            mask_w_len=int(payload.get("mask_w_len", 10)),
            mask_h_len=int(payload.get("mask_h_len", 20)),
            save_pose_meta_json=bool(payload.get("save_pose_meta_json", True)),
        )
