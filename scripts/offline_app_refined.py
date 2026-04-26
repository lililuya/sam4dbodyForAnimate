import argparse
import copy
import glob
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test envs
    torch = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.detector_defaults import resolve_detector_runtime_options
from scripts.offline_reprompt import match_detection_to_track, should_trigger_reprompt


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


def build_davis_palette() -> List[int]:
    palette = [0] * (256 * 3)
    for color in range(256):
        label = color
        bit = 0
        while label:
            palette[color * 3 + 0] |= ((label >> 0) & 1) << (7 - bit)
            palette[color * 3 + 1] |= ((label >> 1) & 1) << (7 - bit)
            palette[color * 3 + 2] |= ((label >> 2) & 1) << (7 - bit)
            bit += 1
            label >>= 3
    return palette


DAVIS_PALETTE = build_davis_palette()


def build_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _autocast_disabled():
    if torch is None:
        return nullcontext()
    return torch.autocast("cuda", enabled=False)


def _module_parameter_dtype(module):
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        return next(module.parameters()).dtype
    except (StopIteration, TypeError, AttributeError):
        return None


def _align_completion_pipeline_dtype(pipeline) -> None:
    if pipeline is None:
        return

    target_dtype = _module_parameter_dtype(getattr(pipeline, "image_encoder", None))
    if target_dtype is None:
        target_dtype = _module_parameter_dtype(getattr(pipeline, "unet", None))
    if target_dtype is None:
        return

    for module_name in ("unet", "vae"):
        module = getattr(pipeline, module_name, None)
        if module is None or not hasattr(module, "to"):
            continue
        if _module_parameter_dtype(module) == target_dtype:
            continue
        module.to(dtype=target_dtype)


def _align_completion_pipeline_dtypes(runtime_app) -> None:
    for pipeline_name in ("pipeline_mask", "pipeline_rgb"):
        _align_completion_pipeline_dtype(getattr(runtime_app, pipeline_name, None))


def build_parser() -> argparse.ArgumentParser:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Refined offline 4D body generation with YOLO and occlusion-aware recovery"
    )
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(repo_root, "configs", "body4d_refined.yaml"),
    )
    parser.add_argument("--detector_backend", type=str, default=None)
    parser.add_argument("--track_chunk_size", type=int, default=None)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--disable_mask_refine", action="store_true")
    parser.add_argument("--disable_auto_reprompt", action="store_true")
    parser.add_argument("--save_debug_metrics", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def load_refined_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


def load_base_offline_module():
    from scripts import offline_app as base_offline_app

    return base_offline_app


def apply_runtime_overrides(args, cfg):
    if args.output_dir:
        cfg.runtime.output_dir = args.output_dir
    if args.detector_backend is not None:
        cfg.detector.backend = args.detector_backend
    if getattr(args, "max_targets", None) is not None:
        cfg.detector.max_targets = int(args.max_targets)
    if args.track_chunk_size is not None:
        cfg.tracking.chunk_size = args.track_chunk_size
    if getattr(args, "disable_mask_refine", False):
        if not hasattr(cfg, "refine") or cfg.refine is None:
            cfg.refine = OmegaConf.create({})
        cfg.refine.enable = False
    if args.disable_auto_reprompt:
        cfg.reprompt.enable = False
    if args.save_debug_metrics:
        cfg.debug.save_metrics = True
    return cfg


def build_reprompt_thresholds(cfg) -> Dict[str, float]:
    return {
        "empty_mask_patience": int(cfg.reprompt.empty_mask_patience),
        "area_drop_ratio": float(cfg.reprompt.area_drop_ratio),
        "edge_touch_ratio": float(cfg.reprompt.edge_touch_ratio),
        "iou_low_threshold": float(cfg.reprompt.iou_low_threshold),
    }


def build_runtime_storage_options(cfg) -> Dict[str, object]:
    return {
        "save_rendered_video": bool(cfg_get(cfg, "runtime.save_rendered_video", True)),
        "save_rendered_video_direct": bool(cfg_get(cfg, "runtime.save_rendered_video_direct", False)),
        "save_rendered_frames": bool(cfg_get(cfg, "runtime.save_rendered_frames", True)),
        "save_rendered_frames_individual": bool(cfg_get(cfg, "runtime.save_rendered_frames_individual", True)),
        "save_mesh_4d_individual": bool(cfg_get(cfg, "runtime.save_mesh_4d_individual", True)),
        "save_focal_4d_individual": bool(cfg_get(cfg, "runtime.save_focal_4d_individual", True)),
        "pose_exports": list(cfg_get(cfg, "runtime.pose_exports", []) or []),
        "wan_export": to_plain_runtime_dict(cfg_get(cfg, "wan_export", {})),
    }


def prepare_output_dirs(
    output_dir: str,
    obj_ids: Iterable[int],
    save_debug_metrics: bool,
    storage_options: Optional[dict] = None,
) -> Dict[str, str]:
    storage_options = dict(storage_options or {})
    save_rendered_frames = bool(storage_options.get("save_rendered_frames", True))
    save_rendered_frames_individual = bool(storage_options.get("save_rendered_frames_individual", True))
    save_mesh_4d_individual = bool(storage_options.get("save_mesh_4d_individual", True))
    save_focal_4d_individual = bool(storage_options.get("save_focal_4d_individual", True))
    paths = {
        "images": os.path.join(output_dir, "images"),
        "masks_raw": os.path.join(output_dir, "masks_raw"),
        "masks_refined": os.path.join(output_dir, "masks_refined"),
        "completion_images": os.path.join(output_dir, "completion_refined", "images"),
        "completion_masks": os.path.join(output_dir, "completion_refined", "masks"),
    }
    if save_rendered_frames:
        paths["rendered_frames"] = os.path.join(output_dir, "rendered_frames")
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    for obj_id in obj_ids:
        if save_mesh_4d_individual:
            os.makedirs(os.path.join(output_dir, "mesh_4d_individual", str(obj_id)), exist_ok=True)
        if save_focal_4d_individual:
            os.makedirs(os.path.join(output_dir, "focal_4d_individual", str(obj_id)), exist_ok=True)
        if save_rendered_frames_individual:
            os.makedirs(os.path.join(output_dir, "rendered_frames_individual", str(obj_id)), exist_ok=True)
    if save_debug_metrics:
        debug_dir = os.path.join(output_dir, "debug_metrics")
        os.makedirs(debug_dir, exist_ok=True)
        paths["debug_metrics"] = debug_dir
    return paths


def resolve_sample_output_dir(explicit_output_dir: Optional[str], configured_output_root: Optional[str]) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)

    output_root = os.path.abspath(configured_output_root or os.path.join(ROOT, "outputs_refined"))
    return os.path.join(output_root, build_run_id())


def write_chunk_manifest(debug_dir: str, chunk_records: List[dict]) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, "chunk_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(chunk_records, handle, indent=2)


def write_debug_json(debug_dir: str, filename: str, payload) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def clone_runtime_config(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def to_plain_runtime_dict(value) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False) or {}
    return dict(value)


def apply_sample_runtime_profile(cfg, runtime_profile: Optional[dict]) -> dict:
    profile = copy.deepcopy(runtime_profile) if runtime_profile else {}
    if "tracking.chunk_size" in profile and hasattr(cfg, "tracking"):
        cfg.tracking.chunk_size = int(profile["tracking.chunk_size"])
    if "sam_3d_body.batch_size" in profile and hasattr(cfg, "sam_3d_body"):
        cfg.sam_3d_body.batch_size = int(profile["sam_3d_body.batch_size"])
    if "batch.initial_search_frames" in profile and hasattr(cfg, "batch"):
        cfg.batch.initial_search_frames = int(profile["batch.initial_search_frames"])
    elif "initial_search_frames" in profile and hasattr(cfg, "batch"):
        cfg.batch.initial_search_frames = int(profile["initial_search_frames"])
    return profile


def cfg_get(cfg, path: str, default=None):
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:  # noqa: BLE001 - tolerate partially mocked configs in unit tests
        return default


def list_input_frames(input_dir: str) -> List[str]:
    frames = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(VALID_IMAGE_EXTENSIONS):
            frames.append(path)
    return frames


def count_video_frames(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return frame_count


def resolve_video_fps(video_path: str) -> Optional[float]:
    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps > 0:
        return fps
    return None


def build_frame_stems(frame_count: int) -> List[str]:
    return [f"{frame_idx:08d}" for frame_idx in range(int(frame_count))]


def save_indexed_mask(mask: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="P")
    mask_image.putpalette(DAVIS_PALETTE)
    mask_image.save(path)


def load_indexed_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("P"), dtype=np.uint8)


def keep_largest_component_for_label(mask: np.ndarray, label: int) -> np.ndarray:
    binary = (mask == int(label)).astype(np.uint8)
    if binary.max() == 0:
        return binary
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        return binary
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(counts.argmax())
    return (labels == largest_label).astype(np.uint8)


def edge_touch_ratio(binary_mask: np.ndarray) -> float:
    foreground = float((binary_mask > 0).sum())
    if foreground <= 0:
        return 0.0
    border = np.zeros_like(binary_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touch = np.logical_and(binary_mask > 0, border).sum()
    return float(touch) / foreground


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a_bin, b_bin).sum()
    return float(inter) / float(union)


def mask_bbox_xyxy(binary_mask: np.ndarray):
    coords = np.column_stack(np.where(binary_mask > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def normalize_detection_outputs(outputs) -> Optional[List[dict]]:
    if outputs is None:
        return None

    if isinstance(outputs, np.ndarray):
        array = np.asarray(outputs, dtype=np.float32)
        if array.size == 0:
            return []
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] < 4:
            return None
        normalized = []
        for row in array:
            score = float(row[4]) if row.shape[0] > 4 else None
            normalized.append(
                {
                    "bbox": [float(value) for value in row[:4]],
                    "score": score,
                }
            )
        return normalized

    if not isinstance(outputs, (list, tuple)):
        return None

    normalized = []
    for output in outputs:
        if isinstance(output, dict):
            bbox = output.get("bbox")
            score = output.get("score")
        else:
            bbox = output
            score = None

        if bbox is None:
            return None

        bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_array.size < 4:
            return None

        normalized.append(
            {
                "bbox": [float(value) for value in bbox_array[:4]],
                "score": None if score is None else float(score),
            }
        )

    return normalized


def limit_detection_outputs(outputs: List[dict], max_targets: int) -> List[dict]:
    if int(max_targets) <= 0 or len(outputs) <= int(max_targets):
        return list(outputs)

    if any(output.get("score") is None for output in outputs):
        return list(outputs[: int(max_targets)])

    ranked_outputs = sorted(
        enumerate(outputs),
        key=lambda item: (-float(item[1]["score"]), item[0]),
    )
    return [output for _, output in ranked_outputs[: int(max_targets)]]


class RefinedOfflineApp:
    def __init__(self, config_path: str, config=None):
        self.config_path = config_path
        self.CONFIG = config if config is not None else load_refined_config(config_path)
        self.reprompt_thresholds = build_reprompt_thresholds(self.CONFIG) if hasattr(self.CONFIG, "reprompt") else {}
        self.should_trigger_reprompt = should_trigger_reprompt
        self.match_detection_to_track = match_detection_to_track
        self._base_module = None
        self._base_app = None
        self._detector_signature = None
        self._face_backend = None
        self.reset_sample_state()

    def reset_sample_state(self):
        self.chunk_records = []
        self.output_paths = {}
        self.reprompt_events = []
        self.sample_summary = {}
        self.sample_config = None
        self.sample_state = {}
        self.initial_targets = {}
        self.OUTPUT_DIR = ""
        self.RUNTIME = {}
        self._tracking_generated = False
        self._last_binary_masks = {}
        self._empty_mask_counts = {}
        self._last_track_boxes = {}

    def prepare_sample_output(self, output_dir: str, obj_ids: Iterable[int]):
        output_dir = os.path.abspath(output_dir)
        self.OUTPUT_DIR = output_dir
        if self.sample_state:
            self.sample_state["output_dir"] = output_dir
            self.sample_state["output_dir_ready"] = True
        if self._base_app is not None:
            self._sync_base_app_runtime(self._base_app, output_dir)
        self.chunk_records = []
        self.output_paths = prepare_output_dirs(
            output_dir,
            obj_ids,
            save_debug_metrics=bool(self.CONFIG.debug.save_metrics),
            storage_options=build_runtime_storage_options(self.CONFIG),
        )
        return self.output_paths

    def finalize_sample(self):
        debug_dir = self.output_paths.get("debug_metrics", "")
        write_chunk_manifest(debug_dir, self.chunk_records)
        if not debug_dir:
            return
        if self.sample_summary:
            write_debug_json(debug_dir, "sample_summary.json", self.sample_summary)
            runtime_profile = self.sample_summary.get("runtime_profile")
            if runtime_profile is not None:
                write_debug_json(debug_dir, "runtime_profile.json", runtime_profile)
        if self.reprompt_events:
            write_debug_json(debug_dir, "reprompt_events.json", self.reprompt_events)

    def _build_sample_fps_summary(self, sample: dict) -> dict:
        input_type = str(sample.get("input_type") or "")
        source_fps = None
        source_fps_source = "unavailable"
        if input_type == "video":
            source_fps = resolve_video_fps(str(sample.get("input_video") or ""))
            source_fps_source = "video_metadata" if source_fps is not None else "unavailable"
        elif input_type == "images":
            source_fps_source = "image_sequence"

        wan_export_cfg = to_plain_runtime_dict(cfg_get(self.CONFIG, "wan_export", {}))
        wan_target_fps = None
        if bool(wan_export_cfg.get("enable", False)):
            wan_target_fps = int(wan_export_cfg.get("fps", 25))

        return {
            "source_fps": source_fps,
            "source_fps_source": source_fps_source,
            "rendered_4d_fps": float(getattr(self, "RUNTIME", {}).get("video_fps", 25) or 25),
            "wan_target_fps": wan_target_fps,
        }

    def _resolve_sample_identity(self, sample: Optional[dict] = None, input_video: Optional[str] = None) -> dict:
        sample = dict(sample or {})
        source_path = os.path.abspath(
            str(sample.get("input_video") or input_video or self.sample_state.get("input_video") or "")
        )
        working_output_dir = sample.get("output_dir") or self.sample_state.get("output_dir")
        sample_id = os.path.splitext(os.path.basename(source_path))[0] or os.path.basename(
            os.path.abspath(str(working_output_dir or "sample"))
        )
        return {
            "sample_id": sample_id,
            "source_path": source_path,
            "working_output_dir": None if not working_output_dir else os.path.abspath(str(working_output_dir)),
        }

    def _resolve_issue_ledger_root(self, sample: Optional[dict] = None) -> str:
        wan_export_cfg = to_plain_runtime_dict(cfg_get(self.CONFIG, "wan_export", {}))
        export_root = str(wan_export_cfg.get("output_dir") or "").strip()
        if export_root:
            return os.path.abspath(export_root)

        configured_output_root = os.path.abspath(
            str(cfg_get(self.CONFIG, "runtime.output_dir", os.path.join(ROOT, "outputs_refined")))
        )
        planned_output_dir = None if not sample else sample.get("output_dir")
        if planned_output_dir:
            planned_output_dir_abs = os.path.abspath(str(planned_output_dir))
            if planned_output_dir_abs == configured_output_root:
                return configured_output_root
            return os.path.dirname(planned_output_dir_abs) or configured_output_root
        return configured_output_root

    def _append_issue_ledger_record(self, record: dict, sample: Optional[dict] = None) -> None:
        from scripts.wan_sample_export import append_wan_issue_records

        append_wan_issue_records(self._resolve_issue_ledger_root(sample), [record])

    def _build_issue_ledger_record(
        self,
        *,
        event_type: str,
        status: str,
        reason: str,
        sample: Optional[dict] = None,
        sample_uuid: Optional[str] = None,
        runtime_profile: Optional[dict] = None,
        details: Optional[dict] = None,
        input_video: Optional[str] = None,
    ) -> dict:
        identity = self._resolve_sample_identity(sample, input_video=input_video)
        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type),
            "status": str(status),
            "reason": str(reason),
            "source_path": identity["source_path"],
            "sample_id": identity["sample_id"],
            "sample_uuid": None if not sample_uuid else str(sample_uuid),
            "working_output_dir": identity["working_output_dir"],
            "runtime_profile": None if runtime_profile is None else copy.deepcopy(runtime_profile),
            "details": dict(details or {}),
        }

    def _ensure_face_backend(self):
        if self._face_backend is None:
            from scripts.wan_face_export import InsightFaceBackend

            self._face_backend = InsightFaceBackend()
        return self._face_backend

    def _probe_sample_face_presence(self, sample: dict) -> dict:
        from scripts.wan_sample_types import WanExportConfig

        wan_cfg = WanExportConfig.from_runtime(cfg_get(self.CONFIG, "wan_export", {}))
        stride = max(1, int(wan_cfg.face_presence_stride))
        summary = {
            "checked_frame_count": 0,
            "face_detected_frame_count": 0,
            "no_face_frame_count": 0,
            "no_face_ratio": 0.0,
            "face_presence_stride": stride,
            "max_no_face_ratio": float(wan_cfg.max_no_face_ratio),
            "skip_sample_without_face": bool(wan_cfg.skip_sample_without_face),
            "probe_executed": False,
        }
        if not bool(wan_cfg.enable) or not bool(wan_cfg.skip_sample_without_face):
            return summary

        frame_count = int(sample.get("frame_count", 0) or 0)
        if frame_count <= 0:
            return summary

        frame_indices = list(range(0, frame_count, stride))
        if not frame_indices:
            frame_indices = [0]

        backend = self._ensure_face_backend()
        face_detected_frame_count = 0
        for frame_idx in frame_indices:
            frame_rgb = self._load_source_frame(sample, frame_idx)
            detections = backend.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if detections:
                face_detected_frame_count += 1

        checked_frame_count = len(frame_indices)
        no_face_frame_count = max(0, checked_frame_count - face_detected_frame_count)
        no_face_ratio = float(no_face_frame_count) / float(max(checked_frame_count, 1))
        summary.update(
            {
                "checked_frame_count": checked_frame_count,
                "face_detected_frame_count": face_detected_frame_count,
                "no_face_frame_count": no_face_frame_count,
                "no_face_ratio": no_face_ratio,
                "probe_executed": True,
            }
        )
        return summary

    def _should_skip_sample_for_face_presence(self, face_presence: dict) -> bool:
        from scripts.wan_sample_types import WanExportConfig

        wan_cfg = WanExportConfig.from_runtime(cfg_get(self.CONFIG, "wan_export", {}))
        if not bool(wan_cfg.enable) or not bool(wan_cfg.skip_sample_without_face):
            return False
        checked_frame_count = int(face_presence.get("checked_frame_count", 0) or 0)
        if checked_frame_count <= 0:
            return False
        return float(face_presence.get("no_face_ratio", 0.0) or 0.0) >= float(wan_cfg.max_no_face_ratio)

    def _ensure_wan_sample_summary(self, sample: dict) -> tuple[Optional[str], Optional[str]]:
        wan_export_cfg = to_plain_runtime_dict(cfg_get(self.CONFIG, "wan_export", {}))
        if not bool(wan_export_cfg.get("enable", False)):
            return None, None

        export_root = str(wan_export_cfg.get("output_dir") or "").strip()
        if not export_root:
            return None, None

        from scripts.wan_sample_export import resolve_or_create_wan_sample_uuid, update_wan_sample_summary

        identity = self._resolve_sample_identity(sample)
        source_path = identity["source_path"]
        sample_id = identity["sample_id"]
        sample_output_dir = identity["working_output_dir"]
        sample_uuid = resolve_or_create_wan_sample_uuid(
            export_root,
            source_path,
            sample_id=sample_id,
            working_output_dir=sample_output_dir,
        )
        update_wan_sample_summary(
            export_root,
            sample_uuid,
            {
                "sample_id": sample_id,
                "source_path": source_path,
                "working_output_dir": None if not sample_output_dir else os.path.abspath(sample_output_dir),
                "export_root": os.path.abspath(export_root),
            },
        )
        self.sample_summary["sample_uuid"] = sample_uuid
        return sample_uuid, os.path.abspath(export_root)

    def _load_wan_exported_target_dirs(self, sample_uuid: Optional[str], wan_export_root: Optional[str]) -> list[str]:
        if not sample_uuid or not wan_export_root:
            return []

        from scripts.wan_sample_export import get_wan_summary_path

        summary_path = get_wan_summary_path(wan_export_root, sample_uuid)
        if not os.path.isfile(summary_path):
            return []

        with open(summary_path, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                return []

        exported_targets = payload.get("exported_targets")
        if not isinstance(exported_targets, list):
            return []

        target_dirs: list[str] = []
        for target in exported_targets:
            if not isinstance(target, dict):
                continue
            sample_dir = str(target.get("sample_dir", "")).strip()
            if not sample_dir:
                continue
            target_dirs.append(os.path.abspath(sample_dir))
        return target_dirs

    def _copy_rendered_4d_to_wan_targets(self, final_4d_path: Optional[str], target_dirs: list[str]) -> None:
        if not final_4d_path or not os.path.isfile(final_4d_path):
            return
        for target_dir in target_dirs:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(final_4d_path, os.path.join(target_dir, "4d.mp4"))

    def _cleanup_sample_workdir_after_export(self, sample_output_dir: Optional[str]) -> None:
        if not sample_output_dir or not os.path.isdir(sample_output_dir):
            return

        removable_dirs = [
            "images",
            "masks",
            "masks_raw",
            "masks_refined",
            "rendered_frames",
            "rendered_frames_individual",
            "mesh_4d_individual",
            "focal_4d_individual",
            "completion_refined",
            "wan_export",
        ]
        for dirname in removable_dirs:
            path = os.path.join(sample_output_dir, dirname)
            if os.path.isdir(path):
                shutil.rmtree(path)

        for video_path in glob.glob(os.path.join(sample_output_dir, "4d_*.mp4")):
            if os.path.isfile(video_path):
                os.remove(video_path)

    def _package_successful_wan_export_outputs(
        self,
        *,
        sample_output_dir: Optional[str],
        sample_uuid: Optional[str],
        wan_export_root: Optional[str],
        final_4d_path: Optional[str],
    ) -> list[str]:
        from scripts.wan_sample_types import WanExportConfig

        wan_cfg = WanExportConfig.from_runtime(cfg_get(self.CONFIG, "wan_export", {}))
        if not bool(wan_cfg.enable):
            return []

        target_dirs = self._load_wan_exported_target_dirs(sample_uuid, wan_export_root)
        if not target_dirs:
            return []

        if bool(wan_cfg.copy_rendered_4d_to_targets):
            self._copy_rendered_4d_to_wan_targets(final_4d_path, target_dirs)
        if bool(wan_cfg.cleanup_sample_workdir_after_export):
            self._cleanup_sample_workdir_after_export(sample_output_dir)
        return target_dirs

    def _persist_sample_runtime(
        self,
        *,
        start_time: float,
        sample_output_dir: Optional[str],
        runtime_profile: Optional[dict],
        sample_uuid: Optional[str],
        wan_export_root: Optional[str],
    ) -> dict:
        pipeline_seconds = max(0.0, float(time.perf_counter() - start_time))
        runtime_payload = {
            "status": self.sample_summary.get("status", "unknown"),
            "pipeline_seconds": pipeline_seconds,
        }
        if runtime_profile is not None:
            runtime_payload["runtime_profile"] = runtime_profile
        if sample_uuid:
            runtime_payload["sample_uuid"] = sample_uuid
        runtime_payload["fps_summary"] = copy.deepcopy(self.sample_summary.get("fps_summary", {}))

        self.sample_summary["pipeline_runtime"] = runtime_payload
        if sample_output_dir and os.path.isdir(sample_output_dir):
            with open(os.path.join(sample_output_dir, "sample_runtime.json"), "w", encoding="utf-8") as handle:
                json.dump(runtime_payload, handle, indent=2)

        if sample_uuid and wan_export_root:
            from scripts.wan_sample_export import update_wan_sample_summary

            update_wan_sample_summary(
                wan_export_root,
                sample_uuid,
                {
                    "pipeline_runtime": runtime_payload,
                    "fps_summary": copy.deepcopy(self.sample_summary.get("fps_summary", {})),
                },
            )
        return runtime_payload

    def _ensure_base_app(self):
        if self._base_app is None:
            self._base_module = load_base_offline_module()
            self._base_app = self._base_module.OfflineApp(config_path=self.config_path)
        output_dir = self.sample_state.get("output_dir") if self.sample_state.get("output_dir_ready") else None
        self._sync_base_app_runtime(self._base_app, output_dir)
        return self._base_app

    def _sync_base_app_runtime(self, runtime_app, output_dir: Optional[str] = None):
        runtime_app.CONFIG = clone_runtime_config(self.CONFIG)
        if output_dir:
            runtime_app.OUTPUT_DIR = os.path.abspath(output_dir)
            os.makedirs(runtime_app.OUTPUT_DIR, exist_ok=True)
        runtime_app.RUNTIME["batch_size"] = int(
            cfg_get(self.CONFIG, "sam_3d_body.batch_size", runtime_app.RUNTIME.get("batch_size", 1))
        )
        runtime_app.RUNTIME["detection_resolution"] = list(
            cfg_get(
                self.CONFIG,
                "completion.detection_resolution",
                runtime_app.RUNTIME.get("detection_resolution", [256, 512]),
            )
        )
        runtime_app.RUNTIME["completion_resolution"] = list(
            cfg_get(
                self.CONFIG,
                "completion.completion_resolution",
                runtime_app.RUNTIME.get("completion_resolution", [512, 1024]),
            )
        )
        runtime_app.RUNTIME["smpl_export"] = bool(
            cfg_get(self.CONFIG, "runtime.smpl_export", runtime_app.RUNTIME.get("smpl_export", False))
        )
        runtime_app.RUNTIME["pose_exports"] = list(
            cfg_get(self.CONFIG, "runtime.pose_exports", runtime_app.RUNTIME.get("pose_exports", [])) or []
        )
        runtime_app.RUNTIME["wan_export"] = to_plain_runtime_dict(
            cfg_get(self.CONFIG, "wan_export", runtime_app.RUNTIME.get("wan_export", {}))
        )
        runtime_app.RUNTIME["save_rendered_video"] = bool(
            cfg_get(self.CONFIG, "runtime.save_rendered_video", runtime_app.RUNTIME.get("save_rendered_video", True))
        )
        runtime_app.RUNTIME["save_rendered_video_direct"] = bool(
            cfg_get(
                self.CONFIG,
                "runtime.save_rendered_video_direct",
                runtime_app.RUNTIME.get("save_rendered_video_direct", False),
            )
        )
        runtime_app.RUNTIME["save_rendered_frames"] = bool(
            cfg_get(self.CONFIG, "runtime.save_rendered_frames", runtime_app.RUNTIME.get("save_rendered_frames", True))
        )
        runtime_app.RUNTIME["save_rendered_frames_individual"] = bool(
            cfg_get(
                self.CONFIG,
                "runtime.save_rendered_frames_individual",
                runtime_app.RUNTIME.get("save_rendered_frames_individual", True),
            )
        )
        runtime_app.RUNTIME["save_mesh_4d_individual"] = bool(
            cfg_get(
                self.CONFIG,
                "runtime.save_mesh_4d_individual",
                runtime_app.RUNTIME.get("save_mesh_4d_individual", True),
            )
        )
        runtime_app.RUNTIME["save_focal_4d_individual"] = bool(
            cfg_get(
                self.CONFIG,
                "runtime.save_focal_4d_individual",
                runtime_app.RUNTIME.get("save_focal_4d_individual", True),
            )
        )
        self._configure_detector(runtime_app)
        self.OUTPUT_DIR = runtime_app.OUTPUT_DIR
        self.RUNTIME = runtime_app.RUNTIME

    def _configure_detector(self, runtime_app):
        backend = str(cfg_get(self.CONFIG, "detector.backend", "vitdet"))
        detector_path = str(cfg_get(self.CONFIG, "sam_3d_body.detector_path", "") or "")
        weights_path = str(cfg_get(self.CONFIG, "detector.weights_path", "") or "")
        device = str(getattr(runtime_app.sam3_3d_body_model, "device", "cuda"))
        backend_normalized = backend.lower()
        signature_weights_path = weights_path if backend_normalized in {"yolo", "yolo11"} else ""
        signature = (backend, detector_path, signature_weights_path, device)
        if signature == self._detector_signature and getattr(runtime_app.sam3_3d_body_model, "detector", None) is not None:
            return

        from models.sam_3d_body.tools.build_detector import HumanDetector

        detector_kwargs = {}
        if backend_normalized in {"yolo", "yolo11"}:
            if weights_path:
                detector_kwargs["weights_path"] = weights_path
            elif detector_path:
                detector_kwargs["path"] = detector_path
        elif detector_path:
            detector_kwargs["path"] = detector_path

        runtime_app.sam3_3d_body_model.detector = HumanDetector(
            name=backend,
            device=device,
            **detector_kwargs,
        )
        self._detector_signature = signature

    def _detect_frame_candidates(self, runtime_app, frame_rgb: np.ndarray, bbox_thr: float, nms_thr: float) -> List[dict]:
        model = getattr(runtime_app, "sam3_3d_body_model", None)
        detector = getattr(model, "detector", None)
        detector_outputs = None
        backend = str(cfg_get(self.CONFIG, "detector.backend", "vitdet")).lower()
        resolved = resolve_detector_runtime_options(
            backend,
            bbox_thresh=bbox_thr,
            iou_thresh=nms_thr,
            max_det=cfg_get(self.CONFIG, "detector.max_det", None),
        )

        if detector is not None and hasattr(detector, "run_human_detection"):
            detector_kwargs = {
                "det_cat_id": 0,
                "bbox_thr": resolved["bbox_thresh"],
                "nms_thr": resolved["iou_thresh"],
                "default_to_full_image": False,
                "return_scores": True,
            }
            if resolved["max_det"] is not None:
                detector_kwargs["max_det"] = int(resolved["max_det"])

            try:
                detector_outputs = detector.run_human_detection(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    **detector_kwargs,
                )
            except TypeError:
                detector_kwargs.pop("return_scores", None)
                detector_outputs = detector.run_human_detection(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    **detector_kwargs,
                )

            normalized_outputs = normalize_detection_outputs(detector_outputs)
            if normalized_outputs is not None:
                return normalized_outputs

        frame_outputs = model.process_one_image(
            frame_rgb,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
        )
        normalized_outputs = normalize_detection_outputs(frame_outputs)
        if normalized_outputs is None:
            return []
        return normalized_outputs

    def _load_source_frame(self, sample: dict, frame_idx: int) -> np.ndarray:
        if sample["input_type"] == "video":
            capture = cv2.VideoCapture(sample["input_video"])
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video: {sample['input_video']}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_path = sample["source_frames"][frame_idx]
        return np.array(Image.open(frame_path).convert("RGB"))

    def _copy_mask_sequence(self, source_dir: str, target_dir: str):
        os.makedirs(target_dir, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(source_dir, "*.png"))):
            shutil.copy2(path, os.path.join(target_dir, os.path.basename(path)))

    def _build_empty_mask(self) -> np.ndarray:
        for candidate_dir in (
            self.output_paths.get("masks_raw", ""),
            os.path.join(self.OUTPUT_DIR, "masks"),
            self.output_paths.get("masks_refined", ""),
        ):
            if not candidate_dir or not os.path.isdir(candidate_dir):
                continue
            candidates = sorted(glob.glob(os.path.join(candidate_dir, "*.png")))
            if candidates:
                return np.zeros_like(load_indexed_mask(candidates[0]), dtype=np.uint8)
        return np.zeros((1, 1), dtype=np.uint8)

    def _generate_tracking_outputs(self):
        if self._tracking_generated:
            return
        runtime_app = self._ensure_base_app()
        runtime_app.on_mask_generation(video_path=self.sample_state["input_video"], start_frame_idx=0)
        raw_source_dir = os.path.join(self.OUTPUT_DIR, "masks")
        self._copy_mask_sequence(raw_source_dir, self.output_paths["masks_raw"])
        self._tracking_generated = True

    def run_sample(
        self,
        input_video: str,
        output_dir: Optional[str],
        skip_existing: bool,
        runtime_profile: Optional[dict] = None,
    ) -> dict:
        self.reset_sample_state()
        shared_cfg = self.CONFIG
        sample_cfg = clone_runtime_config(shared_cfg)
        resolved_runtime_profile = apply_sample_runtime_profile(sample_cfg, runtime_profile)
        self.CONFIG = sample_cfg
        self.sample_config = sample_cfg
        self.sample_summary = {"status": "running", "runtime_profile": resolved_runtime_profile}
        pipeline_start = time.perf_counter()
        sample = None
        sample_output_dir = None
        sample_uuid = None
        wan_export_root = None

        try:
            sample = self.prepare_input(input_video, output_dir, skip_existing)
            sample_output_dir = sample.get("output_dir")
            self.sample_summary["fps_summary"] = self._build_sample_fps_summary(sample)
            face_presence = self._probe_sample_face_presence(sample)
            self.sample_summary["face_presence"] = face_presence
            if self._should_skip_sample_for_face_presence(face_presence):
                self.sample_summary["status"] = "skipped"
                self.sample_summary["skip_reason"] = "face_presence_below_threshold"
                self._append_issue_ledger_record(
                    self._build_issue_ledger_record(
                        event_type="sample_skipped_no_face",
                        status="skipped",
                        reason="face_presence_below_threshold",
                        sample=sample,
                        runtime_profile=resolved_runtime_profile,
                        details=face_presence,
                    ),
                    sample=sample,
                )
                return self.sample_summary

            sample_uuid, wan_export_root = self._ensure_wan_sample_summary(sample)
            initial_targets = self.detect_initial_targets(sample)
            self.prepare_sample_output(sample["output_dir"], initial_targets["obj_ids"])

            for chunk in self.iter_chunks(sample["frames"], sample_cfg.tracking.chunk_size):
                raw_chunk = self.track_chunk(chunk, initial_targets)
                refined_chunk = self.refine_chunk_masks(raw_chunk)
                final_chunk = self.maybe_reprompt_chunk(chunk, refined_chunk, initial_targets)
                self.write_chunk_outputs(chunk, raw_chunk, final_chunk)

            final_4d_path = self.run_refined_4d_generation()
            target_dirs = self._package_successful_wan_export_outputs(
                sample_output_dir=sample_output_dir,
                sample_uuid=sample_uuid,
                wan_export_root=wan_export_root,
                final_4d_path=final_4d_path,
            )
            if target_dirs:
                self.sample_summary["exported_target_dirs"] = list(target_dirs)
            self.sample_summary["status"] = "completed"
        except Exception as exc:
            self.sample_summary["status"] = "failed"
            self._append_issue_ledger_record(
                self._build_issue_ledger_record(
                    event_type="sample_failed",
                    status="failed",
                    reason="pipeline_exception",
                    sample=sample,
                    sample_uuid=sample_uuid,
                    runtime_profile=resolved_runtime_profile,
                    details={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                    input_video=input_video,
                ),
                sample=sample,
            )
            raise
        finally:
            self._persist_sample_runtime(
                start_time=pipeline_start,
                sample_output_dir=sample_output_dir,
                runtime_profile=resolved_runtime_profile,
                sample_uuid=sample_uuid,
                wan_export_root=wan_export_root,
            )
            try:
                self.finalize_sample()
            finally:
                self.CONFIG = shared_cfg
        return self.sample_summary

    def prepare_input(self, input_video: str, output_dir: Optional[str], skip_existing: bool) -> dict:
        del skip_existing

        input_path = os.path.abspath(input_video)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"input path does not exist: {input_video}")

        if os.path.isfile(input_path):
            if not input_path.lower().endswith(".mp4"):
                raise ValueError(f"input file must be an .mp4 video: {input_video}")
            frame_count = count_video_frames(input_path)
            if frame_count <= 0:
                raise ValueError(f"unable to count frames for video: {input_video}")
            sample = {
                "input_video": input_path,
                "input_type": "video",
                "source_frames": None,
                "frames": build_frame_stems(frame_count),
                "frame_count": frame_count,
            }
        elif os.path.isdir(input_path):
            source_frames = list_input_frames(input_path)
            if not source_frames:
                raise ValueError(f"input directory contains no image frames: {input_video}")
            sample = {
                "input_video": input_path,
                "input_type": "images",
                "source_frames": source_frames,
                "frames": build_frame_stems(len(source_frames)),
                "frame_count": len(source_frames),
            }
        else:
            raise ValueError(f"unsupported input path: {input_video}")

        resolved_output_dir = resolve_sample_output_dir(
            output_dir,
            cfg_get(self.CONFIG, "runtime.output_dir", os.path.join(ROOT, "outputs_refined")),
        )
        sample["output_dir"] = resolved_output_dir
        sample["output_dir_ready"] = False

        self.sample_state = dict(sample)
        self.OUTPUT_DIR = resolved_output_dir
        return dict(sample)

    def detect_initial_targets(self, sample: dict) -> dict:
        runtime_app = self._ensure_base_app()
        self.sample_state = dict(sample)

        search_frames = min(
            int(cfg_get(self.CONFIG, "batch.initial_search_frames", 24)),
            int(sample["frame_count"]),
        )
        resolved_detector = resolve_detector_runtime_options(
            cfg_get(self.CONFIG, "detector.backend", "vitdet"),
            bbox_thresh=cfg_get(self.CONFIG, "detector.bbox_thresh", None),
            iou_thresh=cfg_get(self.CONFIG, "detector.iou_thresh", None),
            max_det=cfg_get(self.CONFIG, "detector.max_det", None),
        )
        bbox_thr = resolved_detector["bbox_thresh"]
        nms_thr = resolved_detector["iou_thresh"]

        outputs = []
        start_frame_idx = None
        width = height = None
        max_detection_count = 0
        for frame_idx in range(search_frames):
            frame_rgb = self._load_source_frame(sample, frame_idx)
            frame_height, frame_width = frame_rgb.shape[:2]
            frame_outputs = self._detect_frame_candidates(runtime_app, frame_rgb, bbox_thr, nms_thr)
            detection_count = len(frame_outputs)
            if detection_count > max_detection_count:
                outputs = frame_outputs
                start_frame_idx = frame_idx
                height, width = frame_height, frame_width
                max_detection_count = detection_count

        if start_frame_idx is None or not outputs:
            raise RuntimeError(
                f"no humans detected within the first {search_frames} frames for {sample['input_video']}"
            )

        outputs = limit_detection_outputs(
            outputs,
            int(cfg_get(self.CONFIG, "detector.max_targets", 0)),
        )

        if sample["input_type"] == "video":
            inference_state = runtime_app.predictor.init_state(video_path=sample["input_video"])
        else:
            inference_state = runtime_app.predictor.init_state(video_path=sample["source_frames"])
        runtime_app.predictor.clear_all_points_in_video(inference_state)
        runtime_app.RUNTIME["inference_state"] = inference_state
        runtime_app.RUNTIME["out_obj_ids"] = []

        detected_boxes = []
        for obj_index, output in enumerate(outputs, start=1):
            xmin, ymin, xmax, ymax = [float(value) for value in output["bbox"]]
            rel_box = np.array([[xmin / width, ymin / height, xmax / width, ymax / height]], dtype=np.float32)
            _, runtime_app.RUNTIME["out_obj_ids"], _, _ = runtime_app.predictor.add_new_points_or_box(
                inference_state=runtime_app.RUNTIME["inference_state"],
                frame_idx=start_frame_idx,
                obj_id=obj_index,
                box=rel_box,
            )
            detected_boxes.append([xmin, ymin, xmax, ymax])
            self._last_track_boxes[obj_index] = [xmin, ymin, xmax, ymax]

        self.initial_targets = {
            "obj_ids": list(runtime_app.RUNTIME["out_obj_ids"]),
            "start_frame_idx": int(start_frame_idx),
            "boxes_xyxy": detected_boxes,
        }
        self.sample_summary["initial_targets"] = {
            "obj_ids": list(runtime_app.RUNTIME["out_obj_ids"]),
            "start_frame_idx": int(start_frame_idx),
        }
        return dict(self.initial_targets)

    def iter_chunks(self, frames: List[str], chunk_size: int):
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")

        chunk_id = 0
        for start_frame in range(0, len(frames), int(chunk_size)):
            chunk_frames = frames[start_frame : start_frame + int(chunk_size)]
            yield {
                "chunk_id": chunk_id,
                "start_frame": start_frame,
                "end_frame": start_frame + len(chunk_frames) - 1,
                "frame_indices": list(range(start_frame, start_frame + len(chunk_frames))),
                "frames": chunk_frames,
            }
            chunk_id += 1

    def track_chunk(self, chunk: dict, initial_targets: dict) -> dict:
        del initial_targets

        self._generate_tracking_outputs()
        empty_mask = self._build_empty_mask()
        raw_masks = []
        raw_mask_paths = []
        image_paths = []
        for frame_stem in chunk["frames"]:
            raw_mask_path = os.path.join(self.output_paths["masks_raw"], f"{frame_stem}.png")
            image_path = os.path.join(self.OUTPUT_DIR, "images", f"{frame_stem}.jpg")
            if os.path.isfile(raw_mask_path):
                raw_masks.append(load_indexed_mask(raw_mask_path))
            else:
                raw_masks.append(empty_mask.copy())
            raw_mask_paths.append(raw_mask_path)
            image_paths.append(image_path)

        return {
            "chunk_id": chunk["chunk_id"],
            "frame_indices": list(chunk["frame_indices"]),
            "frame_stems": list(chunk["frames"]),
            "image_paths": image_paths,
            "raw_mask_paths": raw_mask_paths,
            "raw_masks": raw_masks,
        }

    def refine_chunk_masks(self, raw_chunk: dict) -> dict:
        if not bool(cfg_get(self.CONFIG, "refine.enable", True)):
            return self._passthrough_chunk_masks(raw_chunk)

        obj_ids = list(self.initial_targets.get("obj_ids", []))
        refined_masks = []
        frame_metrics = []

        for frame_idx, frame_stem, raw_mask in zip(
            raw_chunk["frame_indices"],
            raw_chunk["frame_stems"],
            raw_chunk["raw_masks"],
        ):
            refined_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            track_metrics = {}

            for obj_id in obj_ids:
                binary = keep_largest_component_for_label(raw_mask, obj_id)
                previous_binary = self._last_binary_masks.get(obj_id)
                previous_area = int(previous_binary.sum()) if previous_binary is not None else int(binary.sum())
                current_area = int(binary.sum())
                empty_mask_count = self._empty_mask_counts.get(obj_id, 0)
                refined_from_previous = False

                if current_area == 0:
                    empty_mask_count += 1
                    if previous_binary is not None and previous_binary.sum() > 0:
                        binary = previous_binary.copy()
                        current_area = int(binary.sum())
                        refined_from_previous = True
                else:
                    empty_mask_count = 0

                if current_area > 0:
                    refined_mask[binary > 0] = int(obj_id)
                    self._last_binary_masks[obj_id] = binary.copy()
                    bbox_xyxy = mask_bbox_xyxy(binary)
                    if bbox_xyxy is not None:
                        self._last_track_boxes[obj_id] = bbox_xyxy

                self._empty_mask_counts[obj_id] = empty_mask_count
                iou_value = 1.0 if previous_binary is None else mask_iou(previous_binary, binary)
                area_ratio = 1.0 if previous_area <= 0 else float(current_area) / float(previous_area)
                track_metrics[str(obj_id)] = {
                    "empty_mask_count": empty_mask_count,
                    "area_ratio": area_ratio,
                    "edge_touch_ratio": edge_touch_ratio(binary),
                    "mask_iou": iou_value,
                    "refined_from_previous": refined_from_previous,
                    "bbox_xyxy": self._last_track_boxes.get(obj_id),
                }

            refined_masks.append(refined_mask)
            frame_metrics.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_stem,
                    "track_metrics": track_metrics,
                }
            )

        return {
            "frame_stems": list(raw_chunk["frame_stems"]),
            "refined_masks": refined_masks,
            "frame_metrics": frame_metrics,
        }

    def _passthrough_chunk_masks(self, raw_chunk: dict) -> dict:
        obj_ids = list(self.initial_targets.get("obj_ids", []))
        passthrough_masks = []
        frame_metrics = []

        for frame_idx, frame_stem, raw_mask in zip(
            raw_chunk["frame_indices"],
            raw_chunk["frame_stems"],
            raw_chunk["raw_masks"],
        ):
            passthrough_mask = np.asarray(raw_mask, dtype=np.uint8).copy()
            track_metrics = {}

            for obj_id in obj_ids:
                binary = (passthrough_mask == int(obj_id)).astype(np.uint8)
                previous_binary = self._last_binary_masks.get(obj_id)
                previous_area = int(previous_binary.sum()) if previous_binary is not None else int(binary.sum())
                current_area = int(binary.sum())
                empty_mask_count = self._empty_mask_counts.get(obj_id, 0)

                if current_area == 0:
                    empty_mask_count += 1
                else:
                    empty_mask_count = 0
                    self._last_binary_masks[obj_id] = binary.copy()
                    bbox_xyxy = mask_bbox_xyxy(binary)
                    if bbox_xyxy is not None:
                        self._last_track_boxes[obj_id] = bbox_xyxy

                self._empty_mask_counts[obj_id] = empty_mask_count
                iou_value = 1.0 if previous_binary is None else mask_iou(previous_binary, binary)
                area_ratio = 1.0 if previous_area <= 0 else float(current_area) / float(previous_area)
                track_metrics[str(obj_id)] = {
                    "empty_mask_count": empty_mask_count,
                    "area_ratio": area_ratio,
                    "edge_touch_ratio": edge_touch_ratio(binary),
                    "mask_iou": iou_value,
                    "refined_from_previous": False,
                    "bbox_xyxy": self._last_track_boxes.get(obj_id),
                }

            passthrough_masks.append(passthrough_mask)
            frame_metrics.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_stem,
                    "track_metrics": track_metrics,
                }
            )

        return {
            "frame_stems": list(raw_chunk["frame_stems"]),
            "refined_masks": passthrough_masks,
            "frame_metrics": frame_metrics,
        }

    def maybe_reprompt_chunk(self, chunk: dict, refined_chunk: dict, initial_targets: dict) -> dict:
        del chunk, initial_targets

        if not bool(cfg_get(self.CONFIG, "reprompt.enable", True)):
            for frame_metric in refined_chunk["frame_metrics"]:
                frame_metric["triggered_reprompt"] = False
            refined_chunk["reprompt_events"] = []
            return refined_chunk

        reprompt_events = []
        for frame_metric in refined_chunk["frame_metrics"]:
            triggered = False
            for obj_id, metrics in frame_metric["track_metrics"].items():
                if self.should_trigger_reprompt(metrics, self.reprompt_thresholds):
                    triggered = True
                    reprompt_events.append(
                        {
                            "frame_idx": frame_metric["frame_idx"],
                            "frame_stem": frame_metric["frame_stem"],
                            "obj_id": int(obj_id),
                            "reason": "mask_quality_trigger",
                            "action": "diagnostic_only",
                            "metrics": metrics,
                        }
                    )
            frame_metric["triggered_reprompt"] = triggered

        refined_chunk["reprompt_events"] = reprompt_events
        return refined_chunk

    def write_chunk_outputs(self, chunk: dict, raw_chunk: dict, final_chunk: dict) -> None:
        del raw_chunk

        refined_dir = self.output_paths.get("masks_refined", os.path.join(self.OUTPUT_DIR, "masks_refined"))
        working_dir = os.path.join(self.OUTPUT_DIR, "masks")
        os.makedirs(refined_dir, exist_ok=True)
        os.makedirs(working_dir, exist_ok=True)

        for frame_stem, refined_mask in zip(final_chunk["frame_stems"], final_chunk["refined_masks"]):
            refined_path = os.path.join(refined_dir, f"{frame_stem}.png")
            working_path = os.path.join(working_dir, f"{frame_stem}.png")
            save_indexed_mask(refined_mask, refined_path)
            save_indexed_mask(refined_mask, working_path)

        reprompt_events = list(final_chunk.get("reprompt_events", []))
        if reprompt_events:
            self.reprompt_events.extend(reprompt_events)

        triggered_reprompt_count = sum(
            1 for frame_metric in final_chunk.get("frame_metrics", []) if frame_metric.get("triggered_reprompt")
        )
        self.chunk_records.append(
            {
                "chunk_id": int(chunk["chunk_id"]),
                "start_frame": int(chunk["start_frame"]),
                "end_frame": int(chunk["end_frame"]),
                "frame_count": len(final_chunk["frame_stems"]),
                "reprompt_event_count": len(reprompt_events),
                "frames_with_reprompt": triggered_reprompt_count,
            }
        )

    def run_refined_4d_generation(self) -> None:
        runtime_app = self._ensure_base_app()
        refined_dir = self.output_paths.get("masks_refined", os.path.join(self.OUTPUT_DIR, "masks_refined"))
        working_dir = os.path.join(self.OUTPUT_DIR, "masks")
        if os.path.isdir(refined_dir):
            self._copy_mask_sequence(refined_dir, working_dir)
        _align_completion_pipeline_dtypes(runtime_app)
        with _autocast_disabled():
            return runtime_app.on_4d_generation(video_path=self.sample_state.get("input_video"))


def run_refined_pipeline(args) -> None:
    cfg = apply_runtime_overrides(args, load_refined_config(args.config))
    print(f"Loaded refined config for detector backend: {cfg.detector.backend}")
    app = RefinedOfflineApp(args.config, config=cfg)
    app.reprompt_thresholds = build_reprompt_thresholds(cfg)
    app.run_sample(args.input_video, args.output_dir, args.skip_existing, runtime_profile=None)
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_refined_pipeline(args)


if __name__ == "__main__":
    main()
