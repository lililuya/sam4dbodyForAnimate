import argparse
import copy
import glob
import json
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.offline_reprompt import match_detection_to_track, should_trigger_reprompt


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


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
    if args.track_chunk_size is not None:
        cfg.tracking.chunk_size = args.track_chunk_size
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


def prepare_output_dirs(output_dir: str, obj_ids: Iterable[int], save_debug_metrics: bool) -> Dict[str, str]:
    paths = {
        "images": os.path.join(output_dir, "images"),
        "masks_raw": os.path.join(output_dir, "masks_raw"),
        "masks_refined": os.path.join(output_dir, "masks_refined"),
        "completion_images": os.path.join(output_dir, "completion_refined", "images"),
        "completion_masks": os.path.join(output_dir, "completion_refined", "masks"),
        "rendered_frames": os.path.join(output_dir, "rendered_frames"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    for obj_id in obj_ids:
        os.makedirs(os.path.join(output_dir, "mesh_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "focal_4d_individual", str(obj_id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rendered_frames_individual", str(obj_id)), exist_ok=True)
    if save_debug_metrics:
        debug_dir = os.path.join(output_dir, "debug_metrics")
        os.makedirs(debug_dir, exist_ok=True)
        paths["debug_metrics"] = debug_dir
    return paths


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


def build_frame_stems(frame_count: int) -> List[str]:
    return [f"{frame_idx:08d}" for frame_idx in range(int(frame_count))]


def save_indexed_mask(mask: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="P").save(path)


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
        self.chunk_records = []
        self.output_paths = prepare_output_dirs(
            output_dir,
            obj_ids,
            save_debug_metrics=bool(self.CONFIG.debug.save_metrics),
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

    def _ensure_base_app(self):
        if self._base_app is None:
            self._base_module = load_base_offline_module()
            self._base_app = self._base_module.OfflineApp(config_path=self.config_path)
        self._sync_base_app_runtime(self._base_app, self.sample_state.get("output_dir"))
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
        self._configure_detector(runtime_app)
        self.OUTPUT_DIR = runtime_app.OUTPUT_DIR
        self.RUNTIME = runtime_app.RUNTIME

    def _configure_detector(self, runtime_app):
        backend = str(cfg_get(self.CONFIG, "detector.backend", "vitdet"))
        detector_path = str(cfg_get(self.CONFIG, "sam_3d_body.detector_path", "") or "")
        weights_path = str(cfg_get(self.CONFIG, "detector.weights_path", "") or "")
        device = str(getattr(runtime_app.sam3_3d_body_model, "device", "cuda"))
        signature = (backend, detector_path, weights_path, device)
        if signature == self._detector_signature and getattr(runtime_app.sam3_3d_body_model, "detector", None) is not None:
            return

        from models.sam_3d_body.tools.build_detector import HumanDetector

        detector_kwargs = {}
        if weights_path:
            detector_kwargs["weights_path"] = weights_path
        elif detector_path:
            detector_kwargs["path"] = detector_path

        runtime_app.sam3_3d_body_model.detector = HumanDetector(
            name=backend,
            device=device,
            **detector_kwargs,
        )
        self._detector_signature = signature

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

        try:
            sample = self.prepare_input(input_video, output_dir, skip_existing)
            initial_targets = self.detect_initial_targets(sample)
            self.prepare_sample_output(sample["output_dir"], initial_targets["obj_ids"])

            for chunk in self.iter_chunks(sample["frames"], sample_cfg.tracking.chunk_size):
                raw_chunk = self.track_chunk(chunk, initial_targets)
                refined_chunk = self.refine_chunk_masks(raw_chunk)
                final_chunk = self.maybe_reprompt_chunk(chunk, refined_chunk, initial_targets)
                self.write_chunk_outputs(chunk, raw_chunk, final_chunk)

            self.run_refined_4d_generation()
            self.sample_summary["status"] = "completed"
        except Exception:
            self.sample_summary["status"] = "failed"
            raise
        finally:
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

        resolved_output_dir = os.path.abspath(
            output_dir or cfg_get(self.CONFIG, "runtime.output_dir", os.path.join(ROOT, "outputs_refined"))
        )
        os.makedirs(resolved_output_dir, exist_ok=True)
        sample["output_dir"] = resolved_output_dir

        self.sample_state = dict(sample)
        self.OUTPUT_DIR = resolved_output_dir
        if self._base_app is not None:
            self._sync_base_app_runtime(self._base_app, resolved_output_dir)
        return dict(sample)

    def detect_initial_targets(self, sample: dict) -> dict:
        runtime_app = self._ensure_base_app()
        self.sample_state = dict(sample)

        search_frames = min(
            int(cfg_get(self.CONFIG, "batch.initial_search_frames", 24)),
            int(sample["frame_count"]),
        )
        bbox_thr = float(cfg_get(self.CONFIG, "detector.bbox_thresh", 0.35))
        nms_thr = float(cfg_get(self.CONFIG, "detector.iou_thresh", 0.50))

        outputs = []
        start_frame_idx = None
        width = height = None
        for frame_idx in range(search_frames):
            frame_rgb = self._load_source_frame(sample, frame_idx)
            height, width = frame_rgb.shape[:2]
            outputs = runtime_app.sam3_3d_body_model.process_one_image(
                frame_rgb,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
            )
            if outputs:
                start_frame_idx = frame_idx
                break

        if start_frame_idx is None or not outputs:
            raise RuntimeError(
                f"no humans detected within the first {search_frames} frames for {sample['input_video']}"
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
        return runtime_app.on_4d_generation(video_path=self.sample_state.get("input_video"))


def run_refined_pipeline(args) -> None:
    cfg = apply_runtime_overrides(args, load_refined_config(args.config))
    print(f"Loaded refined config for detector backend: {cfg.detector.backend}")
    app = RefinedOfflineApp(args.config, config=cfg)
    app.reprompt_thresholds = build_reprompt_thresholds(cfg)
    app.sample_summary = {"status": "running", "runtime_profile": {}}
    try:
        sample = app.prepare_input(args.input_video, cfg.runtime.output_dir, args.skip_existing)
        initial_targets = app.detect_initial_targets(sample)
        app.prepare_sample_output(sample["output_dir"], initial_targets["obj_ids"])

        for chunk in app.iter_chunks(sample["frames"], cfg.tracking.chunk_size):
            raw_chunk = app.track_chunk(chunk, initial_targets)
            refined_chunk = app.refine_chunk_masks(raw_chunk)
            final_chunk = app.maybe_reprompt_chunk(chunk, refined_chunk, initial_targets)
            app.write_chunk_outputs(chunk, raw_chunk, final_chunk)

        app.run_refined_4d_generation()
        app.sample_summary["status"] = "completed"
    except Exception:
        app.sample_summary["status"] = "failed"
        raise
    finally:
        app.finalize_sample()
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_refined_pipeline(args)


if __name__ == "__main__":
    main()
