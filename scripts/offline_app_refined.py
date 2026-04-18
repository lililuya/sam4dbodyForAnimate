import argparse
import copy
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.offline_reprompt import match_detection_to_track, should_trigger_reprompt


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


class RefinedOfflineApp:
    def __init__(self, config_path: str, config=None):
        self.config_path = config_path
        self.CONFIG = config if config is not None else load_refined_config(config_path)
        self.reprompt_thresholds = build_reprompt_thresholds(self.CONFIG) if hasattr(self.CONFIG, "reprompt") else {}
        self.should_trigger_reprompt = should_trigger_reprompt
        self.match_detection_to_track = match_detection_to_track
        self.reset_sample_state()

    def reset_sample_state(self):
        self.chunk_records = []
        self.output_paths = {}
        self.reprompt_events = []
        self.sample_summary = {}
        self.sample_config = None

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
        raise NotImplementedError("prepare_input must collect frames and choose the sample output directory")

    def detect_initial_targets(self, sample: dict) -> dict:
        raise NotImplementedError("detect_initial_targets must run detector-driven initial prompting")

    def iter_chunks(self, frames: List[str], chunk_size: int):
        raise NotImplementedError("iter_chunks must yield bounded tracking windows")

    def track_chunk(self, chunk: dict, initial_targets: dict) -> dict:
        raise NotImplementedError("track_chunk must write raw masks and images for the current chunk")

    def refine_chunk_masks(self, raw_chunk: dict) -> dict:
        raise NotImplementedError("refine_chunk_masks must run the two-stage occlusion refinement flow")

    def maybe_reprompt_chunk(self, chunk: dict, refined_chunk: dict, initial_targets: dict) -> dict:
        raise NotImplementedError("maybe_reprompt_chunk must apply drift checks and re-prompt when needed")

    def write_chunk_outputs(self, chunk: dict, raw_chunk: dict, final_chunk: dict) -> None:
        raise NotImplementedError("write_chunk_outputs must persist masks, metrics, and per-chunk diagnostics")

    def run_refined_4d_generation(self) -> None:
        raise NotImplementedError("run_refined_4d_generation must run mesh reconstruction from refined masks")


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
