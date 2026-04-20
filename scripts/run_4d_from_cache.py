import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from omegaconf import OmegaConf

from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context
from scripts.pose_json_export import write_pose_frame_exports
from scripts.sam3_cache_contract import load_json, validate_cache_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Run offline 4D from an exported SAM3 cache")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_runtime_app(config_path):
    from scripts.offline_app import OfflineApp

    return OfflineApp(config_path=config_path)


def build_cache_runtime(meta):
    runtime_profile = meta.get("runtime_profile", {})
    return {
        "out_obj_ids": list(meta.get("obj_ids", [])),
        "batch_size": int(runtime_profile.get("batch_size", 1)),
        "detection_resolution": list(runtime_profile.get("detection_resolution", [256, 512])),
        "completion_resolution": list(runtime_profile.get("completion_resolution", [512, 1024])),
        "smpl_export": bool(runtime_profile.get("smpl_export", False)),
        "video_fps": float(meta.get("fps", runtime_profile.get("fps", 0.0))),
    }


def resolve_output_root(output_root, config_path):
    if output_root is not None:
        return os.path.abspath(output_root)
    cfg = OmegaConf.load(config_path)
    return os.path.abspath(os.path.join(cfg.runtime.output_dir, "outputs_4d"))


def prepare_output_dir(sample_id, *, output_root, overwrite):
    os.makedirs(output_root, exist_ok=True)
    sample_output_dir = os.path.join(output_root, sample_id)
    if os.path.isdir(sample_output_dir):
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {sample_output_dir}")
        shutil.rmtree(sample_output_dir)
    os.makedirs(sample_output_dir, exist_ok=False)
    return sample_output_dir


def write_run_summary(*, output_dir, summary):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def discover_cache_dirs(root_dir, require_meta=True):
    if not os.path.isdir(root_dir):
        return []
    cache_dirs = []
    for name in sorted(os.listdir(root_dir)):
        cache_dir = os.path.join(root_dir, name)
        if not os.path.isdir(cache_dir):
            continue
        if require_meta and not os.path.isfile(os.path.join(cache_dir, "meta.json")):
            continue
        cache_dirs.append(cache_dir)
    return cache_dirs


def build_pose_frame_writer(output_dir):
    def frame_writer(image_path, mask_output, id_current):
        if not mask_output or not id_current:
            return {"openpose": [], "smpl": []}
        frame_stem = os.path.splitext(os.path.basename(image_path))[0]
        return write_pose_frame_exports(
            output_dir=output_dir,
            frame_stem=frame_stem,
            person_outputs=mask_output,
            track_ids=id_current,
        )

    return frame_writer


def run_cache_sample(*, cache_dir, output_root=None, overwrite=False, config_path=None):
    cache_dir = os.path.abspath(cache_dir)
    ok, errors = validate_cache_dir(cache_dir)
    if not ok:
        raise ValueError(f"invalid cache: {errors}")

    meta = load_json(os.path.join(cache_dir, "meta.json"))
    resolved_config_path = os.path.abspath(config_path or meta["config_path"])
    resolved_output_root = resolve_output_root(output_root, resolved_config_path)
    sample_output_dir = prepare_output_dir(
        meta["sample_id"],
        output_root=resolved_output_root,
        overwrite=overwrite,
    )
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        runtime_app = build_runtime_app(resolved_config_path)
        context = build_4d_context(
            input_dir=cache_dir,
            output_dir=sample_output_dir,
            runtime=build_cache_runtime(meta),
            sam3_3d_body_model=runtime_app.sam3_3d_body_model,
            pipeline_mask=getattr(runtime_app, "pipeline_mask", None),
            pipeline_rgb=getattr(runtime_app, "pipeline_rgb", None),
            depth_model=getattr(runtime_app, "depth_model", None),
            predictor=getattr(runtime_app, "predictor", None),
            generator=getattr(runtime_app, "generator", None),
            frame_writer=build_pose_frame_writer(sample_output_dir),
        )
        out_path = run_4d_pipeline_from_context(context)
        write_run_summary(
            output_dir=sample_output_dir,
            summary={
                "sample_id": meta["sample_id"],
                "cache_dir": cache_dir,
                "status": "completed",
                "config_path": resolved_config_path,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "output_video": out_path,
                "error": None,
            },
        )
        return out_path
    except Exception as exc:
        write_run_summary(
            output_dir=sample_output_dir,
            summary={
                "sample_id": meta["sample_id"],
                "cache_dir": cache_dir,
                "status": "failed",
                "config_path": resolved_config_path,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "output_video": None,
                "error": str(exc),
            },
        )
        raise


def run_cache_batch(root_dir, *, output_root=None, overwrite=False, config_path=None):
    results = []
    for cache_dir in discover_cache_dirs(root_dir):
        try:
            out_path = run_cache_sample(
                cache_dir=cache_dir,
                output_root=output_root,
                overwrite=overwrite,
                config_path=config_path,
            )
            results.append(
                {"cache_dir": cache_dir, "status": "completed", "output": out_path}
            )
        except Exception as exc:
            results.append(
                {"cache_dir": cache_dir, "status": "failed", "error": str(exc)}
            )
    return results


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_cache_sample(
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        overwrite=args.overwrite,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
