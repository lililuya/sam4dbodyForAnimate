import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.face_clip_pipeline import extract_face_clips_from_video, update_face_clip_batch_summary
from scripts.offline_app_refined import RefinedOfflineApp, apply_runtime_overrides, load_refined_config
from scripts.offline_batch_helpers import build_retry_profiles, discover_samples
from scripts.wan_sample_types import WanExportConfig
from scripts.wan_sample_export import resolve_or_create_wan_sample_uuid


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quality-preserving batch runner for refined offline 4D body generation"
    )
    parser.add_argument("--input_root", type=str, default="")
    parser.add_argument("--input_list", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "body4d_refined.yaml"))
    parser.add_argument("--detector_backend", type=str, default=None)
    parser.add_argument("--track_chunk_size", type=int, default=None)
    parser.add_argument("--disable_mask_refine", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--save_debug_metrics", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--retry_mode",
        type=str,
        choices=["never", "quality_safe", "aggressive_safe"],
        default=None,
    )
    return parser


def sample_output_dir(batch_output_dir: str, sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Unsafe sample_id: sample_id must be a non-empty string")

    normalized = sample_id.replace("\\", "/")
    drive, _ = os.path.splitdrive(sample_id)
    if os.path.isabs(sample_id) or drive:
        raise ValueError(
            f"Unsafe sample_id {sample_id!r}: absolute paths are not allowed"
        )
    if "/" in normalized:
        raise ValueError(
            f"Unsafe sample_id {sample_id!r}: path separators are not allowed"
        )
    if sample_id in {".", ".."}:
        raise ValueError(
            f"Unsafe sample_id {sample_id!r}: path traversal is not allowed"
        )

    batch_root = os.path.abspath(batch_output_dir)
    output_dir = os.path.abspath(os.path.join(batch_root, sample_id))
    if os.path.commonpath([batch_root, output_dir]) != batch_root:
        raise ValueError(
            f"Unsafe sample_id {sample_id!r}: sample output path escapes the batch output root"
        )
    return output_dir


def write_batch_manifest(batch_output_dir: str, samples: List[Dict[str, str]]) -> None:
    os.makedirs(batch_output_dir, exist_ok=True)
    manifest_path = os.path.join(batch_output_dir, "batch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)


def append_batch_result(batch_output_dir: str, record: Dict[str, object]) -> None:
    os.makedirs(batch_output_dir, exist_ok=True)
    results_path = os.path.join(batch_output_dir, "batch_results.jsonl")
    with open(results_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _resolve_wan_export_config(cfg) -> WanExportConfig:
    runtime = OmegaConf.select(cfg, "wan_export", default={})
    if runtime is None:
        return WanExportConfig()
    if OmegaConf.is_config(runtime):
        runtime = OmegaConf.to_container(runtime, resolve=False)
    return WanExportConfig.from_runtime(runtime)


def _resolve_face_clip_output_root(cfg, batch_output_dir: str) -> str:
    wan_cfg = _resolve_wan_export_config(cfg)
    if wan_cfg.face_clip_output_dir:
        return os.path.abspath(wan_cfg.face_clip_output_dir)
    return os.path.join(os.path.abspath(batch_output_dir), "face_clips")


def _build_sample_result_base(sample: Dict[str, str], output_dir: str, *, stage: str) -> Dict[str, object]:
    record: Dict[str, object] = {
        "stage": str(stage),
        "sample_id": sample["sample_id"],
        "input": sample["input"],
        "output_dir": output_dir,
    }
    for key in ("clip_id", "source_sample_id", "source_input"):
        value = sample.get(key)
        if value not in {None, ""}:
            record[key] = value
    return record


def _read_json_dict(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_clip_samples_from_ids(
    sample: Dict[str, str],
    clip_output_root: str,
    clip_ids: List[str],
) -> List[Dict[str, str]] | None:
    clip_samples: List[Dict[str, str]] = []
    for raw_clip_id in clip_ids:
        clip_id = str(raw_clip_id or "").strip()
        if not clip_id:
            return None
        clip_dir = os.path.join(clip_output_root, "clips", clip_id)
        required_paths = (
            os.path.join(clip_dir, "clip.mp4"),
            os.path.join(clip_dir, "track.json"),
            os.path.join(clip_dir, "meta.json"),
        )
        if not all(os.path.isfile(path) for path in required_paths):
            return None
        clip_samples.append(
            {
                "input": clip_dir,
                "sample_id": clip_id,
                "clip_id": clip_id,
                "source_sample_id": sample["sample_id"],
                "source_input": sample["input"],
            }
        )
    return clip_samples


def _reuse_existing_face_clip_samples(
    sample: Dict[str, str],
    cfg,
    batch_output_dir: str,
) -> tuple[List[Dict[str, str]], Dict[str, object]] | None:
    clip_output_root = _resolve_face_clip_output_root(cfg, batch_output_dir)
    summary = _read_json_dict(os.path.join(clip_output_root, "batch_summary.json"))
    items = list(summary.get("items") or [])
    input_path = os.path.abspath(str(sample["input"]))

    matched_item = None
    for item in items:
        if not isinstance(item, dict):
            continue
        source_path = os.path.abspath(str(item.get("source_path") or ""))
        if source_path == input_path:
            matched_item = item
            break

    if matched_item is None:
        return None

    existing_status = str(matched_item.get("status") or "").strip()
    clip_ids = [str(value).strip() for value in list(matched_item.get("clip_ids") or []) if str(value).strip()]
    if existing_status not in {"completed", "completed_no_clips", "skipped_no_face_precheck"}:
        return None

    clip_samples = _build_clip_samples_from_ids(sample, clip_output_root, clip_ids)
    if clip_samples is None:
        return None

    record = _build_sample_result_base(sample, clip_output_root, stage="face_clip_extraction")
    record.update(
        {
            "status": "skipped_existing",
            "existing_status": existing_status,
            "kept_clip_count": len(clip_samples),
            "clip_ids": [clip_sample["clip_id"] for clip_sample in clip_samples],
        }
    )
    if "face_presence" in matched_item and isinstance(matched_item["face_presence"], dict):
        record["face_presence"] = dict(matched_item["face_presence"])
    return clip_samples, record


def extract_face_first_clip_samples(app, sample, cfg, batch_output_dir, *, skip_existing: bool = False):
    input_path = str(sample["input"])
    if not input_path.lower().endswith(".mp4"):
        raise ValueError("face-first clip mode only supports .mp4 video inputs")

    wan_cfg = _resolve_wan_export_config(cfg)
    clip_output_root = _resolve_face_clip_output_root(cfg, batch_output_dir)
    if skip_existing:
        reused = _reuse_existing_face_clip_samples(sample, cfg, batch_output_dir)
        if reused is not None:
            return reused
    sample_probe = app.prepare_input(input_path, None, False)
    face_presence = app._probe_sample_face_presence(sample_probe)
    if app._should_skip_sample_for_face_presence(face_presence):
        sample_uuid = resolve_or_create_wan_sample_uuid(
            clip_output_root,
            input_path,
            sample_id=sample["sample_id"],
            working_output_dir=None,
        )
        update_face_clip_batch_summary(
            clip_output_root,
            input_video=input_path,
            item={
                "sample_uuid": sample_uuid,
                "source_path": input_path,
                "status": "skipped_no_face_precheck",
                "kept_clip_count": 0,
                "clip_ids": [],
                "dropped_segment_count": 0,
                "drop_reasons": ["face_presence_below_threshold"],
                "face_presence": dict(face_presence),
            },
        )
        record = _build_sample_result_base(sample, clip_output_root, stage="face_clip_extraction")
        record.update(
            {
                "status": "skipped_no_face_precheck",
                "kept_clip_count": 0,
                "clip_ids": [],
                "face_presence": dict(face_presence),
            }
        )
        return [], record

    clip_dirs = extract_face_clips_from_video(
        input_video=input_path,
        output_root=clip_output_root,
        min_clip_seconds=float(wan_cfg.face_clip_min_clip_seconds),
        face_backend=app._ensure_face_backend(),
        sample_id=sample["sample_id"],
        target_fps=float(wan_cfg.fps),
    )
    clip_ids = [os.path.basename(os.path.abspath(path)) for path in clip_dirs]
    clip_samples = [
        {
            "input": clip_dir,
            "sample_id": clip_id,
            "clip_id": clip_id,
            "source_sample_id": sample["sample_id"],
            "source_input": input_path,
        }
        for clip_dir, clip_id in zip(clip_dirs, clip_ids)
    ]
    record = _build_sample_result_base(sample, clip_output_root, stage="face_clip_extraction")
    record.update(
        {
            "status": "completed" if clip_dirs else "completed_no_clips",
            "kept_clip_count": len(clip_dirs),
            "clip_ids": clip_ids,
            "face_presence": dict(face_presence),
        }
    )
    return clip_samples, record


def should_skip_sample(sample_output_path: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    summary_path = os.path.join(sample_output_path, "debug_metrics", "sample_summary.json")
    if not os.path.isfile(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict):
        return False
    return summary.get("status") == "completed"


def run_sample_with_retries(app, sample, batch_output_dir, cfg, args):
    output_dir = sample_output_dir(batch_output_dir, sample["sample_id"])
    base_record = _build_sample_result_base(sample, output_dir, stage="refined_pipeline")
    if should_skip_sample(output_dir, args.skip_existing):
        return {
            **base_record,
            "status": "skipped",
            "retry_index": 0,
            "runtime_profile": None,
        }

    profiles = build_retry_profiles(cfg, args)
    last_error = None
    last_runtime_profile = None

    for runtime_profile in profiles:
        profile_for_call = dict(runtime_profile)
        try:
            summary = app.run_sample(
                input_video=sample["input"],
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                runtime_profile=profile_for_call,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue through safe retry profiles
            last_error = exc
            last_runtime_profile = profile_for_call
            continue

        return {
            **base_record,
            "status": "completed" if profile_for_call["retry_index"] == 0 else "retry_succeeded",
            "summary_status": summary["status"],
            "retry_index": profile_for_call["retry_index"],
            "runtime_profile": profile_for_call,
        }

    return {
        **base_record,
        "status": "failed",
        "retry_index": None if last_runtime_profile is None else last_runtime_profile["retry_index"],
        "runtime_profile": last_runtime_profile,
        "error": "" if last_error is None else str(last_error),
    }


def run_batch(args):
    runtime_override_args = SimpleNamespace(
        output_dir=args.output_dir,
        detector_backend=args.detector_backend,
        track_chunk_size=args.track_chunk_size,
        disable_mask_refine=getattr(args, "disable_mask_refine", False),
        disable_auto_reprompt=False,
        save_debug_metrics=args.save_debug_metrics,
    )
    cfg = apply_runtime_overrides(runtime_override_args, load_refined_config(args.config))
    samples = discover_samples(input_root=args.input_root, input_list=args.input_list, max_samples=args.max_samples)
    write_batch_manifest(args.output_dir, samples)

    app = RefinedOfflineApp(args.config, config=cfg)
    records = []
    wan_cfg = _resolve_wan_export_config(cfg)

    pipeline_samples = list(samples)
    if bool(wan_cfg.face_clip_enable):
        pipeline_samples = []
        for sample in samples:
            try:
                clip_samples, extraction_record = extract_face_first_clip_samples(
                    app,
                    sample,
                    cfg,
                    args.output_dir,
                    skip_existing=args.skip_existing,
                )
            except Exception as exc:  # noqa: BLE001 - batch runner records stage-1 failures explicitly
                extraction_record = _build_sample_result_base(
                    sample,
                    _resolve_face_clip_output_root(cfg, args.output_dir),
                    stage="face_clip_extraction",
                )
                extraction_record.update(
                    {
                        "status": "failed",
                        "kept_clip_count": 0,
                        "clip_ids": [],
                        "error": str(exc),
                    }
                )
                append_batch_result(args.output_dir, extraction_record)
                records.append(extraction_record)
                if not args.continue_on_error:
                    raise RuntimeError(f"Stopping batch on failed sample: {sample['sample_id']}")
                continue

            append_batch_result(args.output_dir, extraction_record)
            records.append(extraction_record)
            pipeline_samples.extend(clip_samples)

    for sample in pipeline_samples:
        record = run_sample_with_retries(app, sample, args.output_dir, cfg, args)
        append_batch_result(args.output_dir, record)
        records.append(record)
        if record["status"] == "failed" and not args.continue_on_error:
            raise RuntimeError(f"Stopping batch on failed sample: {sample['sample_id']}")

    return records


def main() -> None:
    parser = build_batch_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
