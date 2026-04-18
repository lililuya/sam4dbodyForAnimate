import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.offline_app_refined import RefinedOfflineApp, apply_runtime_overrides, load_refined_config
from scripts.offline_batch_helpers import build_retry_profiles, discover_samples


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
    if should_skip_sample(output_dir, args.skip_existing):
        return {
            "sample_id": sample["sample_id"],
            "input": sample["input"],
            "output_dir": output_dir,
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
            "sample_id": sample["sample_id"],
            "input": sample["input"],
            "output_dir": output_dir,
            "status": "completed" if profile_for_call["retry_index"] == 0 else "retry_succeeded",
            "summary_status": summary["status"],
            "retry_index": profile_for_call["retry_index"],
            "runtime_profile": profile_for_call,
        }

    return {
        "sample_id": sample["sample_id"],
        "input": sample["input"],
        "output_dir": output_dir,
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
        disable_auto_reprompt=False,
        save_debug_metrics=args.save_debug_metrics,
    )
    cfg = apply_runtime_overrides(runtime_override_args, load_refined_config(args.config))
    samples = discover_samples(input_root=args.input_root, input_list=args.input_list, max_samples=args.max_samples)
    write_batch_manifest(args.output_dir, samples)

    app = RefinedOfflineApp(args.config, config=cfg)
    records = []

    for sample in samples:
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
