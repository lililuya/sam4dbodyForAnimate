import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.offline_batch_helpers import is_frame_directory, load_input_list, sample_id_from_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan mp4 inputs with ffmpeg and report corrupted or suspicious videos."
    )
    parser.add_argument("--input_root", type=str, default="")
    parser.add_argument("--input_list", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    parser.add_argument("--max_error_lines", type=int, default=20)
    return parser


def _stderr_preview(stderr_text: str, max_error_lines: int) -> List[str]:
    lines = [line.strip() for line in str(stderr_text or "").splitlines() if line.strip()]
    if max_error_lines <= 0:
        return lines
    return lines[: int(max_error_lines)]


def _resolve_scan_input(sample: Dict[str, str]) -> Dict[str, str] | None:
    input_path = str(sample["input"])
    sample_id = str(sample["sample_id"])

    if os.path.isfile(input_path) and input_path.lower().endswith(".mp4"):
        return {
            "input": os.path.abspath(input_path),
            "sample_id": sample_id,
            "input_kind": "video",
        }

    if os.path.isdir(input_path):
        clip_video_path = os.path.join(input_path, "clip.mp4")
        if os.path.isfile(clip_video_path):
            return {
                "input": os.path.abspath(clip_video_path),
                "sample_id": sample_id,
                "input_kind": "clip_package",
            }
        if is_frame_directory(input_path):
            return {
                "input": os.path.abspath(input_path),
                "sample_id": sample_id,
                "input_kind": "frame_dir",
            }
    return None


def discover_scan_samples(
    *,
    input_root: str = "",
    input_list: str = "",
    max_samples: int | None = None,
) -> List[Dict[str, str]]:
    if bool(input_root) == bool(input_list):
        raise ValueError("Exactly one of input_root or input_list must be provided")

    raw_samples: List[Dict[str, str]] = []
    if input_list:
        raw_samples = load_input_list(input_list)
    else:
        for entry in sorted(os.listdir(input_root)):
            full_path = os.path.join(input_root, entry)
            resolved = _resolve_scan_input(
                {
                    "input": full_path,
                    "sample_id": sample_id_from_input(full_path),
                }
            )
            if resolved is None:
                continue
            raw_samples.append(
                {
                    "input": resolved["input"],
                    "sample_id": resolved["sample_id"],
                    "input_kind": resolved["input_kind"],
                }
            )

    if input_list:
        resolved_samples: List[Dict[str, str]] = []
        for sample in raw_samples:
            resolved = _resolve_scan_input(sample)
            if resolved is None:
                continue
            resolved_samples.append(resolved)
    else:
        resolved_samples = raw_samples

    if max_samples is not None:
        return resolved_samples[: int(max_samples)]
    return resolved_samples


def probe_video(
    input_path: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    max_error_lines: int = 20,
) -> Dict[str, object]:
    command = [str(ffmpeg_bin), "-v", "error", "-i", str(input_path), "-f", "null", os.devnull]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc

    stderr_preview = _stderr_preview(result.stderr, max_error_lines)
    is_bad = bool(result.returncode != 0 or stderr_preview)
    return {
        "input": os.path.abspath(str(input_path)),
        "returncode": int(result.returncode),
        "stderr_preview": stderr_preview,
        "is_bad": bool(is_bad),
    }


def scan_videos(
    *,
    input_root: str = "",
    input_list: str = "",
    max_samples: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    max_error_lines: int = 20,
) -> Dict[str, object]:
    samples = discover_scan_samples(input_root=input_root, input_list=input_list, max_samples=max_samples)

    bad_videos: List[Dict[str, object]] = []
    skipped_non_video_count = 0
    checked_video_count = 0

    for sample in samples:
        input_path = str(sample["input"])
        if str(sample.get("input_kind")) == "frame_dir" or not input_path.lower().endswith(".mp4"):
            skipped_non_video_count += 1
            continue

        checked_video_count += 1
        probe = probe_video(
            input_path,
            ffmpeg_bin=ffmpeg_bin,
            max_error_lines=max_error_lines,
        )
        if not probe["is_bad"]:
            continue
        bad_videos.append(
            {
                "sample_id": str(sample["sample_id"]),
                "input": probe["input"],
                "returncode": int(probe["returncode"]),
                "stderr_preview": list(probe["stderr_preview"]),
            }
        )

    return {
        "input_root": os.path.abspath(input_root) if input_root else "",
        "input_list": os.path.abspath(input_list) if input_list else "",
        "sample_count_total": len(samples),
        "checked_video_count": checked_video_count,
        "bad_video_count": len(bad_videos),
        "good_video_count": checked_video_count - len(bad_videos),
        "skipped_non_video_count": skipped_non_video_count,
        "bad_videos": bad_videos,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print(f"Samples discovered: {int(summary['sample_count_total'])}")
    print(f"Videos checked: {int(summary['checked_video_count'])}")
    print(f"Bad videos: {int(summary['bad_video_count'])}")
    print(f"Skipped non-video samples: {int(summary['skipped_non_video_count'])}")
    print("")
    if not summary["bad_videos"]:
        print("No bad videos detected.")
        return

    print("Bad video list:")
    for index, item in enumerate(summary["bad_videos"], start=1):
        print(f"{index:02d}. [{item['sample_id']}] {item['input']}")
        for line in item["stderr_preview"]:
            print(f"    {line}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = scan_videos(
        input_root=args.input_root,
        input_list=args.input_list,
        max_samples=args.max_samples,
        ffmpeg_bin=args.ffmpeg_bin,
        max_error_lines=args.max_error_lines,
    )
    print_summary(summary)

    if args.output_json:
        output_path = os.path.abspath(str(args.output_json))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("")
        print(f"Saved bad video summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
