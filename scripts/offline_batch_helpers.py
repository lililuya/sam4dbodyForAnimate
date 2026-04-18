import json
import os
from collections import Counter
from collections.abc import Sequence


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VALID_RETRY_MODES = ("never", "quality_safe", "aggressive_safe")


def sample_id_from_input(input_path):
    normalized = os.path.normpath(input_path)
    base_name = os.path.basename(normalized)
    stem, extension = os.path.splitext(base_name)
    return stem if extension else base_name


def is_frame_directory(path):
    if not os.path.isdir(path):
        return False
    for entry in sorted(os.listdir(path)):
        if entry.lower().endswith(IMAGE_EXTENSIONS):
            return True
    return False


def load_input_list(input_list):
    records = []
    with open(input_list, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                data = json.loads(line)
                input_path = data["input"]
                if "sample_id" in data:
                    records.append({"input": input_path, "sample_id": data["sample_id"], "_explicit_sample_id": True})
                    continue
                sample_id = sample_id_from_input(input_path)
            else:
                input_path = line
                sample_id = sample_id_from_input(input_path)
            records.append({"input": input_path, "sample_id": sample_id, "_auto_sample_id": True})
    return _finalize_sample_ids(records)


def discover_samples(input_root="", input_list="", max_samples=None):
    if bool(input_root) == bool(input_list):
        raise ValueError("Exactly one of input_root or input_list must be provided")

    if input_list:
        samples = load_input_list(input_list)
    else:
        samples = []
        for entry in sorted(os.listdir(input_root)):
            full_path = os.path.join(input_root, entry)
            if os.path.isfile(full_path) and full_path.lower().endswith(".mp4"):
                samples.append(
                    {"input": full_path, "sample_id": sample_id_from_input(full_path), "_auto_sample_id": True}
                )
            elif is_frame_directory(full_path):
                samples.append(
                    {"input": full_path, "sample_id": sample_id_from_input(full_path), "_auto_sample_id": True}
                )
        samples = _finalize_sample_ids(samples)

    if max_samples is not None:
        return samples[: int(max_samples)]
    return samples


def build_retry_profiles(cfg, args):
    retry_mode = args.retry_mode or cfg.batch.retry_mode
    if retry_mode not in VALID_RETRY_MODES:
        raise ValueError("retry_mode must be one of never, quality_safe, aggressive_safe")

    base_chunk_size = int(args.track_chunk_size or cfg.tracking.chunk_size)
    base_batch_size = int(cfg.sam_3d_body.batch_size)
    initial_search_frames = int(cfg.batch.initial_search_frames)
    retry_chunk_sizes = _coerce_retry_values(
        cfg.batch.retry_chunk_sizes,
        "retry_chunk_sizes and retry_batch_sizes must each contain at least one value",
        "retry_chunk_sizes must be a sequence of integers",
    )
    retry_batch_sizes = _coerce_retry_values(
        cfg.batch.retry_batch_sizes,
        "retry_chunk_sizes and retry_batch_sizes must each contain at least one value",
        "retry_batch_sizes must be a sequence of integers",
    )

    chunk_sizes = [base_chunk_size] + retry_chunk_sizes
    batch_sizes = [base_batch_size] + retry_batch_sizes

    if retry_mode == "never":
        return [
            {
                "retry_index": 0,
                "reason": "base",
                "tracking.chunk_size": base_chunk_size,
                "sam_3d_body.batch_size": base_batch_size,
                "initial_search_frames": initial_search_frames,
            }
        ]

    safe_chunk_size = chunk_sizes[1]
    safe_batch_size = batch_sizes[1]
    aggressive_chunk_size = chunk_sizes[-1] if retry_mode == "aggressive_safe" else safe_chunk_size
    aggressive_batch_size = batch_sizes[-1] if retry_mode == "aggressive_safe" else safe_batch_size

    return [
        {
            "retry_index": 0,
            "reason": "base",
            "tracking.chunk_size": base_chunk_size,
            "sam_3d_body.batch_size": base_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 1,
            "reason": "safer_tracking",
            "tracking.chunk_size": safe_chunk_size,
            "sam_3d_body.batch_size": base_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 2,
            "reason": "safer_reconstruction",
            "tracking.chunk_size": aggressive_chunk_size,
            "sam_3d_body.batch_size": safe_batch_size,
            "initial_search_frames": initial_search_frames,
        },
        {
            "retry_index": 3,
            "reason": "search_expansion",
            "tracking.chunk_size": aggressive_chunk_size,
            "sam_3d_body.batch_size": aggressive_batch_size,
            "initial_search_frames": initial_search_frames * 2,
        },
    ]


def _finalize_sample_ids(records):
    explicit_ids = [record["sample_id"] for record in records if record.get("_explicit_sample_id")]
    explicit_id_counts = Counter(explicit_ids)
    duplicate_explicit_ids = sorted(
        sample_id for sample_id, count in explicit_id_counts.items() if count > 1
    )
    if duplicate_explicit_ids:
        raise ValueError(
            "Duplicate explicit sample_id values are not allowed: " + ", ".join(duplicate_explicit_ids)
        )

    used_ids = set(explicit_ids)
    auto_counts = {}
    finalized = []

    for record in records:
        normalized = dict(record)
        sample_id = normalized["sample_id"]
        is_auto = normalized.pop("_auto_sample_id", False)
        normalized.pop("_explicit_sample_id", False)

        if is_auto:
            count = auto_counts.get(sample_id, 0)
            candidate = sample_id
            while candidate in used_ids:
                count += 1
                candidate = f"{sample_id}__{count + 1}"
            auto_counts[sample_id] = max(auto_counts.get(sample_id, 0), count)
            normalized["sample_id"] = candidate

        used_ids.add(normalized["sample_id"])
        finalized.append(normalized)

    return finalized


def _coerce_retry_values(values, empty_message, invalid_message):
    if isinstance(values, (str, bytes)):
        raise ValueError(invalid_message)
    if not isinstance(values, Sequence):
        raise ValueError(invalid_message)

    try:
        coerced = [int(value) for value in values]
    except TypeError as exc:
        raise ValueError(invalid_message) from exc
    except ValueError as exc:
        raise ValueError(invalid_message) from exc

    if not coerced:
        raise ValueError(empty_message)
    return coerced
