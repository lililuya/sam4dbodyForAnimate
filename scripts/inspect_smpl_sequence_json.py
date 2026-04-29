import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional


_RECORDS_ARRAY_PATTERN = re.compile(r'"records"\s*:\s*\[')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a large smpl_sequence.json and report which person_output fields dominate the file size."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Optionally inspect only the first N records for a quicker sample-based diagnosis.",
    )
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=4,
        help="Chunk size used while scanning the records array.",
    )
    return parser


def _compact_json_size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _infer_list_shape(value) -> Optional[List[int]]:
    if not isinstance(value, list):
        return None
    shape: List[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def iter_smpl_sequence_records(
    input_json: str,
    *,
    chunk_size_bytes: int = 4 * 1024 * 1024,
    prefix_scan_limit_bytes: int = 8 * 1024 * 1024,
) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    buffer = ""
    records_array_found = False
    prefix_bytes_buffered = 0

    with open(input_json, "r", encoding="utf-8") as handle:
        eof = False
        while True:
            if not eof and len(buffer) < chunk_size_bytes:
                chunk = handle.read(chunk_size_bytes)
                if chunk == "":
                    eof = True
                else:
                    buffer += chunk
                    if not records_array_found:
                        prefix_bytes_buffered += len(chunk.encode("utf-8"))

            if not records_array_found:
                match = _RECORDS_ARRAY_PATTERN.search(buffer)
                if match is None:
                    if eof:
                        raise ValueError(f"Unable to find top-level records array in {input_json}")
                    if prefix_bytes_buffered > prefix_scan_limit_bytes:
                        raise ValueError(
                            f"Unable to find top-level records array within the first {prefix_scan_limit_bytes} bytes"
                        )
                    if len(buffer) > 512:
                        buffer = buffer[-512:]
                    continue
                buffer = buffer[match.end() :]
                records_array_found = True

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer.startswith("]"):
                    return
                try:
                    record, end_index = decoder.raw_decode(buffer)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Malformed records entry in {input_json}") from exc
                    break
                if not isinstance(record, dict):
                    raise ValueError("Expected each records item to be a JSON object")
                yield record
                buffer = buffer[end_index:].lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return
                if not buffer:
                    break
                if eof:
                    raise ValueError(f"Malformed records array delimiter in {input_json}")
                break

            if eof and not buffer:
                raise ValueError(f"Unexpected end of file while reading records array in {input_json}")


def analyze_smpl_sequence_fields(
    input_json: str,
    *,
    max_records: int = 0,
    chunk_size_bytes: int = 4 * 1024 * 1024,
) -> Dict[str, object]:
    field_stats: Dict[str, Dict[str, object]] = {}
    inspected_record_count = 0
    total_field_bytes = 0
    truncated = False

    for record in iter_smpl_sequence_records(input_json, chunk_size_bytes=chunk_size_bytes):
        if max_records > 0 and inspected_record_count >= max_records:
            truncated = True
            break
        inspected_record_count += 1
        person_output = record.get("person_output", {})
        if not isinstance(person_output, dict):
            continue

        for field_name, value in person_output.items():
            field_size = _compact_json_size(value)
            total_field_bytes += field_size
            stats = field_stats.setdefault(
                str(field_name),
                {
                    "field": str(field_name),
                    "count": 0,
                    "total_bytes": 0,
                    "max_bytes": 0,
                    "sample_type": type(value).__name__,
                    "sample_shape": _infer_list_shape(value),
                },
            )
            stats["count"] = int(stats["count"]) + 1
            stats["total_bytes"] = int(stats["total_bytes"]) + field_size
            stats["max_bytes"] = max(int(stats["max_bytes"]), field_size)
            if stats.get("sample_shape") is None:
                stats["sample_shape"] = _infer_list_shape(value)

    fields = []
    for stats in field_stats.values():
        total_bytes = int(stats["total_bytes"])
        count = int(stats["count"])
        fields.append(
            {
                **stats,
                "avg_bytes": 0 if count <= 0 else float(total_bytes) / float(count),
                "share_of_total_bytes": 0.0
                if total_field_bytes <= 0
                else float(total_bytes) / float(total_field_bytes),
            }
        )
    fields.sort(key=lambda item: (-int(item["total_bytes"]), str(item["field"])))

    return {
        "input_json": os.path.abspath(input_json),
        "file_size_bytes": os.path.getsize(input_json),
        "record_count": inspected_record_count,
        "inspected_record_count": inspected_record_count,
        "truncated": bool(truncated),
        "field_count": len(fields),
        "total_field_bytes": int(total_field_bytes),
        "fields": fields,
    }


def _limit_fields(summary: Dict[str, object], top_n: int) -> Dict[str, object]:
    if top_n <= 0:
        return dict(summary)
    limited = dict(summary)
    limited["fields"] = list(summary.get("fields", []))[:top_n]
    limited["reported_top_n"] = int(top_n)
    return limited


def print_summary(summary: Dict[str, object]) -> None:
    print(f"Input JSON: {summary['input_json']}")
    print(f"File size: {_format_bytes(int(summary['file_size_bytes']))}")
    print(
        f"Records inspected: {int(summary['inspected_record_count'])}"
        + (" (truncated sample)" if bool(summary.get("truncated")) else "")
    )
    print(f"Tracked person_output fields: {int(summary['field_count'])}")
    print(f"Summed field bytes: {_format_bytes(int(summary['total_field_bytes']))}")
    print("")
    print("Top fields by total bytes:")
    for index, item in enumerate(summary.get("fields", []), start=1):
        shape = item.get("sample_shape")
        shape_text = "" if shape is None or shape == [] else f" shape={shape}"
        print(
            f"{index:02d}. {item['field']}: total={_format_bytes(int(item['total_bytes']))}, "
            f"avg={_format_bytes(int(round(float(item['avg_bytes']))))}, "
            f"max={_format_bytes(int(item['max_bytes']))}, "
            f"share={float(item['share_of_total_bytes']) * 100.0:.2f}%, "
            f"count={int(item['count'])}, type={item['sample_type']}{shape_text}"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chunk_size_bytes = max(1, int(args.chunk_size_mb)) * 1024 * 1024
    summary = analyze_smpl_sequence_fields(
        args.input_json,
        max_records=max(0, int(args.max_records)),
        chunk_size_bytes=chunk_size_bytes,
    )
    summary = _limit_fields(summary, max(0, int(args.top_n)))
    print_summary(summary)

    if args.output_json:
        output_path = os.path.abspath(str(args.output_json))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("")
        print(f"Saved field summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
