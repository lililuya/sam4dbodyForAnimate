#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

print_usage() {
  cat <<'EOF'
Usage:
  ./run.sh <command> [args...]

Commands:
  detect-debug
      Draw human detector boxes on an image or video.

  offline-refined
      Run the refined offline pipeline for one video or one frame directory.

  offline-refined-batch
      Run the refined offline batch pipeline for a directory or manifest of samples.

  cache-4d
      Run 4D from one exported SAM3 cache directory or a cache root.

Examples:
  ./run.sh detect-debug --input_path data/demo.mp4 --detector_backend yolo --output_path outputs/demo_detected.mp4
  ./run.sh offline-refined --input_video data/demo.mp4 --config configs/body4d_refined.yaml --max_targets 2
  ./run.sh offline-refined-batch --input_root data/batch --output_dir outputs_refined --config configs/body4d_refined_low_memory.yaml --skip_existing --continue_on_error
  ./run.sh cache-4d --cache_root outputs/sam3_cache --output_root outputs/outputs_4d --overwrite
EOF
}

command_name="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  detect-debug)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/debug_human_detection.py" "$@"
    ;;
  offline-refined)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/offline_app_refined.py" "$@"
    ;;
  offline-refined-batch)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/offline_batch_refined.py" "$@"
    ;;
  cache-4d)
    exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_4d_from_cache.py" "$@"
    ;;
  "" | "help" | "-h" | "--help")
    print_usage
    ;;
  *)
    echo "Unknown command: ${command_name}" >&2
    echo >&2
    print_usage >&2
    exit 1
    ;;
esac
