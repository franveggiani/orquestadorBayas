#!/usr/bin/env bash
set -Eeuo pipefail

# Compara detectorBayas (legado) y detectorBayasModern sobre el mismo video.
# Uso:
#   ./comparar_detectores.sh [ruta/al/video.mp4]
# Variables opcionales:
#   SKIP_BUILD=1         Reutiliza las imagenes Docker existentes.
#   STARTUP_TIMEOUT=300  Segundos maximos para esperar cada API.
#   LEGACY_PORT=18001    Puerto host temporal para el detector legado.
#   MODERN_PORT=18002    Puerto host temporal para el detector moderno.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_VOL="${SHARED_VOL:-${PROJECT_ROOT}/shared_vol}"
DEFAULT_VIDEO="${SHARED_VOL}/001_VID_20230322_173221/001_VID_20230322_173221.mp4"
VIDEO_PATH="${1:-${DEFAULT_VIDEO}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
LEGACY_PORT="${LEGACY_PORT:-18001}"
MODERN_PORT="${MODERN_PORT:-18002}"

LEGACY_DIR="${PROJECT_ROOT}/workers/detectorBayas"
MODERN_DIR="${PROJECT_ROOT}/workers/detectorBayasModern"
MODEL_PATH="${MODERN_DIR}/models/2022.11.30_grapes_mix_iou.pth"
LEGACY_IMAGE="detector-bayas-comparison:legacy"
MODERN_IMAGE="detector-bayas-comparison:modern"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RESULTS_DIR="${SHARED_VOL}/detector_comparison/${RUN_ID}"
LEGACY_OUTPUT="${RESULTS_DIR}/legacy"
MODERN_OUTPUT="${RESULTS_DIR}/modern"
LEGACY_CONTAINER="detector-bayas-legacy-${RUN_ID}"
MODERN_CONTAINER="detector-bayas-modern-${RUN_ID}"
LOG_PID=""

cleanup() {
  if [[ -n "${LOG_PID}" ]]; then
    kill "${LOG_PID}" 2>/dev/null || true
    wait "${LOG_PID}" 2>/dev/null || true
  fi
  docker rm -f "${LEGACY_CONTAINER}" "${MODERN_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v docker >/dev/null || fail "docker no esta instalado o no esta en PATH"
command -v curl >/dev/null || fail "curl no esta instalado o no esta en PATH"
command -v python3 >/dev/null || fail "python3 no esta instalado o no esta en PATH"
[[ -d "${SHARED_VOL}" ]] || fail "no existe SHARED_VOL: ${SHARED_VOL}"
[[ -f "${VIDEO_PATH}" ]] || fail "no existe el video: ${VIDEO_PATH}"
[[ -f "${LEGACY_DIR}/2022.11.30_grapes_mix_iou.pth" ]] || fail "falta el peso legado"
[[ -f "${MODEL_PATH}" ]] || fail "falta el peso moderno: ${MODEL_PATH}"

SHARED_REAL="$(realpath "${SHARED_VOL}")"
VIDEO_REAL="$(realpath "${VIDEO_PATH}")"
case "${VIDEO_REAL}" in
  "${SHARED_REAL}"/*) ;;
  *) fail "el video debe estar dentro de SHARED_VOL (${SHARED_REAL})" ;;
esac

VIDEO_RELATIVE="${VIDEO_REAL#"${SHARED_REAL}"/}"
VIDEO_NAME="$(basename "${VIDEO_RELATIVE}" .mp4)"
INPUT_RELATIVE="$(dirname "${VIDEO_RELATIVE}")"
CONTAINER_INPUT="/shared/${INPUT_RELATIVE}"
CONTAINER_RESULTS="/shared/detector_comparison/${RUN_ID}"

mkdir -p "${LEGACY_OUTPUT}" "${MODERN_OUTPUT}"

printf 'Video: %s\nResultados: %s\n' "${VIDEO_REAL}" "${RESULTS_DIR}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  printf '\n[1/4] Construyendo detectorBayas (legado)...\n'
  docker build -t "${LEGACY_IMAGE}" "${LEGACY_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/legacy-build.log"

  printf '\n[2/4] Construyendo detectorBayasModern...\n'
  docker build -t "${MODERN_IMAGE}" "${MODERN_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/modern-build.log"
else
  printf '\n[1/4] Build omitido (SKIP_BUILD=1).\n'
  docker image inspect "${LEGACY_IMAGE}" >/dev/null || fail "no existe ${LEGACY_IMAGE}"
  docker image inspect "${MODERN_IMAGE}" >/dev/null || fail "no existe ${MODERN_IMAGE}"
fi

wait_for_api() {
  local container_name="$1"
  local url="$2"
  local elapsed=0

  until curl -fsS "${url}" >/dev/null 2>&1; do
    if ! docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -q true; then
      return 1
    fi
    if (( elapsed >= STARTUP_TIMEOUT )); then
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
}

run_detector() {
  local label="$1"
  local image="$2"
  local container_name="$3"
  local port="$4"
  local output_folder="$5"
  local log_file="$6"
  shift 6

  printf '\nIniciando %s en localhost:%s...\n' "${label}" "${port}"
  docker run --detach --rm --gpus all \
    --name "${container_name}" \
    --publish "127.0.0.1:${port}:8001" \
    --volume "${SHARED_REAL}:/shared" \
    "$@" \
    "${image}" >/dev/null

  docker logs --follow --timestamps "${container_name}" >"${log_file}" 2>&1 &
  LOG_PID=$!

  if ! wait_for_api "${container_name}" "http://127.0.0.1:${port}/openapi.json"; then
    printf '%s no quedo listo; revisar %s\n' "${label}" "${log_file}" >&2
    docker stop --time 10 "${container_name}" >/dev/null 2>&1 || true
    wait "${LOG_PID}" 2>/dev/null || true
    LOG_PID=""
    return 1
  fi

  local request_file="${RESULTS_DIR}/${label}-request.json"
  local response_file="${RESULTS_DIR}/${label}-response.json"
  local metrics_file="${RESULTS_DIR}/${label}-http-metrics.txt"
  python3 - "${request_file}" "${CONTAINER_INPUT}" "${CONTAINER_RESULTS}/${output_folder}" "${VIDEO_NAME}" <<'PY'
import json
import sys

destination, input_folder, output_folder, video_name = sys.argv[1:]
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "input_folder": input_folder,
            "output_folder": output_folder,
            "video_name": video_name,
        },
        stream,
        indent=2,
    )
PY

  local http_code
  http_code="$(curl -sS \
    --output "${response_file}" \
    --write-out '%{http_code} %{time_total}\n' \
    --request POST "http://127.0.0.1:${port}/detector_task" \
    --header 'Content-Type: application/json' \
    --data-binary "@${request_file}" | tee "${metrics_file}" | awk '{print $1}')"

  docker stop --time 30 "${container_name}" >/dev/null 2>&1 || true
  wait "${LOG_PID}" 2>/dev/null || true
  LOG_PID=""

  [[ "${http_code}" == "200" ]] || {
    printf '%s respondio HTTP %s; revisar %s y %s\n' \
      "${label}" "${http_code}" "${response_file}" "${log_file}" >&2
    return 1
  }
}

printf '\n[3/4] Ejecutando ambos detectores de forma secuencial...\n'
LEGACY_OK=1
MODERN_OK=1
run_detector \
  legacy "${LEGACY_IMAGE}" "${LEGACY_CONTAINER}" "${LEGACY_PORT}" \
  legacy "${RESULTS_DIR}/legacy-service.log" || LEGACY_OK=0

run_detector \
  modern "${MODERN_IMAGE}" "${MODERN_CONTAINER}" "${MODERN_PORT}" \
  modern "${RESULTS_DIR}/modern-service.log" \
  --volume "${MODEL_PATH}:/models/2022.11.30_grapes_mix_iou.pth:ro" || MODERN_OK=0

printf '\n[4/4] Comparando salidas...\n'
LEGACY_JSON="${LEGACY_OUTPUT}/${VIDEO_NAME}.json"
MODERN_JSON="${MODERN_OUTPUT}/${VIDEO_NAME}.json"

python3 - "${LEGACY_JSON}" "${MODERN_JSON}" "${RESULTS_DIR}/comparison-summary.json" \
  "${LEGACY_OK}" "${MODERN_OK}" <<'PY'
import json
import math
from pathlib import Path
import sys

legacy_path, modern_path, summary_path = map(Path, sys.argv[1:4])
legacy_ok = sys.argv[4] == "1"
modern_ok = sys.argv[5] == "1"
summary = {
    "legacy_api_ok": legacy_ok,
    "modern_api_ok": modern_ok,
    "legacy_json": str(legacy_path),
    "modern_json": str(modern_path),
}

if legacy_path.is_file() and modern_path.is_file():
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    modern = json.loads(modern_path.read_text(encoding="utf-8"))
    frame_keys = sorted(set(legacy) | set(modern), key=lambda value: int(value))
    legacy_count = sum(len(items) for items in legacy.values())
    modern_count = sum(len(items) for items in modern.values())
    count_mismatches = 0
    coordinate_errors = []

    for frame in frame_keys:
        legacy_items = legacy.get(frame, {})
        modern_items = modern.get(frame, {})
        if len(legacy_items) != len(modern_items):
            count_mismatches += 1
        shared_items = set(legacy_items) & set(modern_items)
        for item in shared_items:
            left = legacy_items[item]
            right = modern_items[item]
            coordinate_errors.extend(
                abs(float(a) - float(b)) for a, b in zip(left, right)
            )

    summary.update(
        {
            "legacy_frames": len(legacy),
            "modern_frames": len(modern),
            "legacy_detections": legacy_count,
            "modern_detections": modern_count,
            "detection_count_delta": modern_count - legacy_count,
            "frames_with_detection_count_mismatch": count_mismatches,
            "paired_coordinate_values": len(coordinate_errors),
            "coordinate_mae": (
                sum(coordinate_errors) / len(coordinate_errors)
                if coordinate_errors else None
            ),
            "coordinate_max_abs_error": max(coordinate_errors, default=None),
            "json_exact_match": legacy == modern,
        }
    )
else:
    summary["comparison_error"] = "Falta una o ambas salidas JSON"

summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

if not (legacy_ok and modern_ok and legacy_path.is_file() and modern_path.is_file()):
    raise SystemExit(1)
PY

printf '\nComparacion terminada. Artefactos: %s\n' "${RESULTS_DIR}"
