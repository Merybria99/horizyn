#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 CALIBRATION_TAG GPU_LIST MASTER_PORT" >&2
  exit 2
fi

TAG="$1"
GPU_LIST="$2"
MASTER_PORT="$3"
case "$TAG" in
  lambda_0p03|lambda_0p10|lambda_0p30|lambda_1p00) ;;
  *) echo "Unknown calibration tag: $TAG" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$ROOT/runs/reactzyme_f3_biological_no_ec_v1"
CAL_ROOT="$RUN_ROOT/calibration/$TAG"
CONFIG="$CAL_ROOT/train.yaml"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
LOG="$CAL_ROOT/train.stdout.log"
STATUS="$CAL_ROOT/status.jsonl"

mkdir -p "$RUN_ROOT/locks" "$CAL_ROOT" "$RUN_ROOT/cache/huggingface" \
  "$RUN_ROOT/cache/torch" "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/matplotlib" "$RUN_ROOT/wandb"
cd "$ROOT"
exec 9>"$RUN_ROOT/locks/calibration-$TAG.lock"
flock -n 9 || { echo "Calibration already active: $TAG" >&2; exit 1; }

export TMPDIR="/tmp/horizyn-${TAG}-${UID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"

printf '{"time":"%s","tag":"%s","event":"start"}\n' "$(date -Iseconds)" "$TAG" >> "$STATUS"
set +e
CUDA_VISIBLE_DEVICES="$GPU_LIST" MASTER_PORT="$MASTER_PORT" \
  "$PYTHON_BIN" scripts/train_protein_pooling.py --config "$CONFIG" \
  --wandb --wandb-project horizyn-reactzyme-f3-biological-no-ec \
  --wandb-entity omnai --wandb-run-name "F3MC-noEC-aux-calibration-${TAG}-seed42" \
  --wandb-mode online > "$LOG" 2>&1
rc=$?
set -e
printf '{"time":"%s","tag":"%s","event":"end","returncode":%s}\n' \
  "$(date -Iseconds)" "$TAG" "$rc" >> "$STATUS"
exit "$rc"
