#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 VARIANT GPU_LIST MASTER_PORT" >&2
  exit 2
fi

VARIANT="$1"
GPU_LIST="$2"
MASTER_PORT="$3"
case "$VARIANT" in
  F3_no_ec|F3M_no_ec|F3C_no_ec|F3MC_no_ec|F3MC_align_no_ec) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$ROOT/runs/reactzyme_f3_biological_no_ec_v1"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
WANDB_PROJECT="horizyn-reactzyme-f3-biological-no-ec"
CONFIG_DIR="$RUN_ROOT/configs/$VARIANT"
TRAIN_CONFIG="$CONFIG_DIR/train.yaml"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$VARIANT/train"
LOG_DIR="$RUN_ROOT/logs/$VARIANT"
RESULT_DIR="$RUN_ROOT/results/$VARIANT"
SELECTED_JSON="$RESULT_DIR/selected_checkpoint.json"
STATUS_LOG="$LOG_DIR/status.jsonl"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CHECKPOINT_DIR" \
  "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" "$RUN_ROOT/cache/xdg" \
  "$RUN_ROOT/cache/matplotlib" "$RUN_ROOT/wandb"
cd "$ROOT"

export TMPDIR="${F3_NO_EC_TMPDIR:-/tmp/horizyn-${VARIANT}-${UID}}"
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

status() {
  printf '{"time":"%s","variant":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$VARIANT" "$1" "$2" "${3:-}" >> "$STATUS_LOG"
}

select_checkpoint() {
  local log="$1" best=""
  if [[ -s "$log" ]]; then
    best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' "$log" | tr -d '\r')"
  fi
  if [[ -n "$best" && -s "$best" ]]; then
    printf '%s\n' "$best"
  elif [[ -s "$CHECKPOINT_DIR/last.ckpt" ]]; then
    printf '%s\n' "$CHECKPOINT_DIR/last.ckpt"
  else
    find "$CHECKPOINT_DIR" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' \
      | sort -nr | awk 'NR==1{sub(/^[^ ]+ /, ""); print}'
  fi
}

TRAIN_LOG="$LOG_DIR/train.stdout.log"
if [[ ! -s "$SELECTED_JSON" ]]; then
  status retrieval_train start ""
  command=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$TRAIN_CONFIG")
  [[ -s "$CHECKPOINT_DIR/last.ckpt" ]] && command+=(--resume "$CHECKPOINT_DIR/last.ckpt")
  command+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-entity omnai \
    --wandb-run-name "${VARIANT}-reaction-smi-seed42" --wandb-mode online)
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_LIST" MASTER_PORT="$MASTER_PORT" "${command[@]}" > "$TRAIN_LOG" 2>&1
  rc=$?
  set -e
  status retrieval_train end "$rc"
  [[ "$rc" -eq 0 ]] || exit "$rc"
  checkpoint="$(select_checkpoint "$TRAIN_LOG")"
  [[ -n "$checkpoint" && -s "$checkpoint" ]] || { echo "No checkpoint for $VARIANT" >&2; exit 1; }
  "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime write-checkpoint \
    "$SELECTED_JSON" "$VARIANT" "$checkpoint"
else
  checkpoint="$("$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime read-checkpoint "$SELECTED_JSON")"
  status retrieval_train skip 0
fi

for subset in validation test; do
  output="$RESULT_DIR/$subset.json"
  log="$LOG_DIR/$subset.stdout.log"
  config="$CONFIG_DIR/$subset.yaml"
  if [[ ! -s "$output" ]]; then
    status "$subset" start ""
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device cuda \
      --direction both --evaluation-protocol paper_test_candidates \
      --batch-size 512 --target-batch-size 2048 --output "$output" > "$log" 2>&1
    rc=$?
    set -e
    status "$subset" end "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
  fi
done

status variant complete 0
echo "$VARIANT complete"
