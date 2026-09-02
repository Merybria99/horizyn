#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_f3_reaction_smi_paper_v2_4gpu}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../env/bin/python}"
CONFIG="$RUN_ROOT/configs/train.yaml"
TEST_CONFIG="$RUN_ROOT/configs/test.yaml"
RUN_ID="F3_reaction_smi_paper_v2_seed42_4gpu"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$RUN_ID"
TRAIN_LOG="$RUN_ROOT/logs/$RUN_ID/train.stdout.log"
TEST_LOG="$RUN_ROOT/logs/$RUN_ID/test.stdout.log"
TEST_JSON="$RUN_ROOT/results/test.json"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
MASTER_PORT="${MASTER_PORT:-29742}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-80000}"

mkdir -p \
  "$RUN_ROOT/locks" "$RUN_ROOT/logs/$RUN_ID" "$RUN_ROOT/results" \
  "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/matplotlib" "$RUN_ROOT/wandb"
cd "$ROOT"

exec 9>"$RUN_ROOT/locks/controller.lock"
flock -n 9 || { echo "Another F3 paper-v2 controller is already active" >&2; exit 1; }

export TMPDIR="${F3_PAPER_V2_TMPDIR:-/tmp/horizyn-f3-paper-v2-${UID}}"
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
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$RUN_ID" "$1" "$2" "${3:-}" >> "$STATUS_LOG"
}

gpus_have_capacity() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk -v minimum="$GPU_MIN_FREE_MIB" '
        BEGIN {count = 0; ready = 1}
        {count += 1; if (($1 + 0) < minimum) ready = 0}
        END {exit !(count == 4 && ready)}
      '
}

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
[[ "$gpu_count" -eq 4 ]] || { echo "Expected exactly four visible GPUs; found $gpu_count" >&2; exit 1; }

status wait_for_gpus start ""
while ! gpus_have_capacity; do
  echo "[$(date -Iseconds)] Waiting for at least ${GPU_MIN_FREE_MIB} MiB free on every GPU"
  sleep 60
done
status wait_for_gpus end 0

mkdir -p "$CHECKPOINT_DIR"
train_cmd=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$CONFIG")
if [[ -s "$CHECKPOINT_DIR/last.ckpt" ]]; then
  train_cmd+=(--resume "$CHECKPOINT_DIR/last.ckpt")
fi
train_cmd+=(
  --wandb
  --wandb-project horizyn-reactzyme-level1-f3-v2
  --wandb-entity omnai
  --wandb-run-name reactzyme-level1-f3-reaction-smi-paper-v2-seed42
  --wandb-mode online
)

status train start ""
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT="$MASTER_PORT" \
  "${train_cmd[@]}" > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
status train end "$train_rc"
[[ "$train_rc" -eq 0 ]] || exit "$train_rc"

checkpoint="$(
  awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' \
    "$TRAIN_LOG" | tr -d '\r'
)"
[[ -n "$checkpoint" && -s "$checkpoint" ]] || {
  echo "Training completed without a valid best checkpoint" >&2
  exit 1
}
"$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime write-checkpoint \
  "$RUN_ROOT/results/selected_checkpoint.json" "$RUN_ID" "$checkpoint"

status test start ""
set +e
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
  --checkpoint "$checkpoint" \
  --config "$TEST_CONFIG" \
  --device cuda \
  --direction both \
  --evaluation-protocol paper_test_candidates \
  --batch-size 512 \
  --target-batch-size 2048 \
  --output "$TEST_JSON" > "$TEST_LOG" 2>&1
test_rc=$?
set -e
status test end "$test_rc"
[[ "$test_rc" -eq 0 ]] || exit "$test_rc"
status controller complete 0
echo "Training and released-test evaluation complete: $TEST_JSON"
