#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/reactzyme_f3_cluster_proxy_v1"
PYTHON_BIN="$ROOT/../env/bin/python"
TRAIN_PID_FILE="$ROOT/run_pids/reactzyme_f3_cluster_proxy_v1.pid"
TRAIN_CONFIG="$ROOT/configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml"
PANEL_DIR="$ROOT/data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85"
CHEMISTRY_DIR="$PANEL_DIR/features/reaction_set"
SELECTION_ROOT="$RUN_ROOT/cluster_selection"
STATUS_FILE="$RUN_ROOT/chain_status.txt"
CHAIN_PID_FILE="$ROOT/run_pids/reactzyme_f3_cluster_chain.pid"

cd "$ROOT"

mkdir -p \
  "$RUN_ROOT/configs" \
  "$RUN_ROOT/test/target_cache" \
  "$RUN_ROOT/nohome" \
  "$RUN_ROOT/cache/huggingface" \
  "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/xdg"

export HOME="$RUN_ROOT/nohome"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$RUN_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TMPDIR="${HORIZYN_SHORT_TMPDIR:-/tmp/hz-f3-${UID}}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1

printf '%s\n' "$$" > "$CHAIN_PID_FILE"

on_error() {
  local exit_code="$?"
  local stage
  stage="$(cat "$STATUS_FILE" 2>/dev/null || printf 'initializing')"
  printf 'failed: stage=%s line=%s exit_code=%s\n' \
    "$stage" "$1" "$exit_code" > "$STATUS_FILE"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

fail() {
  trap - ERR
  printf 'failed: %s\n' "$1" > "$STATUS_FILE"
  echo "$1" >&2
  exit 1
}

printf 'waiting_for_training\n' > "$STATUS_FILE"
train_pid="$(cat "$TRAIN_PID_FILE")"
while kill -0 "$train_pid" 2>/dev/null; do
  command_line="$(tr '\0' ' ' < "/proc/$train_pid/cmdline" 2>/dev/null || true)"
  [[ "$command_line" == *"$TRAIN_CONFIG"* ]] || break
  sleep 60
done

checkpoint_count="$({ find "$RUN_ROOT/checkpoints" -maxdepth 1 \
  -type f -name 'protein-pooling-epoch=*.ckpt' -print || true; } | wc -l)"
[[ "$checkpoint_count" -eq 30 ]] || fail \
  "Training ended with $checkpoint_count/30 epoch checkpoints; selection was not started"

printf 'evaluating_cluster_validation\n' > "$STATUS_FILE"
"$PYTHON_BIN" scripts/evaluate_f3_reaction_cluster_checkpoints.py \
  --checkpoint-dir "$RUN_ROOT/checkpoints" \
  --minimum-epoch 0 \
  --maximum-epoch 29 \
  --baseline-epoch 28 \
  --r2e-tolerance 0.015 \
  --checkpoint-training-pairs "$PANEL_DIR/train_pairs.csv" \
  --chemistry-npz \
    "$CHEMISTRY_DIR/train_reaction_set_features.npz" \
    "$CHEMISTRY_DIR/validation_reaction_set_features.npz" \
  --output-root "$SELECTION_ROOT" \
  --gpu "${EVAL_GPU:-0}"

report="$SELECTION_ROOT/reports/similarity_0p85.json"
selected_checkpoint="$($PYTHON_BIN - "$report" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
selection = report.get("selection")
if selection is None:
    raise SystemExit("Cluster report did not produce a valid checkpoint selection")
print(selection["selected"]["checkpoint"])
PY
)"
[[ -s "$selected_checkpoint" ]] || fail "Selected checkpoint is missing: $selected_checkpoint"
printf '%s\n' "$selected_checkpoint" > "$RUN_ROOT/checkpoints/selected_checkpoint.txt"

printf 'evaluating_released_test\n' > "$STATUS_FILE"
"$PYTHON_BIN" scripts/create_f3_cluster_test_config.py
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "$PYTHON_BIN" \
  scripts/evaluate_protein_pooling.py \
  --checkpoint "$selected_checkpoint" \
  --config "$RUN_ROOT/configs/test.yaml" \
  --device cuda \
  --direction both \
  --evaluation-protocol paper_test_candidates \
  --batch-size 128 \
  --target-batch-size 512 \
  --target-embeds-cache "$RUN_ROOT/test/target_cache/selected.pt" \
  --output "$RUN_ROOT/test/selected_test.json"

printf 'complete\n' > "$STATUS_FILE"
echo "Training, cluster selection, and released test evaluation completed"
