#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_f3_biological_v2}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-80000}"
WANDB_PROJECT="horizyn-reactzyme-f3-biological-v2"
VARIANTS=(F3M_mechanism F3C_cofactor F3MC_mechanism_cofactor)

mkdir -p "$RUN_ROOT/locks" "$RUN_ROOT/logs/controller" "$RUN_ROOT/results" \
  "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" "$RUN_ROOT/cache/xdg" \
  "$RUN_ROOT/cache/matplotlib" "$RUN_ROOT/wandb"
cd "$ROOT"

exec 9>"$RUN_ROOT/locks/controller.lock"
flock -n 9 || { echo "Another F3 biological-v2 controller is active" >&2; exit 1; }

export TMPDIR="${F3_BIO_TMPDIR:-/tmp/horizyn-f3-bio-v2-${UID}}"
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

STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
status() {
  printf '{"time":"%s","variant":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "$3" "${4:-}" >> "$STATUS_LOG"
}

gpus_have_capacity() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v minimum="$GPU_MIN_FREE_MIB" '
        BEGIN {count = 0; ready = 1}
        {gsub(/ /, "", $1); gsub(/ /, "", $2); if (index("," ENVIRON["GPU_LIST"] ",", "," $1 ",")) {count += 1; if (($2 + 0) < minimum) ready = 0}}
        END {exit !(count == 4 && ready)}
      '
}
export GPU_LIST

wait_for_gpus() {
  while ! gpus_have_capacity; do
    echo "[$(date -Iseconds)] waiting for four GPUs with ${GPU_MIN_FREE_MIB} MiB free"
    sleep 60
  done
}

select_checkpoint() {
  local checkpoint_dir="$1" log="$2" best=""
  if [[ -s "$log" ]]; then
    best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' "$log" | tr -d '\r')"
  fi
  if [[ -n "$best" && -s "$best" ]]; then
    printf '%s\n' "$best"
  elif [[ -s "$checkpoint_dir/last.ckpt" ]]; then
    printf '%s\n' "$checkpoint_dir/last.ckpt"
  else
    find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' \
      | sort -nr | awk 'NR==1{sub(/^[^ ]+ /, ""); print}'
  fi
}

for variant in "${VARIANTS[@]}"; do
  pretrain_config="$RUN_ROOT/configs/$variant/biological_pretrain.yaml"
  train_config="$RUN_ROOT/configs/$variant/train.yaml"
  validation_config="$RUN_ROOT/configs/$variant/validation.yaml"
  test_config="$RUN_ROOT/configs/$variant/test.yaml"
  pretrain_dir="$RUN_ROOT/checkpoints/$variant/biological_pretrain"
  train_dir="$RUN_ROOT/checkpoints/$variant/train"
  pretrain_log="$RUN_ROOT/logs/$variant/biological_pretrain.stdout.log"
  train_log="$RUN_ROOT/logs/$variant/train.stdout.log"
  validation_log="$RUN_ROOT/logs/$variant/validation.stdout.log"
  test_log="$RUN_ROOT/logs/$variant/test.stdout.log"
  selected_json="$RUN_ROOT/results/$variant/selected_checkpoint.json"
  validation_json="$RUN_ROOT/results/$variant/validation.json"
  test_json="$RUN_ROOT/results/$variant/test.json"
  mkdir -p \
    "$pretrain_dir" "$train_dir" "$RUN_ROOT/results/$variant" \
    "$RUN_ROOT/logs/$variant"

  wait_for_gpus
  if [[ ! -s "$pretrain_dir/pretrain_summary.json" ]]; then
    status "$variant" biological_pretrain start ""
    pretrain_cmd=("$PYTHON_BIN" scripts/pretrain_enzyme_biofp_split.py --config "$pretrain_config")
    [[ -s "$pretrain_dir/last.ckpt" ]] && pretrain_cmd+=(--resume "$pretrain_dir/last.ckpt")
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_LIST" MASTER_PORT=29831 "${pretrain_cmd[@]}" > "$pretrain_log" 2>&1
    rc=$?
    set -e
    status "$variant" biological_pretrain end "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
  else
    status "$variant" biological_pretrain skip 0
  fi

  wait_for_gpus
  if [[ ! -s "$selected_json" ]]; then
    status "$variant" retrieval_train start ""
    train_cmd=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$train_config")
    [[ -s "$train_dir/last.ckpt" ]] && train_cmd+=(--resume "$train_dir/last.ckpt")
    train_cmd+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-entity omnai \
      --wandb-run-name "${variant}-reaction-smi-seed42" --wandb-mode online)
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_LIST" MASTER_PORT=29841 "${train_cmd[@]}" > "$train_log" 2>&1
    rc=$?
    set -e
    status "$variant" retrieval_train end "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
    checkpoint="$(select_checkpoint "$train_dir" "$train_log")"
    [[ -n "$checkpoint" && -s "$checkpoint" ]] || { echo "No checkpoint selected for $variant" >&2; exit 1; }
    "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime write-checkpoint \
      "$selected_json" "$variant" "$checkpoint"
  else
    checkpoint="$($PYTHON_BIN -m horizyn.benchmarks.reactzyme_runtime read-checkpoint "$selected_json")"
    status "$variant" retrieval_train skip 0
  fi

  if [[ ! -s "$validation_json" ]]; then
    status "$variant" validation start ""
    set +e
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$validation_config" --device cuda \
      --direction both --evaluation-protocol paper_test_candidates \
      --batch-size 512 --target-batch-size 2048 --output "$validation_json" \
      > "$validation_log" 2>&1
    rc=$?
    set -e
    status "$variant" validation end "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
  fi

  if [[ ! -s "$test_json" ]]; then
    status "$variant" test start ""
    set +e
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$test_config" --device cuda \
      --direction both --evaluation-protocol paper_test_candidates \
      --batch-size 512 --target-batch-size 2048 --output "$test_json" \
      > "$test_log" 2>&1
    rc=$?
    set -e
    status "$variant" test end "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
  fi
done

status campaign controller complete 0
echo "F3 biological-v2 campaign complete"
