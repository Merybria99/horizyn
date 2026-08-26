#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/reactzyme_bidirectional_adapter_v1}"
RUNTIME_ENV="$RUN_ROOT/runtime.env"
if [[ ! -s "$RUNTIME_ENV" ]]; then
  echo "Missing $RUNTIME_ENV; generate the campaign first." >&2
  exit 1
fi

set -a
source "$RUNTIME_ENV"
set +a

VARIANT="Q6_bidirectional_adapters"
STATUS_LOG="$RUN_ROOT/status.jsonl"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cache"
# Python multiprocessing appends a long random socket name, so keep TMPDIR short.
export TMPDIR="${SHORT_TMPDIR:-/datastor2/deep-proteins/EnzymeDiscovery/.q6tmp}"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export WANDB_DIR="$RUN_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$RUN_ROOT/cache/wandb"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR"

status() {
  printf '{"time":"%s","split":"%s","variant":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$VARIANT" "$2" "$3" "${4:-}" >> "$STATUS_LOG"
}

select_checkpoint() {
  local checkpoint_dir="$1"
  local log_path="$2"
  local best=""
  if [[ -s "$log_path" ]]; then
    best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' "$log_path" | tr -d '\r')"
  fi
  if [[ -n "$best" && -s "$best" ]]; then
    printf '%s\n' "$best"
  elif [[ -s "$checkpoint_dir/last.ckpt" ]]; then
    printf '%s\n' "$checkpoint_dir/last.ckpt"
  else
    find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' \
      | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
  fi
}

train_one() {
  local split="$1"
  local split_index="$2"
  local config="$RUN_ROOT/configs/seed42/$VARIANT/$split/train.yaml"
  local checkpoint_dir="$RUN_ROOT/checkpoints/seed42/$split/$VARIANT"
  local selected="$checkpoint_dir/selected.ckpt"
  local log="$RUN_ROOT/logs/seed42/$split/$VARIANT/train.stdout.log"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$selected" ]]; then
    status "$split" train skip 0
    return 0
  fi
  local command=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
  if [[ -s "$checkpoint_dir/last.ckpt" ]]; then
    command+=(--resume "$checkpoint_dir/last.ckpt")
  fi
  if [[ "$WANDB_MODE" != "disabled" ]]; then
    command+=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-run-name "reactzyme-bidirectional-Q6-$split-seed42"
      --wandb-mode "$WANDB_MODE"
    )
  fi
  status "$split" train start ""
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$((BASE_MASTER_PORT + split_index))" \
    "${command[@]}" > "$log" 2>&1
  local rc=$?
  set -e
  status "$split" train end "$rc"
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
  local checkpoint
  checkpoint="$(select_checkpoint "$checkpoint_dir" "$log")"
  if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
    echo "No checkpoint selected for split=$split" >&2
    return 1
  fi
  cp --reflink=auto "$checkpoint" "$selected"
}

evaluate_one() {
  local split="$1"
  local subset="$2"
  local gpu="$3"
  local checkpoint="$RUN_ROOT/checkpoints/seed42/$split/$VARIANT/selected.ckpt"
  local config="$RUN_ROOT/configs/seed42/$VARIANT/$split/$subset.yaml"
  local out_dir="$RUN_ROOT/eval/seed42/$split/$VARIANT"
  local output="$out_dir/$subset.json"
  local log="$out_dir/$subset.stdout.log"
  local protocol="paper_test_candidates"
  if [[ "$subset" == "validation" ]]; then
    protocol="configured_forward_candidates"
  fi
  mkdir -p "$out_dir"
  if [[ -s "$output" ]]; then
    status "$split" "${subset}_eval" skip 0
    return 0
  fi
  status "$split" "${subset}_eval" start ""
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --device cuda:0 \
    --direction both \
    --evaluation-protocol "$protocol" \
    --batch-size 512 \
    --target-batch-size 256 \
    --output "$output" > "$log" 2>&1
  local rc=$?
  set -e
  status "$split" "${subset}_eval" end "$rc"
  return "$rc"
}

cd "$ROOT"
status all campaign start ""

split_index=0
for split in time enzyme_smi reaction_smi; do
  train_one "$split" "$split_index"
  split_index=$((split_index + 1))
done

for subset in validation test; do
  pids=()
  gpu=0
  for split in time enzyme_smi reaction_smi; do
    evaluate_one "$split" "$subset" "$gpu" &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Q6 $subset evaluation failed; inspect $RUN_ROOT/eval" >&2
    exit 1
  fi
done

status all campaign end 0
echo "ReactZyme Q6 bidirectional-adapter campaign complete"
