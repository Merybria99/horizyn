#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/runs/reactzyme_e2r_pareto_v1"

RUN_ROOT_ARG="$DEFAULT_RUN_ROOT"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_ROOT_ARG="$1"
  shift
fi
case "$RUN_ROOT_ARG" in
  /*) RUN_ROOT="$RUN_ROOT_ARG" ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT_ARG" ;;
esac

DETACH=0
WAIT_FOR_GPUS=1
SETUP_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --no-wait) WAIT_FOR_GPUS=0; shift ;;
    --setup-only) SETUP_ONLY=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  pid_file="$RUN_ROOT/logs/controller/pid"
  detach_args=("$RUN_ROOT" --foreground)
  [[ "$WAIT_FOR_GPUS" -eq 0 ]] && detach_args+=(--no-wait)
  [[ "$SETUP_ONLY" -eq 1 ]] && detach_args+=(--setup-only)
  rm -f "$pid_file"
  setsid -f bash -c '
    pid_file="$1"
    shift
    pid_tmp="${pid_file}.tmp.$$"
    printf "%s\n" "$$" > "$pid_tmp"
    mv -f "$pid_tmp" "$pid_file"
    exec bash "$@"
  ' bash "$pid_file" "$0" "${detach_args[@]}" \
    > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  for _ in {1..50}; do
    [[ -s "$pid_file" ]] && break
    sleep 0.1
  done
  echo "Launched ReactZyme E2R Pareto controller"
  echo "PID: $(tr -d '\r\n' < "$pid_file" 2>/dev/null || printf unknown)"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  echo "Status: $RUN_ROOT/logs/status.jsonl"
  exit 0
fi

RUNTIME_ENV="$RUN_ROOT/runtime.env"
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing runtime env: $RUNTIME_ENV" >&2
  echo "Run scripts/create_reactzyme_e2r_pareto_campaign.py first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$RUNTIME_ENV"
HOST_RUNTIME_ENV="$RUN_ROOT/runtime.$(hostname -s).env"
if [[ -f "$HOST_RUNTIME_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$HOST_RUNTIME_ENV"
fi
EVAL_TARGET_BATCH_SIZE="${HORIZYN_EVAL_TARGET_BATCH_SIZE:-512}"
cd "$ROOT"

SHORT_TMPDIR="${HORIZYN_TMPDIR:-/tmp/hz_e2r_pareto_v1}"
mkdir -p \
  "$RUN_ROOT/logs/controller" "$RUN_ROOT/logs/setup" "$RUN_ROOT/nohome" \
  "$RUN_ROOT/tmp" "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/xdg_cache" "$RUN_ROOT/xdg_config" "$RUN_ROOT/xdg_state" \
  "$RUN_ROOT/wandb/data" "$RUN_ROOT/wandb/cache" "$RUN_ROOT/wandb/config" \
  "$RUN_ROOT/matplotlib" "$SHORT_TMPDIR"

export HOME="$RUN_ROOT/nohome"
export TMPDIR="$SHORT_TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export WANDB_ROOT="$RUN_ROOT/wandb"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
export MASTER_ADDR="${HORIZYN_MASTER_ADDR:-127.0.0.1}"
export NCCL_ASYNC_ERROR_HANDLING=1
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "$RUN_ROOT/wandb/api_key")"
fi

STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
PREFLIGHT_LOG="$RUN_ROOT/logs/controller/python_preflight.log"
if ! "$PYTHON_BIN" - <<'PY' > "$PREFLIGHT_LOG" 2>&1
import lightning.pytorch
import torch
import transformers
import wandb
import yaml

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"lightning={lightning.pytorch.__version__}")
print(f"wandb={wandb.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
then
  echo "Training Python preflight failed: $PYTHON_BIN" >&2
  echo "Inspect $PREFLIGHT_LOG" >&2
  exit 1
fi

status() {
  printf '{"time":"%s","seed":"%s","split":"%s","variant":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "$3" "$4" "$5" "${6:-}" >> "$STATUS_LOG"
}

setup_dense_split() {
  local split="$1"
  local out_dir="$RUN_ROOT/data/$split/dense_reaction"
  local report="$out_dir/report.json"
  if [[ -s "$report" ]] \
      && grep -q '"schema_version": "e2r_dense_reaction_features_v2"' "$report" \
      && [[ -s "$out_dir/train_dense_reaction_features.npz" ]] \
      && [[ -s "$out_dir/validation_dense_reaction_features.npz" ]] \
      && [[ -s "$out_dir/test_dense_reaction_features.npz" ]]; then
    status 42 "$split" setup dense skip 0
    return 0
  fi
  mkdir -p "$out_dir"
  status 42 "$split" setup dense start ""
  "$SETUP_PYTHON_BIN" scripts/build_e2r_dense_reaction_features.py \
    --train-reactions "data/revised_protocols/reactzyme_paper/$split/train_rxns.csv" \
    --validation-reactions "data/revised_protocols/reactzyme_paper/$split/validation_rxns.csv" \
    --test-reactions "data/revised_protocols/reactzyme_paper/$split/test_rxns.csv" \
    --train-center-features "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/train/reaction_centers/reaction_features.parquet" \
    --validation-center-features "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/validation/reaction_centers/reaction_features.parquet" \
    --test-center-features "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/test/reaction_centers/reaction_features.parquet" \
    --train-directional-reactions "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/train/rhea_directional_reactions.csv" \
    --validation-directional-reactions "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/validation/rhea_directional_reactions.csv" \
    --test-directional-reactions "$SOURCE_RUN_ROOT/data/$split/reaction_directional/source/test/rhea_directional_reactions.csv" \
    --out-dir "$out_dir" \
    > "$RUN_ROOT/logs/setup/${split}_dense_reaction.log" 2>&1
  status 42 "$split" setup dense end 0
}

setup_dense() {
  local pids=()
  local split
  for split in time enzyme_smi reaction_smi; do
    setup_dense_split "$split" &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Dense reaction setup failed; inspect $RUN_ROOT/logs/setup" >&2
    exit 1
  fi
}

wait_for_free_gpus() {
  [[ "$WAIT_FOR_GPUS" -eq 1 ]] || return 0
  while true; do
    local active
    active="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )"
    if [[ -z "$active" ]]; then
      status 42 all controller wait_for_gpus end 0
      return 0
    fi
    echo "[$(date -Iseconds)] GPUs occupied by PIDs: $(tr '\n' ' ' <<< "$active")"
    status 42 all controller wait_for_gpus poll ""
    sleep 60
  done
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
  local seed="$1"
  local split="$2"
  local variant="$3"
  local port="$4"
  local config="$RUN_ROOT/configs/seed$seed/$variant/$split/train.yaml"
  local checkpoint_dir="$RUN_ROOT/checkpoints/seed$seed/$split/$variant"
  local selected="$checkpoint_dir/selected.ckpt"
  local log="$RUN_ROOT/logs/seed$seed/$split/$variant/train.stdout.log"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$selected" ]]; then
    status "$seed" "$split" "$variant" train skip 0
    return 0
  fi
  local command=(
    "$PYTHON_BIN" scripts/train_protein_pooling.py
    --config "$config"
  )
  if [[ -s "$checkpoint_dir/last.ckpt" ]]; then
    command+=(--resume "$checkpoint_dir/last.ckpt")
  fi
  if [[ "$WANDB_MODE" != "disabled" ]]; then
    command+=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-run-name "reactzyme-e2r-${variant}-${split}-seed${seed}"
      --wandb-mode "$WANDB_MODE"
    )
  fi
  status "$seed" "$split" "$variant" train start ""
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$port" "${command[@]}" > "$log" 2>&1
  local rc=$?
  set -e
  status "$seed" "$split" "$variant" train end "$rc"
  if [[ "$rc" -ne 0 ]]; then return "$rc"; fi
  local checkpoint
  checkpoint="$(select_checkpoint "$checkpoint_dir" "$log")"
  if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
    echo "No checkpoint selected for seed=$seed split=$split variant=$variant" >&2
    return 1
  fi
  cp --reflink=auto "$checkpoint" "$selected"
}

evaluate_one() {
  local seed="$1"
  local split="$2"
  local variant="$3"
  local gpu="$4"
  local subset="$5"
  local protocol="paper_test_candidates"
  if [[ "$subset" == "validation" ]]; then
    protocol="configured_forward_candidates"
  fi
  local checkpoint="$RUN_ROOT/checkpoints/seed$seed/$split/$variant/selected.ckpt"
  local config="$RUN_ROOT/configs/seed$seed/$variant/$split/$subset.yaml"
  local out_dir="$RUN_ROOT/eval/seed$seed/$split/$variant"
  local output="$out_dir/$subset.json"
  local log="$out_dir/$subset.stdout.log"
  mkdir -p "$out_dir"
  if [[ -s "$output" ]]; then
    status "$seed" "$split" "$variant" "${subset}_eval" skip 0
    return 0
  fi
  status "$seed" "$split" "$variant" "${subset}_eval" start ""
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --device cuda:0 \
    --direction both \
    --evaluation-protocol "$protocol" \
    --batch-size 512 \
    --target-batch-size "$EVAL_TARGET_BATCH_SIZE" \
    --output "$output" \
    > "$log" 2>&1
  local rc=$?
  set -e
  status "$seed" "$split" "$variant" "${subset}_eval" end "$rc"
  return "$rc"
}

run_wave() {
  local seed="$1"
  local variant="$2"
  local wave_index="$3"
  local pids=()
  local split_index=0 split port
  echo "[$(date -Iseconds)] Starting seed=$seed wave=$variant"
  for split in time enzyme_smi reaction_smi; do
    port=$((BASE_MASTER_PORT + wave_index * 10 + split_index))
    train_one "$seed" "$split" "$variant" "$port" &
    pids+=("$!")
    split_index=$((split_index + 1))
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Training wave failed: seed=$seed variant=$variant" >&2
    return 1
  fi

  for subset in validation test; do
    pids=()
    split_index=0
    for split in time enzyme_smi reaction_smi; do
      evaluate_one "$seed" "$split" "$variant" "$split_index" "$subset" &
      pids+=("$!")
      split_index=$((split_index + 1))
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then failed=1; fi
    done
    if [[ "$failed" -ne 0 ]]; then
      echo "Evaluation failed: seed=$seed variant=$variant subset=$subset" >&2
      return 1
    fi
  done
  "$PYTHON_BIN" scripts/report_reactzyme_e2r_pareto.py "$RUN_ROOT" --seed "$seed" \
    > "$RUN_ROOT/logs/controller/report_seed${seed}.log"
  echo "[$(date -Iseconds)] Completed seed=$seed wave=$variant"
}

mine_q1_hard_negatives() {
  local pids=()
  local split_index=0 split
  for split in time enzyme_smi reaction_smi; do
    (
      local checkpoint="$RUN_ROOT/checkpoints/seed42/$split/Q1_e2r_adapter/selected.ckpt"
      local config="$RUN_ROOT/configs/seed42/Q1_e2r_adapter/$split/mine_train.yaml"
      local out_dir="$RUN_ROOT/hard_negatives/seed42/$split"
      local output="$out_dir/Q1_e2r_adapter.json"
      local metrics="$out_dir/Q1_train_metrics.json"
      local log="$out_dir/Q1_mining.stdout.log"
      mkdir -p "$out_dir"
      if [[ -s "$output" ]]; then
        status 42 "$split" Q1_e2r_adapter hard_negative_mining skip 0
        exit 0
      fi
      status 42 "$split" Q1_e2r_adapter hard_negative_mining start ""
      CUDA_VISIBLE_DEVICES="$split_index" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
        --checkpoint "$checkpoint" \
        --config "$config" \
        --device cuda:0 \
        --direction enzyme_to_reaction \
        --evaluation-protocol paper_test_candidates \
        --batch-size 512 \
        --target-batch-size "$EVAL_TARGET_BATCH_SIZE" \
        --hard-negative-output "$output" \
        --hard-negative-top-k 256 \
        --output "$metrics" \
        > "$log" 2>&1
      status 42 "$split" Q1_e2r_adapter hard_negative_mining end 0
    ) &
    pids+=("$!")
    split_index=$((split_index + 1))
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Q1 hard-negative mining failed" >&2
    return 1
  fi
}

status 42 all controller campaign start ""
setup_dense
if [[ "$SETUP_ONLY" -eq 1 ]]; then
  status 42 all controller campaign setup_only
  exit 0
fi
wait_for_free_gpus

variants=(
  Q0_f3
  Q1_e2r_adapter
  Q2_e2r_hardneg
  Q3_dense_transform
  Q4_factorized_modalities
  Q5_combined
)
wave_index=0
for variant in "${variants[@]}"; do
  run_wave 42 "$variant" "$wave_index"
  if [[ "$variant" == "Q1_e2r_adapter" ]]; then
    mine_q1_hard_negatives
  fi
  wave_index=$((wave_index + 1))
done

promotion="$(
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
payload=json.loads(Path("$RUN_ROOT/reports/promotion.json").read_text())
print(" ".join(payload.get("variants", [])))
PY
)"
if [[ -z "$promotion" ]]; then
  echo "No variants passed validation promotion guardrails" >&2
  exit 1
fi

for seed in 7 23; do
  run_wave "$seed" Q0_f3 "$wave_index"
  wave_index=$((wave_index + 1))
  for variant in $promotion; do
    run_wave "$seed" "$variant" "$wave_index"
    wave_index=$((wave_index + 1))
  done
done

status 42 all controller campaign end 0
echo "ReactZyme E2R Pareto campaign complete"
