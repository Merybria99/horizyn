#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/bio_aux_minimal_v1}"
[[ "$RUN_ROOT" = /* ]] || RUN_ROOT="$ROOT/$RUN_ROOT"
[[ $# -gt 0 ]] && shift

PHASE="train"
DETACH=0
FORCE_SETUP=0
ONLY_VARIANTS=()
SELECTION_MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --force-setup) FORCE_SETUP=1; shift ;;
    --variant) ONLY_VARIANTS+=("${2:?missing variant}"); shift 2 ;;
    --selection-manifest) SELECTION_MANIFEST="${2:?missing manifest}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in prepare|pretrain|retrieval|train|freeze|test) ;; *)
  echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  args=("$RUN_ROOT" --phase "$PHASE" --foreground)
  [[ "$FORCE_SETUP" -eq 1 ]] && args+=(--force-setup)
  [[ -n "$SELECTION_MANIFEST" ]] && args+=(--selection-manifest "$SELECTION_MANIFEST")
  for variant in "${ONLY_VARIANTS[@]}"; do args+=(--variant "$variant"); done
  setsid -f "$0" "${args[@]}" >"$RUN_ROOT/logs/controller/nohup.log" 2>&1
  echo "Launched controller; log=$RUN_ROOT/logs/controller/nohup.log"
  exit 0
fi

mkdir -p "$RUN_ROOT/nohome"
export HOME="$RUN_ROOT/nohome"
source "$RUN_ROOT/runtime.env"
cd "$ROOT"
PRETRAIN_PLAN="$RUN_ROOT/pretrain_plan.tsv"
TRAIN_PLAN="$RUN_ROOT/train_plan.tsv"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
RUNTIME_HELPER="$ROOT/horizyn/benchmarks/reactzyme_runtime.py"

mkdir -p "$RUN_ROOT"/{logs/controller,logs/setup,nohome,tmp,cache/huggingface,cache/torch,xdg_cache,xdg_config,xdg_state,wandb/data,wandb/cache,wandb/config,matplotlib}

# Python's multiprocessing resource sharer creates AF_UNIX sockets below
# TMPDIR. Keep that path short and node-local; all durable artifacts stay in RUN_ROOT.
SHORT_TMPDIR="${HORIZYN_SHORT_TMPDIR:-/tmp/hz-${UID}-${BASE_MASTER_PORT}}"
[[ ! -e "$SHORT_TMPDIR" || -d "$SHORT_TMPDIR" ]] || {
  echo "Short TMPDIR is not a directory: $SHORT_TMPDIR" >&2
  exit 1
}
mkdir -p "$SHORT_TMPDIR"
[[ "$(stat -c %u "$SHORT_TMPDIR")" == "$UID" ]] || {
  echo "Short TMPDIR is owned by another user: $SHORT_TMPDIR" >&2
  exit 1
}
chmod 700 "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
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
export WANDB_MODE TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
export MASTER_ADDR="${HORIZYN_MASTER_ADDR:-127.0.0.1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HORIZYN_CAPABILITY_SITE_PACKAGES="$(
  "$SETUP_PYTHON_BIN" -c 'from importlib.util import find_spec; from pathlib import Path; spec = find_spec("pandas"); assert spec and spec.origin; print(Path(spec.origin).resolve().parents[1])'
)"
if [[ "$PHASE" == prepare || "$PHASE" == train ]]; then
  "$PYTHON_BIN" -c 'import rxnmapper' >/dev/null
fi
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' <"$RUN_ROOT/wandb/api_key")"
fi

status() {
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "$3" "${4:-}" >>"$STATUS_LOG"
}

config_value() {
  "$PYTHON_BIN" -c \
    'import sys, yaml; from functools import reduce; data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); print(reduce(lambda value, key: value[key], sys.argv[2].split("."), data))' \
    "$1" "$2"
}

selected_variant() {
  [[ "${#ONLY_VARIANTS[@]}" -eq 0 ]] && return 0
  local requested
  for requested in "${ONLY_VARIANTS[@]}"; do [[ "$requested" == "$1" ]] && return 0; done
  return 1
}

run_or_skip() {
  local output_spec="$1" log="$2"; shift 2
  local outputs=() output all_present=1
  IFS='|' read -r -a outputs <<<"$output_spec"
  for output in "${outputs[@]}"; do
    [[ -s "$output" ]] || all_present=0
  done
  if [[ "$FORCE_SETUP" -eq 0 && "$all_present" -eq 1 ]]; then return 0; fi
  mkdir -p "$(dirname "${outputs[0]}")" "$(dirname "$log")"
  "$@" >"$log" 2>&1
  for output in "${outputs[@]}"; do
    [[ -s "$output" ]] || { echo "Expected setup output is missing: $output" >&2; return 1; }
  done
}

prepare_split() {
  local split="$1" train_pairs="$2" train_reactions="$3" biofp_npz="$4" biofp_vocab="$5" enzyme_ec_labels="$6" ec_report="$7"
  local base="$RUN_ROOT/data/$split"
  local matches="$base/rhea_matches"
  status "$split" prepare start
  run_or_skip "$matches/matched_members.csv|$matches/matched_pairs.csv|$matches/matched_reactions.csv|$matches/report.json" "$RUN_ROOT/logs/setup/${split}_rhea_matches.log" \
    "$SETUP_PYTHON_BIN" scripts/build_reactzyme_train_rhea_matches.py \
      --protocol-root "$PROTOCOL_ROOT" --split "$split" \
      --cleaned-uniprot-rhea "$CLEANED_UNIPROT_RHEA" --rhea-molecules "$RHEA_MOLECULES" \
      --out-dir "$matches"
  run_or_skip "$matches/directional_features/reaction_features.parquet" "$RUN_ROOT/logs/setup/${split}_directional_features.log" \
    "$PYTHON_BIN" scripts/build_reaction_features.py \
      --reaction-smiles "$matches/matched_reactions.csv" \
      --train-pairs "$matches/matched_pairs.csv" \
      --cofactor-dictionary "$COFACTOR_DICTIONARY" \
      --out-dir "$matches/directional_features" --use-rxnmapper true
  run_or_skip "$biofp_npz|$biofp_vocab" "$RUN_ROOT/logs/setup/${split}_biological_targets.log" \
    "$SETUP_PYTHON_BIN" scripts/build_enzyme_biological_targets.py \
      --train-pairs "$train_pairs" \
      --matched-rhea-members "$matches/matched_members.csv" \
      --directional-reaction-features "$matches/directional_features/reaction_features.parquet" \
      --enzyme-cofactor-labels "$ENZYME_COFACTOR_LABELS" \
      --out-npz "$biofp_npz" --out-vocab "$biofp_vocab"
  run_or_skip "$enzyme_ec_labels|$ec_report" "$RUN_ROOT/logs/setup/${split}_ec.log" \
    "$SETUP_PYTHON_BIN" "$RUNTIME_HELPER" build-ec-subset \
      "$train_pairs" "$EC_SOURCE" "$enzyme_ec_labels" "$ec_report"
  status "$split" prepare end 0
}

prepare_all() {
  local header split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report; do
      prepare_split "$split" "$ROOT/$train_pairs" "$ROOT/$train_reactions" "$biofp_npz" "$biofp_vocab" "$enzyme_ec_labels" "$ec_report"
    done
  } <"$RUN_ROOT/setup_plan.tsv"
}

run_pretrain_job() {
  local run_id="$1" stage="$2" config="$3" port="$4" log="$5"
  [[ -z "$config" || "$config" == "-" ]] && return 0
  local checkpoint checkpoint_dir summary
  if [[ "$stage" == ec_pretrain ]]; then
    checkpoint="$(config_value "$config" output_checkpoint)"
    checkpoint_dir="$(dirname "$checkpoint")"
  else
    checkpoint_dir="$(config_value "$config" logging.checkpoint_dir)"
    checkpoint="$checkpoint_dir/last.ckpt"
  fi
  summary="$checkpoint_dir/pretrain_summary.json"
  if [[ -s "$checkpoint" && -s "$summary" ]]; then
    status "$run_id" "$stage" skip 0
    return 0
  fi
  status "$run_id" "$stage" start
  set +e
  if [[ "$stage" == ec_pretrain ]]; then
    local ec_resume=()
    [[ -s "$checkpoint" ]] && ec_resume=(--resume-checkpoint "$checkpoint")
    if [[ "${#ec_resume[@]}" -gt 0 ]]; then
      printf '\n===== resuming %s at %s =====\n' "$run_id" "$(date -Iseconds)" >>"$log"
      CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc-per-node=4 --master-port="$port" scripts/pretrain_hyperbolic_enzyme.py \
        --config "$config" "${ec_resume[@]}" >>"$log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc-per-node=4 --master-port="$port" scripts/pretrain_hyperbolic_enzyme.py \
        --config "$config" >"$log" 2>&1
    fi
  else
    local biological_resume=()
    [[ -s "$checkpoint" ]] && biological_resume=(--resume "$checkpoint")
    if [[ "${#biological_resume[@]}" -gt 0 ]]; then
      printf '\n===== resuming %s at %s =====\n' "$run_id" "$(date -Iseconds)" >>"$log"
      CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$port" "$PYTHON_BIN" \
        scripts/pretrain_enzyme_biofp_split.py --config "$config" \
        "${biological_resume[@]}" >>"$log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$port" "$PYTHON_BIN" \
        scripts/pretrain_enzyme_biofp_split.py --config "$config" >"$log" 2>&1
    fi
  fi
  local rc=$?
  set -e
  status "$run_id" "$stage" end "$rc"
  return "$rc"
}

run_pretrain_wave() {
  local variant="$1" stage="$2" pids=()
  local header wave row_variant label split run_id ec_config biological_config port config log
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r wave row_variant label split run_id ec_config biological_config port; do
      [[ "$row_variant" == "$variant" ]] || continue
      config="$biological_config"; [[ "$stage" == ec_pretrain ]] && config="$ec_config"
      [[ -n "$config" && "$config" != "-" ]] || continue
      log="$RUN_ROOT/logs/$split/$row_variant/${stage}.log"
      mkdir -p "$(dirname "$log")"
      (run_pretrain_job "$run_id" "$stage" "$config" "$port" "$log") & pids+=("$!")
    done
  } <"$PRETRAIN_PLAN"
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
}

pretrain_all() {
  local variants variant
  variants="$($PYTHON_BIN "$RUNTIME_HELPER" list-variants "$PRETRAIN_PLAN")"
  for variant in $variants; do
    selected_variant "$variant" || continue
    run_pretrain_wave "$variant" ec_pretrain
    run_pretrain_wave "$variant" biological_pretrain
  done
}

select_checkpoint() {
  local checkpoint_dir="$1" log="$2" best=""
  [[ -f "$log" ]] && best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {x=$2} END {print x}' "$log" | tr -d '\r')"
  if [[ -n "$best" && -f "$best" ]]; then printf '%s\n' "$best"
  elif [[ -f "$checkpoint_dir/last.ckpt" ]]; then printf '%s\n' "$checkpoint_dir/last.ckpt"
  else find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,"");print}'; fi
}

run_retrieval_job() {
  local variant="$1" split="$2" run_id="$3" config="$4" checkpoint_dir="$5" log="$6" eval_json="$7" wandb_name="$8" port="$9"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")" "$(dirname "$eval_json")"
  local sidecar="$eval_json.checkpoint.json" checkpoint=""
  [[ -s "$sidecar" ]] && checkpoint="$($PYTHON_BIN "$RUNTIME_HELPER" read-checkpoint "$sidecar")"
  if [[ -z "$checkpoint" ]]; then
    status "$run_id" retrieval start
    local cmd=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
    [[ -s "$checkpoint_dir/last.ckpt" ]] && cmd+=(--resume "$checkpoint_dir/last.ckpt")
    if [[ "$WANDB_MODE" != disabled ]]; then
      cmd+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$wandb_name" --wandb-mode "$WANDB_MODE")
      [[ -n "${WANDB_ENTITY:-}" ]] && cmd+=(--wandb-entity "$WANDB_ENTITY")
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$port" "${cmd[@]}" >"$log" 2>&1
    local rc=$?
    set -e
    status "$run_id" retrieval end "$rc"
    [[ "$rc" -eq 0 ]] || return "$rc"
    checkpoint="$(select_checkpoint "$checkpoint_dir" "$log")"
    [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
    "$PYTHON_BIN" "$RUNTIME_HELPER" write-checkpoint "$sidecar" "$run_id" "$checkpoint"
  fi
}

run_retrieval_wave() {
  local variant="$1" pids=()
  local header wave row_variant label split run_id config test_config checkpoint_dir log eval_json wandb_name port description
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r wave row_variant label split run_id config test_config checkpoint_dir log eval_json wandb_name port description; do
      [[ "$row_variant" == "$variant" ]] || continue
      (run_retrieval_job "$row_variant" "$split" "$run_id" "$config" "$checkpoint_dir" "$log" "$eval_json" "$wandb_name" "$port") & pids+=("$!")
    done
  } <"$TRAIN_PLAN"
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
}

retrieval_all() {
  local variants variant
  variants="$($PYTHON_BIN "$RUNTIME_HELPER" list-variants "$TRAIN_PLAN")"
  for variant in $variants; do selected_variant "$variant" || continue; run_retrieval_wave "$variant"; done
}

freeze_selection() {
  local output="${SELECTION_MANIFEST:-$RUN_ROOT/selection/frozen_selection.json}"
  local cmd=("$PYTHON_BIN" "$RUNTIME_HELPER" freeze-selection "$TRAIN_PLAN" "$output")
  [[ "${#ONLY_VARIANTS[@]}" -gt 0 ]] && cmd+=(--variants "${ONLY_VARIANTS[@]}")
  "${cmd[@]}"
  echo "Frozen selection: $output"
}

run_test_job() {
  local variant="$1" split="$2" run_id="$3" checkpoint="$4" test_config="$5" eval_json="$6" device="$7"
  [[ -s "$eval_json" ]] && return 0
  local returncode
  mkdir -p "$(dirname "$eval_json")"
  status "$run_id" test start
  if OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$GPUS" \
      "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$test_config" --device "$device" \
      --direction both --evaluation-protocol "$EVALUATION_PROTOCOL" \
      --batch-size 512 --target-batch-size 2048 --output "$eval_json" \
      >"${eval_json%.json}.stdout.log" 2>&1; then
    status "$run_id" test end 0
  else
    returncode=$?
    status "$run_id" test end "$returncode"
    return "$returncode"
  fi
}

run_test_worker() {
  local rows="$1" slot="$2"
  local index=0
  local header variant label split run_id checkpoint sha test_config eval_json port
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r variant label split run_id checkpoint sha test_config eval_json port; do
      if (( index % TEST_GPU_COUNT == slot )); then
        run_test_job \
          "$variant" "$split" "$run_id" "$checkpoint" "$test_config" "$eval_json" \
          "cuda:$slot"
      fi
      index=$((index + 1))
    done
  } <"$rows"
}

test_frozen() {
  [[ -n "$SELECTION_MANIFEST" ]] || { echo "--selection-manifest is required for test" >&2; exit 2; }
  local rows="$RUN_ROOT/selection/verified_rows.tsv"
  mkdir -p "$(dirname "$rows")"
  "$PYTHON_BIN" "$RUNTIME_HELPER" selection-rows "$SELECTION_MANIFEST" >"$rows"
  local gpu_ids pids=() slot pid failed=0
  IFS=',' read -r -a gpu_ids <<<"$GPUS"
  TEST_GPU_COUNT="${#gpu_ids[@]}"
  export TEST_GPU_COUNT
  for ((slot = 0; slot < TEST_GPU_COUNT; slot++)); do
    (run_test_worker "$rows" "$slot") & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1
  "$PYTHON_BIN" "$RUNTIME_HELPER" write-summary "$TRAIN_PLAN" "$RUN_ROOT/eval/summary.csv" "$RUN_ROOT/eval/summary.md"
}

echo "Run root: $RUN_ROOT"
echo "Phase: $PHASE; three independent 4-GPU DDP split jobs per wave"
case "$PHASE" in
  prepare) prepare_all ;;
  pretrain) pretrain_all ;;
  retrieval) retrieval_all ;;
  train) prepare_all; pretrain_all; retrieval_all ;;
  freeze) freeze_selection ;;
  test) test_frozen ;;
esac
