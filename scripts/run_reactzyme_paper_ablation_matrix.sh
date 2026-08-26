#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/outputs/reactzyme_latent_organization_paper_ablation_20260717"

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
SKIP_SETUP=0
SKIP_TRAIN=0
FORCE_SETUP=0
ONLY_VARIANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
      ;;
    --foreground)
      DETACH=0
      shift
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --force-setup)
      FORCE_SETUP=1
      shift
      ;;
    --variant)
      ONLY_VARIANT="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  pid_file="$RUN_ROOT/logs/controller/pid"
  rm -f "$pid_file"
  detach_args=("$RUN_ROOT" --foreground)
  [[ "$SKIP_SETUP" -eq 1 ]] && detach_args+=(--skip-setup)
  [[ "$SKIP_TRAIN" -eq 1 ]] && detach_args+=(--skip-train)
  [[ "$FORCE_SETUP" -eq 1 ]] && detach_args+=(--force-setup)
  [[ -n "$ONLY_VARIANT" ]] && detach_args+=(--variant "$ONLY_VARIANT")
  setsid -f bash -c '
    pid_file="$1"
    shift
    printf "%s\n" "$$" > "$pid_file"
    exec bash "$@"
  ' bash "$pid_file" "$0" "${detach_args[@]}" > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  for _ in {1..50}; do
    [[ -s "$pid_file" ]] && break
    sleep 0.1
  done
  echo "Launched ReactZyme paper ablation controller"
  echo "PID: $(cat "$pid_file" 2>/dev/null || printf unknown)"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  echo "Status: $RUN_ROOT/logs/status.jsonl"
  exit 0
fi

RUNTIME_ENV="$RUN_ROOT/runtime.env"
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing runtime env: $RUNTIME_ENV" >&2
  echo "Run scripts/create_reactzyme_paper_ablation_matrix.py first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$RUNTIME_ENV"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-$PYTHON_BIN}"
MATRIX_NAME="${MATRIX_NAME:-latent}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-paper_test_candidates}"
cd "$ROOT"

SETUP_PLAN="$RUN_ROOT/setup_plan.tsv"
TRAIN_PLAN="$RUN_ROOT/train_plan.tsv"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
SHORT_TMPDIR="${HORIZYN_TMPDIR:-/tmp/hz_latent_ablate_20260714}"

mkdir -p \
  "$RUN_ROOT/logs/controller" \
  "$RUN_ROOT/logs/setup" \
  "$RUN_ROOT/nohome" \
  "$RUN_ROOT/tmp" \
  "$RUN_ROOT/cache/huggingface" \
  "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/xdg_cache" \
  "$RUN_ROOT/xdg_config" \
  "$RUN_ROOT/xdg_state" \
  "$RUN_ROOT/wandb" \
  "$RUN_ROOT/wandb/data" \
  "$RUN_ROOT/wandb/cache" \
  "$RUN_ROOT/wandb/config" \
  "$RUN_ROOT/matplotlib" \
  "$SHORT_TMPDIR"

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
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "$RUN_ROOT/wandb/api_key")"
fi
export WANDB_MODE="$WANDB_MODE"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
# These ablation jobs are single-node DDP jobs. Do not inherit a cluster-level
# MASTER_ADDR here, because that can make concurrent local jobs try to rendezvous
# through a stale external address and fail during NCCL setup.
export MASTER_ADDR="${HORIZYN_MASTER_ADDR:-127.0.0.1}"
export NCCL_ASYNC_ERROR_HANDLING=1

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT" "$1" ;;
  esac
}

status() {
  local run_id="$1"
  local stage="$2"
  local event="$3"
  local port="${4:-}"
  local rc="${5:-}"
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","master_port":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$run_id" "$stage" "$event" "$port" "$rc" >> "$STATUS_LOG"
}

for path in "$PYTHON_BIN" "$SETUP_PYTHON_BIN" "$SETUP_PLAN" "$TRAIN_PLAN"; do
  require_file "$path"
done

run_setup_stage() {
  if [[ "$SKIP_SETUP" -eq 1 ]]; then
    echo "Skipping setup stage"
    return
  fi

  echo "Starting train-only artifact setup"
  if [[ "$MATRIX_NAME" == "representation" ]]; then
    status "global" "setup_factorized_capability" "start" "" ""
    if [[ "$FORCE_SETUP" -eq 0 && -s "$FACTORIZED_CAPABILITY_VECTORS" ]]; then
      echo "[global] factorized capability vectors already exist"
    else
      mkdir -p "$(dirname "$FACTORIZED_CAPABILITY_VECTORS")"
      "$SETUP_PYTHON_BIN" scripts/build_factorized_capability_vectors.py \
        --flat-vectors "$CAPABILITY_VECTORS" \
        --out-npz "$FACTORIZED_CAPABILITY_VECTORS" \
        --out-metadata "$FACTORIZED_CAPABILITY_METADATA" \
        > "$RUN_ROOT/logs/setup/factorized_capability.log" 2>&1
    fi
    status "global" "setup_factorized_capability" "end" "" "0"
  fi

  local header split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report; do
      [[ -z "${split:-}" ]] && continue
      local train_pairs_abs train_reactions_abs candidate_ids_abs
      train_pairs_abs="$(abs_path "$train_pairs")"
      train_reactions_abs="$(abs_path "$train_reactions")"
      candidate_ids_abs="$(abs_path "$candidate_ids")"
      mkdir -p "$(dirname "$reaction_features")" "$(dirname "$biofp_npz")" "$(dirname "$hardneg_json")" "$(dirname "$enzyme_ec_labels")"

      status "$split" "setup_reaction_features" "start" "" ""
      if [[ "$FORCE_SETUP" -eq 0 && -s "$reaction_features" ]]; then
        echo "[$split] reaction capability features already exist"
      else
        "$SETUP_PYTHON_BIN" scripts/build_reaction_features.py \
          --reaction-smiles "$train_reactions_abs" \
          --train-pairs "$train_pairs_abs" \
          --cofactor-dictionary "$COFACTOR_DICTIONARY" \
          --out-dir "$(dirname "$reaction_features")" \
          --use-rxnmapper false \
          > "$RUN_ROOT/logs/setup/${split}_reaction_features.log" 2>&1
      fi
      status "$split" "setup_reaction_features" "end" "" "0"

      if [[ "$MATRIX_NAME" == "representation" ]]; then
        local validation_reactions_abs test_reactions_abs validation_reaction_features test_reaction_features validation_reaction_feature_dir test_reaction_feature_dir reaction_chemistry_dir train_chemistry_npz validation_chemistry_npz test_chemistry_npz chemistry_vocab
        validation_reactions_abs="$PROTOCOL_ROOT/$split/validation_rxns.csv"
        test_reactions_abs="$PROTOCOL_ROOT/$split/test_rxns.csv"
        validation_reaction_feature_dir="$RUN_ROOT/data/$split/capability/validation_reaction_features"
        validation_reaction_features="$validation_reaction_feature_dir/reaction_features.parquet"
        test_reaction_feature_dir="$RUN_ROOT/data/$split/capability/test_reaction_features"
        test_reaction_features="$test_reaction_feature_dir/reaction_features.parquet"
        reaction_chemistry_dir="$RUN_ROOT/data/$split/reaction_chemistry"
        train_chemistry_npz="$reaction_chemistry_dir/train_reaction_chemistry_vectors.npz"
        validation_chemistry_npz="$reaction_chemistry_dir/validation_reaction_chemistry_vectors.npz"
        test_chemistry_npz="$reaction_chemistry_dir/test_reaction_chemistry_vectors.npz"
        chemistry_vocab="$reaction_chemistry_dir/reaction_chemistry_vocab.json"
        mkdir -p "$validation_reaction_feature_dir" "$test_reaction_feature_dir" "$reaction_chemistry_dir"

        status "$split" "setup_validation_reaction_features" "start" "" ""
        if [[ "$FORCE_SETUP" -eq 0 && -s "$validation_reaction_features" ]]; then
          echo "[$split] validation reaction capability features already exist"
        else
          "$SETUP_PYTHON_BIN" scripts/build_reaction_features.py \
            --reaction-smiles "$validation_reactions_abs" \
            --cofactor-dictionary "$COFACTOR_DICTIONARY" \
            --out-dir "$validation_reaction_feature_dir" \
            --use-rxnmapper false \
            > "$RUN_ROOT/logs/setup/${split}_validation_reaction_features.log" 2>&1
        fi
        status "$split" "setup_validation_reaction_features" "end" "" "0"

        status "$split" "setup_test_reaction_features" "start" "" ""
        if [[ "$FORCE_SETUP" -eq 0 && -s "$test_reaction_features" ]]; then
          echo "[$split] test reaction capability features already exist"
        else
          "$SETUP_PYTHON_BIN" scripts/build_reaction_features.py \
            --reaction-smiles "$test_reactions_abs" \
            --cofactor-dictionary "$COFACTOR_DICTIONARY" \
            --out-dir "$test_reaction_feature_dir" \
            --use-rxnmapper false \
            > "$RUN_ROOT/logs/setup/${split}_test_reaction_features.log" 2>&1
        fi
        status "$split" "setup_test_reaction_features" "end" "" "0"

        status "$split" "setup_reaction_chemistry_vectors" "start" "" ""
        if [[ "$FORCE_SETUP" -eq 0 && -s "$train_chemistry_npz" && -s "$validation_chemistry_npz" && -s "$test_chemistry_npz" && -s "$chemistry_vocab" ]]; then
          echo "[$split] reaction chemistry vectors already exist"
        else
          "$SETUP_PYTHON_BIN" scripts/build_reaction_chemistry_vectors.py \
            --reaction-features "$reaction_features" \
            --out-npz "$train_chemistry_npz" \
            --out-vocab "$chemistry_vocab" \
            --output-dim 256 \
            > "$RUN_ROOT/logs/setup/${split}_reaction_chemistry_train.log" 2>&1
          "$SETUP_PYTHON_BIN" scripts/build_reaction_chemistry_vectors.py \
            --reaction-features "$validation_reaction_features" \
            --vocab-json "$chemistry_vocab" \
            --out-npz "$validation_chemistry_npz" \
            --output-dim 256 \
            > "$RUN_ROOT/logs/setup/${split}_reaction_chemistry_validation.log" 2>&1
          "$SETUP_PYTHON_BIN" scripts/build_reaction_chemistry_vectors.py \
            --reaction-features "$test_reaction_features" \
            --vocab-json "$chemistry_vocab" \
            --out-npz "$test_chemistry_npz" \
            --output-dim 256 \
            > "$RUN_ROOT/logs/setup/${split}_reaction_chemistry_test.log" 2>&1
        fi
        status "$split" "setup_reaction_chemistry_vectors" "end" "" "0"
      fi

      status "$split" "setup_biofp" "start" "" ""
      if [[ "$FORCE_SETUP" -eq 0 && -s "$biofp_npz" && -s "$biofp_vocab" ]]; then
        echo "[$split] BioFP targets already exist"
      else
        "$SETUP_PYTHON_BIN" scripts/build_enzyme_biofp_soft_targets.py \
          --train-pairs "$train_pairs_abs" \
          --reaction-features "$reaction_features" \
          --enzyme-cofactor-labels "$ENZYME_COFACTOR_LABELS" \
          --glycosyl-transfer-cofactor-rule type_or_transition \
          --cofactor-label-set expanded \
          --out-npz "$biofp_npz" \
          --out-vocab "$biofp_vocab" \
          > "$RUN_ROOT/logs/setup/${split}_biofp.log" 2>&1
      fi
      status "$split" "setup_biofp" "end" "" "0"

      status "$split" "setup_hardneg" "start" "" ""
      if [[ "$FORCE_SETUP" -eq 0 && -s "$hardneg_json" && -s "$hardneg_parquet" && -s "$hardneg_report" ]]; then
        echo "[$split] hard-negative pools already exist"
      else
        "$SETUP_PYTHON_BIN" scripts/mine_r2e_hard_negatives.py \
          --train-pairs "$train_pairs_abs" \
          --candidate-ids "$candidate_ids_abs" \
          --biofp-targets "$biofp_npz" \
          --out-json "$hardneg_json" \
          --out-parquet "$hardneg_parquet" \
          --out-report "$hardneg_report" \
          --max-negatives-per-query "$HARD_NEGATIVE_MAX_PER_QUERY" \
          --seed "$SEED" \
          > "$RUN_ROOT/logs/setup/${split}_hardneg.log" 2>&1
      fi
      status "$split" "setup_hardneg" "end" "" "0"

      status "$split" "setup_ec" "start" "" ""
      if [[ "$FORCE_SETUP" -eq 0 && -s "$enzyme_ec_labels" && -s "$ec_report" ]]; then
        echo "[$split] EC labels already exist"
      else
        "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime build-ec-subset \
          "$train_pairs_abs" \
          "$EC_SOURCE" \
          "$enzyme_ec_labels" \
          "$ec_report" \
          > "$RUN_ROOT/logs/setup/${split}_ec.log" 2>&1
      fi
      status "$split" "setup_ec" "end" "" "0"
    done
  } < "$SETUP_PLAN"
  echo "Artifact setup complete"
}

select_checkpoint() {
  local checkpoint_dir="$1"
  local log_path="$2"
  local best=""
  if [[ -f "$log_path" ]]; then
    best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' "$log_path" | tr -d '\r')"
  fi
  if [[ -n "$best" && -f "$best" ]]; then
    printf '%s\n' "$best"
  elif [[ -f "$checkpoint_dir/last.ckpt" ]]; then
    printf '%s\n' "$checkpoint_dir/last.ckpt"
  else
    find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
  fi
}

run_train_eval_job() {
  local variant="$1"
  local split="$2"
  local run_id="$3"
  local config="$4"
  local test_config="$5"
  local checkpoint_dir="$6"
  local log_path="$7"
  local eval_json="$8"
  local wandb_run_name="$9"
  local master_port="${10}"
  mkdir -p "$checkpoint_dir" "$(dirname "$log_path")" "$(dirname "$eval_json")"
  if [[ -s "$eval_json" ]]; then
    status "$run_id" "test_eval" "skip" "$master_port" "0"
    echo "[$(date -Iseconds)] SKIP $run_id existing eval=$eval_json"
    return 0
  fi
  local checkpoint
  local checkpoint_sidecar="$eval_json.checkpoint.json"
  if [[ -s "$checkpoint_sidecar" ]]; then
    checkpoint="$(
      "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime \
        read-checkpoint "$checkpoint_sidecar"
    )"
  else
    checkpoint=""
  fi
  if [[ -n "$checkpoint" ]]; then
    status "$run_id" "train" "skip" "$master_port" "0"
    echo "[$(date -Iseconds)] SKIP train $run_id existing checkpoint=$checkpoint"
  else
    status "$run_id" "train" "start" "$master_port" ""
    echo "[$(date -Iseconds)] START train $run_id variant=$variant split=$split gpus=$GPUS port=$master_port"
    local train_cmd=(
      "$PYTHON_BIN" scripts/train_protein_pooling.py
      --config "$config"
    )
    local resume_checkpoint="$checkpoint_dir/last.ckpt"
    if [[ -s "$resume_checkpoint" ]]; then
      echo "[$(date -Iseconds)] RESUME train $run_id checkpoint=$resume_checkpoint"
      train_cmd+=(--resume "$resume_checkpoint")
    fi
    if [[ "$WANDB_MODE" != "disabled" ]]; then
      train_cmd+=(
        --wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "$wandb_run_name"
        --wandb-mode "$WANDB_MODE"
      )
      if [[ -n "${WANDB_ENTITY:-}" ]]; then
        train_cmd+=(--wandb-entity "$WANDB_ENTITY")
      fi
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$master_port" "${train_cmd[@]}" > "$log_path" 2>&1
    local train_rc=$?
    set -e
    status "$run_id" "train" "end" "$master_port" "$train_rc"
    echo "[$(date -Iseconds)] END train $run_id rc=$train_rc log=$log_path"
    if [[ "$train_rc" -ne 0 ]]; then
      return "$train_rc"
    fi
    checkpoint="$(select_checkpoint "$checkpoint_dir" "$log_path")"
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "No checkpoint found for $run_id in $checkpoint_dir" >&2
    return 1
  fi
  "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime write-checkpoint \
    "$checkpoint_sidecar" "$run_id" "$checkpoint"

  local eval_log="${eval_json%.json}.stdout.log"
  local eval_device="cuda"
  case "$split" in
    time)
      eval_device="cuda:0"
      ;;
    enzyme_smi)
      eval_device="cuda:1"
      ;;
    reaction_smi)
      eval_device="cuda:2"
      ;;
  esac
  status "$run_id" "test_eval" "start" "$master_port" ""
  echo "[$(date -Iseconds)] START test_eval $run_id checkpoint=$checkpoint device=$eval_device protocol=$EVALUATION_PROTOCOL"
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" \
    --config "$test_config" \
    --device "$eval_device" \
    --direction both \
    --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --batch-size 512 \
    --target-batch-size 2048 \
    --output "$eval_json" \
    > "$eval_log" 2>&1
  local eval_rc=$?
  set -e
  status "$run_id" "test_eval" "end" "$master_port" "$eval_rc"
  echo "[$(date -Iseconds)] END test_eval $run_id rc=$eval_rc eval=$eval_json"
  return "$eval_rc"
}

run_wave() {
  local variant="$1"
  local pids=()
  local header wave row_variant label split run_id config test_config checkpoint_dir log_path eval_json wandb_run_name master_port description
  echo "Starting wave $variant"
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r wave row_variant label split run_id config test_config checkpoint_dir log_path eval_json wandb_run_name master_port description; do
      [[ -z "${run_id:-}" ]] && continue
      [[ "$row_variant" == "$variant" ]] || continue
      (
        run_train_eval_job "$row_variant" "$split" "$run_id" "$config" "$test_config" "$checkpoint_dir" "$log_path" "$eval_json" "$wandb_run_name" "$master_port"
      ) &
      pids+=("$!")
    done
  } < "$TRAIN_PLAN"
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "No rows found for variant $variant" >&2
    return 1
  fi
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Wave $variant failed" >&2
    return 1
  fi
  echo "Wave $variant complete"
}

run_train_stage() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "Skipping train/eval stage"
    return
  fi
  local variants
  if [[ -n "$ONLY_VARIANT" ]]; then
    variants="$ONLY_VARIANT"
  else
    variants="$(
      "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime \
        list-variants "$TRAIN_PLAN"
    )"
  fi
  local variant
  for variant in $variants; do
    run_wave "$variant"
    write_summary
  done
}

write_summary() {
  "$PYTHON_BIN" -m horizyn.benchmarks.reactzyme_runtime write-summary \
    "$TRAIN_PLAN" \
    "$RUN_ROOT/eval/summary.csv" \
    "$RUN_ROOT/eval/summary.md"
}

write_reaction_feature_tiger_comparison() {
  if [[ "$MATRIX_NAME" != "reaction_features" || -n "$ONLY_VARIANT" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/report_reactzyme_reaction_feature_ablation.py \
    --run-root "$RUN_ROOT"
}

echo "Run root: $RUN_ROOT"
echo "W&B: ${WANDB_ENTITY:-}/$WANDB_PROJECT mode=$WANDB_MODE"
echo "GPUs per run: $GPUS"
echo "Execution: 3 independent 4-GPU DDP runs per ablation wave"
echo "Evaluation protocol: $EVALUATION_PROTOCOL"
echo "TMPDIR: $TMPDIR"
status "controller" "controller" "start" "" ""
run_setup_stage
run_train_stage
write_summary
write_reaction_feature_tiger_comparison
status "controller" "controller" "end" "" "0"
echo "ReactZyme paper ablation complete"
echo "Summary: $RUN_ROOT/eval/summary.md"
