#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/outputs/reactzyme_no_text_ablation_20260713_full_matrix"

RUN_ROOT_ARG="$DEFAULT_RUN_ROOT"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_ROOT_ARG="$1"
  shift
fi

DETACH=0
SKIP_SETUP=0
SKIP_PRETRAIN=0
SKIP_TRAIN=0
FORCE_SETUP=0
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
    --skip-pretrain)
      SKIP_PRETRAIN=1
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
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_ROOT_ARG" != /* ]]; then
  RUN_ROOT="$ROOT/$RUN_ROOT_ARG"
else
  RUN_ROOT="$RUN_ROOT_ARG"
fi

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  detach_args=("$RUN_ROOT" --foreground)
  [[ "$SKIP_SETUP" -eq 1 ]] && detach_args+=(--skip-setup)
  [[ "$SKIP_PRETRAIN" -eq 1 ]] && detach_args+=(--skip-pretrain)
  [[ "$SKIP_TRAIN" -eq 1 ]] && detach_args+=(--skip-train)
  [[ "$FORCE_SETUP" -eq 1 ]] && detach_args+=(--force-setup)
  pid_file="$RUN_ROOT/logs/controller/pid"
  rm -f "$pid_file"
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
  controller_pid="$(cat "$pid_file" 2>/dev/null || true)"
  echo "Launched no-text ablation matrix controller"
  echo "PID: ${controller_pid:-unknown}"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  echo "Status: $RUN_ROOT/logs/status.jsonl"
  exit 0
fi

RUNTIME_ENV="$RUN_ROOT/runtime.env"
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing runtime env: $RUNTIME_ENV" >&2
  echo "Run scripts/create_reactzyme_no_text_ablation_matrix.py first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$RUNTIME_ENV"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-$PYTHON_BIN}"
cd "$ROOT"

SETUP_PLAN="$RUN_ROOT/setup_plan.tsv"
PRETRAIN_PLAN="$RUN_ROOT/pretrain_plan.tsv"
TRAIN_PLAN="$RUN_ROOT/train_plan.tsv"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
SHORT_TMPDIR="${HORIZYN_TMPDIR:-/tmp/hz_nt_ablate_20260713}"

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
  local gpu="${4:-}"
  local rc="${5:-}"
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","gpu":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$run_id" "$stage" "$event" "$gpu" "$rc" >> "$STATUS_LOG"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

for path in "$PYTHON_BIN" "$SETUP_PYTHON_BIN" "$SETUP_PLAN" "$PRETRAIN_PLAN" "$TRAIN_PLAN"; do
  require_file "$path"
done

run_setup_stage() {
  if [[ "$SKIP_SETUP" -eq 1 ]]; then
    echo "Skipping setup stage"
    return
  fi

  echo "Starting setup stage"
  local header split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r split train_pairs train_reactions candidate_ids reaction_features biofp_npz biofp_vocab hardneg_json hardneg_parquet hardneg_report enzyme_ec_labels ec_report; do
      [[ -z "${split:-}" ]] && continue
      local train_pairs_abs train_reactions_abs candidate_ids_abs feature_log biofp_log hardneg_log ec_log
      train_pairs_abs="$(abs_path "$train_pairs")"
      train_reactions_abs="$(abs_path "$train_reactions")"
      candidate_ids_abs="$(abs_path "$candidate_ids")"
      feature_log="$RUN_ROOT/logs/setup/${split}_reaction_features.log"
      biofp_log="$RUN_ROOT/logs/setup/${split}_biofp.log"
      hardneg_log="$RUN_ROOT/logs/setup/${split}_hardneg.log"
      ec_log="$RUN_ROOT/logs/setup/${split}_ec.log"
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
          > "$feature_log" 2>&1
      fi
      status "$split" "setup_reaction_features" "end" "" "0"

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
          > "$biofp_log" 2>&1
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
          > "$hardneg_log" 2>&1
      fi
      status "$split" "setup_hardneg" "end" "" "0"

      status "$split" "setup_ec" "start" "" ""
      if [[ "$FORCE_SETUP" -eq 0 && -s "$enzyme_ec_labels" && -s "$ec_report" ]]; then
        echo "[$split] EC labels already exist"
      else
        "$SETUP_PYTHON_BIN" - "$train_pairs_abs" "$EC_SOURCE" "$enzyme_ec_labels" "$ec_report" > "$ec_log" 2>&1 <<'PY'
import csv
import json
import sys
from pathlib import Path

train_pairs = Path(sys.argv[1])
ec_source = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
out_report = Path(sys.argv[4])

def suffix(value: str) -> str:
    value = str(value)
    return value.split("_", 1)[1] if "_" in value else value

train_proteins = []
seen = set()
with train_pairs.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        protein_id = str(row.get("protein_id", "")).strip()
        if protein_id and protein_id not in seen:
            seen.add(protein_id)
            train_proteins.append(protein_id)

suffix_to_ec = {}
with ec_source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        protein_id = str(row.get("protein_id", "")).strip()
        ecs = [ec.strip() for ec in str(row.get("ec_number", "")).split(";") if ec.strip()]
        if protein_id and ecs:
            suffix_to_ec.setdefault(suffix(protein_id), set()).update(ecs)

rows = []
for protein_id in train_proteins:
    ecs = sorted(suffix_to_ec.get(suffix(protein_id), set()))
    if ecs:
        rows.append({"protein_id": protein_id, "ec_number": ";".join(ecs)})

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["protein_id", "ec_number"])
    writer.writeheader()
    writer.writerows(rows)

report = {
    "source_train_pairs": str(train_pairs),
    "source_ec_labels": str(ec_source),
    "train_proteins": len(train_proteins),
    "train_proteins_with_ec": len(rows),
    "coverage": (len(rows) / len(train_proteins)) if train_proteins else 0.0,
}
out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PY
      fi
      status "$split" "setup_ec" "end" "" "0"
    done
  } < "$SETUP_PLAN"
  echo "Setup stage complete"
}

GPU_FIFO=""
PIDS=()

init_gpu_pool() {
  GPU_FIFO="$RUN_ROOT/tmp/gpu_tokens.$$"
  rm -f "$GPU_FIFO"
  mkfifo "$GPU_FIFO"
  exec 9<>"$GPU_FIFO"
  rm -f "$GPU_FIFO"

  local count=0
  local gpu
  for gpu in $GPUS; do
    if [[ "$count" -lt "$MAX_PARALLEL" ]]; then
      printf '%s\n' "$gpu" >&9
      count=$((count + 1))
    fi
  done
  if [[ "$count" -eq 0 ]]; then
    echo "No GPU tokens were configured" >&2
    exit 1
  fi
}

launch_async() {
  local func="$1"
  shift
  local gpu
  read -r gpu <&9
  (
    set +e
    "$func" "$gpu" "$@"
    local rc=$?
    printf '%s\n' "$gpu" >&9
    exit "$rc"
  ) &
  PIDS+=("$!")
}

wait_for_jobs() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one queued job failed" >&2
    exit 1
  fi
}

run_pretrain_job() {
  local gpu="$1"
  local split="$2"
  local run_id="$3"
  local config="$4"
  local checkpoint_dir="$5"
  local log_path="$6"
  local wandb_run_name="$7"
  mkdir -p "$checkpoint_dir" "$(dirname "$log_path")"
  status "$run_id" "pretrain" "start" "$gpu" ""
  echo "[$(date -Iseconds)] START pretrain $run_id gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/pretrain_enzyme_biofp_split.py \
    --config "$config" \
    > "$log_path" 2>&1
  local rc=$?
  status "$run_id" "pretrain" "end" "$gpu" "$rc"
  echo "[$(date -Iseconds)] END pretrain $run_id rc=$rc log=$log_path"
  return "$rc"
}

run_pretrain_stage() {
  if [[ "$SKIP_PRETRAIN" -eq 1 ]]; then
    echo "Skipping pretrain stage"
    return
  fi
  echo "Starting BioFP pretrain stage"
  local header split run_id config checkpoint_dir log_path wandb_run_name
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r split run_id config checkpoint_dir log_path wandb_run_name; do
      [[ -z "${run_id:-}" ]] && continue
      launch_async run_pretrain_job "$split" "$run_id" "$config" "$checkpoint_dir" "$log_path" "$wandb_run_name"
    done
  } < "$PRETRAIN_PLAN"
  wait_for_jobs
  echo "BioFP pretrain stage complete"
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
  local gpu="$1"
  local run_id="$2"
  local split="$3"
  local variant="$4"
  local label="$5"
  local config="$6"
  local test_config="$7"
  local checkpoint_dir="$8"
  local log_path="$9"
  local eval_json="${10}"
  local wandb_run_name="${11}"
  local requires_pretrain="${12}"
  mkdir -p "$checkpoint_dir" "$(dirname "$log_path")" "$(dirname "$eval_json")"

  if [[ "$requires_pretrain" == "true" ]]; then
    local pretrain_ckpt="$RUN_ROOT/checkpoints/$split/biofp_pretrain/last.ckpt"
    if [[ ! -f "$pretrain_ckpt" ]]; then
      echo "Missing required pretrain checkpoint for $run_id: $pretrain_ckpt" >&2
      return 1
    fi
  fi

  status "$run_id" "train" "start" "$gpu" ""
  echo "[$(date -Iseconds)] START train $run_id gpu=$gpu"
  local train_cmd=(
    "$PYTHON_BIN" scripts/train_protein_pooling.py
    --config "$config"
  )
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
  CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}" > "$log_path" 2>&1
  local train_rc=$?
  status "$run_id" "train" "end" "$gpu" "$train_rc"
  echo "[$(date -Iseconds)] END train $run_id rc=$train_rc log=$log_path"
  if [[ "$train_rc" -ne 0 ]]; then
    return "$train_rc"
  fi

  local checkpoint
  checkpoint="$(select_checkpoint "$checkpoint_dir" "$log_path")"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "No checkpoint found for $run_id in $checkpoint_dir" >&2
    return 1
  fi
  "$PYTHON_BIN" - "$checkpoint" "$eval_json.checkpoint.json" "$run_id" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = sys.argv[1]
out = Path(sys.argv[2])
run_id = sys.argv[3]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"run_id": run_id, "checkpoint": checkpoint}, indent=2) + "\n")
PY

  local eval_log="${eval_json%.json}.stdout.log"
  status "$run_id" "test_eval" "start" "$gpu" ""
  echo "[$(date -Iseconds)] START test_eval $run_id checkpoint=$checkpoint"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" \
    --config "$test_config" \
    --device cuda \
    --batch-size 128 \
    --target-batch-size 512 \
    --output "$eval_json" \
    > "$eval_log" 2>&1
  local eval_rc=$?
  status "$run_id" "test_eval" "end" "$gpu" "$eval_rc"
  echo "[$(date -Iseconds)] END test_eval $run_id rc=$eval_rc eval=$eval_json"
  return "$eval_rc"
}

run_train_stage() {
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    echo "Skipping train/eval stage"
    return
  fi
  echo "Starting train/eval stage"
  local header run_id split variant label config test_config checkpoint_dir log_path eval_json wandb_run_name requires_pretrain
  {
    IFS=$'\t' read -r header
    while IFS=$'\t' read -r run_id split variant label config test_config checkpoint_dir log_path eval_json wandb_run_name requires_pretrain; do
      [[ -z "${run_id:-}" ]] && continue
      launch_async run_train_eval_job "$run_id" "$split" "$variant" "$label" "$config" "$test_config" "$checkpoint_dir" "$log_path" "$eval_json" "$wandb_run_name" "$requires_pretrain"
    done
  } < "$TRAIN_PLAN"
  wait_for_jobs
  echo "Train/eval stage complete"
}

write_summary() {
  "$PYTHON_BIN" - "$TRAIN_PLAN" "$RUN_ROOT/eval/summary.csv" "$RUN_ROOT/eval/summary.md" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])

rows = []
with plan_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        eval_path = Path(row["eval_json"])
        payload = {}
        if eval_path.exists():
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "split": row["split"],
                "variant": row["variant"],
                "label": row["label"],
                "run_id": row["run_id"],
                "mrr": payload.get("mrr", ""),
                "top_1": payload.get("top_1", ""),
                "top_10": payload.get("top_10", ""),
                "top_100": payload.get("top_100", ""),
                "top_1000": payload.get("top_1000", ""),
                "r_precision": payload.get("r_precision", ""),
                "avg_precision": payload.get("avg_precision", ""),
                "num_queries": payload.get("num_queries", ""),
                "num_targets": payload.get("num_targets", ""),
                "eval_json": str(eval_path),
            }
        )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "split",
    "variant",
    "label",
    "run_id",
    "mrr",
    "top_1",
    "top_10",
    "top_100",
    "top_1000",
    "r_precision",
    "avg_precision",
    "num_queries",
    "num_targets",
    "eval_json",
]
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value == "" or value is None:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.4f}"

lines = ["# ReactZyme No-Text Ablation Test Summary", ""]
for split in sorted({row["split"] for row in rows}):
    lines.extend(
        [
            f"## {split}",
            "",
            "| Variant | MRR | Top1 | Top10 | Top100 | Top1000 | AP | Queries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [item for item in rows if item["split"] == split]:
        lines.append(
            "| {variant} | {mrr} | {top_1} | {top_10} | {top_100} | {top_1000} | {avg_precision} | {num_queries} |".format(
                variant=row["variant"],
                mrr=fmt(row["mrr"]),
                top_1=fmt(row["top_1"]),
                top_10=fmt(row["top_10"]),
                top_100=fmt(row["top_100"]),
                top_1000=fmt(row["top_1000"]),
                avg_precision=fmt(row["avg_precision"]),
                num_queries=row["num_queries"],
            )
        )
    lines.append("")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {summary_csv}")
print(f"Wrote {summary_md}")
PY
}

echo "Run root: $RUN_ROOT"
echo "W&B: ${WANDB_ENTITY:-}/$WANDB_PROJECT mode=$WANDB_MODE"
echo "GPUs: $GPUS max_parallel=$MAX_PARALLEL"
echo "Setup Python: $SETUP_PYTHON_BIN"
echo "Training Python: $PYTHON_BIN"
echo "TMPDIR: $TMPDIR"
status "controller" "controller" "start" "" ""

run_setup_stage
init_gpu_pool
run_pretrain_stage
run_train_stage
write_summary

status "controller" "controller" "end" "" "0"
echo "All requested stages complete"
echo "Summary: $RUN_ROOT/eval/summary.md"
