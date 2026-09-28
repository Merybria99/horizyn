#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/runs/reactzyme_tiger_top10_hitk_v1"
RUN_ROOT="$DEFAULT_RUN_ROOT"
if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_ROOT="$1"
  shift
fi
if [[ "$RUN_ROOT" != /* ]]; then RUN_ROOT="$ROOT/$RUN_ROOT"; fi

DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$RUN_ROOT/logs/controller"
if [[ "$DETACH" -eq 1 ]]; then
  pid_file="$RUN_ROOT/logs/controller/pid"
  rm -f "$pid_file"
  setsid -f bash -c '
    pid_file="$1"
    shift
    printf "%s\n" "$$" > "$pid_file"
    exec bash "$@"
  ' bash "$pid_file" "$0" "$RUN_ROOT" \
    > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  for _ in {1..50}; do [[ -s "$pid_file" ]] && break; sleep 0.1; done
  echo "Launched TIGER Hit@k evaluation controller PID $(cat "$pid_file")"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  exit 0
fi

HOST="$(hostname -s)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../.machine-envs/$HOST/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$ROOT/../env/bin/python"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "No usable workspace Python interpreter found" >&2
  exit 1
fi

GPU="${HITK_GPU:-0}"
MAX_PARALLEL="${HITK_MAX_PARALLEL:-2}"
if (( MAX_PARALLEL < 1 )); then
  echo "HITK_MAX_PARALLEL must be positive" >&2
  exit 2
fi

mkdir -p \
  "$RUN_ROOT/eval" "$RUN_ROOT/logs" "$RUN_ROOT/nohome/$HOST" \
  "$RUN_ROOT/tmp/$HOST" "$RUN_ROOT/cache/$HOST/huggingface" \
  "$RUN_ROOT/cache/$HOST/torch" "$RUN_ROOT/cache/$HOST/xdg" \
  "$RUN_ROOT/cache/$HOST/matplotlib"
export HOME="$RUN_ROOT/nohome/$HOST"
export TMPDIR="$RUN_ROOT/tmp/$HOST"
export HF_HOME="$RUN_ROOT/cache/$HOST/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/$HOST/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/$HOST/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/$HOST/matplotlib"
export PYTHONPATH="$ROOT"
export UNIMOL_WEIGHT_DIR="$ROOT/../unimol_weights"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=(
  Q5_combined
  Q4_factorized_modalities
  Q1_e2r_adapter
  Q0_f3
  Q3_dense_transform
  F3_set_chemistry
  Q2_e2r_hardneg
  F0_directional_delta_control
  F1_molecule_set
  F6_rhea_center
)
SPLITS=(time enzyme_smi reaction_smi)
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"

source_result() {
  local model="$1" split="$2"
  if [[ "$model" == Q* ]]; then
    printf '%s\n' "$ROOT/runs/reactzyme_e2r_pareto_v1/eval/seed42/$split/$model/test.json"
  else
    printf '%s\n' "$ROOT/runs/reactzyme_reaction_features_v1/eval/$split/$model/test_both.json"
  fi
}

is_complete() {
  local output="$1"
  [[ -s "$output" ]] && "$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1]))
required = {
    f"{direction}/top_{cutoff}"
    for direction in ("enzyme_to_reaction", "reaction_to_enzyme")
    for cutoff in (1, 2, 3, 4, 5, 10, 20)
}
raise SystemExit(0 if required <= d.keys() else 1)
' "$output"
}

record_status() {
  local model="$1" split="$2" event="$3" rc="${4:-}"
  printf '{"time":"%s","host":"%s","model":"%s","split":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$HOST" "$model" "$split" "$event" "$rc" >> "$STATUS_LOG"
}

evaluate_one() {
  local model="$1" split="$2" source_json output log checkpoint config metadata rc
  source_json="$(source_result "$model" "$split")"
  output="$RUN_ROOT/eval/$model/$split.json"
  log="$RUN_ROOT/logs/$model/$split.log"
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  if is_complete "$output"; then
    record_status "$model" "$split" skip 0
    return 0
  fi
  if [[ ! -s "$source_json" ]]; then
    echo "Missing source result: $source_json" > "$log"
    record_status "$model" "$split" end 1
    return 1
  fi
  metadata="$($PYTHON_BIN -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["checkpoint"])
print(d["config"])
' "$source_json")"
  checkpoint="$(sed -n '1p' <<< "$metadata")"
  config="$(sed -n '2p' <<< "$metadata")"
  if [[ ! -s "$checkpoint" || ! -s "$config" ]]; then
    echo "Missing checkpoint or config: $checkpoint | $config" > "$log"
    record_status "$model" "$split" end 1
    return 1
  fi

  record_status "$model" "$split" start
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --device cuda:0 \
    --direction both \
    --evaluation-protocol paper_test_candidates \
    --batch-size 512 \
    --target-batch-size 512 \
    --output "$output" \
    > "$log" 2>&1
  rc=$?
  set -e
  record_status "$model" "$split" end "$rc"
  return "$rc"
}

cd "$ROOT"
record_status all all campaign_start
pids=()
labels=()
failed=0
for model in "${MODELS[@]}"; do
  for split in "${SPLITS[@]}"; do
    evaluate_one "$model" "$split" &
    pids+=("$!")
    labels+=("$model/$split")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      if ! wait "${pids[0]}"; then
        echo "Evaluation failed: ${labels[0]}" >&2
        failed=1
      fi
      pids=("${pids[@]:1}")
      labels=("${labels[@]:1}")
    fi
  done
done
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Evaluation failed: ${labels[$index]}" >&2
    failed=1
  fi
done

if (( failed )); then
  record_status all all campaign_end 1
  exit 1
fi

"$PYTHON_BIN" scripts/report_reactzyme_tiger_top10_hitk.py \
  --run-root "$RUN_ROOT" \
  --output "$ROOT/../documents/README_reactzyme_tiger_top10.md" \
  > "$RUN_ROOT/logs/controller/report.log" 2>&1
record_status all all campaign_end 0
