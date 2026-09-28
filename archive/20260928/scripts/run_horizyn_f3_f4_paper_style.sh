#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/horizyn_f3_f4_paper_v1"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
REACTION_T5_MODEL_PATH="${REACTION_T5_MODEL_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc}"
COFACTOR_DICTIONARY="${COFACTOR_DICTIONARY:-$ROOT/data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv}"
GPUS="${GPUS:-0,1,2,3}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-f3-f4-paper-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-omnai}"
WANDB_MODE="${WANDB_MODE:-online}"
MASTER_PORT_F3="${MASTER_PORT_F3:-27930}"
MASTER_PORT_F4="${MASTER_PORT_F4:-27940}"
SHORT_TMPDIR="${HORIZYN_PAPER_TMPDIR:-/tmp/hz_f3_f4_paper_${UID}}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
WAIT_FOR_M_CAMPAIGN="${WAIT_FOR_M_CAMPAIGN:-1}"
M_CAMPAIGN_ROOT="${M_CAMPAIGN_ROOT:-$ROOT/runs/reactzyme_f3_geometry_v1}"
M_CAMPAIGN_PHASE="${M_CAMPAIGN_PHASE:-screen}"
M_CAMPAIGN_POLL_SECONDS="${M_CAMPAIGN_POLL_SECONDS:-60}"

mkdir -p \
  "$RUN_ROOT"/{checkpoints/F3,checkpoints/F4,logs/F3,logs/F4,features/test_bidirectional,features/reaction_set,protocol} \
  "$RUN_ROOT"/{cache/huggingface,cache/torch,xdg_cache,xdg_config,xdg_state,wandb,matplotlib,tmp} \
  "$SHORT_TMPDIR"

exec 9>"$RUN_ROOT/runner.lock"
if ! flock -n 9; then
  echo "Another Horizyn F3/F4 paper-style runner already holds $RUN_ROOT/runner.lock" >&2
  exit 1
fi

export TMPDIR="$SHORT_TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UNIMOL_WEIGHT_DIR="$ROOT/../unimol_weights"
export PYTHONPATH="$ROOT"

if [[ -z "${WANDB_API_KEY:-}" && -s "$ROOT/runs/reactzyme_reaction_features_v1/wandb/api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "$ROOT/runs/reactzyme_reaction_features_v1/wandb/api_key")"
fi

status() {
  printf '{"time":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "${3:-}" >> "$RUN_ROOT/logs/status.jsonl"
}

run_logged() {
  local stage="$1"
  local log="$2"
  shift 2
  status "$stage" start
  set +e
  "$@" > "$log" 2>&1
  local rc=$?
  set -e
  status "$stage" end "$rc"
  return "$rc"
}

wait_for_idle_gpus() {
  if [[ "$WAIT_FOR_IDLE_GPUS" != 1 ]]; then
    return
  fi
  status gpu_queue start
  while [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null || true)" =~ [0-9] ]]; do
    sleep "$GPU_POLL_SECONDS"
  done
  status gpu_queue end 0
}

m_campaign_succeeded() {
  "$PYTHON_BIN" - "$M_CAMPAIGN_ROOT/logs/status.jsonl" \
    "$M_CAMPAIGN_PHASE" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
phase = sys.argv[2]
if not status_path.is_file():
    raise SystemExit(1)
matches = []
for raw_line in status_path.read_text(encoding="utf-8").splitlines():
    if not raw_line.strip():
        continue
    event = json.loads(raw_line)
    if (
        event.get("run_id") == "controller"
        and event.get("stage") == phase
        and event.get("event") == "end"
    ):
        matches.append(event)
if not matches or str(matches[-1].get("returncode")) != "0":
    raise SystemExit(1)
PY
}

wait_for_m_campaign() {
  if [[ "$WAIT_FOR_M_CAMPAIGN" != 1 ]]; then
    return
  fi
  local pid_file="$M_CAMPAIGN_ROOT/logs/controller/pid"
  if [[ ! -s "$pid_file" ]]; then
    echo "M campaign controller PID file is missing: $pid_file" >&2
    return 1
  fi
  local campaign_pid
  campaign_pid="$(tr -d '\r\n' < "$pid_file")"
  if [[ ! "$campaign_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid M campaign controller PID: $campaign_pid" >&2
    return 1
  fi
  status m_campaign_dependency start
  while kill -0 "$campaign_pid" 2>/dev/null; do
    if m_campaign_succeeded; then
      break
    fi
    sleep "$M_CAMPAIGN_POLL_SECONDS"
  done
  if ! m_campaign_succeeded; then
    echo "M campaign phase '$M_CAMPAIGN_PHASE' did not complete successfully" >&2
    status m_campaign_dependency end 1
    return 1
  fi
  if [[ ! -s "$M_CAMPAIGN_ROOT/reports/screen_selection.stdout.json" ]]; then
    echo "M campaign screen-selection report is missing" >&2
    status m_campaign_dependency end 1
    return 1
  fi
  status m_campaign_dependency end 0
}

cd "$ROOT"

if [[ ! -s "$RUN_ROOT/protocol/published_candidate_ids.txt" ]]; then
  cp runs/horizyn_f4_in_domain_v1/protocol/test_candidate_ids.txt \
    "$RUN_ROOT/protocol/published_candidate_ids.txt"
fi

if [[ ! -s "$RUN_ROOT/features/reaction_set/train_reaction_set_features.npz" ]]; then
  run_logged reaction_set "$RUN_ROOT/logs/reaction_set.log" \
    "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
      --train-reactions data/sota/train_rxns.csv \
      --validation-reactions data/sota/test_rxns.csv \
      --test-reactions data/sota/test_rxns.csv \
      --cofactor-dictionary "$COFACTOR_DICTIONARY" \
      --out-dir "$RUN_ROOT/features/reaction_set"
fi

wait_for_m_campaign
wait_for_idle_gpus

feature_pids=()
feature_names=()
if [[ ! -s "$RUN_ROOT/features/test_bidirectional/reactiont5v2.h5" ]]; then
  run_logged reactiont5_test "$RUN_ROOT/logs/reactiont5_test.log" \
    env CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test_bidirectional/reactiont5v2.h5" \
      --model-name "$REACTION_T5_MODEL_PATH" \
      --batch-size 64 --max-length 512 --pooling mean \
      --device cuda --dtype float16 --bidirectional \
      --no-allow-pseudo-reactions --force &
  feature_pids+=("$!")
  feature_names+=(reactiont5_test)
fi
if [[ ! -s "$RUN_ROOT/features/test_bidirectional/unimol2.h5" ]]; then
  run_logged unimol2_test "$RUN_ROOT/logs/unimol2_test.log" \
    env PYTHONPATH="$ROOT/.deps/unimol_tools:$ROOT/../env/unimol2_site:$ROOT" \
      CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test_bidirectional/unimol2.h5" \
      --batch-size 64 --dtype float16 --compression none --bidirectional \
      --no-allow-pseudo-reactions --skip-invalid-molecules \
      --skip-invalid-reactions --force &
  feature_pids+=("$!")
  feature_names+=(unimol2_test)
fi
if [[ ! -s "$RUN_ROOT/features/test_bidirectional/chiro.h5" ]]; then
  run_logged chiro_test "$RUN_ROOT/logs/chiro_test.log" \
    env PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
      CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test_bidirectional/chiro.h5" \
      --device cuda --num-workers 4 --batch-size 64 --bidirectional \
      --no-allow-pseudo-reactions --force &
  feature_pids+=("$!")
  feature_names+=(chiro_test)
fi

for index in "${!feature_pids[@]}"; do
  if ! wait "${feature_pids[$index]}"; then
    echo "Feature extraction failed: ${feature_names[$index]}" >&2
    exit 1
  fi
done

for variant in F3 F4; do
  config="$ROOT/configs/horizyn_${variant,,}_paper_style.yaml"
  "$PYTHON_BIN" -c \
    "from horizyn.config import load_config; load_config('$config'); print('Validated: $config')"

  if [[ -s "$RUN_ROOT/checkpoints/$variant/last.ckpt" ]]; then
    resume_args=(--resume "$RUN_ROOT/checkpoints/$variant/last.ckpt")
  else
    resume_args=()
  fi
  if [[ "$WANDB_MODE" == disabled ]]; then
    wandb_args=()
  else
    wandb_args=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-run-name "horizyn-paper-${variant}-seed42"
      --wandb-mode "$WANDB_MODE"
    )
  fi
  if [[ "$variant" == F3 ]]; then
    master_port="$MASTER_PORT_F3"
  else
    master_port="$MASTER_PORT_F4"
  fi
  run_logged "train_$variant" "$RUN_ROOT/logs/$variant/train.log" \
    env CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$master_port" \
      "$PYTHON_BIN" scripts/train_protein_pooling.py \
      --config "$config" "${resume_args[@]}" "${wandb_args[@]}"
  if [[ ! -s "$RUN_ROOT/checkpoints/$variant/last.ckpt" ]]; then
    echo "$variant training ended without a final checkpoint" >&2
    exit 1
  fi
done

echo "Horizyn F3/F4 paper-style training complete"
echo "F3 final checkpoint: $RUN_ROOT/checkpoints/F3/last.ckpt"
echo "F4 final checkpoint: $RUN_ROOT/checkpoints/F4/last.ckpt"
