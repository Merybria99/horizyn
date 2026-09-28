#!/usr/bin/env bash
set -u

ROOT="/datastor2/deep-proteins/EnzymeDiscovery/horizyn"
RUN_ENV="/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python"
SESSION="enzyme_capability_factorized_rhea_4gpu"
LOG="$ROOT/logs/enzyme_capability_factorized_rhea_4gpu.log"
WATCHDOG_LOG="$ROOT/logs/enzyme_capability_factorized_rhea_4gpu.watchdog.log"
CHECKPOINT="$ROOT/outputs/enzyme_capability_pretrain/train_exact_rhea_factorized_biological/last.ckpt"
CONFIG="configs/enzyme_capability_pretrain.yaml"
DONE_EPOCH="${DONE_EPOCH:-29}"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-300}"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs" "/datastor2/deep-proteins/EnzymeDiscovery/.cache/matplotlib-capability-run"

checkpoint_complete() {
  [[ -f "$CHECKPOINT" ]] || return 1
  "$RUN_ENV" - "$CHECKPOINT" "$DONE_EPOCH" <<'PY'
import sys
import torch

checkpoint_path, done_epoch = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(checkpoint_path, map_location="cpu")
sys.exit(0 if int(checkpoint.get("epoch", -1)) >= done_epoch else 1)
PY
}

while true; do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf '%s %s alive\n' "$(date -Is)" "$SESSION" >> "$WATCHDOG_LOG"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  if checkpoint_complete; then
    printf '%s last checkpoint is at or beyond epoch %s; watchdog exiting\n' \
      "$(date -Is)" "$DONE_EPOCH" >> "$WATCHDOG_LOG"
    exit 0
  fi

  printf '%s %s missing before completion; relaunching\n' \
    "$(date -Is)" "$SESSION" >> "$WATCHDOG_LOG"
  CUDA_VISIBLE_DEVICES=0,1,2,3 tmux new -d -s "$SESSION" \
    "cd $ROOT && \
     export MPLCONFIGDIR=/datastor2/deep-proteins/EnzymeDiscovery/.cache/matplotlib-capability-run && \
     export WANDB_START_METHOD=thread && \
     export PYTHONPATH=. && \
     $RUN_ENV scripts/train_enzyme_capability_encoder.py --config $CONFIG 2>&1 | tee -a $LOG"
  sleep "$INTERVAL_SECONDS"
done
