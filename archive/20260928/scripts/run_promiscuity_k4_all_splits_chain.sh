#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
DEVICES="${DEVICES:-3}"
PROTOTYPE_COUNT="${PROTOTYPE_COUNT:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_TARGET_BATCH_SIZE="${EVAL_TARGET_BATCH_SIZE:-1024}"

if [[ ! "$PROTOTYPE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PROTOTYPE_COUNT must be a positive integer, got %s\n' "$PROTOTYPE_COUNT" >&2
  exit 1
fi

PROTOTYPE_TAG="k${PROTOTYPE_COUNT}"
cd "$PROJECT_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

milestone() {
  printf '\n[%s] %s\n' "$(timestamp)" "$1"
}

parent_checkpoint() {
  case "$1" in
    reaction_smi)
      printf '%s\n' "runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt"
      ;;
    enzyme_smi)
      printf '%s\n' "runs/reactzyme_e2r_pareto_v1/checkpoints/seed42/enzyme_smi/Q0_f3/selected.ckpt"
      ;;
    time)
      printf '%s\n' "runs/reactzyme_e2r_pareto_v1/checkpoints/seed42/time/Q0_f3/selected.ckpt"
      ;;
    *)
      printf 'Unsupported split: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

link_reusable_artifacts() {
  local split="$1"
  local run_root="runs/promiscuity_${PROTOTYPE_TAG}_${split}"
  local reuse_root="runs/promiscuity_k2_${split}"
  local artifact source target resolved_source resolved_target
  mkdir -p "$run_root"
  for artifact in data cache; do
    source="${reuse_root}/${artifact}"
    target="${run_root}/${artifact}"
    if [[ ! -d "$source" ]]; then
      printf 'Reusable K=2 artifact directory not found: %s\n' "$source" >&2
      return 1
    fi
    resolved_source="$(realpath "$source")"
    if [[ -L "$target" ]]; then
      resolved_target="$(realpath "$target")"
      if [[ "$resolved_target" != "$resolved_source" ]]; then
        printf 'Existing link %s resolves to %s, expected %s\n' \
          "$target" "$resolved_target" "$resolved_source" >&2
        return 1
      fi
    elif [[ -e "$target" ]]; then
      printf 'Refusing to replace existing non-link path: %s\n' "$target" >&2
      return 1
    else
      ln -s "$resolved_source" "$target"
    fi
  done
}

best_checkpoint_from_last() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import sys
import torch

last_path = Path(sys.argv[1]).resolve()
checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
candidates = []
for state in checkpoint.get("callbacks", {}).values():
    if isinstance(state, dict) and state.get("best_model_path"):
        candidates.append(Path(state["best_model_path"]).resolve())
existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
if len(existing) != 1:
    raise RuntimeError(
        f"Expected exactly one validation-selected checkpoint in {last_path}; "
        f"found {existing}"
    )
print(existing[0])
PY
}

prepare_split() {
  local split="$1"
  local run_root="runs/promiscuity_${PROTOTYPE_TAG}_${split}"
  local parent
  parent="$(parent_checkpoint "$split")"
  milestone "Preparing ${split} K=${PROTOTYPE_COUNT}"
  if [[ ! -f "$parent" ]]; then
    printf 'Parent checkpoint not found: %s\n' "$parent" >&2
    return 1
  fi
  link_reusable_artifacts "$split"
  "$PYTHON_BIN" scripts/materialize_promiscuity_k2_split_configs.py \
    --split "$split" \
    --prototype-count "$PROTOTYPE_COUNT" \
    --parent-checkpoint "$parent" \
    --run-root "$run_root"
}

train_split() {
  local split="$1"
  local run_root="runs/promiscuity_${PROTOTYPE_TAG}_${split}"
  local parent source_config finetune_config source_checkpoint
  parent="$(parent_checkpoint "$split")"
  source_config="${run_root}/configs/source_pretrain.yaml"
  finetune_config="${run_root}/configs/finetune.yaml"

  milestone "Source-collapse K=${PROTOTYPE_COUNT} prototype pretraining for ${split}"
  "$PYTHON_BIN" -u scripts/train_protein_pooling.py \
    --config "$source_config" \
    --training.init_from_checkpoint "$parent" \
    --training.devices "$DEVICES"

  source_checkpoint="${run_root}/checkpoints/source_pretrain/last.ckpt"
  if [[ ! -f "$source_checkpoint" ]]; then
    printf 'Source prototype checkpoint not found: %s\n' "$source_checkpoint" >&2
    return 1
  fi

  milestone "ReactZyme K=${PROTOTYPE_COUNT} prototype finetuning for ${split}"
  "$PYTHON_BIN" -u scripts/train_protein_pooling.py \
    --config "$finetune_config" \
    --training.init_from_checkpoint "$source_checkpoint" \
    --training.devices "$DEVICES"
}

test_split() {
  local split="$1"
  local run_root="runs/promiscuity_${PROTOTYPE_TAG}_${split}"
  local last_checkpoint="${run_root}/checkpoints/reactzyme_finetune/last.ckpt"
  local best_checkpoint test_config evaluation_dir
  if [[ ! -f "$last_checkpoint" ]]; then
    printf 'Finetuning checkpoint not found: %s\n' "$last_checkpoint" >&2
    return 1
  fi
  best_checkpoint="$(best_checkpoint_from_last "$last_checkpoint")"
  test_config="${run_root}/configs/test.yaml"
  evaluation_dir="${run_root}/evaluation/best_validation"
  mkdir -p "$evaluation_dir"

  milestone "Paper-protocol bidirectional test for ${split} using ${best_checkpoint}"
  "$PYTHON_BIN" -u scripts/evaluate_protein_pooling.py \
    --checkpoint "$best_checkpoint" \
    --config "$test_config" \
    --direction both \
    --evaluation-protocol paper_test_candidates \
    --batch-size "$EVAL_BATCH_SIZE" \
    --target-batch-size "$EVAL_TARGET_BATCH_SIZE" \
    --target-embeds-cache "${evaluation_dir}/target_embeddings.pt" \
    --output "${evaluation_dir}/test_both.json"
}

milestone "Starting all-split K=${PROTOTYPE_COUNT} chain on ${DEVICES} GPUs"
for split in reaction_smi enzyme_smi time; do
  prepare_split "$split"
done
for split in reaction_smi enzyme_smi time; do
  train_split "$split"
done
for split in reaction_smi enzyme_smi time; do
  test_split "$split"
done
milestone "All K=${PROTOTYPE_COUNT} training and tests completed"
