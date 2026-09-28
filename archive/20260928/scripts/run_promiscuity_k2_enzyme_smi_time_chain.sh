#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
DEVICES="${DEVICES:-3}"
CACHE_PRECISION="${CACHE_PRECISION:-32}"
CACHE_ENZYME_BATCH_SIZE="${CACHE_ENZYME_BATCH_SIZE:-96}"
CACHE_REACTION_BATCH_SIZE="${CACHE_REACTION_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_TARGET_BATCH_SIZE="${EVAL_TARGET_BATCH_SIZE:-1024}"

cd "$PROJECT_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

milestone() {
  printf '\n[%s] %s\n' "$(timestamp)" "$1"
}

parent_checkpoint() {
  case "$1" in
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

best_checkpoint_from_last() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import sys
import torch

last_path = Path(sys.argv[1]).resolve()
checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
candidates = []
for state in checkpoint.get("callbacks", {}).values():
    if not isinstance(state, dict):
        continue
    candidate = state.get("best_model_path")
    if candidate:
        candidates.append(Path(candidate).resolve())
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
  local run_root="runs/promiscuity_k2_${split}"
  local parent
  parent="$(parent_checkpoint "$split")"

  milestone "Preparing ${split}"
  if [[ ! -f "$parent" ]]; then
    printf 'Parent checkpoint not found: %s\n' "$parent" >&2
    return 1
  fi

  "$PYTHON_BIN" scripts/materialize_promiscuity_k2_split_configs.py \
    --split "$split" \
    --parent-checkpoint "$parent" \
    --run-root "$run_root"

  "$PYTHON_BIN" scripts/build_promiscuity_reaction_smi_pretrain_split.py \
    --validation-pairs "data/revised_protocols/reactzyme_paper/${split}/validation_pairs.csv" \
    --heldout-test-reactions "data/revised_protocols/reactzyme_official/${split}/test_rxns.csv" \
    --output-dir "${run_root}/data/source_pretrain"

  "$PYTHON_BIN" scripts/build_annotation_negative_pools.py \
    --train-pairs "data/revised_protocols/reactzyme_paper/${split}/train_pairs.csv" \
    --ec-labels data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv \
    --biofp-targets runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/biology/enzyme_biofp_soft_targets.npz \
    --output "${run_root}/data/reactzyme_negative_pools.json" \
    --report "${run_root}/data/reactzyme_negative_pools_report.json" \
    --max-biological 32 \
    --max-random 32 \
    --ec-prefix-depth 2 \
    --biofp-threshold 0.5 \
    --min-biofp-similarity 0.5 \
    --seed 42
}

train_split() {
  local split="$1"
  local run_root="runs/promiscuity_k2_${split}"
  local parent source_config finetune_config
  parent="$(parent_checkpoint "$split")"
  source_config="${run_root}/configs/source_pretrain.yaml"
  finetune_config="${run_root}/configs/finetune.yaml"

  milestone "Caching frozen source-collapse towers for ${split}"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$DEVICES" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$source_config" \
    --checkpoint "$parent" \
    --output-dir "${run_root}/cache/source" \
    --enzyme-batch-size "$CACHE_ENZYME_BATCH_SIZE" \
    --reaction-batch-size "$CACHE_REACTION_BATCH_SIZE" \
    --precision "$CACHE_PRECISION"

  milestone "Caching frozen ReactZyme towers for ${split}"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$DEVICES" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$finetune_config" \
    --checkpoint "$parent" \
    --output-dir "${run_root}/cache/reactzyme" \
    --enzyme-batch-size "$CACHE_ENZYME_BATCH_SIZE" \
    --reaction-batch-size "$CACHE_REACTION_BATCH_SIZE" \
    --precision "$CACHE_PRECISION"

  milestone "Source-collapse K=2 prototype pretraining for ${split}"
  "$PYTHON_BIN" -u scripts/train_protein_pooling.py \
    --config "$source_config" \
    --training.init_from_checkpoint "$parent" \
    --data.cached_enzyme_base_embeds_path "${run_root}/cache/source/enzyme_base.h5" \
    --data.cached_train_reaction_base_embeds_path "${run_root}/cache/source/train_reaction_base.h5" \
    --training.devices "$DEVICES"

  local source_checkpoint="${run_root}/checkpoints/source_pretrain/last.ckpt"
  if [[ ! -f "$source_checkpoint" ]]; then
    printf 'Source prototype checkpoint not found: %s\n' "$source_checkpoint" >&2
    return 1
  fi

  milestone "ReactZyme K=2 prototype finetuning for ${split}"
  "$PYTHON_BIN" -u scripts/train_protein_pooling.py \
    --config "$finetune_config" \
    --training.init_from_checkpoint "$source_checkpoint" \
    --data.cached_enzyme_base_embeds_path "${run_root}/cache/reactzyme/enzyme_base.h5" \
    --data.cached_train_reaction_base_embeds_path "${run_root}/cache/reactzyme/train_reaction_base.h5" \
    --data.cached_validation_reaction_base_embeds_path "${run_root}/cache/reactzyme/validation_reaction_base.h5" \
    --training.devices "$DEVICES"
}

test_split() {
  local split="$1"
  local run_root="runs/promiscuity_k2_${split}"
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

milestone "Starting enzyme-SMILES and time K=2 chain on ${DEVICES} GPUs"
prepare_split enzyme_smi
prepare_split time
train_split enzyme_smi
train_split time
test_split enzyme_smi
test_split time
milestone "All training and tests completed"
