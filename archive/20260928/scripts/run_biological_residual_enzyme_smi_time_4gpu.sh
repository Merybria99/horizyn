#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
GPU_COUNT="${GPU_COUNT:-4}"
SLEEC_CHECKPOINT="${SLEEC_CHECKPOINT:-checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt}"
SOURCE_RESIDUES="outputs/biofp_from_scratch_chains/biofp-fresh-chain-nohome-20260709_172734/data/standardized/retrieval_training_source_collapse/train_exact/fit_proteins_with_clipzyme_eval_prott5_residue.h5"
REACTZYME_RESIDUES="data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5"
RAW_SOURCE_PAIRS="runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/split/train_pairs.csv"
SHARED_SOURCE_TOKENS="${SHARED_SOURCE_TOKENS:-runs/biological_residual_shared/cache/source_full/protein_functional_tokens.h5}"
SHARED_REACTZYME_TOKENS="${SHARED_REACTZYME_TOKENS:-runs/biological_residual_reaction_smi/cache/reactzyme/protein_functional_tokens.h5}"
SPLIT_LIST="${SPLITS:-enzyme_smi time}"
read -r -a SPLITS_TO_RUN <<<"$SPLIT_LIST"

for split in "${SPLITS_TO_RUN[@]}"; do
  if [[ "$split" != "enzyme_smi" && "$split" != "time" ]]; then
    echo "Unsupported split in SPLITS: $split" >&2
    exit 2
  fi
done

run_training_stage() {
  local config_path="$1"
  local checkpoint_path="$2"
  local expected_epochs
  expected_epochs=$("$PYTHON_BIN" -c \
    'import sys; from horizyn.config import load_config; print(int(load_config(sys.argv[1]).training.max_epochs))' \
    "$config_path")
  if [[ ! -s "$checkpoint_path" ]]; then
    "$PYTHON_BIN" scripts/train_protein_pooling.py \
      --config "$config_path" \
      --training.devices="$GPU_COUNT"
    return
  fi

  local saved_epoch
  saved_epoch=$("$PYTHON_BIN" -c \
    'import sys,torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("epoch", -1)))' \
    "$checkpoint_path")
  if (( saved_epoch + 1 >= expected_epochs )); then
    echo "Reusing completed checkpoint: $checkpoint_path (epoch=$saved_epoch)"
  else
    echo "Resuming incomplete checkpoint: $checkpoint_path (epoch=$saved_epoch)"
    "$PYTHON_BIN" scripts/train_protein_pooling.py \
      --config "$config_path" \
      --training.devices="$GPU_COUNT" \
      --resume "$checkpoint_path"
  fi
}

validate_functional_coverage() {
  local token_path="$1"
  shift
  "$PYTHON_BIN" - "$token_path" "$@" <<'PY'
import csv
import sys
from pathlib import Path

import h5py

token_path = Path(sys.argv[1])
if not token_path.is_file() or token_path.stat().st_size == 0:
    raise FileNotFoundError(f"Functional-token cache is missing: {token_path}")
with h5py.File(token_path, "r") as handle:
    available = {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in handle["ids"][:]
    }
requested = set()
for pair_path in map(Path, sys.argv[2:]):
    with pair_path.open(newline="", encoding="utf-8") as stream:
        requested.update(str(row["protein_id"]).strip() for row in csv.DictReader(stream))
missing = sorted(requested - available)
if missing:
    raise ValueError(
        f"{token_path} misses {len(missing)} requested proteins; examples: {missing[:5]}"
    )
print(f"Validated functional-token coverage: {len(requested)}/{len(requested)} at {token_path}")
PY
}

mkdir -p "$(dirname "$SHARED_SOURCE_TOKENS")" "$(dirname "$SHARED_REACTZYME_TOKENS")"

# Functional residue extraction is independent of the ReactZyme split. Build
# one superset cache for all source proteins and reuse the already-complete
# ReactZyme cache, whose 178,327 proteins are shared by all three paper splits.
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_sleec_functional_tokens.py \
  --residue-h5 "$SOURCE_RESIDUES" \
  --sleec-checkpoint "$SLEEC_CHECKPOINT" \
  --pairs "$RAW_SOURCE_PAIRS" \
  --output "$SHARED_SOURCE_TOKENS" \
  --top-k 48 \
  --context-k 16 \
  --bf16

if [[ ! -s "$SHARED_REACTZYME_TOKENS" ]]; then
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    scripts/cache_sleec_functional_tokens.py \
    --residue-h5 "$REACTZYME_RESIDUES" \
    --sleec-checkpoint "$SLEEC_CHECKPOINT" \
    --pairs data/revised_protocols/reactzyme_paper/enzyme_smi/train_pairs.csv \
    --pairs data/revised_protocols/reactzyme_paper/enzyme_smi/validation_pairs.csv \
    --pairs data/revised_protocols/reactzyme_paper/enzyme_smi/test_pairs.csv \
    --output "$SHARED_REACTZYME_TOKENS" \
    --top-k 48 \
    --context-k 16 \
    --bf16
fi

for split in "${SPLITS_TO_RUN[@]}"; do
  run_root="runs/biological_residual_${split}"
  source_config="$run_root/configs/source_pretrain.yaml"
  finetune_config="$run_root/configs/reactzyme_finetune.yaml"
  f3_checkpoint="runs/reactzyme_reaction_features_v1/checkpoints/$split/F3_set_chemistry/protein-pooling-epoch=29.ckpt"
  f3_test_config="runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/$split/test.yaml"
  source_cache="$run_root/cache/source"
  reactzyme_cache="$run_root/cache/reactzyme"
  validation_cache="$run_root/cache/reactzyme_validation_eval"
  test_cache="$run_root/cache/reactzyme_test"
  source_checkpoint="$run_root/checkpoints/source_pretrain/last.ckpt"
  final_checkpoint="$run_root/checkpoints/reactzyme_finetune/last.ckpt"
  validation_results="$run_root/results/validation_alpha_sweep.json"
  test_results="$run_root/results/test_selected_alpha.json"

  echo "Starting biological-residual pipeline for $split"
  mkdir -p "$run_root/logs" "$run_root/results"
  "$PYTHON_BIN" scripts/materialize_biological_residual_split_configs.py \
    --split "$split" \
    --shared-source-tokens "$SHARED_SOURCE_TOKENS" \
    --shared-reactzyme-tokens "$SHARED_REACTZYME_TOKENS"

  "$PYTHON_BIN" scripts/build_biological_residual_pretrain_split.py \
    --heldout-pairs "data/revised_protocols/reactzyme_paper/$split/validation_pairs.csv" \
    --heldout-pairs "data/revised_protocols/reactzyme_paper/$split/test_pairs.csv" \
    --heldout-reactions "data/revised_protocols/reactzyme_paper/$split/validation_rxns.csv" \
    --heldout-reactions "data/revised_protocols/reactzyme_paper/$split/test_rxns.csv" \
    --output-dir "$run_root/data/source_pretrain"

  validate_functional_coverage \
    "$SHARED_SOURCE_TOKENS" \
    "$run_root/data/source_pretrain/train_pairs.csv"
  validate_functional_coverage \
    "$SHARED_REACTZYME_TOKENS" \
    "data/revised_protocols/reactzyme_paper/$split/train_pairs.csv" \
    "data/revised_protocols/reactzyme_paper/$split/validation_pairs.csv" \
    "data/revised_protocols/reactzyme_paper/$split/test_pairs.csv"

  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$source_config" \
    --checkpoint "$f3_checkpoint" \
    --output-dir "$source_cache" \
    --precision bf16

  run_training_stage "$source_config" "$source_checkpoint"

  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$finetune_config" \
    --checkpoint "$f3_checkpoint" \
    --output-dir "$reactzyme_cache" \
    --precision bf16

  run_training_stage "$finetune_config" "$final_checkpoint"

  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$finetune_config" \
    --checkpoint "$f3_checkpoint" \
    --output-dir "$validation_cache" \
    --validation-only \
    --precision 32 \
    --output-dtype float32

  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    scripts/cache_frozen_prototype_embeddings.py \
    --config "$f3_test_config" \
    --checkpoint "$f3_checkpoint" \
    --output-dir "$test_cache" \
    --validation-only \
    --precision 32 \
    --output-dtype float32

  "$PYTHON_BIN" scripts/evaluate_biological_residual.py \
    --checkpoint "$final_checkpoint" \
    --config "$finetune_config" \
    --pairs "data/revised_protocols/reactzyme_paper/$split/validation_pairs.csv" \
    --reactions "data/revised_protocols/reactzyme_paper/$split/validation_rxns.csv" \
    --reaction-base-cache "$validation_cache/validation_reaction_base.h5" \
    --enzyme-base-cache "$validation_cache/enzyme_base.h5" \
    --functional-token-cache "$SHARED_REACTZYME_TOKENS" \
    --candidate-ids "data/revised_protocols/reactzyme_paper/$split/validation_candidate_ids.txt" \
    --alphas 0 0.025 0.05 0.075 0.1 \
    --output "$validation_results"

  best_alpha=$("$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_alpha"])' \
    "$validation_results")

  "$PYTHON_BIN" scripts/evaluate_biological_residual.py \
    --checkpoint "$final_checkpoint" \
    --config "$f3_test_config" \
    --pairs "data/revised_protocols/reactzyme_paper/$split/test_pairs.csv" \
    --reactions "data/revised_protocols/reactzyme_paper/$split/test_rxns.csv" \
    --reaction-base-cache "$test_cache/validation_reaction_base.h5" \
    --enzyme-base-cache "$test_cache/enzyme_base.h5" \
    --functional-token-cache "$SHARED_REACTZYME_TOKENS" \
    --candidate-ids "data/revised_protocols/reactzyme_paper/$split/test_candidate_ids.txt" \
    --alphas 0 "$best_alpha" \
    --output "$test_results"

  echo "Completed $split. Validation: $validation_results"
  echo "Completed $split. Test: $test_results"
done
