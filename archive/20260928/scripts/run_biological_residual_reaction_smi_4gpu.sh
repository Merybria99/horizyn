#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
GPU_COUNT="${GPU_COUNT:-4}"
RUN_ROOT="runs/biological_residual_reaction_smi"
F3_CHECKPOINT="${F3_CHECKPOINT:-runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry/protein-pooling-epoch=29.ckpt}"
F3_TEST_CONFIG="${F3_TEST_CONFIG:-runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml}"
SLEEC_CHECKPOINT="${SLEEC_CHECKPOINT:-checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt}"
SOURCE_CONFIG="configs/source_collapse_biological_residual_reaction_smi_pretrain.yaml"
REACTZYME_CONFIG="configs/reactzyme_reaction_smi_biological_residual.yaml"

SOURCE_CACHE="$RUN_ROOT/cache/source"
REACTZYME_CACHE="$RUN_ROOT/cache/reactzyme"
VALIDATION_EVAL_CACHE="$RUN_ROOT/cache/reactzyme_validation_eval"
TEST_CACHE="$RUN_ROOT/cache/reactzyme_test"
SOURCE_TOKENS="$SOURCE_CACHE/protein_functional_tokens.h5"
REACTZYME_TOKENS="$REACTZYME_CACHE/protein_functional_tokens.h5"
SOURCE_CHECKPOINT="$RUN_ROOT/checkpoints/source_pretrain/last.ckpt"
FINAL_CHECKPOINT="$RUN_ROOT/checkpoints/reactzyme_finetune/last.ckpt"
VALIDATION_RESULTS="$RUN_ROOT/results/validation_alpha_sweep.json"
TEST_RESULTS="$RUN_ROOT/results/test_selected_alpha.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

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

"$PYTHON_BIN" scripts/build_biological_residual_pretrain_split.py

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$SOURCE_CONFIG" \
  --checkpoint "$F3_CHECKPOINT" \
  --output-dir "$SOURCE_CACHE" \
  --precision bf16

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_sleec_functional_tokens.py \
  --residue-h5 outputs/biofp_from_scratch_chains/biofp-fresh-chain-nohome-20260709_172734/data/standardized/retrieval_training_source_collapse/train_exact/fit_proteins_with_clipzyme_eval_prott5_residue.h5 \
  --sleec-checkpoint "$SLEEC_CHECKPOINT" \
  --pairs "$RUN_ROOT/data/source_pretrain/train_pairs.csv" \
  --output "$SOURCE_TOKENS" \
  --top-k 48 \
  --context-k 16 \
  --bf16

run_training_stage "$SOURCE_CONFIG" "$SOURCE_CHECKPOINT"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$REACTZYME_CONFIG" \
  --checkpoint "$F3_CHECKPOINT" \
  --output-dir "$REACTZYME_CACHE" \
  --precision bf16

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_sleec_functional_tokens.py \
  --residue-h5 data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5 \
  --sleec-checkpoint "$SLEEC_CHECKPOINT" \
  --pairs data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv \
  --pairs data/revised_protocols/reactzyme_paper/reaction_smi/validation_pairs.csv \
  --pairs data/revised_protocols/reactzyme_paper/reaction_smi/test_pairs.csv \
  --output "$REACTZYME_TOKENS" \
  --top-k 48 \
  --context-k 16 \
  --bf16

run_training_stage "$REACTZYME_CONFIG" "$FINAL_CHECKPOINT"

# Keep the large training cache compact, but recompute the much smaller
# validation/test CIRCE baselines in FP32 so alpha=0 matches normal evaluation.
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$REACTZYME_CONFIG" \
  --checkpoint "$F3_CHECKPOINT" \
  --output-dir "$VALIDATION_EVAL_CACHE" \
  --validation-only \
  --precision 32 \
  --output-dtype float32

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$F3_TEST_CONFIG" \
  --checkpoint "$F3_CHECKPOINT" \
  --output-dir "$TEST_CACHE" \
  --validation-only \
  --precision 32 \
  --output-dtype float32

"$PYTHON_BIN" scripts/evaluate_biological_residual.py \
  --checkpoint "$FINAL_CHECKPOINT" \
  --config "$REACTZYME_CONFIG" \
  --pairs data/revised_protocols/reactzyme_paper/reaction_smi/validation_pairs.csv \
  --reactions data/revised_protocols/reactzyme_paper/reaction_smi/validation_rxns.csv \
  --reaction-base-cache "$VALIDATION_EVAL_CACHE/validation_reaction_base.h5" \
  --enzyme-base-cache "$VALIDATION_EVAL_CACHE/enzyme_base.h5" \
  --functional-token-cache "$REACTZYME_TOKENS" \
  --candidate-ids data/revised_protocols/reactzyme_paper/reaction_smi/validation_candidate_ids.txt \
  --alphas 0 0.025 0.05 0.075 0.1 \
  --output "$VALIDATION_RESULTS"

BEST_ALPHA=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_alpha"])' "$VALIDATION_RESULTS")

"$PYTHON_BIN" scripts/evaluate_biological_residual.py \
  --checkpoint "$FINAL_CHECKPOINT" \
  --config "$F3_TEST_CONFIG" \
  --pairs data/revised_protocols/reactzyme_paper/reaction_smi/test_pairs.csv \
  --reactions data/revised_protocols/reactzyme_paper/reaction_smi/test_rxns.csv \
  --reaction-base-cache "$TEST_CACHE/validation_reaction_base.h5" \
  --enzyme-base-cache "$TEST_CACHE/enzyme_base.h5" \
  --functional-token-cache "$REACTZYME_TOKENS" \
  --candidate-ids data/revised_protocols/reactzyme_paper/reaction_smi/test_candidate_ids.txt \
  --alphas 0 "$BEST_ALPHA" \
  --output "$TEST_RESULTS"

echo "Completed. Validation: $VALIDATION_RESULTS"
echo "Completed. Test: $TEST_RESULTS"
