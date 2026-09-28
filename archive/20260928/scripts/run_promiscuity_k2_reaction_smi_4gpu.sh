#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
DEVICES="${DEVICES:-4}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/source_collapse_promiscuity_k2_reaction_smi_pretrain.yaml}"
REACTZYME_CONFIG="${REACTZYME_CONFIG:-configs/reactzyme_reaction_smi_promiscuity_k2.yaml}"
SOURCE_PROTOTYPE_CHECKPOINT="${SOURCE_PROTOTYPE_CHECKPOINT:-runs/promiscuity_k2_reaction_smi/checkpoints/source_pretrain/last.ckpt}"
CACHE_PRECISION="${CACHE_PRECISION:-32}"
CACHE_ENZYME_BATCH_SIZE="${CACHE_ENZYME_BATCH_SIZE:-96}"
CACHE_REACTION_BATCH_SIZE="${CACHE_REACTION_BATCH_SIZE:-512}"
SOURCE_CACHE_DIR="${SOURCE_CACHE_DIR:-runs/promiscuity_k2_reaction_smi/cache/source}"
REACTZYME_CACHE_DIR="${REACTZYME_CACHE_DIR:-runs/promiscuity_k2_reaction_smi/cache/reactzyme}"

cd "$PROJECT_ROOT"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/build_promiscuity_reaction_smi_pretrain_split.py

"$PYTHON_BIN" scripts/build_annotation_negative_pools.py \
  --train-pairs data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv \
  --ec-labels data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv \
  --biofp-targets runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/biology/enzyme_biofp_soft_targets.npz \
  --output runs/promiscuity_k2_reaction_smi/data/reactzyme_negative_pools.json \
  --report runs/promiscuity_k2_reaction_smi/data/reactzyme_negative_pools_report.json \
  --max-biological 32 \
  --max-random 32 \
  --ec-prefix-depth 2 \
  --biofp-threshold 0.5 \
  --min-biofp-similarity 0.5 \
  --seed 42

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$DEVICES" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$SOURCE_CONFIG" \
  --checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$SOURCE_CACHE_DIR" \
  --enzyme-batch-size "$CACHE_ENZYME_BATCH_SIZE" \
  --reaction-batch-size "$CACHE_REACTION_BATCH_SIZE" \
  --precision "$CACHE_PRECISION"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$DEVICES" \
  scripts/cache_frozen_prototype_embeddings.py \
  --config "$REACTZYME_CONFIG" \
  --checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$REACTZYME_CACHE_DIR" \
  --enzyme-batch-size "$CACHE_ENZYME_BATCH_SIZE" \
  --reaction-batch-size "$CACHE_REACTION_BATCH_SIZE" \
  --precision "$CACHE_PRECISION"

"$PYTHON_BIN" scripts/train_protein_pooling.py \
  --config "$SOURCE_CONFIG" \
  --training.init_from_checkpoint "$BASE_CHECKPOINT" \
  --data.cached_enzyme_base_embeds_path "$SOURCE_CACHE_DIR/enzyme_base.h5" \
  --data.cached_train_reaction_base_embeds_path "$SOURCE_CACHE_DIR/train_reaction_base.h5" \
  --training.devices "$DEVICES"

if [[ ! -f "$SOURCE_PROTOTYPE_CHECKPOINT" ]]; then
  echo "Source prototype checkpoint not found after pretraining: $SOURCE_PROTOTYPE_CHECKPOINT" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/train_protein_pooling.py \
  --config "$REACTZYME_CONFIG" \
  --training.init_from_checkpoint "$SOURCE_PROTOTYPE_CHECKPOINT" \
  --data.cached_enzyme_base_embeds_path "$REACTZYME_CACHE_DIR/enzyme_base.h5" \
  --data.cached_train_reaction_base_embeds_path "$REACTZYME_CACHE_DIR/train_reaction_base.h5" \
  --data.cached_validation_reaction_base_embeds_path "$REACTZYME_CACHE_DIR/validation_reaction_base.h5" \
  --training.devices "$DEVICES"
