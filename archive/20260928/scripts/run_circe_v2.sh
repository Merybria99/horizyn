#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
CIRCE_DEVICES="${CIRCE_DEVICES:-4}"
CIRCE_CONFIG="${CIRCE_CONFIG:-configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml}"
CIRCE_NEGATIVE_TRAIN_PAIRS="${CIRCE_NEGATIVE_TRAIN_PAIRS:-data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv}"
CIRCE_NEGATIVE_POOL_OUTPUT="${CIRCE_NEGATIVE_POOL_OUTPUT:-runs/circe_v2_annotation_negatives/data/train_negative_pools.json}"
CIRCE_NEGATIVE_POOL_REPORT="${CIRCE_NEGATIVE_POOL_REPORT:-runs/circe_v2_annotation_negatives/data/train_negative_pools_report.json}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/build_annotation_negative_pools.py \
  --train-pairs "$CIRCE_NEGATIVE_TRAIN_PAIRS" \
  --ec-labels data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv \
  --biofp-targets runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/biology/enzyme_biofp_soft_targets.npz \
  --output "$CIRCE_NEGATIVE_POOL_OUTPUT" \
  --report "$CIRCE_NEGATIVE_POOL_REPORT" \
  --max-biological 32 \
  --max-random 32 \
  --ec-prefix-depth 2 \
  --biofp-threshold 0.5 \
  --min-biofp-similarity 0.5 \
  --seed 42

"$PYTHON_BIN" scripts/train_protein_pooling.py \
  --config "$CIRCE_CONFIG" \
  --training.devices "$CIRCE_DEVICES"
