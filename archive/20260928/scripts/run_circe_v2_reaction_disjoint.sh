#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
export PROJECT_ROOT
export CIRCE_CONFIG="configs/reactzyme_reaction_smi_prott5_annotation_negatives_symmetric_reaction_blocks_reaction_disjoint.yaml"
export CIRCE_NEGATIVE_TRAIN_PAIRS="data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/train_pairs.csv"
export CIRCE_NEGATIVE_POOL_OUTPUT="runs/circe_v2_symmetric_reaction_blocks_reaction_disjoint/data/train_negative_pools.json"
export CIRCE_NEGATIVE_POOL_REPORT="runs/circe_v2_symmetric_reaction_blocks_reaction_disjoint/data/train_negative_pools_report.json"

exec bash "$PROJECT_ROOT/scripts/run_circe_v2.sh"
