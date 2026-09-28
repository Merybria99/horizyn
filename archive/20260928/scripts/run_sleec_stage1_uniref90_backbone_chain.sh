#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-sleec-stage1-uniref90}"
PSEUDO_LABELS_PATH="${PSEUDO_LABELS_PATH:-data/sleec_stage1/msa_pseudo_residue_labels_uniref90.balanced.csv}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "$PROJECT_ROOT"

run_backbone() {
  local backbone="$1"
  local run_name="$2"
  local output_dir="$3"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  scripts/run_sleec_stage1_backbone_4gpu.sh \
    --backbone "$backbone" \
    --run-name "$run_name" \
    --output-dir "$output_dir" \
    --pseudo-labels "$PSEUDO_LABELS_PATH" \
    --train-only \
    --wandb \
    --wandb-project "$WANDB_PROJECT"
}

echo "[$(date --iso-8601=seconds)] Starting UniRef90 SLEEC Stage 1 backbone chain"
echo "Pseudo labels: $PSEUDO_LABELS_PATH"
echo "W&B project: $WANDB_PROJECT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

run_backbone \
  prott5 \
  "sleec_stage1_prott5_uniref90_msa_4gpu_${RUN_STAMP}" \
  "checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_${RUN_STAMP}"

run_backbone \
  esm2 \
  "sleec_stage1_esm2_650m_uniref90_msa_4gpu_${RUN_STAMP}" \
  "checkpoints/SLEEC/sleec_stage1_esm2_650m_uniref90_msa_4gpu_${RUN_STAMP}"

run_backbone \
  esmc \
  "sleec_stage1_esmc_6b_uniref90_msa_4gpu_${RUN_STAMP}" \
  "checkpoints/SLEEC/sleec_stage1_esmc_6b_uniref90_msa_4gpu_${RUN_STAMP}"

echo "[$(date --iso-8601=seconds)] UniRef90 SLEEC Stage 1 backbone chain complete"
