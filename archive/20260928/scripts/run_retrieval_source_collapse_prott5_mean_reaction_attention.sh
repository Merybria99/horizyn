#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/datastor2/deep-proteins/EnzymeDiscovery"
PROJECT_DIR="${ROOT_DIR}/horizyn"
ENV_PYTHON="${ROOT_DIR}/env/bin/python"
CONFIG_PATH="configs/retrieval_source_collapse_nr90_prott5_mean_reaction_attention.yaml"
RUN_NAME="${1:-horizyn-source-collapse-nr90-prott5-mean-reaction-attention-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_DIR}"

export WANDB_API_KEY="$(cat "${HOME}/.config/horizyn_wandb_api_key")"
export WANDB_DIR="${ROOT_DIR}/wandb"
export WANDB_CACHE_DIR="${ROOT_DIR}/wandb/.cache"
export WANDB_CONFIG_DIR="${ROOT_DIR}/wandb/.config"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
mkdir -p logs/retrieval_source_collapse_nr90_prott5_mean_reaction_attention
mkdir -p checkpoints/retrieval_source_collapse_nr90_prott5_mean_reaction_attention

"${ENV_PYTHON}" scripts/train_protein_pooling.py \
  --config "${CONFIG_PATH}" \
  --wandb \
  --wandb-project horizyn-retrieval-source-collapse \
  --wandb-run-name "${RUN_NAME}"
