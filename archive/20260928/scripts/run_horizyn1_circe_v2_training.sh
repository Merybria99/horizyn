#!/usr/bin/env bash
# Run in tmux/nohup; --profile h200 is opt-in and defaults to a separate run root.
# Preparation work stays on shared storage unless --scratch-root is explicit.
# No shell backgrounding, old-database migration or GPU scheduling is hidden here.
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPU_COUNT="${GPU_COUNT:-4}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/../hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UNIMOL_WEIGHT_DIR="${UNIMOL_WEIGHT_DIR:-${PROJECT_ROOT}/../unimol_weights}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-${PROJECT_ROOT}/../.capability-run-py/bin/python}"
cd "$PROJECT_ROOT"
exec "$PIPELINE_PYTHON" scripts/horizyn1_circe_v2_pipeline.py "${@:-all}"
