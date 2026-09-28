#!/usr/bin/env bash
# Keep W&B run files, cache, and artifact staging off the home quota.

WANDB_ROOT="${WANDB_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/wandb}"

export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_ROOT/config}"

mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
