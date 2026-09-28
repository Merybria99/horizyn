#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_enzymecage_f3.sh [prepare|features|train|all]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-all}"
[[ "$stage" =~ ^(prepare|features|train|all)$ ]] || { echo "Invalid stage: $stage" >&2; exit 2; }
run_root="${RUN_ROOT:-$PWD/runs/enzymecage_f3_seed42}"
mkdir -p "$run_root/logs" "$run_root/features"
# One pipeline owns this run directory, including across detached sessions.
exec 9>"$run_root/pipeline.lock"
flock -n 9 || { echo "This run already has an active pipeline" >&2; exit 1; }
python_bin="${PIPELINE_PYTHON:-$PWD/../.capability-run-py/bin/python}"
setup_python="${SETUP_PYTHON:-$PWD/../env/bin/python}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/../hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UNIMOL_WEIGHT_DIR="${UNIMOL_WEIGHT_DIR:-$PWD/../unimol_weights}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
IFS=, read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu_count="${#gpu_ids[@]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ "$stage" == prepare || "$stage" == all ]]; then
  if [[ ! -s "$run_root/data/manifest.json" ]]; then
    "$python_bin" scripts/prepare_enzymecage_f3.py \
      --train ../EnzymeCAGE/dataset/training/train.csv \
      --valid ../EnzymeCAGE/dataset/training/valid.csv \
      --output "$run_root/data"
  fi
  if [[ ! -s "$run_root/data/normalized/manifest.json" ]]; then
    "$python_bin" scripts/normalize_enzymecage_f3_reactions.py --data-dir "$run_root/data"
  fi
  "$python_bin" scripts/configure_enzymecage_f3.py \
    --template runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml \
    --run-root "$run_root" --devices "$gpu_count"
fi

if [[ "$stage" == features || "$stage" == all ]]; then
  [[ -s "$run_root/data/normalized/manifest.json" ]] || { echo "Run prepare first" >&2; exit 1; }
  prott5="${PROTT5_MODEL:-$PWD/data/external/cyp_specificity_2026/models/prott5_safetensors}"
  reaction_t5="${REACTION_T5_MODEL:-$PWD/../hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc}"
  common=(--fasta "$run_root/data/proteins.fasta" --output "$run_root/features/proteins_prott5_residue.h5"
    --model-name "$prott5" --world-size "$gpu_count" --max-sequence-length 1022
    --sequence-truncation ends_center --dtype float16 --compression none --batch-size 256
    --max-tokens-per-batch 65536 --length-sort --padded-token-budget --merge-order shard
    --merge-storage virtual --resume)
  if [[ ! -s "$run_root/features/proteins_prott5_residue.h5" ]]; then
    pids=()
    for rank in "${!gpu_ids[@]}"; do
      CUDA_VISIBLE_DEVICES="${gpu_ids[$rank]}" "$python_bin" scripts/extract_prott5_residue_embeddings.py \
        "${common[@]}" --rank "$rank" --device cuda \
        > "$run_root/logs/prott5_rank${rank}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || { echo "ProtT5 worker failed; see rank logs" >&2; exit 1; }
    "$python_bin" scripts/extract_prott5_residue_embeddings.py "${common[@]}" --merge-only
  fi
  rxns=("$run_root/data/normalized/train_rxns.csv" "$run_root/data/normalized/validation_rxns.csv")
  if [[ ! -s "$run_root/features/reactiont5v2.h5" ]]; then
    [[ ! -e "$run_root/features/reactiont5v2.h5.partial" ]] || { echo "Inspect incomplete ReactionT5 file before retry" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions "${rxns[@]}" --output "$run_root/features/reactiont5v2.h5.partial" \
      --model-name "$reaction_t5" --batch-size 64 --max-length 512 --pooling mean \
      --device cuda --dtype float16 --no-bidirectional --no-allow-pseudo-reactions --no-standardize
    mv "$run_root/features/reactiont5v2.h5.partial" "$run_root/features/reactiont5v2.h5"
  fi
  if [[ ! -s "$run_root/features/unimol2.h5" ]]; then
    [[ ! -e "$run_root/features/unimol2.h5.partial" ]] || { echo "Inspect incomplete Uni-Mol2 file before retry" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" PYTHONPATH="$PWD/.deps/unimol_tools:$PWD/../env/unimol2_site:$PWD" \
      "$python_bin" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions "${rxns[@]}" --output "$run_root/features/unimol2.h5.partial" \
      --batch-size 64 --dtype float16 --compression none --skip-invalid-molecules \
      --skip-invalid-reactions --no-bidirectional --no-allow-pseudo-reactions --no-standardize
    mv "$run_root/features/unimol2.h5.partial" "$run_root/features/unimol2.h5"
  fi
  if [[ ! -s "$run_root/features/chiro.h5" ]]; then
    [[ ! -e "$run_root/features/chiro.h5.partial" ]] || { echo "Inspect incomplete ChIRo file before retry" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" PYTHONPATH="$PWD/.deps/python:$PWD/.deps/ChIRo:$PWD" \
      "$python_bin" scripts/extract_chiro_reaction_embeddings.py \
      --reactions "${rxns[@]}" --output "$run_root/features/chiro.h5.partial" \
      --device cuda --batch-size 64 --num-workers 4 --dtype float16 \
      --molecule-cache "$run_root/features/chiro_molecules.sqlite" \
      --no-bidirectional --no-allow-pseudo-reactions --no-standardize
    mv "$run_root/features/chiro.h5.partial" "$run_root/features/chiro.h5"
  fi
  if [[ ! -s "$run_root/features/chemistry/schema.json" ]]; then
    "$setup_python" scripts/build_reaction_set_features.py \
      --train-reactions "${rxns[0]}" --validation-reactions "${rxns[1]}" \
      --test-reactions "${rxns[1]}" \
      --cofactor-dictionary data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv \
      --out-dir "$run_root/features/chemistry"
  fi
fi

if [[ "$stage" == train || "$stage" == all ]]; then
  [[ -s "$run_root/configs/train.yaml" && -s "$run_root/features/chemistry/schema.json" ]] || {
    echo "Run prepare and features first" >&2; exit 1;
  }
  "$python_bin" scripts/configure_enzymecage_f3.py \
    --template runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml \
    --run-root "$run_root" --devices "$gpu_count"
  "$python_bin" scripts/align_enzymecage_f3_features.py --run-root "$run_root" --check-loader
  "$python_bin" scripts/train_protein_pooling.py --config "$run_root/configs/train.yaml"
fi
