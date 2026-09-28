#!/usr/bin/env bash
# Restart only the exact legacy test, preserving all existing logs and caches.
set -euo pipefail
circe_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$circe_project"
circe_session=horizyn_test_best16k_4gpu_v2
circe_out="$circe_project/runs/horizyn1_tyrosine_test_best16k_4gpu_v2"
circe_data="$circe_project/runs/horizyn1_circe_v2_reaction_holdout_h200_paired/data"
circe_old="$circe_project/runs/horizyn1_tyrosine_test_best16k/test.json"
command -v tmux >/dev/null
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "The four-GPU test session already exists; no duplicate launched."
  exit 1
fi
mkdir -p "$circe_out"
circe_args=(
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$circe_project/../env/bin/python" -u "$circe_project/scripts/evaluate_horizyn1_parallel.py" run
  --checkpoint "$circe_project/runs/horizyn1_circe_v2_reaction_holdout_tyrosine_validation_fixed/checkpoints/protein-pooling-epoch=00-v1.ckpt"
  --config "$circe_project/runs/horizyn1_circe_v2_reaction_holdout_h200_paired/configs/train.yaml"
  --pairs "$circe_data/test_query_gold.csv" --reactions "$circe_data/test_rxns.csv"
  --protein-candidates "$circe_data/all_candidate_ids.txt"
  --reaction-candidates "$circe_data/test_reaction_ids.txt"
  --enzyme-query-ids "$circe_data/test_enzyme_query_ids.txt"
  --reaction-query-ids "$circe_data/test_reaction_query_ids.txt"
  --output "$circe_out/test.json" --gpus "${CIRCE_TEST_GPUS:-0,1,2,3}"
  --stop-previous-output "$circe_old" --catalog-on-gpu
)
tmux new-session -d -s "$circe_session" -c "$circe_project" \
  /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c \
  'exec >> "$1" 2>&1; shift; exec "$@"' \
  circe-parallel "$circe_out/pipeline.log" "${circe_args[@]}"
echo "Detached four-GPU test requested: $circe_session"
echo "GPU preflight must pass before any remaining old evaluator is stopped; tmux is never a stop target."
echo "Watch: tail -f $circe_out/pipeline.log"
