#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/cyp_external_v1
exec 9>runs/cyp_external_v1/enzymecage_recovery.lock
flock -n 9 || { echo "EnzymeCAGE recovery is already running"; exit 1; }
exec >> runs/cyp_external_v1/enzymecage_recovery_pipeline.log 2>&1
echo "Recovering released complexes; no benchmark candidates will be removed."
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
../.capability-run-py/bin/hf download lizmahood/cyp_pred_repos boltzcyp_data_dir.tar.gz \
  --repo-type dataset --revision a5c15469c6caae1328ef777ba141215a8c7c14b5 \
  --local-dir data/external/cyp_specificity_2026/release
/usr/bin/python3 -u scripts/prepare_cyp_external_assets.py boltzcyp_recovery
.deps/cyp-external-env/bin/python -u scripts/recover_cyp_pockets.py --allow-missing
/usr/bin/python3 -u scripts/prepare_cyp_external_assets.py boltzcyp_generation_inputs
.deps/cyp-boltz-env/bin/python -u scripts/generate_cyp_complexes.py --gpu "${1:-1}" --generate-missing-msas
.deps/cyp-external-env/bin/python -u scripts/recover_cyp_pockets.py
.deps/cyp-external-env/bin/python -u scripts/run_cyp_external.py run \
  --models enzymecage_pretrained --gpus "${1:-1}"
