#!/usr/bin/env bash
# CPU-only comparison; never touches existing GPU jobs or source feature caches.
set -euo pipefail
unset BASH_ENV ENV
circe_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
circe_python="$circe_project/../.capability-run-py/bin/python"
circe_run="$circe_project/runs/cyp_baselines_v1"
circe_session=cyp_baselines_v1
circe_workers=${CYP_CPU_WORKERS:-4}
command -v tmux >/dev/null
[[ "$circe_workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CYP_CPU_WORKERS'; exit 1; }
if tmux has-session -t "=$circe_session" 2>/dev/null; then
    echo "Session already exists: $circe_session. No duplicate launched."
    exit 1
fi
[[ -x "$circe_python" ]] || { echo "Missing $circe_python"; exit 1; }
[[ -x "$circe_project/.deps/cyp-checkpoint-converter/bin/python" ]] || {
    echo 'Missing isolated torch>=2.6 CPU environment; see docs/cyp_baselines.md'; exit 1;
}
[[ -f "$circe_project/.deps/cyp-baseline-deps/Bio/Align/__init__.py" ]] || {
    echo 'Missing scoped Biopython dependency; see docs/cyp_baselines.md'; exit 1;
}
[[ -s "$circe_project/data/external/cyp_specificity_2026/release/fusionesp_model_ckpts.tar.gz" ]] || {
    echo 'Missing pinned FusionESP checkpoints; see download command in docs/cyp_baselines.md'; exit 1;
}
mkdir -p "$circe_run"
printf -v circe_command '%q ' /usr/bin/env -u BASH_ENV -u ENV \
    CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$circe_python" "$circe_project/scripts/run_cyp_baselines.py" all \
    --run-root "$circe_run" --workers "$circe_workers"
printf -v circe_log '%q' "$circe_run/pipeline.log"
# One quoted command and a clean environment prevent stale tmux BASH_ENV hooks.
tmux new-session -d -s "$circe_session" -c / \
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C /bin/bash --noprofile --norc -c \
    "exec $circe_command >> $circe_log 2>&1"
echo "Detached CPU comparison requested: $circe_session"
echo "Watch: tail -f $circe_run/pipeline.log"
