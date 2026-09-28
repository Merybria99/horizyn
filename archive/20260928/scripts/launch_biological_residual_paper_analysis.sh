#!/usr/bin/env bash
# Existing checkpoints/features only. One evaluation per selected free GPU.
set -euo pipefail
circe_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
circe_run=${CIRCE_ANALYSIS_RUN_ROOT:-"$circe_project/runs/biological_residual_paper_analysis_v1"}
circe_gpus=${CIRCE_ANALYSIS_GPUS:-0,1,2}
circe_python="$circe_project/../env/bin/python"
circe_run=$(realpath -m "$circe_run")
[[ "$(dirname "$circe_run")" == "$circe_project/runs" && "$(basename "$circe_run")" == biological_residual_paper_analysis_* ]] || {
  echo "Use a separate runs/biological_residual_paper_analysis_* directory." >&2; exit 1;
}
command -v tmux >/dev/null
[[ -x "$circe_python" && -x "$circe_project/../.capability-run-py/bin/python" ]]
circe_tag=$(printf '%s' "$circe_run" | sha256sum | cut -c1-10)
circe_session="bio_paper_$circe_tag"
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Session already exists: $circe_session; no duplicate launched." >&2; exit 1;
fi
mkdir -p "$circe_run"
tmux new-session -d -s "$circe_session" -c "$circe_project" \
  /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /bin/bash --noprofile --norc -c \
  'circe_output=$1; shift; exec "$@" >> "$circe_output/pipeline.log" 2>&1' \
  circe-analysis "$circe_run" "$circe_python" -u \
  "$circe_project/scripts/run_biological_residual_paper_analysis.py" run \
  --run-root "$circe_run" --gpus "$circe_gpus"
echo "Detached session: $circe_session"
echo "Log: $circe_run/pipeline.log"
echo "Launch requested; selected GPUs must pass occupancy checks. No training/extraction is scheduled."
