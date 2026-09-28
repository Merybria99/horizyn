#!/usr/bin/env bash
# One detached controller; no cancellation of existing jobs and no backbone extraction.
set -euo pipefail
circe_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
circe_python="$circe_project/../env/bin/python"
circe_run=${CIRCE_CONTROL_RUN_ROOT:-"$circe_project/runs/biological_residual_controls_reaction_smi_v1"}
circe_gpus=${CIRCE_CONTROL_GPUS:-0,1,2,3}

if [[ "${1:-launch}" == run ]]; then
  circe_run=$2
  circe_gpus=$3
  exec >> "$circe_run/pipeline.log" 2>&1
  echo "Controller starting on $(hostname -s); requested GPUs: $circe_gpus"
  exec "$circe_python" -u "$circe_project/scripts/run_biological_residual_controls.py" \
    run --run-root "$circe_run" --gpus "$circe_gpus"
fi
[[ "${1:-launch}" == launch ]] || { echo "Usage: bash $0 [launch]" >&2; exit 1; }
command -v tmux >/dev/null
[[ -x "$circe_python" ]] || { echo "Missing Python: $circe_python" >&2; exit 1; }
circe_run=$(realpath -m "$circe_run")
[[ "$circe_run" != "$circe_project" && "$circe_run" != / && "$circe_run" != "$circe_project/runs" ]] || exit 1
[[ "$circe_run" != "$circe_project/runs/biological_residual_reaction_smi"* ]] || {
  echo "Use a separate control-run directory; original results must be preserved." >&2; exit 1;
}
circe_tag=$(printf '%s' "$circe_run" | sha256sum | cut -c1-10)
circe_session="bio_controls_$circe_tag"
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Session already exists: $circe_session. No duplicate launched." >&2
  exit 1
fi
mkdir -p "$circe_run"
tmux new-session -d -s "$circe_session" -c "$circe_project" \
  /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /bin/bash --noprofile --norc "$circe_project/scripts/launch_biological_residual_controls.sh" \
  run "$circe_run" "$circe_gpus"
echo "Detached session: $circe_session"
echo "Log: $circe_run/pipeline.log"
echo "Launch requested; the log confirms preflight/training or explains any refusal."
