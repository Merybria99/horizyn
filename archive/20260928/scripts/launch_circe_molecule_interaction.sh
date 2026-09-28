#!/usr/bin/env bash
set -euo pipefail
circe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command -v tmux >/dev/null
exec /usr/bin/env -u BASH_ENV -u ENV PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$circe_root/../env/bin/python" "$circe_root/scripts/run_circe_molecule_interaction.py" launch "$@"
