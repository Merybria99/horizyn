#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_e2r_pareto_v1}"

case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT" ;;
esac

HOST="$(hostname -s)"
ENV_ROOT="/datastor2/deep-proteins/EnzymeDiscovery/.machine-envs/$HOST"
MACHINE_HOME="$RUN_ROOT/nohome/$HOST"
UV_CACHE_ROOT="$RUN_ROOT/cache/uv/$HOST"
HOST_RUNTIME_ENV="$RUN_ROOT/runtime.$HOST.env"
CAMPAIGN_SCRIPT="$ROOT/scripts/run_reactzyme_e2r_pareto_campaign.sh"

mkdir -p "$ENV_ROOT" "$MACHINE_HOME" "$UV_CACHE_ROOT"

export HOME="$MACHINE_HOME"
export UV_CACHE_DIR="$UV_CACHE_ROOT"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache/$HOST"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config/$HOST"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state/$HOST"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x /lusr/bin/uv ]]; then
  UV_BIN="/lusr/bin/uv"
else
  echo "uv was not found in PATH or at /lusr/bin/uv." >&2
  exit 1
fi

if command -v python3.12 >/dev/null 2>&1; then
  SYSTEM_PYTHON="$(command -v python3.12)"
elif [[ -x /usr/bin/python3.12 ]]; then
  SYSTEM_PYTHON="/usr/bin/python3.12"
else
  echo "Python 3.12 was not found on host $HOST." >&2
  exit 1
fi

echo "Host: $HOST"
echo "System Python: $SYSTEM_PYTHON"
echo "Machine environment: $ENV_ROOT"
echo "uv cache: $UV_CACHE_ROOT"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$UV_BIN" venv --python "$SYSTEM_PYTHON" "$ENV_ROOT"
fi

(
  cd "$ROOT"
  VIRTUAL_ENV="$ENV_ROOT" "$UV_BIN" sync --active --frozen --no-dev
)

cat > "$HOST_RUNTIME_ENV" <<EOF
PYTHON_BIN=$ENV_ROOT/bin/python
SETUP_PYTHON_BIN=$ENV_ROOT/bin/python
EOF

echo "Validating the machine environment..."
"$ENV_ROOT/bin/python" - <<'PY'
import lightning.pytorch
import torch
import transformers
import wandb
import yaml

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"lightning={lightning.pytorch.__version__}")
print(f"wandb={wandb.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")
if torch.cuda.device_count() < 4:
    raise SystemExit(
        f"The campaign requires four visible GPUs; found {torch.cuda.device_count()}."
    )
PY

if [[ ! -x "$CAMPAIGN_SCRIPT" ]]; then
  chmod +x "$CAMPAIGN_SCRIPT"
fi

echo "Launching the ReactZyme E2R Pareto campaign..."
"$CAMPAIGN_SCRIPT" "$RUN_ROOT" --detach

echo
echo "Controller log:"
echo "  $RUN_ROOT/logs/controller/nohup.log"
echo "Python preflight:"
echo "  $RUN_ROOT/logs/controller/python_preflight.log"
echo "Campaign status:"
echo "  $RUN_ROOT/logs/status.jsonl"
