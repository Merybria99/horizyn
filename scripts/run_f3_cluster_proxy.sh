#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/reactzyme_f3_cluster_proxy_v1"
CONFIG="$ROOT/configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml"
PYTHON_BIN="$ROOT/../env/bin/python"
REAL_HOME="${HOME:?HOME must identify the existing W&B profile}"

mkdir -p \
  "$RUN_ROOT/checkpoints" \
  "$RUN_ROOT/logs" \
  "$RUN_ROOT/nohome" \
  "$RUN_ROOT/cache/huggingface" \
  "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/xdg" \
  "$RUN_ROOT/cache/wandb" \
  "$ROOT/run_pids"

if find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f -name '*.ckpt' | grep -q .; then
  echo "Refusing to start from scratch: checkpoints already exist in $RUN_ROOT/checkpoints" >&2
  exit 2
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$($PYTHON_BIN - "$REAL_HOME/.netrc" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("No api.wandb.ai credential in the existing netrc profile")
print(credentials[2])
PY
)"
  export WANDB_API_KEY
fi

export HOME="$RUN_ROOT/nohome"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$RUN_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export WANDB_DIR="$RUN_ROOT/cache/wandb"
export WANDB_CACHE_DIR="$RUN_ROOT/cache/wandb"
# Python appends long random names for multiprocessing sockets, whose complete
# AF_UNIX path must stay below the kernel's 108-byte limit.
export TMPDIR="${HORIZYN_SHORT_TMPDIR:-/tmp/hz-f3-${UID}}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1

printf '%s\n' "$$" > "$ROOT/run_pids/reactzyme_f3_cluster_proxy_v1.pid"
cd "$ROOT"
exec "$PYTHON_BIN" -u scripts/train_protein_pooling.py --config "$CONFIG"
