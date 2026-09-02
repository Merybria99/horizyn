#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 VARIANT GPU_LIST MASTER_PORT" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
READY="$ROOT/runs/reactzyme_f3_biological_no_ec_v1/READY"
while [[ ! -e "$READY" ]]; do
  echo "[$(date -Iseconds)] waiting for finalized no-EC configs"
  sleep 30
done
exec bash "$SCRIPT_DIR/run_f3_no_ec_variant.sh" "$1" "$2" "$3"
