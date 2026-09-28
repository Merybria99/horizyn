#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_reactzyme_f3_loss_campaign.sh" "$@" --variant L7
