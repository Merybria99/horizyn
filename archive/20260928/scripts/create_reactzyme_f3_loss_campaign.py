#!/usr/bin/env python3
"""Generate the canonical-isomeric F3 loss-ablation campaign."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.benchmarks.reactzyme_f3_loss_campaign import main


if __name__ == "__main__":
    if not sys.argv[1:] or sys.argv[1].startswith("--"):
        sys.argv.insert(1, "generate")
    main()

