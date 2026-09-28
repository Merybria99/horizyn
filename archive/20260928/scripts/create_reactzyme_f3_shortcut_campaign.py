#!/usr/bin/env python3
"""Generate the single-head F3 shortcut-ablation campaign."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.benchmarks.reactzyme_f3_shortcut_campaign import main


if __name__ == "__main__":
    if not sys.argv[1:] or sys.argv[1].startswith("--"):
        sys.argv.insert(1, "generate")
    main()
