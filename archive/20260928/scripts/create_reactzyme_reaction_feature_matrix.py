#!/usr/bin/env python3
"""Generate the F0-F6 ReactZyme reaction-feature ablation matrix."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.benchmarks.reactzyme_matrix import main


if __name__ == "__main__":
    if "--matrix" not in sys.argv:
        sys.argv[1:1] = ["--matrix", "reaction_features"]
    main()
