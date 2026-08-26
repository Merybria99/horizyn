#!/usr/bin/env python3
"""Generate the minimal biological-supervision ReactZyme matrix."""

from __future__ import annotations

import sys

from horizyn.benchmarks.reactzyme_matrix import main


if __name__ == "__main__":
    if "--matrix" not in sys.argv:
        sys.argv.extend(["--matrix", "biological"])
    main()
