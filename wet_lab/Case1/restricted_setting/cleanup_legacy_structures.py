#!/usr/bin/env python3
"""Remove legacy Case1 structure layouts after preserving the canonical tree."""

from __future__ import annotations

import shutil
from pathlib import Path

from organize_structure_tree import SCRIPT_DIR, TREE_ROOT, organize


def cleanup() -> Path:
    if not (TREE_ROOT / "manifest.csv").is_file():
        raise FileNotFoundError(f"Canonical structure manifest is missing: {TREE_ROOT}")

    structure_root = SCRIPT_DIR / "structures"
    for path in structure_root.iterdir():
        if path.name == "F3_set_chemistry":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    consensus_structures = SCRIPT_DIR / "all_splits/structures_consensus"
    shutil.rmtree(consensus_structures, ignore_errors=True)

    # Rebuild manifests using canonical PDBs as their own provenance sources.
    return organize()


def main() -> None:
    print(cleanup())


if __name__ == "__main__":
    main()
