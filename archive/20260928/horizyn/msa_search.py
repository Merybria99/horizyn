"""
Thin adapter for running the Ligns MMseqs2 MSA pipeline from Horizyn.

The MSA implementation intentionally lives in the cloned ``Ligns`` checkout.
This module only resolves paths, exposes a stable Horizyn-facing function, and
keeps imports lazy so normal Horizyn training does not depend on MSA tooling.
"""

from __future__ import annotations

import sys
import os
import shutil
from pathlib import Path
from typing import Literal


def default_ligns_root() -> Path:
    return Path(__file__).resolve().parents[1] / "Ligns"


def default_mmseqs_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "mmseqs" / "bin" / "mmseqs"


def ensure_ligns_importable(ligns_root: Path | None = None) -> Path:
    root = (ligns_root or default_ligns_root()).resolve()
    package_dir = root / "Ligns"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Ligns checkout not found at {root}. Clone https://github.com/danny305/Ligns "
            "into the Horizyn repo root, or pass --ligns-root."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_mmseqs_on_path(mmseqs_bin: Path | None = None) -> Path:
    candidates = []
    if mmseqs_bin is not None:
        candidates.append(mmseqs_bin)
    candidates.append(default_mmseqs_path())

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            bin_dir = str(candidate.parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if bin_dir not in path_entries:
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate

    found = shutil.which("mmseqs")
    if found is not None:
        return Path(found).resolve()

    raise FileNotFoundError(
        "The Ligns MSA pipeline requires the 'mmseqs' executable on PATH. "
        f"Expected repo-local binary at {default_mmseqs_path()} or pass --mmseqs."
    )


def run_ligns_msa_search(
    query_fasta: Path,
    target_db: Path,
    out_dir: Path,
    name: str,
    *,
    ligns_root: Path | None = None,
    mmseqs_bin: Path | None = None,
    mode: Literal["monomer", "multimer"] = "monomer",
    cache: bool = False,
    keep_insertions: bool = False,
    allow_deletions: bool = False,
    expand_aln: bool = True,
    sensitivity: float = 7.5,
    max_seq_id: float = 0.95,
    threads: int = 12,
) -> list[Path]:
    """
    Run Ligns' MMseqs2 MSA search for a query FASTA.

    Args:
        query_fasta: FASTA file containing one or more query protein sequences.
        target_db: MMseqs2 target database prefix. This is the file path accepted
            by Ligns/MMseqs, not a directory.
        out_dir: Directory where Ligns should write generated A3M files.
        name: Run name used by Ligns for output subdirectories/files.
        ligns_root: Optional path to the cloned Ligns repository.
        mode: Ligns MSA mode, ``monomer`` or ``multimer``.
        cache: Reuse Ligns intermediate databases in the output directory.
        keep_insertions: Keep insertions in split A3M files.
        allow_deletions: Forwarded to Ligns/MMseqs result filtering.
        expand_aln: Whether to run Ligns' expandaln step.
        sensitivity: MMseqs search sensitivity.
        max_seq_id: Max sequence identity for Ligns filtering.
        threads: Number of MMseqs threads.

    Returns:
        Sorted list of generated ``.a3m`` files under ``out_dir / name``.
    """
    if mode not in {"monomer", "multimer"}:
        raise ValueError("mode must be one of: monomer, multimer")

    query_fasta = query_fasta.resolve()
    target_db = target_db.resolve()
    out_dir = out_dir.resolve()

    if not query_fasta.is_file():
        raise FileNotFoundError(f"Query FASTA does not exist: {query_fasta}")
    if query_fasta.suffix not in {".fa", ".fasta"}:
        raise ValueError(f"Query FASTA must end in .fa or .fasta: {query_fasta}")
    if not target_db.is_file():
        raise FileNotFoundError(f"MMseqs target database prefix does not exist: {target_db}")
    ensure_mmseqs_on_path(mmseqs_bin)

    ensure_ligns_importable(ligns_root)
    try:
        from Ligns.cli.make_msa import generate_msa
    except ImportError as exc:
        raise ImportError(
            "Could not import Ligns MSA code. Install Ligns' Python dependencies "
            "(at minimum biopython and pandas) in the active environment."
        ) from exc

    generate_msa(
        seq_data=[query_fasta],
        db=target_db,
        out_dir=out_dir,
        name=name,
        cache=cache,
        rm_insertions=not keep_insertions,
        allow_deletions=allow_deletions,
        n_threads=threads,
        sensitivity=sensitivity,
        max_seq_id=max_seq_id,
        expand_aln=expand_aln,
        mode=mode,
    )
    return sorted((out_dir / name).rglob("*.a3m"))
