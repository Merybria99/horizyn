#!/usr/bin/env python3
"""
Run MSA search through the cloned Ligns MMseqs2 pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def _load_run_ligns_msa_search():
    module_path = project_root / "horizyn" / "msa_search.py"
    spec = importlib.util.spec_from_file_location("horizyn_msa_search", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MSA adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_ligns_msa_search


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate A3M MSAs using the Ligns MMseqs2 pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument(
        "--target-db",
        required=True,
        type=Path,
        help="MMseqs2 database prefix accepted by Ligns, e.g. uniref30_2302_db",
    )
    parser.add_argument("--out-dir", default=Path("data/sota/msas_ligns"), type=Path)
    parser.add_argument("--name", default="horizyn_ligns_msa")
    parser.add_argument("--ligns-root", default=Path("Ligns"), type=Path)
    parser.add_argument(
        "--mmseqs",
        default=None,
        type=Path,
        help="Optional path to the mmseqs executable. Defaults to tools/mmseqs/bin/mmseqs.",
    )
    parser.add_argument("--mode", choices=["monomer", "multimer"], default="monomer")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--keep-insertions", action="store_true")
    parser.add_argument("--allow-deletions", action="store_true")
    parser.add_argument("--dont-expand-aln", action="store_true")
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seq-id", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=12)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_ligns_msa_search = _load_run_ligns_msa_search()
    generated = run_ligns_msa_search(
        query_fasta=args.query_fasta,
        target_db=args.target_db,
        out_dir=args.out_dir,
        name=args.name,
        ligns_root=args.ligns_root,
        mmseqs_bin=args.mmseqs,
        mode=args.mode,
        cache=args.cache,
        keep_insertions=args.keep_insertions,
        allow_deletions=args.allow_deletions,
        expand_aln=not args.dont_expand_aln,
        sensitivity=args.sensitivity,
        max_seq_id=args.max_seq_id,
        threads=args.threads,
    )
    print(f"Generated {len(generated)} A3M file(s):")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
