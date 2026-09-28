#!/usr/bin/env python3
"""Reconstruct and audit the public-source approximation of Horizyn-1 data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "horizyn/datasets/horizyn1_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("horizyn1_reconstruction", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load reconstruction implementation from {MODULE_PATH}")
reconstruction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reconstruction
SPEC.loader.exec_module(reconstruction)

build_rhea_direction_map = reconstruction.build_rhea_direction_map
cluster_and_collapse = reconstruction.cluster_and_collapse
development_rhea_ids = reconstruction.development_rhea_ids
disk_preflight = reconstruction.disk_preflight
finalize_enzymemap_pairs = reconstruction.finalize_enzymemap_pairs
merge_sources = reconstruction.merge_sources
parse_uniprot_release = reconstruction.parse_uniprot_release
prepare_enzymemap = reconstruction.prepare_enzymemap
read_fasta_ids = reconstruction.read_fasta_ids
read_id_file = reconstruction.read_id_file
write_json = reconstruction.write_json


def _emit(stats: dict[str, object], manifest: Path | None = None) -> None:
    if manifest is not None:
        write_json(manifest, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check conservative disk capacity")
    preflight.add_argument("--path", type=Path, required=True)
    preflight.add_argument("--reserve-tib", type=float, default=5.0)
    preflight.add_argument("--manifest", type=Path)

    rhea = subparsers.add_parser("rhea-map", help="extract Rhea directional-to-master mapping")
    rhea.add_argument("--input", type=Path, required=True)
    rhea.add_argument("--output", type=Path, required=True)
    rhea.add_argument("--manifest", type=Path)

    enzymemap = subparsers.add_parser("prepare-enzymemap", help="extract EnzymeMap pairs")
    enzymemap.add_argument("--processed", type=Path, required=True)
    enzymemap.add_argument("--pairs", type=Path, required=True)
    enzymemap.add_argument("--reactions", type=Path, required=True)
    enzymemap.add_argument("--wanted-accessions", type=Path, required=True)
    enzymemap.add_argument("--manifest", type=Path)

    uniprot = subparsers.add_parser("parse-uniprot", help="stream the UniProt release archive")
    uniprot.add_argument("--input", type=Path, required=True)
    uniprot.add_argument("--source-kind", choices=["auto", "sprot", "trembl"], default="auto")
    uniprot.add_argument("--rhea-map", type=Path, required=True)
    uniprot.add_argument("--development-reactions", type=Path, nargs="+", required=True)
    uniprot.add_argument("--wanted-accessions", type=Path, required=True)
    uniprot.add_argument("--existing-fasta", type=Path, nargs="+", required=True)
    uniprot.add_argument("--pairs", type=Path, required=True)
    uniprot.add_argument("--selected-fasta", type=Path, required=True)
    uniprot.add_argument("--extra-fasta", type=Path, required=True)
    uniprot.add_argument("--accession-map", type=Path, required=True)
    uniprot.add_argument("--manifest", type=Path)

    finalize = subparsers.add_parser(
        "finalize-enzymemap", help="resolve EnzymeMap accessions to parsed sequences"
    )
    finalize.add_argument("--raw-pairs", type=Path, required=True)
    finalize.add_argument("--accession-map", type=Path, required=True)
    finalize.add_argument("--existing-fasta", type=Path, nargs="+", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path)

    merge = subparsers.add_parser("merge", help="merge and deduplicate all raw graph sources")
    merge.add_argument("--development-pairs", type=Path, nargs="+", required=True)
    merge.add_argument("--development-reactions", type=Path, nargs="+", required=True)
    merge.add_argument("--fasta", type=Path, nargs="+", required=True)
    merge.add_argument("--trembl-pairs", type=Path, required=True)
    merge.add_argument("--enzymemap-pairs", type=Path, required=True)
    merge.add_argument("--enzymemap-reactions", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--sort-threads", type=int, default=16)

    cluster = subparsers.add_parser("cluster", help="MMseqs2 cluster and source-collapse edges")
    cluster.add_argument("--raw-fasta", type=Path, required=True)
    cluster.add_argument("--raw-pairs", type=Path, required=True)
    cluster.add_argument("--output-dir", type=Path, required=True)
    cluster.add_argument("--mmseqs", type=Path, required=True)
    cluster.add_argument("--threads", type=int, default=64)
    cluster.add_argument("--min-seq-id", type=float, default=0.8)
    cluster.add_argument(
        "--extra-cluster-arg",
        action="append",
        default=[],
        help="additional MMseqs cluster argument; repeat once per token",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "preflight":
        stats = disk_preflight(args.path, reserve_tib=args.reserve_tib)
        _emit(stats, args.manifest)
        if not stats["enough_space"]:
            raise SystemExit(2)
    elif args.command == "rhea-map":
        _emit(build_rhea_direction_map(args.input, args.output), args.manifest)
    elif args.command == "prepare-enzymemap":
        _emit(
            prepare_enzymemap(args.processed, args.pairs, args.reactions, args.wanted_accessions),
            args.manifest,
        )
    elif args.command == "parse-uniprot":
        _emit(
            parse_uniprot_release(
                input_path=args.input,
                rhea_map_path=args.rhea_map,
                valid_master_ids=development_rhea_ids(args.development_reactions),
                wanted_accessions=read_id_file(args.wanted_accessions),
                existing_accessions=read_fasta_ids(args.existing_fasta),
                pair_output=args.pairs,
                selected_fasta_output=args.selected_fasta,
                extra_fasta_output=args.extra_fasta,
                accession_map_output=args.accession_map,
                source_kind=args.source_kind,
            ),
            args.manifest,
        )
    elif args.command == "finalize-enzymemap":
        _emit(
            finalize_enzymemap_pairs(
                args.raw_pairs,
                args.accession_map,
                read_fasta_ids(args.existing_fasta),
                args.output,
            ),
            args.manifest,
        )
    elif args.command == "merge":
        _emit(
            merge_sources(
                development_pair_csvs=args.development_pairs,
                development_reaction_csvs=args.development_reactions,
                fasta_inputs=args.fasta,
                trembl_pairs=args.trembl_pairs,
                enzymemap_pairs=args.enzymemap_pairs,
                enzymemap_reactions=args.enzymemap_reactions,
                output_dir=args.output_dir,
                sort_threads=args.sort_threads,
            )
        )
    elif args.command == "cluster":
        _emit(
            cluster_and_collapse(
                raw_fasta=args.raw_fasta,
                raw_pairs=args.raw_pairs,
                output_dir=args.output_dir,
                mmseqs=args.mmseqs,
                threads=args.threads,
                min_seq_id=args.min_seq_id,
                extra_cluster_args=args.extra_cluster_arg,
            )
        )


if __name__ == "__main__":
    main()
