#!/usr/bin/env python3
"""Prepare split-isolated ReactZyme reaction feature extraction.

This utility preserves the clean ReactZyme protocol boundaries. It never
writes a train+test reaction union. For each protocol/split it
writes a manifest, coverage report, and extraction commands, and optionally
runs the existing ReactionT5v2, UniMol2, and ChIRo extractors.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = PROJECT_ROOT / "data/revised_protocols/reactzyme_official"
DEFAULT_FEATURE_ROOT = DEFAULT_PROTOCOL_ROOT / "features"
DEFAULT_CONFIG_ROOT = DEFAULT_PROTOCOL_ROOT / "config_fragments"
DEFAULT_PROTEIN_RESIDUE_H5 = (
    PROJECT_ROOT
    / "data/standardized/retrieval_training_source_collapse/test/"
    "horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5"
)
PROTOCOLS = ("time", "enzyme_smi", "reaction_smi")
SPLITS = ("train", "test")
MODALITIES = ("reactiont5v2", "unimol2", "chiro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=list(MODALITIES),
        help="Modalities to command, validate, and optionally run.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run extraction commands. Without this flag, only manifests and commands are written.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing HDF5 outputs.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an existing HDF5 has missing or extra split keys.",
    )
    parser.add_argument(
        "--allow-pseudo-reactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expect and extract molecule-set strings as self-reactions.",
    )
    parser.add_argument("--python", default=sys.executable, help="Default Python executable.")
    parser.add_argument("--reactiont5-python", default=None)
    parser.add_argument("--unimol-python", default=None)
    parser.add_argument("--chiro-python", default=None)
    parser.add_argument("--reactiont5-model", default="sagawa/ReactionT5v2-forward")
    parser.add_argument("--reactiont5-batch-size", type=int, default=64)
    parser.add_argument("--reactiont5-max-length", type=int, default=512)
    parser.add_argument("--reactiont5-pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--reactiont5-device", default="cuda")
    parser.add_argument("--reactiont5-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--unimol-batch-size", type=int, default=64)
    parser.add_argument("--unimol-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--unimol-compression", choices=("none", "gzip", "lzf"), default="none")
    parser.add_argument("--chiro-device", default="cuda")
    parser.add_argument("--chiro-batch-size", type=int, default=64)
    parser.add_argument("--chiro-num-workers", type=int, default=4)
    parser.add_argument(
        "--protein-residue-embeds-path",
        type=Path,
        default=DEFAULT_PROTEIN_RESIDUE_H5,
        help="Path written into generated config fragments.",
    )
    parser.add_argument(
        "--no-config-fragments",
        action="store_true",
        help="Do not write train/validation and final-test config fragments.",
    )
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_reactions(path: Path) -> OrderedDict[str, str]:
    reactions: OrderedDict[str, str] = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"reaction_id", "reaction_smiles"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            reaction_id = str(row["reaction_id"]).strip()
            reaction_smiles = str(row["reaction_smiles"]).strip()
            if not reaction_id:
                continue
            previous = reactions.get(reaction_id)
            if previous is not None and previous != reaction_smiles:
                raise ValueError(f"{path} has conflicting SMILES for {reaction_id}")
            reactions.setdefault(reaction_id, reaction_smiles)
    return reactions


def expected_directional_ids(
    reactions: OrderedDict[str, str],
    *,
    allow_pseudo_reactions: bool,
) -> list[str]:
    ids: list[str] = []
    for reaction_id, smiles in reactions.items():
        ids.append(f"{reaction_id}_f")
        has_direction = ">>" in smiles and len(smiles.split(">>")) == 2
        if has_direction or allow_pseudo_reactions:
            ids.append(f"{reaction_id}_r")
    return ids


def read_h5_ids(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        if "ids" not in handle:
            raise KeyError(f"{path} does not contain an 'ids' dataset")
        ids = []
        for value in handle["ids"][:]:
            if isinstance(value, bytes):
                ids.append(value.decode("utf-8"))
            else:
                ids.append(str(value))
        return ids


def coverage_for(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    observed_ids = read_h5_ids(path)
    if observed_ids is None:
        return {
            "path": relpath(path),
            "exists": False,
            "expected": len(expected_ids),
            "observed": 0,
            "missing": len(expected_ids),
            "extra": 0,
            "missing_examples": expected_ids[:20],
            "extra_examples": [],
        }
    expected = set(expected_ids)
    observed = set(observed_ids)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "path": relpath(path),
        "exists": True,
        "expected": len(expected_ids),
        "observed": len(observed_ids),
        "unique_observed": len(observed),
        "missing": len(missing),
        "extra": len(extra),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
    }


def output_paths(features_root: Path, protocol: str, split: str) -> dict[str, Path]:
    out_dir = features_root / protocol / split
    return {
        "dir": out_dir,
        "reactiont5v2": out_dir / "reactiont5v2.h5",
        "unimol2": out_dir / "unimol2.h5",
        "chiro": out_dir / "chiro.h5",
    }


def command_for(
    modality: str,
    *,
    args: argparse.Namespace,
    reactions_path: Path,
    output_path: Path,
) -> list[str]:
    python = {
        "reactiont5v2": args.reactiont5_python or args.python,
        "unimol2": args.unimol_python or args.python,
        "chiro": args.chiro_python or args.python,
    }[modality]
    if modality == "reactiont5v2":
        cmd = [
            python,
            str(PROJECT_ROOT / "scripts/extract_reaction_t5v2_embeddings.py"),
            "--reactions",
            str(reactions_path),
            "--output",
            str(output_path),
            "--model-name",
            args.reactiont5_model,
            "--batch-size",
            str(args.reactiont5_batch_size),
            "--max-length",
            str(args.reactiont5_max_length),
            "--pooling",
            args.reactiont5_pooling,
            "--device",
            args.reactiont5_device,
            "--dtype",
            args.reactiont5_dtype,
        ]
    elif modality == "unimol2":
        cmd = [
            python,
            str(PROJECT_ROOT / "scripts/extract_unimol2_reaction_embeddings.py"),
            "--reactions",
            str(reactions_path),
            "--output",
            str(output_path),
            "--batch-size",
            str(args.unimol_batch_size),
            "--dtype",
            args.unimol_dtype,
            "--compression",
            args.unimol_compression,
            "--skip-invalid-molecules",
            "--skip-invalid-reactions",
        ]
    elif modality == "chiro":
        cmd = [
            python,
            str(PROJECT_ROOT / "scripts/extract_chiro_reaction_embeddings.py"),
            "--reactions",
            str(reactions_path),
            "--output",
            str(output_path),
            "--device",
            args.chiro_device,
            "--num-workers",
            str(args.chiro_num_workers),
            "--batch-size",
            str(args.chiro_batch_size),
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if args.allow_pseudo_reactions:
        cmd.append("--allow-pseudo-reactions")
    else:
        cmd.append("--no-allow-pseudo-reactions")
    if args.force:
        cmd.append("--force")
    return cmd


def write_command_file(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(" ".join(shlex.quote(part) for part in cmd) for cmd in commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o775)


def write_markdown_report(
    path: Path,
    *,
    protocol: str,
    split: str,
    num_reactions: int,
    expected_directional: int,
    coverage: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"# ReactZyme {protocol} {split} feature coverage",
        "",
        f"- Reactions: {num_reactions}",
        f"- Expected directional IDs: {expected_directional}",
        "",
        "| Modality | H5 exists | Observed | Missing | Extra |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for modality in MODALITIES:
        if modality not in coverage:
            continue
        item = coverage[modality]
        lines.append(
            f"| {modality} | {item['exists']} | {item['observed']} | "
            f"{item['missing']} | {item['extra']} |"
        )
    for modality, item in coverage.items():
        if item["missing_examples"]:
            lines.extend(["", f"## {modality} Missing Examples", ""])
            lines.extend(f"- `{value}`" for value in item["missing_examples"])
        if item["extra_examples"]:
            lines.extend(["", f"## {modality} Extra Examples", ""])
            lines.extend(f"- `{value}`" for value in item["extra_examples"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_fragment(
    *,
    protocol_root: Path,
    features_root: Path,
    protocol: str,
    validation_split: str,
    protein_residue_embeds_path: Path,
) -> dict[str, Any]:
    protocol_dir = protocol_root / protocol
    train_features = output_paths(features_root, protocol, "train")
    validation_features = output_paths(features_root, protocol, validation_split)
    return {
        "data": {
            "train_pairs_path": relpath(protocol_dir / "train_pairs.csv"),
            "train_reactions_path": relpath(protocol_dir / "train_rxns.csv"),
            "test_pairs_path": relpath(protocol_dir / f"{validation_split}_pairs.csv"),
            "test_reactions_path": relpath(protocol_dir / f"{validation_split}_rxns.csv"),
            "protein_residue_embeds_path": relpath(protein_residue_embeds_path),
            "reaction_representation": "multimodal_reaction_attention",
            "train_reaction_t5v2_embeds_path": relpath(train_features["reactiont5v2"]),
            "validation_reaction_t5v2_embeds_path": relpath(
                validation_features["reactiont5v2"]
            ),
            "train_reaction_unimol2_embeds_path": relpath(train_features["unimol2"]),
            "validation_reaction_unimol2_embeds_path": relpath(validation_features["unimol2"]),
            "train_reaction_chiro_embeds_path": relpath(train_features["chiro"]),
            "validation_reaction_chiro_embeds_path": relpath(validation_features["chiro"]),
            "reaction_model_dim": 768,
            "reaction_unimol_dim": 768,
            "reaction_chiro_dim": 256,
            "reaction_chirality_dim": 256,
            "reaction_chienn_dim": 256,
            "reaction_use_chiro": True,
            "reaction_use_chirality": True,
            "reaction_use_chienn": True,
            "reaction_allow_missing_unimol2": True,
            "reaction_allow_missing_chiro": True,
            "reaction_allow_missing_chirality": True,
            "reaction_allow_missing_chienn": True,
            "normalize_molecule_sets_as_self_reactions": True,
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": relpath(
                protocol_dir / "candidate_ids.txt"
            ),
        }
    }


def write_config_fragments(args: argparse.Namespace) -> None:
    args.config_root.mkdir(parents=True, exist_ok=True)
    for protocol in args.protocols:
        train_test = config_fragment(
            protocol_root=args.protocol_root,
            features_root=args.features_root,
            protocol=protocol,
            validation_split="test",
            protein_residue_embeds_path=args.protein_residue_embeds_path,
        )
        (args.config_root / f"{protocol}_train_test.yaml").write_text(
            yaml.safe_dump(train_test, sort_keys=False),
            encoding="utf-8",
        )
        (args.config_root / f"{protocol}_final_test.yaml").write_text(
            yaml.safe_dump(train_test, sort_keys=False),
            encoding="utf-8",
        )


def process_split(
    *,
    args: argparse.Namespace,
    protocol: str,
    split: str,
) -> dict[str, Any]:
    reactions_path = args.protocol_root / protocol / f"{split}_rxns.csv"
    if not reactions_path.exists():
        raise FileNotFoundError(f"Missing reaction split CSV: {reactions_path}")
    reactions = read_reactions(reactions_path)
    expected_ids = expected_directional_ids(
        reactions,
        allow_pseudo_reactions=args.allow_pseudo_reactions,
    )
    paths = output_paths(args.features_root, protocol, split)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for modality in args.modalities:
        commands[modality] = command_for(
            modality,
            args=args,
            reactions_path=reactions_path,
            output_path=paths[modality],
        )
        if args.run and (args.force or not paths[modality].exists()):
            subprocess.run(commands[modality], check=True, cwd=PROJECT_ROOT)
        coverage[modality] = coverage_for(paths[modality], expected_ids)

    write_command_file(paths["dir"] / "commands.sh", list(commands.values()))
    write_markdown_report(
        paths["dir"] / "coverage_report.md",
        protocol=protocol,
        split=split,
        num_reactions=len(reactions),
        expected_directional=len(expected_ids),
        coverage=coverage,
    )
    manifest = {
        "protocol": protocol,
        "split": split,
        "reactions_csv": relpath(reactions_path),
        "num_reactions": len(reactions),
        "expected_directional_ids": len(expected_ids),
        "allow_pseudo_reactions": args.allow_pseudo_reactions,
        "outputs": {modality: relpath(paths[modality]) for modality in args.modalities},
        "commands": {
            modality: " ".join(shlex.quote(part) for part in command)
            for modality, command in commands.items()
        },
        "coverage": coverage,
    }
    (paths["dir"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.strict:
        bad = [
            modality
            for modality, item in coverage.items()
            if not item["exists"] or item["missing"] or item["extra"]
        ]
        if bad:
            raise ValueError(f"{protocol}/{split} has incomplete split coverage: {bad}")
    return manifest


def write_top_level_report(
    features_root: Path,
    manifests: list[dict[str, Any]],
) -> None:
    lines = [
        "# ReactZyme split-isolated feature coverage",
        "",
        "| Protocol | Split | Reactions | Expected directional | Modality | Exists | Missing | Extra |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for manifest in manifests:
        for modality, item in manifest["coverage"].items():
            lines.append(
                f"| {manifest['protocol']} | {manifest['split']} | "
                f"{manifest['num_reactions']} | {manifest['expected_directional_ids']} | "
                f"{modality} | {item['exists']} | {item['missing']} | {item['extra']} |"
            )
    features_root.mkdir(parents=True, exist_ok=True)
    (features_root / "coverage_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifests: list[dict[str, Any]] = []
    all_commands: list[list[str]] = []
    for protocol in args.protocols:
        for split in args.splits:
            manifest = process_split(args=args, protocol=protocol, split=split)
            manifests.append(manifest)
            all_commands.extend(
                shlex.split(command) for command in manifest["commands"].values()
            )

    args.features_root.mkdir(parents=True, exist_ok=True)
    (args.features_root / "summary_manifest.json").write_text(
        json.dumps({"splits": manifests}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_top_level_report(args.features_root, manifests)
    write_command_file(args.features_root / "run_all_split_feature_extraction.sh", all_commands)
    if not args.no_config_fragments:
        write_config_fragments(args)


if __name__ == "__main__":
    main()
