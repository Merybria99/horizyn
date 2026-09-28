"""Command-line interface; run `python -m horizyn.pipelines --help`."""

import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Delegate phase-1 flags verbatim to the shared trainer, including overrides.
    if argv and argv[0] == "train":
        from .training import main as train

        previous = sys.argv
        try:
            sys.argv = ["circe train", *argv[1:]]
            train()
        finally:
            sys.argv = previous
        return
    parser = argparse.ArgumentParser(description="V4 (default), F3, and CIRCEv2")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Train phase-1 encoders; append --help for training flags")
    commands.add_parser("list", help="List the maintained pipelines")
    export = commands.add_parser("export-refinement", help="Export phase-1 banks and training edges for V4")
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--device", default="cpu")
    export.add_argument("--batch-size", type=int, default=128)
    phase2 = commands.add_parser("refine", help="Train V4 heads on the frozen full training graph")
    phase2.add_argument("--config", type=Path, help="Phase-2 YAML; explicit CLI options override it")
    phase2.add_argument("--features", required=True, type=Path)
    phase2.add_argument("--output", required=True, type=Path)
    phase2.add_argument("--device", default="cpu")
    phase2.add_argument("--steps", type=int)
    phase2.add_argument("--identity-weight", type=float)
    phase2.add_argument("--temperature", type=float)
    phase2.add_argument("--learning-rate", type=float)
    phase2.add_argument("--seed", type=int)
    encode = commands.add_parser("encode", help="Encode both banks using a frozen model manifest")
    encode.add_argument("--manifest", required=True, type=Path)
    encode.add_argument("--reactions", required=True, type=Path, help="reaction_id,reaction_smiles CSV")
    encode.add_argument("--proteins", required=True, type=Path, help="Enzyme IDs, one per line")
    encode.add_argument("--residues", required=True, type=Path)
    encode.add_argument("--features", required=True, type=Path, help="Reaction feature directory")
    encode.add_argument("--output", required=True, type=Path)
    encode.add_argument("--device", default="cpu")
    encode.add_argument("--batch-size", type=int, default=128)
    evaluate = commands.add_parser("evaluate", help="Score aligned banks using official metric definitions")
    evaluate.add_argument("--embeddings", required=True, type=Path)
    evaluate.add_argument("--pairs", required=True, type=Path, help="reaction_id,protein_id CSV")
    evaluate.add_argument("--protocol", choices=["reactzyme", "screening"], required=True)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.command == "list":
        from .registry import PIPELINES

        for name, description in PIPELINES.items():
            print(f"{name.value}: {description}")
        return
    if args.command == "export-refinement":
        from .export import export_refinement

        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        export_refinement(args.config, args.checkpoint, args.output,
                          device=args.device, batch_size=args.batch_size)
        print(args.output / "manifest.json")
        return
    if args.command == "refine":
        from .refinement import train_refinement
        import yaml

        options = yaml.safe_load(args.config.read_text()) if args.config else {}
        if not isinstance(options, dict):
            raise ValueError("Refinement config must be a mapping")
        allowed = {"steps", "hidden", "scale", "temperature", "identity_weight",
                   "learning_rate", "weight_decay", "seed"}
        if set(options) - allowed:
            raise ValueError(f"Unknown refinement options: {sorted(set(options) - allowed)}")
        options.update({key: getattr(args, key) for key in
                        ("steps", "identity_weight", "temperature", "learning_rate", "seed")
                        if getattr(args, key) is not None})
        train_refinement(args.features, args.output, device=args.device, **options)
        print(args.output / f"step{options.get('steps', 100):04d}.pt")
        return
    if args.output.exists():
        parser.error("Choose a fresh output path")
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.command == "encode":
        from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs
        from .inference import RetrievalPipeline

        pipeline = RetrievalPipeline.from_manifest(args.manifest, args.device)
        pipeline.config.data.protein_residue_embeds_path = str(args.residues)
        pipeline.config.data.reaction_chemistry_vectors_path = str(args.features / "reaction_set_features.npz")
        task = BenchmarkTask(name="inference", task_type="retrieval", dataset="external",
                             task_label="inference", split="external", pairs=args.reactions,
                             reactions=args.reactions,
                             reaction_model_embeds_h5=args.features / "reactiont5v2.h5",
                             reaction_unimol2_embeds_h5=args.features / "unimol2.h5",
                             reaction_chiro_embeds_h5=args.features / "chiro.h5")
        with args.reactions.open(newline="") as stream:
            r_ids = [row["reaction_id"] for row in csv.DictReader(stream)]
        e_ids = args.proteins.read_text().splitlines()
        if not r_ids or not e_ids or len(set(r_ids)) != len(r_ids) or len(set(e_ids)) != len(e_ids) or any(not x for x in e_ids):
            raise ValueError("Reaction/enzyme IDs must be nonempty and unique")
        inputs = build_reaction_inputs(task, pipeline.config)
        try:
            r = pipeline.encode_reactions(inputs, r_ids, args.batch_size)
            e = pipeline.encode_enzymes(e_ids, args.batch_size)
        finally:
            if hasattr(inputs, "close"):
                inputs.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as stream:
            np.savez(stream, reactions=r.cpu().numpy(), enzymes=e.cpu().numpy(),
                     reaction_ids=np.asarray(r_ids), enzyme_ids=np.asarray(e_ids))
    else:
        from .evaluation import evaluate_embeddings

        with np.load(args.embeddings, allow_pickle=False) as data:
            r = torch.as_tensor(data["reactions"], device=args.device)
            e = torch.as_tensor(data["enzymes"], device=args.device)
            r_ids, e_ids = data["reaction_ids"].tolist(), data["enzyme_ids"].tolist()
        if len(set(r_ids)) != len(r_ids) or len(set(e_ids)) != len(e_ids) or len(r_ids) != len(r) or len(e_ids) != len(e):
            raise ValueError("Embedding IDs must be unique and aligned with their banks")
        ri, ei = {key: i for i, key in enumerate(r_ids)}, {key: i for i, key in enumerate(e_ids)}
        with args.pairs.open(newline="") as stream:
            edges = np.asarray([(ri[row["reaction_id"]], ei[row["protein_id"]]) for row in csv.DictReader(stream)], dtype=np.int64)
        result = evaluate_embeddings(r, e, edges, protocol=args.protocol, batch_size=args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
