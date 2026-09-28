#!/usr/bin/env python3
"""
Evaluate a reaction-conditioned residue-pooling Horizyn checkpoint.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.metrics import average_precision, r_precision, top_k_hit_rate
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.reaction_conditioned_lightning_module import ReactionConditionedLitModule
from horizyn.utils import residue_collate_fn, unimol2_reaction_collate_fn


def build_bidirectional_pairs(val_pairs: CSVDataset) -> BaseDataset:
    augmented_keys = []
    augmented_data = []
    for pair_key in val_pairs.keys:
        pair_data = val_pairs[pair_key]
        query_id = pair_data["query_id"]
        target_id = pair_data["target_id"]
        augmented_keys.append(f"{pair_key}_f")
        augmented_data.append({"query_id": f"{query_id}_f", "target_id": target_id})
        augmented_keys.append(f"{pair_key}_r")
        augmented_data.append({"query_id": f"{query_id}_r", "target_id": target_id})
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_bidirectional_reactions(reactions: CSVDataset) -> BaseDataset:
    augmented_keys = []
    augmented_data = []
    for rxn_id in reactions.keys:
        rxn_data = reactions[rxn_id]
        smiles = rxn_data["reaction_smiles"]
        augmented_keys.append(f"{rxn_id}_f")
        augmented_data.append({"reaction_smiles": smiles})
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) == 2:
                augmented_keys.append(f"{rxn_id}_r")
                augmented_data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_reaction_fingerprints(config):
    return build_reaction_feature_dataset(
        reactions_path=config.data.test_reactions_path,
        config=config,
        bidirectional=True,
    )


def build_query_inputs(reaction_features, query_ids: list[str], device: str):
    samples = [reaction_features[query_id] for query_id in query_ids]
    if isinstance(samples[0], dict):
        batch = unimol2_reaction_collate_fn(samples)
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return torch.stack(samples).to(device, non_blocking=True)


def format_results_table(results: dict) -> str:
    lines = [
        "=" * 70,
        "REACTION-CONDITIONED HORIZYN EVALUATION RESULTS",
        "=" * 70,
        "",
        f"Checkpoint: {results.get('checkpoint', 'N/A')}",
        f"Config: {results.get('config', 'N/A')}",
        f"Queries evaluated: {results.get('num_queries', 'N/A')}",
        f"Screening set size: {results.get('num_targets', 'N/A')}",
        "",
        "-" * 70,
        "RETRIEVAL METRICS",
        "-" * 70,
        "",
        f"{'Metric':<20} {'Value':<15} {'Value (%)':<15}",
        "-" * 50,
    ]
    for display_name, key in [
        ("Top-1 HR", "top_1"),
        ("Top-10 HR", "top_10"),
        ("Top-100 HR", "top_100"),
        ("Top-1000 HR", "top_1000"),
        ("R-precision", "r_precision"),
        ("Avg. precision", "avg_precision"),
    ]:
        value = results.get(key, 0.0)
        lines.append(f"{display_name:<20} {value:.4f}         {value * 100:.1f}%")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str,
    batch_size: int,
    target_chunk_size: int,
) -> dict:
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print(f"Loading checkpoint from: {checkpoint_path}")
    module = ReactionConditionedLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module.eval()
    module.to(device)

    print("Setting up validation data...")
    val_pairs = CSVDataset(
        file_path=config.data.test_pairs_path,
        key_column="pr_id",
        columns=["reaction_id", "protein_id"],
        rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
    )
    val_pairs = build_bidirectional_pairs(val_pairs)

    query_to_targets = defaultdict(list)
    for pair_key in val_pairs.keys:
        pair = val_pairs[pair_key]
        query_to_targets[pair["query_id"]].append(pair["target_id"])

    reaction_fps = build_reaction_fingerprints(config)

    residue_dataset = ResidueEmbedDataset(
        file_path=config.data.protein_residue_embeds_path,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )
    target_id_to_idx = {target_id: idx for idx, target_id in enumerate(residue_dataset.keys)}

    available_query_ids = set(reaction_fps.keys)
    filtered_query_to_targets = defaultdict(list)
    missing_query_pairs = 0
    missing_target_pairs = 0
    for query_id, target_ids in query_to_targets.items():
        if query_id not in available_query_ids:
            missing_query_pairs += len(target_ids)
            continue
        for target_id in target_ids:
            if target_id in target_id_to_idx:
                filtered_query_to_targets[query_id].append(target_id)
            else:
                missing_target_pairs += 1

    query_to_targets = filtered_query_to_targets
    unique_query_ids = sorted(query_to_targets.keys())
    if missing_query_pairs or missing_target_pairs:
        print(
            "Filtered validation pairs unavailable in evaluation data: "
            f"missing_queries={missing_query_pairs}, "
            f"missing_targets={missing_target_pairs}"
        )

    metric_results = defaultdict(list)
    print(f"Screening set size: {len(residue_dataset)} proteins")
    print(f"Evaluating {len(unique_query_ids)} queries")

    with torch.no_grad():
        for query_start in tqdm(
            range(0, len(unique_query_ids), batch_size),
            desc="Evaluating query batches",
        ):
            query_end = min(query_start + batch_size, len(unique_query_ids))
            query_ids = unique_query_ids[query_start:query_end]
            query_inputs = build_query_inputs(reaction_fps, query_ids, device)
            score_chunks = []

            for target_start in range(0, len(residue_dataset), target_chunk_size):
                target_end = min(target_start + target_chunk_size, len(residue_dataset))
                target_keys = residue_dataset.keys[target_start:target_end]
                samples = []
                for target_id in target_keys:
                    sample = residue_dataset[target_id]
                    sample["target_id"] = target_id
                    samples.append(sample)
                target_batch = residue_collate_fn(samples)
                residues = target_batch["residue_embeddings"].to(device)
                mask = target_batch["residue_padding_mask"].to(device)
                scores = module.model.score_matrix(query_inputs, residues, mask)
                score_chunks.append(scores.cpu())

            all_scores = torch.cat(score_chunks, dim=1)

            for row_idx, query_id in enumerate(query_ids):
                target_indices = [
                    target_id_to_idx[target_id]
                    for target_id in query_to_targets[query_id]
                    if target_id in target_id_to_idx
                ]
                if not target_indices:
                    continue
                target_idx = torch.tensor(target_indices, dtype=torch.long)
                scores = all_scores[row_idx]
                metric_results["top_1"].append(top_k_hit_rate(scores, target_idx, k=1).item())
                metric_results["top_10"].append(top_k_hit_rate(scores, target_idx, k=10).item())
                metric_results["top_100"].append(top_k_hit_rate(scores, target_idx, k=100).item())
                metric_results["top_1000"].append(top_k_hit_rate(scores, target_idx, k=1000).item())
                metric_results["r_precision"].append(r_precision(scores, target_idx).item())
                metric_results["avg_precision"].append(average_precision(scores, target_idx).item())

    results = {
        metric_name: sum(values) / len(values) if values else 0.0
        for metric_name, values in metric_results.items()
    }
    results["num_queries"] = len(metric_results["top_1"])
    results["num_targets"] = len(residue_dataset)
    results["checkpoint"] = checkpoint_path
    results["config"] = config_path
    results["target_chunk_size"] = target_chunk_size
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reaction-conditioned Horizyn checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Query batch size")
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=64,
        help="Number of target proteins to score per chunk",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
        target_chunk_size=args.target_chunk_size,
    )
    print()
    print(format_results_table(results))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
