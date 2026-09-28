#!/usr/bin/env python3
"""Evaluate CIRCE plus the biological residual over a protected alpha sweep.

Alpha must be selected on validation.  The same command can then be run on the
released test set with that single fixed alpha.  Alpha zero is evaluated from
the frozen CIRCE caches and asserted to be an exact score-level recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.biological_residual import FunctionalTokenH5Dataset  # noqa: E402
from horizyn.config import load_config  # noqa: E402
from horizyn.datasets.hdf5 import EmbedDataset  # noqa: E402
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule  # noqa: E402
from horizyn.reaction_features import build_reaction_feature_dataset  # noqa: E402
from horizyn.utils import dict_collate_fn  # noqa: E402

RANK_CUTOFFS = (1, 2, 3, 4, 5, 10, 20, 100, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--reactions", required=True, type=Path)
    parser.add_argument("--reaction-base-cache", required=True, type=Path)
    parser.add_argument("--enzyme-base-cache", required=True, type=Path)
    parser.add_argument("--functional-token-cache", required=True, type=Path)
    parser.add_argument("--candidate-ids", required=True, type=Path)
    parser.add_argument(
        "--split-name",
        choices=("train", "validation"),
        default="validation",
        help="Select the matching split-specific raw reaction HDF5 paths",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.075, 0.1])
    parser.add_argument(
        "--max-directional-mrr-drop",
        type=float,
        default=0.005,
        help="Validation guard relative to alpha=0 for each retrieval direction",
    )
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection-metric",
        choices=("first_positive_mrr", "reactzyme_mrr"),
        default="first_positive_mrr",
    )
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--selection-from",
        type=Path,
        help="Validation JSON fixing test alpha; never select on test scores",
    )
    parser.add_argument("--per-query-output", type=Path)
    parser.add_argument(
        "--train-pairs", type=Path, help="Training-only association counts for diagnostics"
    )
    return parser.parse_args()


def _read_ids(path: Path) -> list[str]:
    return list(
        dict.fromkeys(
            line.strip().split(",")[0].split()[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )


def _read_positive_graph(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    graph: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"reaction_id", "protein_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pair CSV is missing columns {sorted(missing)}: {path}")
        for row in reader:
            query_id = f"{str(row['reaction_id']).strip()}_f"
            target_id = str(row["protein_id"]).strip()
            if query_id and target_id and target_id not in graph[query_id]:
                graph[query_id].append(target_id)
    return sorted(graph), dict(graph)


def _stack_cache(dataset: Any, ids: list[str], device: torch.device) -> torch.Tensor:
    missing = [value for value in ids if value not in set(dataset.keys)]
    if missing:
        raise ValueError(
            f"Cache is missing {len(missing)} requested IDs; examples: {', '.join(missing[:5])}"
        )
    return torch.stack([dataset[value] for value in ids]).float().to(device)


def validate_exact_base_cache(path: Path) -> None:
    """Reject reduced-precision caches for the CIRCE preservation baseline."""

    with h5py.File(path, "r") as handle:
        if "vectors" not in handle:
            raise ValueError(f"Base cache has no vectors dataset: {path}")
        if str(handle["vectors"].dtype) != "float32":
            raise ValueError(
                f"Exact CIRCE evaluation requires a float32 cache, got "
                f"{handle['vectors'].dtype} at {path}"
            )
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Exact CIRCE cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest.get("signature", {})
    if signature.get("precision") != "32" or signature.get("output_dtype") != "float32":
        raise ValueError(
            "Exact CIRCE evaluation requires cache extraction with "
            "--precision 32 --output-dtype float32"
        )


def _mean_metrics(values: dict[str, list[float]]) -> dict[str, float]:
    return {name: (sum(items) / len(items) if items else 0.0) for name, items in values.items()}


def evaluate_score_matrix(
    scores: torch.Tensor,
    query_ids: list[str],
    candidate_ids: list[str],
    query_to_targets: dict[str, list[str]],
    *,
    per_query: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute both retrieval directions from one reaction-by-enzyme matrix."""

    # Ranking emits many scalar diagnostics. Transfer once instead of forcing
    # a CUDA synchronization for every metric of every individual query.
    scores = scores.detach().cpu()
    target_to_index = {value: index for index, value in enumerate(candidate_ids)}
    query_to_index = {value: index for index, value in enumerate(query_ids)}
    r2e_values: dict[str, list[float]] = defaultdict(list)
    target_to_queries: dict[str, list[str]] = defaultdict(list)
    for row, query_id in enumerate(query_ids):
        target_indices = [
            target_to_index[target]
            for target in query_to_targets.get(query_id, [])
            if target in target_to_index
        ]
        if not target_indices:
            continue
        metrics = rank_metrics(scores[row], target_indices)
        for key, value in metrics.items():
            r2e_values[key].append(value)
        if per_query is not None:
            per_query.append(
                {
                    "direction": "reaction_to_enzyme",
                    "query_id": query_id,
                    "known_positive_count": len(target_indices),
                    **metrics,
                }
            )
        for target in query_to_targets[query_id]:
            if target in target_to_index:
                target_to_queries[target].append(query_id)

    e2r_values: dict[str, list[float]] = defaultdict(list)
    for target_id, positive_queries in target_to_queries.items():
        positive_indices = [query_to_index[value] for value in positive_queries]
        metrics = rank_metrics(scores[:, target_to_index[target_id]], positive_indices)
        for key, value in metrics.items():
            e2r_values[key].append(value)
        if per_query is not None:
            per_query.append(
                {
                    "direction": "enzyme_to_reaction",
                    "query_id": target_id,
                    "known_positive_count": len(positive_indices),
                    **metrics,
                }
            )
    r2e = _mean_metrics(r2e_values)
    e2r = _mean_metrics(e2r_values)
    result = {
        **{f"reaction_to_enzyme/{key}": value for key, value in r2e.items()},
        **{f"enzyme_to_reaction/{key}": value for key, value in e2r.items()},
        "reaction_to_enzyme/num_queries": len(r2e_values["mrr"]),
        "enzyme_to_reaction/num_queries": len(e2r_values["mrr"]),
    }
    for metric in ("mrr", "first_positive_mrr", "reactzyme_mrr", "top_1"):
        result[f"balanced_{metric}"] = 0.5 * (r2e.get(metric, 0.0) + e2r.get(metric, 0.0))
    return result


def rank_metrics(scores: torch.Tensor, positives: list[int]) -> dict[str, float]:
    """One stable sort per query; mean_rank averages all known positive ranks."""
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite retrieval scores")
    order = torch.argsort(scores, descending=True, stable=True)
    relevant = torch.zeros_like(scores, dtype=torch.bool)
    relevant[positives] = True
    ranks = (relevant[order].nonzero().flatten() + 1).float()
    if not len(ranks):
        raise ValueError("Query has no valid positives")
    result = {
        "mrr": float(1 / ranks[0]),
        "first_positive_mrr": float(1 / ranks[0]),
        "reactzyme_mrr": float((1 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "first_positive_rank": float(ranks[0]),
        "r_precision": float((ranks <= len(ranks)).float().mean()),
        "avg_precision": float(
            (torch.arange(1, len(ranks) + 1, device=ranks.device) / ranks).mean()
        ),
    }
    for k in RANK_CUTOFFS:
        result[f"top_{k}"] = float(ranks[0] <= k)
        # Fixed-k precision: denominator is k, including when the pool is smaller.
        result[f"precision_at_{k}"] = float((ranks <= k).sum()) / k
    return result


def select_alpha(
    results: dict[str, dict[str, float]], metric: str, tolerance: float
) -> tuple[float, list[float]]:
    eligible = [
        float(alpha)
        for alpha, values in results.items()
        if all(
            values[f"{direction}/{metric}"] >= results["0"][f"{direction}/{metric}"] - tolerance
            for direction in ("enzyme_to_reaction", "reaction_to_enzyme")
        )
    ]
    # Prefer the smaller correction on exact ties, independently of grid ordering.
    best = max(eligible, key=lambda alpha: (results[f"{alpha:g}"][f"balanced_{metric}"], -alpha))
    return best, sorted(eligible)


def fixed_test_alpha(path: Path, checkpoint: Path, metric: str) -> float:
    selection = json.loads(path.read_text())
    if selection.get("evaluation_split") != "validation":
        raise ValueError("Test selection must come from a validation evaluation")
    if selection.get("selection_metric") != f"balanced_{metric}":
        raise ValueError("Validation/test selection metrics disagree")
    if Path(selection["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Validation/test checkpoints disagree")
    alpha = float(selection["best_alpha"])
    if f"{alpha:g}" not in selection["alpha_results"]:
        raise ValueError("Selected alpha was not evaluated on validation")
    return alpha


def training_degrees(path: Path | None) -> dict[tuple[str, str], int]:
    if path is None:
        return {}
    graph = _read_positive_graph(path)[1]
    degrees = {
        ("reaction_to_enzyme", reaction): len(enzymes) for reaction, enzymes in graph.items()
    }
    for enzymes in graph.values():
        for enzyme in enzymes:
            key = ("enzyme_to_reaction", enzyme)
            degrees[key] = degrees.get(key, 0) + 1
    return degrees


def _encode_functional_targets(
    module: ProteinPooledLitModule,
    dataset: FunctionalTokenH5Dataset,
    target_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = module.model.biological_residual
    if residual is None:
        raise ValueError("Checkpoint does not contain an enabled biological residual")
    tokens_out: list[torch.Tensor] = []
    pooled_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor] = []
    scores_out: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(target_ids), batch_size), desc="Encoding functional residues"
        ):
            ids = target_ids[start : start + batch_size]
            samples = [dataset[value] for value in ids]
            tokens = torch.stack([sample["biological_residue_tokens"] for sample in samples]).to(
                device
            )
            mask = torch.stack([sample["biological_residue_mask"] for sample in samples]).to(device)
            sleec = torch.stack([sample["biological_sleec_scores"] for sample in samples]).to(
                device
            )
            encoded, pooled = residual.enzyme_encoder(tokens, mask, sleec)
            # Preserve the learned residual exactly.  The raw functional-token
            # cache is already compact; quantizing the encoder output again here
            # would make offline evaluation differ unnecessarily from training.
            tokens_out.append(encoded.float().cpu())
            pooled_out.append(pooled.float().cpu())
            masks_out.append(mask.cpu())
            scores_out.append(sleec.float().cpu())
    return (
        torch.cat(tokens_out),
        torch.cat(masks_out),
        torch.cat(pooled_out),
        torch.cat(scores_out),
    )


def _biological_scores(
    module: ProteinPooledLitModule,
    reaction_dataset: Any,
    query_ids: list[str],
    encoded_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    query_batch_size: int,
    target_batch_size: int,
) -> torch.Tensor:
    residual = module.model.biological_residual
    if residual is None:
        raise ValueError("Checkpoint does not contain an enabled biological residual")
    target_tokens, target_masks, target_pooled, target_sleec = encoded_targets
    output = torch.empty((len(query_ids), target_tokens.shape[0]), dtype=torch.float32)
    with torch.inference_mode():
        for query_start in tqdm(
            range(0, len(query_ids), query_batch_size), desc="Scoring biological residual"
        ):
            batch_ids = query_ids[query_start : query_start + query_batch_size]
            batch = dict_collate_fn([reaction_dataset[value] for value in batch_ids])
            reaction_inputs = module._cached_biological_query_inputs(
                {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }
            )
            reaction_tokens, reaction_mask, reaction_pooled = residual.reaction_encoder(
                reactant_embeddings=reaction_inputs["reactant_embeddings"],
                product_embeddings=reaction_inputs["product_embeddings"],
                reactant_padding_mask=reaction_inputs.get("reactant_padding_mask"),
                product_padding_mask=reaction_inputs.get("product_padding_mask"),
                reactant_chirality_embeddings=reaction_inputs.get("reactant_chirality_embeddings"),
                product_chirality_embeddings=reaction_inputs.get("product_chirality_embeddings"),
                reactant_chirality_padding_mask=reaction_inputs.get(
                    "reactant_chirality_padding_mask"
                ),
                product_chirality_padding_mask=reaction_inputs.get(
                    "product_chirality_padding_mask"
                ),
                has_chirality=reaction_inputs.get(
                    "has_chirality",
                    reaction_inputs.get("has_chiro", reaction_inputs.get("has_chienn")),
                ),
            )
            chunks: list[torch.Tensor] = []
            availability = reaction_inputs.get("has_unimol2")
            for target_start in range(0, target_tokens.shape[0], target_batch_size):
                target_end = min(target_start + target_batch_size, target_tokens.shape[0])
                chunk = residual.scorer(
                    reaction_tokens,
                    reaction_mask,
                    reaction_pooled,
                    target_tokens[target_start:target_end].float().to(device),
                    target_masks[target_start:target_end].to(device),
                    target_pooled[target_start:target_end].float().to(device),
                    target_sleec[target_start:target_end].float().to(device),
                )
                if availability is not None:
                    chunk = chunk * availability.to(chunk.dtype).unsqueeze(1)
                chunks.append(chunk.cpu())
            output[query_start : query_start + len(batch_ids)] = torch.cat(chunks, dim=1)
    return output


def main() -> None:
    args = parse_args()
    selected_test_alpha = None
    if args.evaluation_split == "test":
        if args.selection_from is None:
            raise ValueError("Test evaluation requires --selection-from validation.json")
        selected_test_alpha = fixed_test_alpha(
            args.selection_from, args.checkpoint, args.selection_metric
        )
        args.alphas = sorted({0.0, selected_test_alpha})
    elif args.selection_from is not None:
        raise ValueError("--selection-from is only valid for test evaluation")
    if args.query_batch_size <= 0 or args.target_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.max_directional_mrr_drop < 0.0:
        raise ValueError("--max-directional-mrr-drop must be non-negative")
    if 0.0 not in args.alphas:
        raise ValueError("The alpha sweep must include 0 for the CIRCE preservation guard")
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    config = load_config(str(args.config))
    module = ProteinPooledLitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval().to(device)
    residual = module.model.biological_residual
    if residual is None:
        raise ValueError("Checkpoint has no biological residual")
    for alpha in args.alphas:
        if abs(alpha) > residual.fusion.max_alpha:
            raise ValueError(
                f"alpha={alpha} exceeds checkpoint max alpha={residual.fusion.max_alpha}"
            )

    query_ids, query_to_targets = _read_positive_graph(args.pairs)
    candidate_ids = _read_ids(args.candidate_ids)
    candidate_set = set(candidate_ids)
    missing_positive_targets = sorted(
        {
            target
            for targets in query_to_targets.values()
            for target in targets
            if target not in candidate_set
        }
    )
    if missing_positive_targets:
        raise ValueError(
            f"Candidate set omits {len(missing_positive_targets)} positive proteins; "
            f"examples: {', '.join(missing_positive_targets[:5])}"
        )

    validate_exact_base_cache(args.reaction_base_cache)
    validate_exact_base_cache(args.enzyme_base_cache)
    reaction_base = _stack_cache(
        EmbedDataset(str(args.reaction_base_cache), in_memory=True), query_ids, device
    )
    enzyme_base = _stack_cache(
        EmbedDataset(str(args.enzyme_base_cache), in_memory=True), candidate_ids, device
    )
    base_scores = F.normalize(reaction_base, dim=-1) @ F.normalize(enzyme_base, dim=-1).t()

    functional_dataset = FunctionalTokenH5Dataset(args.functional_token_cache)
    missing_functional = [
        value for value in candidate_ids if value not in set(functional_dataset.keys)
    ]
    if missing_functional:
        raise ValueError(
            f"Functional cache omits {len(missing_functional)} candidates; "
            f"examples: {', '.join(missing_functional[:5])}"
        )
    encoded_targets = _encode_functional_targets(
        module, functional_dataset, candidate_ids, device, args.target_batch_size
    )
    reaction_dataset = build_reaction_feature_dataset(
        reactions_path=args.reactions,
        config=config,
        bidirectional=True,
        split_name=args.split_name,
    )
    missing_reactions = [value for value in query_ids if value not in set(reaction_dataset.keys)]
    if missing_reactions:
        raise ValueError(
            f"Raw reaction features omit {len(missing_reactions)} queries; "
            f"examples: {', '.join(missing_reactions[:5])}"
        )
    local_scores = _biological_scores(
        module,
        reaction_dataset,
        query_ids,
        encoded_targets,
        device,
        args.query_batch_size,
        args.target_batch_size,
    ).to(device)

    alpha_results: dict[str, dict[str, float]] = {}
    query_rows: list[dict[str, Any]] = []
    exact_alpha_zero = False
    for alpha in args.alphas:
        residual.fusion.set_alpha(alpha)
        fused_scores = residual.fusion(base_scores, local_scores)
        if alpha == 0.0:
            exact_alpha_zero = fused_scores.data_ptr() == base_scores.data_ptr() and torch.equal(
                fused_scores, base_scores
            )
            if not exact_alpha_zero:
                raise RuntimeError("alpha=0 failed exact CIRCE score recovery")
        rows: list[dict[str, Any]] | None = [] if args.per_query_output else None
        alpha_results[f"{alpha:g}"] = evaluate_score_matrix(
            fused_scores, query_ids, candidate_ids, query_to_targets, per_query=rows
        )
        if rows is not None:
            query_rows.extend({"alpha": alpha, **row} for row in rows)

    if selected_test_alpha is None:
        best_alpha, eligible_alphas = select_alpha(
            alpha_results, args.selection_metric, args.max_directional_mrr_drop
        )
    else:
        best_alpha, eligible_alphas = selected_test_alpha, []
    if args.per_query_output:
        degrees = training_degrees(args.train_pairs)
        reaction_flags = {}
        for query_id in query_ids:
            sample = reaction_dataset[query_id]
            reaction_flags[query_id] = {
                "has_unimol2": bool(sample.get("has_unimol2", True)),
                "has_chiro": bool(sample.get("has_chirality", False)),
            }
        for row in query_rows:
            row["train_known_association_count"] = (
                degrees.get((row["direction"], row["query_id"]), 0) if args.train_pairs else ""
            )
            row.update(reaction_flags.get(row["query_id"], {"has_unimol2": "", "has_chiro": ""}))
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_csv = args.per_query_output.with_suffix(f".tmp.{os.getpid()}")
        with temporary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
            writer.writeheader()
            writer.writerows(query_rows)
        os.replace(temporary_csv, args.per_query_output)
    result = {
        "schema_version": "biological_residual_alpha_sweep_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "pairs": str(args.pairs.resolve()),
        "candidate_count": len(candidate_ids),
        "reaction_count": len(query_ids),
        "alpha_zero_exact_circe_recovery": exact_alpha_zero,
        "selection_metric": f"balanced_{args.selection_metric}",
        "evaluation_split": args.evaluation_split,
        "selection_from": str(args.selection_from.resolve()) if args.selection_from else None,
        "max_directional_mrr_drop": args.max_directional_mrr_drop,
        "eligible_alphas": [float(value) for value in eligible_alphas],
        "best_alpha": float(best_alpha),
        "alpha_results": alpha_results,
        "rank_definition": "mean_rank averages all known positive ranks per query, then queries",
        "precision_definition": "known positives in top k divided by k",
        "tie_policy": "stable descending sort; original candidate order breaks ties",
        "checkpoint_signature": {
            "size": args.checkpoint.stat().st_size,
            "mtime_ns": args.checkpoint.stat().st_mtime_ns,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
