#!/usr/bin/env python3
"""Mine BioFP-neighborhood hard-negative enzyme targets for R->E training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ("center", "cofactor", "transition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--candidate-ids", required=True, type=Path)
    parser.add_argument("--biofp-targets", required=True, type=Path)
    parser.add_argument("--top-false-positives", type=Path, default=None)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-parquet", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--max-negatives-per-query", type=int, default=256)
    parser.add_argument("--max-candidates-per-signature", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reaction_id = str(row.get("reaction_id", "")).strip()
            protein_id = str(row.get("protein_id", "")).strip()
            if reaction_id and protein_id:
                pairs.append((reaction_id, protein_id))
    return pairs


def read_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                ids.append(value.split(",")[0].split()[0])
    return ids


def id_aliases(protein_id: str) -> list[str]:
    protein_id = str(protein_id)
    aliases = [protein_id]
    if "_" in protein_id:
        _, suffix = protein_id.split("_", 1)
        for prefix in ("prot", "uprot", "nr90"):
            alias = f"{prefix}_{suffix}"
            if alias not in aliases:
                aliases.append(alias)
    return aliases


class BioFPTargets:
    def __init__(self, path: Path, threshold: float) -> None:
        payload = np.load(path, allow_pickle=True)
        self.ids = [str(value) for value in payload["ids"]]
        self.index: dict[str, int] = {}
        for idx, protein_id in enumerate(self.ids):
            for alias in id_aliases(protein_id):
                self.index.setdefault(alias, idx)
        self.targets = {
            family: payload[f"{family}_targets"].astype(np.float32)
            for family in FAMILIES
        }
        self.masks = {family: payload[f"{family}_mask"].astype(bool) for family in FAMILIES}
        self.threshold = float(threshold)

    def signature(self, protein_id: str, family: str) -> tuple[int, ...] | None:
        idx = None
        for alias in id_aliases(protein_id):
            if alias in self.index:
                idx = self.index[alias]
                break
        if idx is None or not bool(self.masks[family][idx]):
            return None
        return tuple(int(i) for i in np.flatnonzero(self.targets[family][idx] > self.threshold))

    def combined_signature(self, protein_id: str) -> tuple[tuple[int, ...], ...] | None:
        parts = []
        for family in FAMILIES:
            signature = self.signature(protein_id, family)
            if signature is None:
                return None
            parts.append(signature)
        return tuple(parts)


def add_candidate(
    rows: list[dict[str, Any]],
    seen: set[str],
    *,
    reaction_id: str,
    protein_id: str,
    source: str,
    priority: int,
    positives: set[str],
    max_negatives: int,
) -> None:
    if len(seen) >= max_negatives:
        return
    if protein_id in positives or protein_id in seen:
        return
    seen.add(protein_id)
    rows.append(
        {
            "reaction_id": reaction_id,
            "hard_negative_protein_id": protein_id,
            "source": source,
            "priority": priority,
        }
    )


def deterministic_bucket_window(
    values: list[str],
    *,
    salt: str,
    limit: int,
) -> list[str]:
    if limit <= 0 or len(values) <= limit:
        return values
    offset = int(stable_hash(salt)[:12], 16) % len(values)
    end = offset + limit
    if end <= len(values):
        return values[offset:end]
    return values[offset:] + values[: end - len(values)]


def read_top_false_positive_rows(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception:
        import pandas as pd

        frame = pd.read_csv(path)
    required = {"reaction_id", "protein_id", "is_positive"}
    if not required.issubset(frame.columns):
        return {}
    result: dict[str, list[str]] = defaultdict(list)
    for row in frame[["reaction_id", "protein_id", "is_positive"]].itertuples(index=False):
        if bool(row.is_positive):
            continue
        reaction_id = str(row.reaction_id)
        protein_id = str(row.protein_id)
        if protein_id not in result[reaction_id]:
            result[reaction_id].append(protein_id)
    return result


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.train_pairs)
    candidate_ids = [str(protein_id) for protein_id in read_ids(args.candidate_ids)]
    candidate_ids = sorted(set(candidate_ids), key=lambda value: stable_hash(f"{args.seed}:cand:{value}"))
    candidate_set = set(candidate_ids)
    biofp = BioFPTargets(args.biofp_targets, threshold=args.threshold)
    top_false_positive_map = read_top_false_positive_rows(args.top_false_positives)

    reaction_to_positives: dict[str, set[str]] = defaultdict(set)
    for reaction_id, protein_id in pairs:
        reaction_to_positives[reaction_id].add(str(protein_id))

    family_maps: dict[str, dict[tuple[int, ...], list[str]]] = {
        family: defaultdict(list) for family in FAMILIES
    }
    combined_map: dict[tuple[tuple[int, ...], ...], list[str]] = defaultdict(list)
    for protein_id in candidate_ids:
        for family in FAMILIES:
            signature = biofp.signature(protein_id, family)
            if signature:
                family_maps[family][signature].append(protein_id)
        combined = biofp.combined_signature(protein_id)
        if combined is not None and any(part for part in combined):
            combined_map[combined].append(protein_id)
    for family in FAMILIES:
        for signature, values in family_maps[family].items():
            values.sort(key=lambda value: stable_hash(f"{args.seed}:bucket:{family}:{value}"))
    for signature, values in combined_map.items():
        values.sort(key=lambda value: stable_hash(f"{args.seed}:bucket:combined:{value}"))

    mined_rows: list[dict[str, Any]] = []
    query_to_negatives: dict[str, list[str]] = {}
    source_counts: dict[str, int] = defaultdict(int)
    missing_biofp_queries = 0
    for reaction_id in sorted(reaction_to_positives):
        positives = reaction_to_positives[reaction_id]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for protein_id in top_false_positive_map.get(reaction_id, []):
            if protein_id in candidate_set:
                add_candidate(
                    rows,
                    seen,
                    reaction_id=reaction_id,
                    protein_id=protein_id,
                    source="top_false_positive",
                    priority=0,
                    positives=positives,
                    max_negatives=args.max_negatives_per_query,
                )

        positive_combined = [
            signature
            for protein_id in positives
            if (signature := biofp.combined_signature(protein_id)) is not None
            and any(part for part in signature)
        ]
        if not positive_combined:
            missing_biofp_queries += 1
        for signature in sorted(set(positive_combined), key=repr):
            bucket = deterministic_bucket_window(
                combined_map.get(signature, []),
                salt=f"{args.seed}:combined:{reaction_id}",
                limit=args.max_candidates_per_signature,
            )
            for protein_id in bucket:
                add_candidate(
                    rows,
                    seen,
                    reaction_id=reaction_id,
                    protein_id=protein_id,
                    source="same_combined_biofp",
                    priority=1,
                    positives=positives,
                    max_negatives=args.max_negatives_per_query,
                )
                if len(seen) >= args.max_negatives_per_query:
                    break
            if len(seen) >= args.max_negatives_per_query:
                break

        for family_idx, family in enumerate(FAMILIES, start=2):
            positive_signatures = [
                signature
                for protein_id in positives
                if (signature := biofp.signature(protein_id, family)) is not None and signature
            ]
            for signature in sorted(set(positive_signatures), key=repr):
                bucket = deterministic_bucket_window(
                    family_maps[family].get(signature, []),
                    salt=f"{args.seed}:{family}:{reaction_id}",
                    limit=args.max_candidates_per_signature,
                )
                for protein_id in bucket:
                    add_candidate(
                        rows,
                        seen,
                        reaction_id=reaction_id,
                        protein_id=protein_id,
                        source=f"same_{family}_biofp",
                        priority=family_idx,
                        positives=positives,
                        max_negatives=args.max_negatives_per_query,
                    )
                if len(seen) >= args.max_negatives_per_query:
                    break
            if len(seen) >= args.max_negatives_per_query:
                break

        if rows:
            rows = rows[: args.max_negatives_per_query]
            mined_rows.extend(rows)
            query_to_negatives[reaction_id] = [
                row["hard_negative_protein_id"] for row in rows
            ]
            for row in rows:
                source_counts[str(row["source"])] += 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "query_to_negatives": query_to_negatives,
                "metadata": {
                    "source_train_pairs": str(args.train_pairs),
                    "source_candidate_ids": str(args.candidate_ids),
                    "source_biofp_targets": str(args.biofp_targets),
                    "source_top_false_positives": (
                        None if args.top_false_positives is None else str(args.top_false_positives)
                    ),
                    "max_negatives_per_query": args.max_negatives_per_query,
                    "max_candidates_per_signature": args.max_candidates_per_signature,
                    "seed": args.seed,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    import pandas as pd

    pd.DataFrame(mined_rows).to_parquet(args.out_parquet, index=False)
    counts = [len(values) for values in query_to_negatives.values()]
    report = {
        "train_pairs": len(pairs),
        "train_queries": len(reaction_to_positives),
        "candidate_count": len(candidate_ids),
        "queries_with_negatives": len(query_to_negatives),
        "queries_without_negatives": len(reaction_to_positives) - len(query_to_negatives),
        "queries_without_positive_biofp": missing_biofp_queries,
        "hard_negative_rows": len(mined_rows),
        "mean_negatives_per_query": float(np.mean(counts)) if counts else 0.0,
        "median_negatives_per_query": float(np.median(counts)) if counts else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
    }
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
