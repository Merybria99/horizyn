#!/usr/bin/env python3
"""Evaluate every declared frozen method on the six official ReactZyme cells.

Score manifest: {"freeze":{"path":"frozen_recipe.json","sha256":"..."},
 "baseline_method":"F3", "methods":[{"label":"F3","primary":false,
 "splits":{"reaction_smi":{"path":".../scores.npz","score_key":"baseline"},
 "enzyme_smi":{...},"time":{...}}}, {"label":"combined_seed42",
 "primary":true,"seed":42,"splits":{...}}]}.

Paths are relative to the score manifest. Each entry needs a native prediction
receipt, or explicit score and catalog SHA256 values. This program reports all
methods and performs no selection, fitting, calibration, or checkpoint search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_export import atomic_json, identity, digest, read_pairs
from scripts.generalization_metrics import evaluate_scores
from scripts.generalization_test_export import EXPECTED

SPLITS = ("reaction_smi", "enzyme_smi", "time")
DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")
METRICS = ("reactzyme_mrr", "first_positive_mrr", "top_1", "top_5", "top_10")


def resolve(directory, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def bootstrap_difference(method, baseline, replicates, seed):
    if not np.array_equal(method["query_index"], baseline["query_index"]):
        raise ValueError("Paired contrasts require identical query IDs")
    count = len(method["query_index"])
    if not count:
        raise ValueError("An official evaluation cell cannot have zero queries")
    def column(values, key):
        return 1 / np.asarray(values["first_rank"], np.float64) if key == "first_positive_mrr" else np.asarray(values[key], np.float64)
    differences = np.stack([column(method, key) - column(baseline, key) for key in METRICS], axis=1)
    samples = np.empty((replicates, len(METRICS)), np.float64)
    rng = np.random.default_rng(seed)
    # Bounded memory even for the 14,688 enzyme-query cell. The RNG restarts
    # with the same cell seed for every method so all contrasts are paired.
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        draws = rng.integers(0, count, size=(stop - start, count))
        samples[start:stop] = differences[draws].mean(1)
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {key: dict(delta=float(differences[:, i].mean()), lower_95=float(bounds[0, i]),
                     upper_95=float(bounds[1, i]), queries=count, replicates=replicates,
                     seed=seed, unit="retrieval query") for i, key in enumerate(METRICS)}


def reaction_cluster_bootstrap(method, baseline, group_by_enzyme, replicates, seed):
    """Resample reaction groups, retaining the pooled enzyme-query estimand."""
    if not np.array_equal(method["query_index"], baseline["query_index"]):
        raise ValueError("Cluster contrasts require identical enzyme query IDs")
    query_ids = method["query_index"]
    groups, inverse = np.unique(group_by_enzyme[query_ids], return_inverse=True)
    if not len(groups) or (groups < 0).any():
        raise ValueError("Every enzyme query requires a positive reaction group")
    counts = np.bincount(inverse).astype(np.float64)
    differences = np.stack([
        (1 / np.asarray(method["first_rank"], np.float64) - 1 / np.asarray(baseline["first_rank"], np.float64))
        if key == "first_positive_mrr" else np.asarray(method[key], np.float64) - np.asarray(baseline[key], np.float64)
        for key in METRICS], axis=1)
    sums = np.stack([np.bincount(inverse, weights=differences[:, i]) for i in range(len(METRICS))], axis=1)
    samples = np.empty((replicates, len(METRICS)), np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        selected = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        samples[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {key: dict(delta=float(differences[:, i].mean()), lower_95=float(bounds[0, i]),
        upper_95=float(bounds[1, i]), queries=len(query_ids), clusters=len(groups), replicates=replicates,
        seed=seed, unit="positive reaction group", assignment="Smallest sorted positive reaction index per enzyme; one deterministic group for rare multi-reaction enzymes.",
        estimand="Sum of sampled group query-metric differences divided by sampled query count; preserves pooled enzyme-query weighting.")
        for i, key in enumerate(METRICS)}


def checked_scores(entry, directory, catalog_path, shape, freeze_sha, provenance):
    path = resolve(directory, entry["path"])
    info = identity(path, True)
    if entry.get("sha256") and info["sha256"] != entry["sha256"]:
        raise ValueError(f"Score checksum mismatch: {path}")
    catalog_sha = digest(catalog_path)
    catalog_bound = entry.get("catalog_sha256") == catalog_sha
    if entry.get("catalog_sha256") and not catalog_bound:
        raise ValueError("Explicit score catalog checksum mismatch")
    receipt_path = resolve(directory, entry["receipt"]) if entry.get("receipt") else path.parent / "complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        expected_sha = receipt.get("output_sha256") or receipt.get("scores_sha256") or entry.get("sha256")
        if expected_sha != info["sha256"]:
            raise ValueError("Score receipt is missing or disagrees with its output hash")
        source_catalog = receipt.get("inputs", {}).get("catalog")
        if source_catalog:
            if source_catalog["sha256"] != catalog_sha:
                raise ValueError("Prediction IDs/order differ from the official catalog")
            catalog_bound = True
        if receipt.get("labels_used") or receipt.get("activity_labels_used"):
            raise ValueError("Predictions must be produced without held-out labels")
        if "bundle" in receipt:
            bundle = resolve(receipt_path.parent, receipt["bundle"]["path"])
            if digest(bundle) != receipt["bundle"]["sha256"]:
                raise ValueError("Prediction bundle changed")
            if json.loads(bundle.read_text())["frozen_recipe"]["sha256"] != freeze_sha:
                raise ValueError("Prediction bundle belongs to another frozen recipe")
            provenance[str(bundle)] = identity(bundle, True)
        provenance[str(receipt_path)] = identity(receipt_path, True)
    elif not entry.get("sha256"):
        raise ValueError("Scores require a native receipt or an explicit SHA256")
    if not catalog_bound:
        raise ValueError("Scores must be bound to the exact official catalog order")
    with np.load(path, allow_pickle=False) as source:
        key = entry.get("score_key", "scores")
        scores = np.asarray(source[key])
        catalog = json.loads(catalog_path.read_text())
        for array_key, ids in (("protein_ids", catalog["proteins"]), ("reaction_ids", catalog["reactions"])):
            if array_key in source and source[array_key].astype(str).tolist() != ids:
                raise ValueError("Score ID array order mismatch")
    if scores.shape != shape or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite full score matrix {shape}, got {scores.shape}")
    provenance[str(path)] = info
    return scores


def checked_official_truth(feature_root, split, freeze_sha, provenance):
    """Reconstruct labels from the export's hashed official association source."""
    catalog_path = feature_root / "catalog.json"
    completion = json.loads((feature_root / "complete.json").read_text())
    manifest = json.loads((feature_root / "manifest.json").read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if completion["manifest_signature"] != signature:
        raise ValueError("Official export manifest changed after completion")
    if (completion["freeze_sha256"] != freeze_sha or completion["split"] != split
            or manifest["freeze"]["freeze_sha256"] != freeze_sha or manifest["split"] != split):
        raise ValueError("Official feature export belongs to another freeze or split")
    receipt_path = feature_root / "feature_bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["inputs"]["catalog"]["sha256"] != digest(catalog_path):
        raise ValueError("Official feature catalog changed after integrity binding")
    catalog = json.loads(catalog_path.read_text())
    shape = (len(catalog["reactions"]), len(catalog["proteins"]))
    if (shape != EXPECTED[split] or catalog["reactions"] != sorted(set(catalog["reactions"]))
            or catalog["proteins"] != sorted(set(catalog["proteins"]))):
        raise ValueError("Official pool size or stable candidate order changed")
    source_pairs = Path(manifest["inputs"]["pairs"]["path"])
    if digest(source_pairs) != manifest["inputs"]["pairs"]["sha256"]:
        raise ValueError("Official source associations changed after export")
    associations = read_pairs(source_pairs)
    if (sorted({q for q, _ in associations}) != catalog["reactions"]
            or sorted({p for _, p in associations}) != catalog["proteins"]):
        raise ValueError("Official catalog does not equal the source positive universe")
    qi = {key: i for i, key in enumerate(catalog["reactions"])}
    pi = {key: i for i, key in enumerate(catalog["proteins"])}
    expected_edges = np.asarray([(qi[q], pi[p]) for q, p in associations], np.int64)
    with np.load(feature_root / "pairs.npz", allow_pickle=False) as pairs:
        edges = pairs["test"]
    if not np.array_equal(edges, expected_edges):
        raise ValueError("Compact test labels differ from the hashed official associations")
    for path in (catalog_path, feature_root / "pairs.npz", feature_root / "manifest.json",
                 feature_root / "complete.json", receipt_path, source_pairs):
        provenance[str(path)] = identity(path, True)
    return catalog, edges


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True, help="Parent containing features_test_{split}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--tiger-reference", type=Path, help="Fixed literature values; defaults to the campaign's audited target JSON")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")
    source = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    freeze_path = resolve(directory, source["freeze"]["path"])
    freeze_sha = digest(freeze_path)
    if freeze_sha != source["freeze"]["sha256"]:
        raise ValueError("Frozen recipe checksum mismatch")
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("frozen_before_held_out_evaluation"):
        raise ValueError("The recipe must have been frozen before held-out evaluation")
    methods = [{**method, "splits": method.get("splits", method.get("panels", {}))} for method in source["methods"]]
    labels = [method["label"] for method in methods]
    baseline = source["baseline_method"]
    primary = [m["label"] for m in methods if m.get("primary")]
    if len(labels) != len(set(labels)) or baseline not in labels or len(primary) != 1 or primary[0] == baseline:
        raise ValueError("Require unique method labels, one baseline, and one nonbaseline primary")
    primary_specification = next(method for method in methods if method["label"] == primary[0])
    if primary_specification.get("seed") != freeze.get("primary_seed", 42):
        raise ValueError("The primary method must declare the seed fixed by the frozen recipe")
    if any(set(method["splits"]) != set(SPLITS) for method in methods):
        raise ValueError("Every method must report all six official cells")
    if any(not label or any(not (c.isalnum() or c in "_-") for c in label) for label in labels):
        raise ValueError("Use nonempty safe method labels")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation outputs are immutable; use an empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    provenance = {str(args.manifest.resolve()): identity(args.manifest.resolve(), True),
                  str(freeze_path): identity(freeze_path, True)}
    tiger_path = args.tiger_reference or args.feature_root / "case1_audit/tiger_reference_targets.json"
    tiger = json.loads(tiger_path.read_text())
    provenance[str(tiger_path)] = identity(tiger_path, True)
    results = {label: {} for label in labels}
    rows, contrasts, cluster_contrasts, literature = [], [], [], []
    for si, split in enumerate(SPLITS):
        feature_root = args.feature_root / f"features_test_{split}"
        catalog_path = feature_root / "catalog.json"
        catalog, edges = checked_official_truth(feature_root, split, freeze_sha, provenance)
        shape = (len(catalog["reactions"]), len(catalog["proteins"]))
        truth = dict(reaction_index=edges[:, 0], enzyme_index=edges[:, 1])
        enzyme_groups = np.full(shape[1], shape[0], np.int64)
        np.minimum.at(enzyme_groups, edges[:, 1], edges[:, 0])
        if (enzyme_groups == shape[0]).any():
            raise ValueError("An official enzyme candidate has no positive reaction")
        per_method = {}
        for method in methods:
            label = method["label"]
            scores = checked_scores(method["splits"][split], directory, catalog_path, shape, freeze_sha, provenance)
            evaluation = evaluate_scores(torch.from_numpy(scores).to(args.device), truth)
            per_method[label] = {direction: evaluation["per_query"][direction]["all"] for direction in DIRECTIONS}
            packed = {f"{direction}__{key}": values for direction, values_by_key in per_method[label].items() for key, values in values_by_key.items()}
            np.savez_compressed(args.output / f"{label}_{split}_per_query.npz", **packed)
            results[label][split] = {}
            for direction in DIRECTIONS:
                summary = evaluation["summary"][direction]["all"]
                results[label][split][direction] = summary
                rows.append(dict(method=label, primary=bool(method.get("primary")), seed=method.get("seed", ""),
                                 split=split, direction=direction, **summary))
                reference = tiger[split][direction]
                for metric, reference_key in (("reactzyme_mrr", "mrr"), ("top_1", "hit1"), ("top_10", "hit10")):
                    literature.append(dict(method=label, split=split, direction=direction, metric=metric,
                        observed=summary[metric], tiger_reference=reference[reference_key],
                        numerical_difference=summary[metric] - reference[reference_key], source=tiger["source"],
                        comparison="Literature only; TIGER candidate assets and MRR implementation not independently verified; no paired significance claim."))
            print(f"Evaluated {label} {split}", flush=True)
        for method in methods:
            label = method["label"]
            if label == baseline:
                continue
            for di, direction in enumerate(DIRECTIONS):
                stats = bootstrap_difference(per_method[label][direction], per_method[baseline][direction],
                                             args.bootstrap_replicates, args.seed + si * 2 + di)
                for metric, value in stats.items():
                    contrasts.append(dict(method=label, baseline=baseline, primary=bool(method.get("primary")),
                                          split=split, direction=direction, metric=metric, **value))
                if direction == "enzyme_to_reaction":
                    cluster_stats = reaction_cluster_bootstrap(per_method[label][direction], per_method[baseline][direction],
                        enzyme_groups, args.bootstrap_replicates, args.seed + 100 + si)
                    for metric, value in cluster_stats.items():
                        cluster_contrasts.append(dict(method=label, baseline=baseline, primary=bool(method.get("primary")),
                            split=split, direction=direction, metric=metric, **value))
    summary = dict(schema="frozen_official_six_cell_evaluation_v1", primary_method=primary[0], baseline_method=baseline,
        frozen_recipe_sha256=freeze_sha, methods=results,
        six_cell_macro={label: float(np.mean([results[label][split][direction]["reactzyme_mrr"]
                                              for split in SPLITS for direction in DIRECTIONS])) for label in labels},
        metric="ReactZyme MRR averages reciprocal rank over every known positive, then over queries; Hit@k uses the first positive.",
        tie_policy="Stable descending score with sorted candidate ID order, preserving full candidate universes.",
        uncertainty="Paired percentile query bootstrap, shared draws across method contrasts within each cell. Query independence is an approximation; reaction/protein dependence and multiple comparisons are not corrected.",
        enzyme_to_reaction_cluster_sensitivity="Additional reaction-group bootstrap preserves enzyme-query weighting by pooling sampled group sums/counts. Groups use the smallest positive reaction index per enzyme; captures reaction grouping but not all protein homology dependence.",
        tiger_reference=tiger,
        model_selection="None; all methods were declared before this evaluation, and the supplied primary method is retained.")
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "paired_bootstrap.json", contrasts)
    atomic_json(args.output / "reaction_cluster_bootstrap.json", cluster_contrasts)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "paired_bootstrap.csv", contrasts)
    write_csv(args.output / "reaction_cluster_bootstrap.csv", cluster_contrasts)
    write_csv(args.output / "tiger_literature_comparison.csv", literature)
    lines = ["# Frozen official ReactZyme evaluation", "", f"Primary: **{primary[0]}**; baseline: **{baseline}**.", "",
             "| Method | Reaction R→E | Reaction E→R | Enzyme R→E | Enzyme E→R | Time R→E | Time E→R | Macro |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        values = [results[label][split][direction]["reactzyme_mrr"] for split in SPLITS for direction in DIRECTIONS]
        lines.append("| " + label + " | " + " | ".join(f"{value:.6f}" for value in [*values, summary["six_cell_macro"][label]]) + " |")
    tiger_values = [tiger[split][direction]["mrr"] for split in SPLITS for direction in DIRECTIONS]
    lines.append("| TIGER ESM2Text (literature) | " + " | ".join(f"{value:.6f}" for value in [*tiger_values, float(np.mean(tiger_values))]) + " |")
    lines += ["", "All values are all-positive MRR; every declared method is reported. No checkpoint or method was chosen from these results.", "",
              "Paired query-bootstrap intervals for every metric and cell are in paired_bootstrap.csv. E→R reaction-group sensitivity intervals are in reaction_cluster_bootstrap.csv; pooled sums/counts retain enzyme-query weighting. Query dependence and unadjusted multiple comparisons limit interpretation.", "",
              f"TIGER numbers are a literature comparison from [{tiger['method']}]({tiger['source']}); candidate manifests and the published MRR implementation were not independently verified. No paired statistical claim is made against these reported values."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    if digest(freeze_path) != freeze_sha:
        raise ValueError("Frozen recipe changed during evaluation")
    atomic_json(args.output / "complete.json", dict(schema="frozen_official_evaluation_receipt_v1", freeze_sha256=freeze_sha,
        inputs=provenance, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.seed,
        evaluator=identity(__file__, True), metrics_source=identity(ROOT / "scripts/generalization_metrics.py", True),
        output=identity(args.output / "summary.json", True)))


if __name__ == "__main__":
    main()
