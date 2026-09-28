#!/usr/bin/env python3
"""CPU-only biological metadata and paired reports for the frozen residual recipe."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("time", "enzyme_smi", "reaction_smi")
DIRECTIONS = ("enzyme_to_reaction", "reaction_to_enzyme")
# Published Table 1 point estimates; not local reproductions or bootstrap inputs.
TIGER_TABLE1 = {
    "time": {"ESM2Text": (0.581, 0.690, 0.454, 0.366), "ProtT3": (0.583, 0.683, 0.454, 0.372)},
    "enzyme_smi": {
        "ESM2Text": (0.931, 0.956, 0.792, 0.592),
        "ProtT3": (0.908, 0.940, 0.784, 0.579),
    },
    "reaction_smi": {
        "ESM2Text": (0.416, 0.518, 0.430, 0.319),
        "ProtT3": (0.386, 0.472, 0.428, 0.337),
    },
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream)


def write_csv(path, records):
    records = list(records)
    if not records:
        raise ValueError(f"Refusing empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pair_inventory(path):
    edges, proteins, reactions = set(), {}, {}
    for row in rows(path):
        p, r = row["protein_id"], row["reaction_id"]
        edges.add((p, r))
        sequence_hash = hashlib.sha256(row["protein_sequence"].encode()).hexdigest()
        if p in proteins and proteins[p] != sequence_hash:
            raise ValueError(f"Conflicting protein sequence: {p}")
        if r in reactions and reactions[r] != row["reaction_smiles"]:
            raise ValueError(f"Conflicting reaction SMILES: {r}")
        proteins[p], reactions[r] = sequence_hash, row["reaction_smiles"]
    return edges, proteins, reactions


def fasta_hashes(path):
    result, identifier, sequence = {}, None, []
    with path.open() as stream:
        for line in stream:
            if line.startswith(">"):
                if identifier is not None:
                    if identifier in result:
                        raise ValueError(f"Duplicate FASTA ID: {identifier}")
                    result[identifier] = hashlib.sha256("".join(sequence).encode()).hexdigest()
                identifier, sequence = line[1:].strip().split()[0], []
            else:
                sequence.append(line.strip())
    if identifier is not None:
        if identifier in result:
            raise ValueError(f"Duplicate FASTA ID: {identifier}")
        result[identifier] = hashlib.sha256("".join(sequence).encode()).hexdigest()
    return result


def similarity_bin(value):
    if value is None:
        return "unknown"
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid similarity: {value}")
    for boundary, label in ((0.4, "<0.4"), (0.6, "0.4-0.6"), (0.8, "0.6-0.8"), (1.0, "0.8-1.0")):
        if value < boundary:
            return label
    return "1.0 (fingerprint-equal)"


def sequence_bin(hit):
    if hit is None:
        return "no_reported_hit"
    identity, qcov, tcov = (
        float(hit[k]) for k in ("local_sequence_identity", "query_coverage", "target_coverage")
    )
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (identity, qcov, tcov)):
        raise ValueError("Invalid MMseqs identity/coverage")
    if min(qcov, tcov) < 0.8:
        return "coverage<80%"
    return (
        "<30%"
        if identity < 0.3
        else "30-50%" if identity < 0.5 else "50-80%" if identity < 0.8 else ">=80%"
    )


def annotation_text(value):
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in ("", "nan", "none", "null", "unknown") else text


def ec_coverage(value):
    labels = [
        label
        for label in annotation_text(value).split("|")
        if len(label.split(".")) == 4 and all(p.isdigit() or p == "-" for p in label.split("."))
    ]
    if any(
        len(label.split(".")) == 4 and all(p.isdigit() for p in label.split("."))
        for label in labels
    ):
        return "complete_EC4_present"
    return "partial_EC_only" if labels else "unknown"


def protein_key(value):
    for prefix in ("prot_", "uprot_", "nr90_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def verified_sequence_hits(audit, split, train_proteins, test_proteins):
    directory = audit / "mmseqs" / split
    paths = [directory / name for name in ("train.fasta", "test.fasta", "test_to_train.tsv")]
    summary = audit / "protein_nearest_train_mmseqs.csv"
    if not all(path.is_file() for path in [*paths, summary]):
        return None
    if fasta_hashes(paths[0]) != train_proteins or fasta_hashes(paths[1]) != test_proteins:
        raise ValueError(f"{split}: alignment FASTA differs from actual train/test sequences")
    columns = [
        "protein_id",
        "nearest_train_protein_id",
        "local_sequence_identity",
        "alignment_length",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
    ]
    raw = {}
    with paths[2].open() as stream:
        for values in csv.reader(stream, delimiter="\t"):
            if len(values) != len(columns):
                raise ValueError("Malformed alignment row")
            row = dict(zip(columns, values))
            if row["protein_id"] in raw:
                raise ValueError("Ambiguous best-hit alignment")
            if (
                row["protein_id"] not in test_proteins
                or row["nearest_train_protein_id"] not in train_proteins
            ):
                raise ValueError("Alignment references entities outside its train/test split")
            raw[row["protein_id"]] = row
    exported = {}
    for row in rows(summary):
        if row["protocol"] == split:
            if row["protein_id"] in exported:
                raise ValueError("Duplicate exported sequence hit")
            exported[row["protein_id"]] = row
    if raw.keys() != exported.keys():
        raise ValueError("Alignment export is incomplete or has extra rows")
    for key, row in raw.items():
        for column in columns[1:]:
            same = (
                row[column] == exported[key][column]
                if column == "nearest_train_protein_id"
                else np.isclose(float(row[column]), float(exported[key][column]), rtol=1e-5, atol=0)
            )
            if not same:
                raise ValueError(f"Alignment export disagrees with raw output: {key}/{column}")
    return raw


@lru_cache(maxsize=100000)
def participant_fingerprint(smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    fp = DataStructs.ExplicitBitVect(2048)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for molecule in str(smiles).replace(">>", ".").split("."):
        mol = Chem.MolFromSmiles(molecule)
        if mol is None:
            return None
        fp |= generator.GetFingerprint(mol)
    return fp


def nearest_chemistry(queries, training):
    from rdkit import DataStructs

    reference = [participant_fingerprint(s) for s in training.values()]
    if not reference or any(fp is None for fp in reference):
        raise ValueError(
            "Training chemistry is empty or unparsable; nearest-neighbor audit would be incomplete"
        )
    result = {}
    for key, smiles in queries.items():
        fp = participant_fingerprint(smiles)
        result[key] = None if fp is None else max(DataStructs.BulkTanimotoSimilarity(fp, reference))
    return result


def metadata(project, output, audit, splits=SPLITS):
    import pyarrow
    import pyarrow.parquet as pq
    from rdkit import RDLogger, rdBase

    RDLogger.DisableLog("rdApp.warning")

    annotations = pq.read_table(
        audit / "pair_annotations.parquet",
        columns=["protocol", "subset", "protein_id", "reaction_id"],
    ).to_pylist()
    enzymes = {
        r["protein_id"]: r for r in pq.read_table(audit / "enzyme_annotations.parquet").to_pylist()
    }
    reactions = {
        r["reaction_id"]: r
        for r in pq.read_table(audit / "reaction_annotations.parquet").to_pylist()
    }
    evidence = {
        "audit_hashes": {p.name: digest(p) for p in audit.glob("*.parquet")},
        "runtime": dict(pyarrow=pyarrow.__version__, rdkit=rdBase.rdkitVersion),
        "splits": {},
    }
    for split in splits:
        print(
            f"Biology: {split}; verifying split membership and sequence alignment inputs",
            flush=True,
        )
        protocol = project / "data/revised_protocols/reactzyme_paper" / split
        inventories = {
            s: pair_inventory(protocol / f"{s}_pairs.csv") for s in ("train", "validation", "test")
        }
        for subset, (edges, _, _) in inventories.items():
            old = {
                (r["protein_id"], r["reaction_id"])
                for r in annotations
                if r["protocol"] == split and r["subset"] == subset
            }
            if edges != old:
                raise ValueError(
                    f"{split}/{subset}: annotation audit uses a different pair partition"
                )
        _, train_proteins, train_reactions = inventories["train"]
        _, test_proteins, test_reactions = inventories["test"]
        hits = verified_sequence_hits(audit, split, train_proteins, test_proteins)
        source = project / "runs" / f"biological_residual_{split}" / "data/source_pretrain"
        manifest = json.loads((source / "manifest.json").read_text())
        for name, field in (
            ("train_pairs.csv", "output_pairs_sha256"),
            ("train_rxns.csv", "output_reactions_sha256"),
        ):
            if digest(source / name) != manifest[field]:
                raise ValueError(f"{split}: source-pretraining graph changed")
        # These are entity identifiers, not a sequence-homology measurement.
        source_ids = {
            protein_key(r.get("protein_uid") or r["protein_id"])
            for r in rows(source / "train_pairs.csv")
        }
        source_reactions = {
            r["reaction_id"]: r["reaction_smiles"] for r in rows(source / "train_rxns.csv")
        }
        print(
            f"Biology: {split}; CPU chemistry neighbors against ReactZyme and source training",
            flush=True,
        )
        chemistry = nearest_chemistry(test_reactions, train_reactions)
        source_chemistry = nearest_chemistry(test_reactions, source_reactions)
        records = []
        for direction, identifiers in (
            (DIRECTIONS[0], test_proteins),
            (DIRECTIONS[1], test_reactions),
        ):
            for identifier in sorted(identifiers):
                enzyme = direction == DIRECTIONS[0]
                annotation = (enzymes if enzyme else reactions).get(identifier, {})
                ec = ec_coverage(
                    annotation.get("ec_numbers" if enzyme else "associated_ec_numbers")
                )
                native = (
                    bool(annotation_text(annotation.get("enzyme_core_cofactor_labels")))
                    if enzyme
                    else False
                )
                row = dict(
                    direction=direction, query_id=identifier if enzyme else identifier + "_f"
                )
                row.update(
                    stratum_reactzyme_sequence_similarity=(
                        "not_applicable"
                        if not enzyme
                        else "unknown" if hits is None else sequence_bin(hits.get(identifier))
                    ),
                    stratum_reactzyme_chemistry_similarity=(
                        "not_applicable" if enzyme else similarity_bin(chemistry[identifier])
                    ),
                    stratum_source_chemistry_similarity=(
                        "not_applicable" if enzyme else similarity_bin(source_chemistry[identifier])
                    ),
                    stratum_combined_chemistry_similarity=(
                        "not_applicable"
                        if enzyme
                        else similarity_bin(
                            None
                            if chemistry[identifier] is None or source_chemistry[identifier] is None
                            else max(chemistry[identifier], source_chemistry[identifier])
                        )
                    ),
                    stratum_source_entity_id_exposure=(
                        "not_applicable"
                        if not enzyme
                        else ("present" if protein_key(identifier) in source_ids else "not_present")
                    ),
                    stratum_source_sequence_similarity="unknown" if enzyme else "not_applicable",
                    stratum_ec_annotation=ec,
                    stratum_native_cofactor_annotation=(
                        "not_applicable" if not enzyme else "annotated" if native else "unknown"
                    ),
                    stratum_annotation_coverage=(
                        "EC+native_cofactor"
                        if ec != "unknown" and native
                        else (
                            "EC_only"
                            if ec != "unknown"
                            else "native_cofactor_only" if native else "unknown"
                        )
                    ),
                    stratum_generic_chemistry=(
                        "not_applicable"
                        if enzyme
                        else (
                            "unknown"
                            if not annotation
                            else (
                                "wildcard_in_source"
                                if annotation["source_had_generic_wildcard"]
                                else "no_recorded_wildcard"
                            )
                        )
                    ),
                    reactzyme_chemistry_similarity=(
                        "" if enzyme or chemistry[identifier] is None else chemistry[identifier]
                    ),
                    source_chemistry_similarity=(
                        ""
                        if enzyme or source_chemistry[identifier] is None
                        else source_chemistry[identifier]
                    ),
                )
                records.append(row)
        write_csv(output / split / "query_metadata.csv", records)
        evidence["splits"][split] = {
            "pair_sha256": {s: digest(protocol / f"{s}_pairs.csv") for s in inventories},
            "sequence_alignment_verified": hits is not None,
            "source_manifest_sha256": digest(source / "manifest.json"),
            "query_count": len(records),
            "missing_analysis": [
                "sequence homology against source-pretraining proteins",
                "upstream F3/SLEEC pretraining exposure",
            ],
        }
    (output / "metadata_provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")


def report(output, splits=SPLITS):
    from scripts.report_biological_residual_controls import (
        bootstrap_difference,
        count_bin,
        selected_rows,
    )

    table, subgroup_rows, curve, coverage = [], [], [], []
    for split in splits:
        directory = output / split
        validation = json.loads((directory / "validation.json").read_text())
        result = json.loads((directory / "test.json").read_text())
        if (
            result["evaluation_split"] != "test"
            or validation["evaluation_split"] != "validation"
            or result["selection_metric"] != "balanced_reactzyme_mrr"
            or validation["selection_metric"] != result["selection_metric"]
            or result["checkpoint"] != validation["checkpoint"]
            or result["best_alpha"] != validation["best_alpha"]
        ):
            raise ValueError(f"{split}: validation/test contract mismatch")
        alpha = result["best_alpha"]
        baseline = selected_rows(directory / "test.per_query.csv", 0)
        selected = selected_rows(directory / "test.per_query.csv", alpha)
        meta = {}
        for row in rows(directory / "query_metadata.csv"):
            key = row["direction"], row["query_id"]
            if key in meta:
                raise ValueError("Duplicate biological metadata key")
            meta[key] = row
        if baseline.keys() != selected.keys() or baseline.keys() != meta.keys():
            raise ValueError(f"{split}: baseline/residual/metadata query sets differ")
        for a, frame in ((0, baseline), (alpha, selected)):
            for direction in DIRECTIONS:
                records = [r for k, r in frame.items() if k[0] == direction]
                scores = result["alpha_results"][f"{a:g}"]
                if int(scores[f"{direction}/num_queries"]) != len(records):
                    raise ValueError("Aggregate and per-query counts disagree")
                for metric in (
                    "reactzyme_mrr",
                    "first_positive_mrr",
                    "top_1",
                    "precision_at_10",
                    "mean_rank",
                ):
                    if not np.isclose(
                        np.mean([r[metric] for r in records]),
                        scores[f"{direction}/{metric}"],
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError(f"Aggregate and per-query metrics disagree: {metric}")
        for a, scores in validation["alpha_results"].items():
            for direction in DIRECTIONS:
                curve.append(
                    dict(
                        split=split,
                        direction=direction,
                        alpha=float(a),
                        selected=float(a) == alpha,
                        reactzyme_mrr=scores[f"{direction}/reactzyme_mrr"],
                    )
                )
        for variant, a in (("F3_exact_parent", 0), ("Biological_residual", alpha)):
            scores = result["alpha_results"][f"{a:g}"]
            for direction in DIRECTIONS:
                table.append(
                    dict(
                        split=split,
                        variant=variant,
                        direction=direction,
                        alpha=a,
                        **{
                            k.split("/", 1)[1]: v
                            for k, v in scores.items()
                            if k.startswith(direction + "/")
                        },
                    )
                )
        for direction in DIRECTIONS:
            groups = defaultdict(list)
            for key, row in selected.items():
                if key[0] != direction:
                    continue
                annotations = {k: v for k, v in meta[key].items() if k.startswith("stratum_")}
                annotations.update(
                    stratum_known_associations=count_bin(row["known_positive_count"]),
                    stratum_reactzyme_training_associations=count_bin(
                        row["train_known_association_count"]
                    ),
                    stratum_unimol2_available=row["has_unimol2"] or "not_applicable",
                    stratum_chiro_available=row["has_chiro"] or "not_applicable",
                )
                groups[("overall", "all")].append(key)
                for column, group in annotations.items():
                    groups[column, group or "unknown"].append(key)
            for (column, group), keys in sorted(groups.items()):
                delta = np.array(
                    [selected[k]["reactzyme_mrr"] - baseline[k]["reactzyme_mrr"] for k in keys]
                )
                interval = bootstrap_difference(delta)
                subgroup_rows.append(
                    dict(
                        split=split,
                        direction=direction,
                        stratum=column,
                        group=group,
                        n_queries=len(keys),
                        low_support=len(keys) < 20,
                        F3_mrr=np.mean([baseline[k]["reactzyme_mrr"] for k in keys]),
                        residual_mrr=np.mean([selected[k]["reactzyme_mrr"] for k in keys]),
                        delta=interval["delta"],
                        ci95_low=interval["ci95"][0],
                        ci95_high=interval["ci95"][1],
                    )
                )
                coverage.append(
                    dict(
                        split=split,
                        direction=direction,
                        stratum=column,
                        group=group,
                        n_queries=len(keys),
                    )
                )
    reports = output / "reports"
    write_csv(reports / "test_metrics.csv", table)
    write_csv(reports / "biological_strata.csv", subgroup_rows)
    write_csv(reports / "metadata_coverage.csv", coverage)
    write_csv(reports / "validation_alpha_curves.csv", curve)
    lines = [
        "# Biological residual: harmonized three-split evaluation",
        "",
        "Fixed existing checkpoints; alpha selected on validation all-positive MRR only. Both directions use the released test-positive candidate universe.",
        "",
        "| Split | Model | Alpha | E→R Hit@1 | E→R MRR | R→E Hit@1 | R→E MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        for variant in ("F3_exact_parent", "Biological_residual"):
            e, r = [
                next(
                    x
                    for x in table
                    if (x["split"], x["variant"], x["direction"]) == (split, variant, d)
                )
                for d in DIRECTIONS
            ]
            lines.append(
                f"| {split} | {variant} | {e['alpha']:g} | {e['top_1']:.4f} | {e['reactzyme_mrr']:.4f} | {r['top_1']:.4f} | {r['reactzyme_mrr']:.4f} |"
            )
        for name, (ehit, emrr, rhit, rmrr) in TIGER_TABLE1[split].items():
            lines.append(
                f"| {split} | TIGER {name} (published) | — | {ehit:.4f} | {emrr:.4f} | {rhit:.4f} | {rmrr:.4f} |"
            )
    lines += [
        "",
        "TIGER entries are [published Table 1 point estimates](https://aclanthology.org/2026.acl-long.1643.pdf), not local reproductions. Local MRR is all-positive ReactZyme MRR; evaluator equivalence with TIGER remains unverified. Do not infer statistical significance against TIGER from our paired-query intervals.",
    ]
    lines += [
        "",
        "## Biological analysis",
        "",
        "All subgroup effects and paired-query 95% bootstrap intervals are in biological_strata.csv. Groups with fewer than 20 queries are flagged; intervals are descriptive, not corrected for multiple comparisons or training-seed variability.",
        "",
        "Sequence bins use existing MMseqs best reported local hits, requiring ≥80% query AND target coverage. No reported hit is not proof of low identity. Their reference is ReactZyme training ONLY.",
        "",
        "Chemistry is recomputed on CPU: OR of radius-2, 2048-bit Morgan participant fingerprints, separately against ReactZyme training, source pretraining, and their union. It does not encode reaction direction or prove mechanistic novelty.",
        "",
        "Source protein identifier exposure is NOT sequence homology. Source-pretraining sequence similarity and upstream F3/SLEEC exposure remain explicitly unknown. No remote-homology/generalization claim across all training sources is supported by these bins.",
        "",
        "EC and native cofactor presence describe annotation coverage, not functional absence. Reaction EC values aggregate associated source annotations. Unknown labels are never negative labels or model inputs.",
        "",
        "Known-positive association counts measure annotated support/hubs, not complete enzyme promiscuity. Molecular feature availability is reported separately. Related queries can violate independent-query bootstrap assumptions.",
        "",
        "Older published-in-workspace scores are not silently mixed with these re-evaluations. This is harmonization after prior test inspection, not a fresh untouched-test study. The fixed F3 parent is epoch 29, not the validation-selected checkpoint used by some older F3 tables.",
    ]
    (reports / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {reports / 'summary.md'}", flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("metadata", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/reactzyme_split_biology")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "metadata":
        metadata(args.project, args.output, args.audit)
    else:
        report(args.output)
