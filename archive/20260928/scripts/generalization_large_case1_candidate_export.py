#!/usr/bin/env python3
"""Attach source annotations to completed, unchanged Case 1 rank tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def checked(record):
    path = Path(record["path"])
    if identity(path)["sha256"] != record["sha256"]:
        raise ValueError(f"Source changed: {path}")
    return path


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    repo = Path(__file__).resolve().parents[1]
    primary_path = campaign / "large_case1_evaluation/complete.json"
    primary = json.loads(primary_path.read_text())
    scan_path = campaign / "large_case1_scan/complete.json"
    audit_path = campaign / "large_case1_scan/completion_audit.json"
    audit = json.loads(audit_path.read_text())
    if (primary.get("schema") != "large_case1_evaluation_complete_v1"
        or audit.get("all_exact") is not True
        or primary["scan_receipt"]["sha256"] != identity(scan_path)["sha256"]
        or audit["complete"]["sha256"] != identity(scan_path)["sha256"]):
        raise ValueError("Require matching completed original primary and numerical audit")
    checked(primary["protocol"])
    checked(primary["evaluator"])
    names = ("top100_per_method.csv", "known_positive_ranks.csv")
    tables = {name: read_csv(checked(primary["outputs"][name])) for name in names}
    methods = {"phase2", "phase4", "f3_native", "f3_fp64", "circev2"}
    for name, rows in tables.items():
        count = 100 if name.startswith("top100") else 24
        if len(rows) != len(methods) * count or {r["method"] for r in rows} != methods:
            raise ValueError("Require every fixed method and original rank row")
        if any(sum(r["method"] == method for r in rows) != count for method in methods):
            raise ValueError("Incomplete per-method export")
    targets = {r["candidate_id"] for rows in tables.values() for r in rows}
    length_plan_path = campaign / "large_case1_length_audit/plan.json"
    sources = json.loads(length_plan_path.read_text())["metadata_sources"]
    aliases = json.loads(checked(sources["aliases"]).read_text())["groups"]
    by_candidate = {row["candidate_id"]: row for row in aliases}
    literature_catalog = json.loads(checked(sources["literature_catalog"]).read_text())
    literature_metadata_path = repo / literature_catalog["source_metadata"]
    literature_metadata = {r["entry_id"]: r for r in read_csv(literature_metadata_path)}
    annotations = {}
    for candidate in targets & by_candidate.keys():
        group = by_candidate[candidate]
        row = literature_metadata[group["representative_id"]]
        if row["sha256"] != group["sequence_sha256"]:
            raise ValueError("Literature annotation does not match the scored sequence")
        annotations[candidate] = dict(source_accession="", source_description=row["name"],
            source_organism="", full_aa_length=int(row["length"]),
            source_annotation="Literature workbook name; evidence is in the separate source audit",
            literature_representative=group["representative_id"],
            literature_aliases=";".join(group["all_entry_ids"]))
    refseq_record = sources["full_aa_metadata_table"]
    digest = hashlib.sha256()
    # Hash exactly the bytes parsed, in one bounded sequential metadata pass.
    with Path(refseq_record["path"]).open("rb") as handle:
        def lines():
            for raw in handle:
                digest.update(raw)
                yield raw.decode("utf-8")
        for row in csv.DictReader(lines()):
            candidate = row["protein_id"]
            if candidate not in targets or candidate in by_candidate:
                continue
            if candidate in annotations:
                raise ValueError("Duplicate selected metadata ID")
            annotations[candidate] = dict(source_accession=row["accession"],
                source_description=row["description"], source_organism=row["organism"],
                full_aa_length=int(row["sequence_length"]),
                source_annotation="RefSeq description; no catalytic verification inferred",
                literature_representative="", literature_aliases="")
    if digest.hexdigest() != refseq_record["sha256"] or set(annotations) != targets:
        raise ValueError("Metadata identity mismatch or incomplete ID join")
    if any(row["full_aa_length"] <= 0 for row in annotations.values()):
        raise ValueError("Invalid source sequence length")
    out = campaign / "large_case1_annotated_candidates"
    if out.exists():
        raise ValueError("Use a fresh annotation-export directory")
    out.mkdir()
    outputs = {}
    for name, rows in tables.items():
        enriched = [{**row, **annotations[row["candidate_id"]]} for row in rows]
        path = out / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
            writer.writeheader()
            writer.writerows(enriched)
        replay = read_csv(path)
        if [{k: row[k] for k in rows[0]} for row in replay] != rows:
            raise ValueError("Annotation export changed an original rank-table field")
        outputs[name] = identity(path)
    (out / "README.md").write_text(
        "# Annotated Case 1 rank tables\n\n"
        "These tables attach source descriptions and full sequence lengths to every original "
        "top-100 row for all five fixed methods, and to all 24 paper/patent catalysts per method. "
        "Every original field, score, rank and row order is preserved and checked after CSV replay. "
        "No annotation is used to filter or rerank candidates.\n\n"
        "The top-100 table uses the original stable exact-tie order; it is a review artifact, "
        "not a claim that all boundary-tied entries are distinguishable. Known-positive tables "
        "retain best/worst tie ranks. RefSeq descriptions are annotations, not verified activity "
        "for this reaction. Background activity remains unknown. The literature evidence audit "
        "governs paper/patent claims; workbook names alone do not establish them.\n\n"
        "This descriptive export follows the completed primary evaluation and changes no experiment. "
        "Its completion receipt records the original tables, source metadata and exact join checks.\n")
    receipt = dict(schema="large_case1_annotation_export_v1", primary=identity(primary_path),
        scan=identity(scan_path), completion_audit=identity(audit_path),
        original_tables={name: primary["outputs"][name] for name in names},
        metadata_sources={"length_plan": identity(length_plan_path), "refseq": refseq_record,
            "literature": identity(literature_metadata_path), "aliases": sources["aliases"],
            "literature_catalog": sources["literature_catalog"]},
        implementation=identity(Path(__file__)), outputs=outputs,
        original_fields_and_row_order_exact=True, unique_annotated_candidates=len(targets),
        no_candidates_filtered=True, no_activity_inferred=True)
    (out / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(out), "unique_candidates": len(targets), "all_original_fields_exact": True}))


if __name__ == "__main__":
    main()
