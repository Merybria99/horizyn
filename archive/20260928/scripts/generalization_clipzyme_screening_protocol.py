#!/usr/bin/env python3
"""Pin CLIPZyme notebook's unique, train-unseen EnzymeMap screening queries."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identifier(reaction: str) -> str:
    return "r_" + hashlib.sha256(reaction.encode()).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-enzymemap", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "fixed_architecture_clipzyme_manifest_v1":
        raise ValueError("Expected audited CLIPZyme manifest")
    candidate_map = args.catalog / "screening_candidate_map.csv"
    with candidate_map.open(newline="") as handle:
        candidates = [row["uniprot_id"] for row in csv.DictReader(handle)]
    if len(candidates) != 261907 or len(set(candidates)) != len(candidates):
        raise ValueError("Official screening candidate order changed")
    candidate_set = set(candidates)
    source = {}
    for split in ("train", "test"):
        info = manifest["associations"][split]
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Official {split} association file changed")
        with path.open(newline="") as handle:
            source[split] = list(csv.DictReader(handle))
        if len(source[split]) != info["count"]:
            raise ValueError(f"Official {split} row count changed")
    train_reactions = {row["reaction"] for row in source["train"]}
    train_enzymes = {row["protein_id"] for row in source["train"]}
    test_reactions = {row["reaction"] for row in source["test"]}
    eligible = test_reactions - train_reactions
    query_reactions = list(dict.fromkeys(row["reaction"] for row in source["test"]
                                         if row["reaction"] in eligible))
    if (len(test_reactions) != 1551 or len(test_reactions & train_reactions) != 30
            or len(query_reactions) != 1521):
        raise ValueError("Official unique train-unseen screening query set changed")
    args.output.mkdir(parents=True)
    input_path = args.output / "query_inputs.csv"
    with input_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_index", "reaction_id", "reaction"))
        writer.writerows((i, identifier(reaction), reaction)
                         for i, reaction in enumerate(query_reactions))
    raw = json.loads(args.raw_enzymemap.read_text())
    if len(raw) != 349431:
        raise ValueError("Official raw EnzymeMap reaction collection changed")
    labels = defaultdict(set)
    eligible_set = set(query_reactions)
    for item in raw:
        if item["protein_db"] not in ("swissprot", "uniprot"):
            continue
        reaction = ".".join(sorted(item["reactants"])) + ">>" + ".".join(sorted(item["products"]))
        if reaction in eligible_set:
            refs = ast.literal_eval(item["protein_refs"])
            if not isinstance(refs, list):
                raise ValueError("Expected list of UniProt protein references")
            labels[reaction].update(refs)
    query_ids = {row["reaction_id"] for row in csv.DictReader(
        (args.catalog / "test_rxns.csv").open(newline=""))}
    query_path = args.output / "queries.csv"
    positives_total = 0
    nontrain_positives_total = 0
    nontrain_queries = 0
    with query_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_index", "reaction_id", "reaction", "positive_uniprot_ids_json"))
        for i, reaction in enumerate(query_reactions):
            reaction_id = identifier(reaction)
            if reaction_id not in query_ids:
                raise ValueError("Query reaction is absent from F3 test feature catalog")
            positives = sorted(labels[reaction] & candidate_set)
            if not positives:
                raise ValueError("Screening query has no positive in official candidate pool")
            positives_total += len(positives)
            other = set(positives) - train_enzymes
            nontrain_positives_total += len(other)
            nontrain_queries += bool(other)
            writer.writerow((i, reaction_id, reaction, json.dumps(positives)))
    if positives_total != 4544:
        raise ValueError("Official notebook label construction changed")
    train_path = args.output / "train_uniprot_ids.txt"
    train_path.write_text("\n".join(sorted(train_enzymes)) + "\n")
    receipt = {"schema": "clipzyme_screening_notebook_protocol_v1",
               "source_notebook": "data/external/cyp_specificity_2026/clipzyme_official/analysis/Results.ipynb",
               "source_notebook_sha256": sha256(Path("data/external/cyp_specificity_2026/clipzyme_official/analysis/Results.ipynb")),
               "raw_enzymemap_sha256": sha256(args.raw_enzymemap),
               "manifest_sha256": sha256(args.manifest),
               "candidate_map_sha256": sha256(candidate_map),
               "test_association_rows": len(source["test"]),
               "unique_test_reactions": len(test_reactions),
               "train_overlap_reactions_excluded": len(test_reactions & train_reactions),
               "table1_queries": len(query_reactions),
               "table1_candidate_ids": len(candidates),
               "table1_candidate_positive_labels": positives_total,
               "table2_candidate_ids": len(candidate_set - train_enzymes),
               "table2_queries_with_positive": nontrain_queries,
               "table2_candidate_positive_labels": nontrain_positives_total,
               "query_inputs_sha256": sha256(input_path),
               "queries_sha256": sha256(query_path),
               "train_uniprot_ids_sha256": sha256(train_path),
               "source_sha256": sha256(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ("table1_queries", "table1_candidate_ids",
                       "table1_candidate_positive_labels", "table2_queries_with_positive")}), flush=True)


if __name__ == "__main__":
    main()
