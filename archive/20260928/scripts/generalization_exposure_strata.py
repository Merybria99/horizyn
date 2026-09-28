"""Post-hoc endpoint-exposure strata of already frozen full-pool ranks.

No scores are reranked, candidates removed, or model parameters selected.
Run only after the independent input-only exposure audit has completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = args.campaign
    audit = campaign / "official_exposure_audit"
    inputs = [audit / "summary.json", audit / "per_endpoint.csv"]
    summary = json.loads(inputs[0].read_text())
    assert summary["retrieval_scores_read"] is False
    assert summary["schema"] == "official_endpoint_exposure_audit_v2_atom_map_independent"
    with inputs[1].open() as handle:
        exposure = list(csv.DictReader(handle))
    methods = ["F3_native", "F3_fp64", "phase2_seed42", "phase4_seed42"]
    rows, assignments = [], []
    for split in ("reaction_smi", "enzyme_smi", "time"):
        features = campaign / f"features_test_{split}"
        inputs += [features / "catalog.json", features / "pairs.npz"]
        catalog = json.loads((features / "catalog.json").read_text())
        pairs = np.load(features / "pairs.npz", allow_pickle=False)["test"]
        assert len(np.unique(pairs, axis=0)) == len(pairs)
        reaction_ids, protein_ids = catalog["reactions"], catalog["proteins"]
        assert reaction_ids == catalog["test_reactions"]
        assert protein_ids == catalog["test_candidates"]
        lookup = {(r["endpoint"], r["identifier"]): r for r in exposure
                  if r["split"] == split and r["panel"] == "test"}
        assert len(lookup) == len(reaction_ids) + len(protein_ids)
        chem = np.array([lookup["reaction", r]["canonical_stereo_participant_seen"] == "True"
                         for r in reaction_ids])
        assert all(lookup["reaction", r]["canonical_parse_ok"] == "True" for r in reaction_ids)
        sequence = np.array([lookup["protein", p]["full_sequence_seen"] == "True" for p in protein_ids])
        declared = summary["splits"][split]["panels"]["test"]
        assert chem.sum() == declared["canonical_stereochemical_complete_participant_string_seen"]["count"]
        assert sequence.sum() == declared["exact_full_sequence_seen"]["count"]
        edge_count = np.bincount(pairs[:, 1], minlength=len(protein_ids))
        seen_edge_count = np.bincount(pairs[:, 1], weights=chem[pairs[:, 0]], minlength=len(protein_ids))
        assert np.all(edge_count > 0)
        positive_exposure = np.where(seen_edge_count == 0, "all_absent",
                                    np.where(seen_edge_count == edge_count, "all_seen", "mixed"))
        factors = {
            "reaction_to_enzyme": {"query_complete_participant_exposure": np.where(chem, "seen", "absent")},
            "enzyme_to_reaction": {
                "positive_complete_participant_exposure": positive_exposure,
                "query_full_sequence_exposure": np.where(sequence, "seen", "absent"),
            },
        }
        for direction, factor_values in factors.items():
            ids = reaction_ids if direction == "reaction_to_enzyme" else protein_ids
            for factor, values in factor_values.items():
                assignments += [dict(split=split, direction=direction, query_id=identifier,
                                     factor=factor, stratum=str(value))
                                for identifier, value in zip(ids, values)]
            for method in methods:
                path = campaign / "phase4/official_evaluation" / f"{method}_{split}_per_query.npz"
                if path not in inputs:
                    inputs.append(path)
                with np.load(path, allow_pickle=False) as data:
                    assert np.array_equal(data[f"{direction}__query_index"], np.arange(len(ids)))
                    mrr = data[f"{direction}__reactzyme_mrr"].astype(np.float64)
                    hit1 = data[f"{direction}__top_1"].astype(np.float64)
                    hit10 = data[f"{direction}__top_10"].astype(np.float64)
                    positives = data[f"{direction}__positive_count"].astype(np.int64)
                expected = np.bincount(pairs[:, 0 if direction == "reaction_to_enzyme" else 1], minlength=len(ids))
                assert np.array_equal(positives, expected)
                for factor, values in factor_values.items():
                    for value in sorted(set(values)):
                        mask = values == value
                        rows.append(dict(split=split, direction=direction, method=method, factor=factor,
                                         stratum=str(value), query_count=int(mask.sum()),
                                         full_query_count=len(ids), positive_edge_count=int(positives[mask].sum()),
                                         full_candidate_count=len(protein_ids) if direction == "reaction_to_enzyme" else len(reaction_ids),
                                         mrr=float(mrr[mask].mean()), hit1=float(hit1[mask].mean()),
                                         hit10=float(hit10[mask].mean())))
                    subtotal = [r for r in rows if r["split"] == split and r["direction"] == direction
                                and r["method"] == method and r["factor"] == factor]
                    assert sum(r["query_count"] for r in subtotal) == len(ids)
                    assert np.isclose(sum(r["mrr"] * r["query_count"] for r in subtotal) / len(ids),
                                      mrr.mean(), rtol=0, atol=1e-12)
    for name, records in [("metrics.csv", rows), ("query_assignments.csv", assignments)]:
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    receipt = {
        "status": "complete", "study": "post_evaluation_descriptive_fixed_candidate_pool_stratification",
        "candidate_pools_changed": False, "scores_reranked": False, "used_for_selection": False,
        "metrics": "Original per-query all-positive MRR and any-positive Hit@1/10, arithmetic means within input-defined strata.",
        "enzyme_to_reaction_strata": "All recorded positive reaction endpoints seen/absent or mixed; preserves every positive and every original candidate.",
        "novelty_limit": "Exact participant/sequence absence does not establish scaffold, family, mechanism or pretraining novelty.",
        "uncertainty": "Descriptive point estimates only; query groups can be small and dependent. No new significance claim.",
        "sources": [sha(p) for p in inputs], "implementation": sha(Path(__file__)),
        "outputs": [sha(args.output / name) for name in ("metrics.csv", "query_assignments.csv")],
        "metric_rows": len(rows), "assignment_rows": len(assignments),
    }
    (args.output / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "complete", "metric_rows": len(rows), "assignment_rows": len(assignments)}))


if __name__ == "__main__":
    main()
