#!/usr/bin/env python3
"""Check ReactZyme metric comparability against FGW-CLIP oracle table rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER_ORACLE_MRR = {
    "reaction_smi": {"reaction_to_enzyme": 0.6715, "enzyme_to_reaction": 1.0000},
    "enzyme_smi": {"reaction_to_enzyme": 0.7321, "enzyme_to_reaction": 0.9999},
    "time": {"reaction_to_enzyme": 0.7497, "enzyme_to_reaction": 0.9998},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-evaluation", type=Path, required=True)
    parser.add_argument("--fgw-values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fgw = json.loads(args.fgw_values.read_text())
    if fgw["schema"] != "fgw_clip_v2_published_reactzyme_mrr_transcription_v1":
        raise ValueError("FGW-CLIP paper transcription changed")
    rows = []
    for split in ("reaction_smi", "enzyme_smi", "time"):
        path = args.phase2_evaluation / f"phase2_seed42_{split}_per_query.npz"
        data = np.load(path, allow_pickle=False)
        for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
            counts = data[f"{direction}__positive_count"].astype(np.int64)
            if len(counts) == 0 or np.any(counts < 1):
                raise ValueError("Expected at least one positive for every test query")
            harmonic = np.cumsum(1.0 / np.arange(1, counts.max() + 1, dtype=np.float64))
            perfect_all_positive = float(np.mean(harmonic[counts - 1] / counts))
            paper_oracle = PAPER_ORACLE_MRR[split][direction]
            if abs(perfect_all_positive - paper_oracle) >= 0.00005:
                raise ValueError(f"Perfect-ranking MRR does not reproduce FGW {split}/{direction}")
            all_positive = float(data[f"{direction}__reactzyme_mrr"].mean())
            first_positive = float(np.reciprocal(
                data[f"{direction}__first_rank"].astype(np.float64)).mean())
            paper_best = fgw["values"][split][direction]["cellwise_best"]
            rows.append({"split": split, "direction": direction,
                         "queries": len(counts),
                         "positive_labels": int(counts.sum()),
                         "paper_perfect_mrr": paper_oracle,
                         "local_perfect_all_positive_mrr": perfect_all_positive,
                         "phase2_all_positive_mrr": all_positive,
                         "phase2_first_positive_mrr": first_positive,
                         "fgw_paper_best_mrr": paper_best,
                         "phase2_all_positive_delta_vs_fgw": all_positive - paper_best,
                         "phase2_better_than_fgw_paper_point": all_positive > paper_best,
                         "per_query_sha256": sha256(path)})
    receipt = {"schema": "reactzyme_fgw_clip_groundtruth_metric_audit_v1",
               "source_paper": "https://arxiv.org/pdf/2512.08508",
               "paper_oracle_tables": {"time": "Table 3", "enzyme_smi": "Table 7",
                                       "reaction_smi": "Table 8"},
               "paper_prose_metric_conflict": "Appendix B.2 says first positive; the published perfect-ranking rows below 1 on multi-positive R-to-E queries match mean reciprocal rank over all positives instead.",
               "fgw_values_sha256": sha256(args.fgw_values),
               "rows": rows,
               "all_six_phase2_paper_point_wins": all(
                   row["phase2_better_than_fgw_paper_point"] for row in rows),
               "source_sha256": sha256(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"all_six_phase2_paper_point_wins":
                      receipt["all_six_phase2_paper_point_wins"],
                      "oracle_matches": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
