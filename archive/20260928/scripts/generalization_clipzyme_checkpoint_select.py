#!/usr/bin/env python3
"""Choose a predeclared F3 snapshot using validation-only full-pool BEDROC85."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = ("bedroc85", "bedroc20", "ef0.05", "ef0.1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, action="append", required=True,
                   help="Full-library validation summary.json, in the predeclared grid")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = []
    manifest_hash = None
    for path in a.evaluation:
        result = json.loads(path.read_text())
        if (result.get("schema") != "clipzyme_f3_full_library_validation_v1" or
                not result.get("validation_only") or result.get("test_labels_read") or
                result.get("selection_metric") != "table1.bedroc85" or
                result["summary"]["table1"]["queries"] != 2652 or
                result["summary"]["table1"]["candidate_ids"] != 261907 or
                result["summary"]["table2"]["queries"] != 2216 or
                result["summary"]["table2"]["candidate_ids"] != 252113):
            raise ValueError(f"Not a complete, validation-only full-library result: {path}")
        if manifest_hash is None:
            manifest_hash = result["association_manifest_sha256"]
        elif result["association_manifest_sha256"] != manifest_hash:
            raise ValueError("Compared snapshots use different train/dev associations")
        reported = {table: {metric: result["summary"][table][metric] for metric in METRICS}
                    for table in ("table1", "table2")}
        if not all(math.isfinite(value) for values in reported.values() for value in values.values()):
            raise ValueError(f"Nonfinite screening validation metric: {path}")
        if result["selection_value"] != reported["table1"]["bedroc85"]:
            raise ValueError(f"Selection value does not match validation BEDROC85: {path}")
        rows.append({"evaluation": str(path),
                     "checkpoint_sha256": result["checkpoint_sha256"],
                     "table1_bedroc85": result["selection_value"],
                     "table2_bedroc85": result["summary"]["table2"]["bedroc85"],
                     "summary": reported})
    if len({row["checkpoint_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate model checkpoint in comparison")
    # The input order is the predeclared checkpoint grid; ties prefer earlier.
    winner = max(range(len(rows)), key=lambda i: rows[i]["table1_bedroc85"])
    selection = {"schema": "clipzyme_f3_validation_selection_v1",
                 "metric": "table1.bedroc85", "library_size": 261907,
                 "reported_metrics": list(METRICS), "selection_mode": "max",
                 "test_labels_read": False,
                 "association_manifest_sha256": manifest_hash,
                 "predeclared_grid": rows,
                 "selected_index": winner,
                 "selected": rows[winner]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection["selected"]))


if __name__ == "__main__":
    main()
