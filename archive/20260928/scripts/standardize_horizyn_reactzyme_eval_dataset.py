#!/usr/bin/env python3
"""Create a shared Horizyn + ReactZyme evaluation protein catalog."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data/standardized/horizyn_reactzyme_eval"


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks).upper()
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records[current_id] = "".join(chunks).upper()
    return records


def write_fasta(path: Path, records: OrderedDict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records.items():
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split(",")[0].split()[0])
    return ids


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sota_fasta = PROJECT_ROOT / "data/sota/prots.fasta"
    reactzyme_fasta = PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta"
    sota_records = read_fasta(sota_fasta)
    reactzyme_records = read_fasta(reactzyme_fasta)

    combined: OrderedDict[str, str] = OrderedDict()
    conflicts: list[str] = []
    for source_records in (sota_records, reactzyme_records):
        for protein_id, sequence in source_records.items():
            previous = combined.get(protein_id)
            if previous is not None and previous != sequence:
                conflicts.append(protein_id)
                continue
            combined.setdefault(protein_id, sequence)
    if conflicts:
        examples = ", ".join(conflicts[:5])
        raise ValueError(f"Conflicting duplicate protein IDs: {examples}")

    write_fasta(OUTPUT_ROOT / "proteins.fasta", combined)
    write_ids(OUTPUT_ROOT / "candidate_ids.txt", list(combined.keys()))

    candidate_specs = {
        "horizyn_sota": list(sota_records.keys()),
        "reactzyme_time": read_ids(
            PROJECT_ROOT / "data/paper/reactzyme/eval/time/candidate_ids.txt"
        ),
        "reactzyme_enzyme_smi": read_ids(
            PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/candidate_ids.txt"
        ),
        "reactzyme_reaction_smi": read_ids(
            PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/candidate_ids.txt"
        ),
    }
    for name, ids in candidate_specs.items():
        missing = sorted(set(ids) - set(combined))
        if missing:
            examples = ", ".join(missing[:5])
            raise ValueError(f"{name} has {len(missing)} missing candidate IDs: {examples}")
        write_ids(OUTPUT_ROOT / f"{name}_candidate_ids.txt", ids)

    task_counts = {
        "horizyn_sota": {
            "pairs": count_csv_rows(PROJECT_ROOT / "data/sota/test_pairs.csv"),
            "reactions": count_csv_rows(PROJECT_ROOT / "data/sota/test_rxns.csv"),
            "candidates": len(candidate_specs["horizyn_sota"]),
        },
        "reactzyme_time": {
            "pairs": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/time/test_pairs.csv"
            ),
            "reactions": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/time/reactions.csv"
            ),
            "candidates": len(candidate_specs["reactzyme_time"]),
        },
        "reactzyme_enzyme_smi": {
            "pairs": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/test_pairs.csv"
            ),
            "reactions": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/reactions.csv"
            ),
            "candidates": len(candidate_specs["reactzyme_enzyme_smi"]),
        },
        "reactzyme_reaction_smi": {
            "pairs": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/test_pairs.csv"
            ),
            "reactions": count_csv_rows(
                PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/reactions.csv"
            ),
            "candidates": len(candidate_specs["reactzyme_reaction_smi"]),
        },
    }
    metadata = {
        "output_root": str(OUTPUT_ROOT),
        "sota_fasta": str(sota_fasta),
        "reactzyme_fasta": str(reactzyme_fasta),
        "sota_proteins": len(sota_records),
        "reactzyme_proteins": len(reactzyme_records),
        "combined_proteins": len(combined),
        "task_counts": task_counts,
    }
    (OUTPUT_ROOT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Standardized Horizyn + ReactZyme Evaluation Catalog",
        "",
        "This folder contains a shared protein FASTA and candidate ID lists for",
        "testing a model on the original Horizyn SOTA and ReactZyme benchmark",
        "test splits after training on a separate unique training dataset.",
        "",
        "The shared `proteins.fasta` should be used to extract one residue HDF5",
        "whose keys are the original Horizyn/ReactZyme protein IDs.",
        "",
        "| Task | Test pairs | Reactions | Candidate proteins |",
        "|---|---:|---:|---:|",
    ]
    for name, row in task_counts.items():
        lines.append(
            f"| `{name}` | {row['pairs']:,} | {row['reactions']:,} | "
            f"{row['candidates']:,} |"
        )
    lines.extend(
        [
            "",
            "| Artifact | Path |",
            "|---|---|",
            f"| Shared FASTA | `{OUTPUT_ROOT / 'proteins.fasta'}` |",
            f"| All candidate IDs | `{OUTPUT_ROOT / 'candidate_ids.txt'}` |",
            f"| Horizyn SOTA candidate IDs | `{OUTPUT_ROOT / 'horizyn_sota_candidate_ids.txt'}` |",
            f"| ReactZyme time candidate IDs | `{OUTPUT_ROOT / 'reactzyme_time_candidate_ids.txt'}` |",
            f"| ReactZyme enzyme-SMI candidate IDs | `{OUTPUT_ROOT / 'reactzyme_enzyme_smi_candidate_ids.txt'}` |",
            f"| ReactZyme reaction-SMI candidate IDs | `{OUTPUT_ROOT / 'reactzyme_reaction_smi_candidate_ids.txt'}` |",
            "",
            "The benchmark suite using these files is",
            "`configs/benchmarks/horizyn_reactzyme_eval.yaml`.",
            "",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
