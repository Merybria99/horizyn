"""Search the measured esterase sequences against official Horizyn-1 train-only proteins."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "runs/generalization_20260919_2251/esterase_audit"
TRAIN = ROOT / "data/sota"
OUT = PANEL / "public_horizyn1_dev/homology"
MMSEQS = ROOT / ".deps/mmseqs/bin/mmseqs"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fasta(path):
    sequences = {}
    key, parts = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if key is not None:
                    sequences[key] = "".join(parts)
                key, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if key is not None:
        sequences[key] = "".join(parts)
    return sequences


def main():
    target = OUT / "summary.json"
    if target.exists():
        raise ValueError("Preserve completed homology audit")
    exact = json.loads((OUT.parent / "training_exposure.json").read_text())
    for name, meta in exact["assets"].items():
        if sha256(TRAIN / name) != meta["sha256"]:
            raise ValueError("Official training asset changed")
    panel_rows = rows(PANEL / "inputs/proteins.csv")
    train_ids = {row["protein_id"] for row in rows(TRAIN / "train_pairs.csv")}
    train_sequences = fasta(TRAIN / "prots.fasta")
    if len(panel_rows) != 145 or not train_ids <= train_sequences.keys():
        raise ValueError("Protein pool changed")
    OUT.mkdir(parents=True, exist_ok=True)
    query_path, reference_path, hits_path = (OUT / "panel.fasta", OUT / "train_only.fasta", OUT / "hits.tsv")
    query_path.write_text("".join(f">{row['protein_id']}\n{row['sequence']}\n" for row in panel_rows))
    reference_path.write_text("".join(f">{pid}\n{train_sequences[pid]}\n" for pid in sorted(train_ids)))
    command = [str(MMSEQS), "easy-search", str(query_path), str(reference_path), str(hits_path), str(OUT / "tmp"),
               "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits", "--max-seqs", "10",
               "-s", "7.5", "--threads", "16"]
    subprocess.run(command, check=True, stdout=(OUT / "mmseqs.log").open("w"), stderr=subprocess.STDOUT)
    hits = rows_tsv(hits_path)
    grouped = {row["protein_id"]: [] for row in panel_rows}
    for hit in hits:
        grouped[hit["query"]].append(hit)
    best = {pid: max(group, key=lambda h: float(h["bits"])) if group else None for pid, group in grouped.items()}
    counts = {f"identity_at_least_{pct}_query_coverage_at_least_80pct": sum(
        any(float(hit["fident"]) >= pct / 100 and float(hit["qcov"]) >= 0.8 for hit in group)
        for group in grouped.values()) for pct in (30, 50, 70, 90)}
    report = {
        "schema": "esterase_horizyn1_train_only_mmseqs_v1",
        "method": "MMseqs2 easy-search, top 10 per query, sensitivity 7.5, 16 threads",
        "command": command,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (query_path, reference_path, hits_path)},
        "mmseqs_version": subprocess.check_output([str(MMSEQS), "version"], text=True).strip(),
        "panel_query_count": len(grouped),
        "training_reference_count": len(train_ids),
        "queries_with_hit": sum(bool(x) for x in grouped.values()),
        "counts": counts,
        "best_hits": best,
        "limits": "Local sequence homology only; coverage/identity thresholds are post-evaluation descriptive strata, not model-selection gates.",
        "implementation_sha256": sha256(Path(__file__)),
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": counts, "output": str(target)}), flush=True)


def rows_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", fieldnames=("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")))


if __name__ == "__main__":
    main()
