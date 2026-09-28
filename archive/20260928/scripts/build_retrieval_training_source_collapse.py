#!/usr/bin/env python3
"""Build collapsed retrieval-training datasets from selected source splits.

The output is intentionally separate from the leakage-audit datasets. It follows
the user's requested split semantics:

* collapse only the training sources into one retrieval train set;
* keep Horizyn/ReactZyme tests separate as benchmark test sources;
* keep CLIPZyme eval separate as validation;
* write exact, nr90, and nr50 training variants.

AI4Protein/EC is written as an auxiliary enzyme-only EC catalog because it has
enzyme sequences and EC labels, but no reaction SMILES and therefore cannot be
used directly as Horizyn MLNCE reaction-protein pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENZYME_DISCOVERY_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data/standardized/retrieval_training_source_collapse"
DEFAULT_MMSEQS = PROJECT_ROOT / "tools/mmseqs/bin/mmseqs"


@dataclass(frozen=True)
class TrainSource:
    name: str
    dataset: str
    split: str
    pairs_path: Path
    reactions_path: Path | None
    sequence_source: str


@dataclass
class PairRecord:
    reaction_id: str
    reaction_smiles: str
    protein_uid: str
    protein_sequence: str
    source_name: str
    source_dataset: str
    source_split: str
    source_reaction_id: str
    source_protein_id: str
    source_pair_index: int


def clean_sequence(sequence: str) -> str:
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def protein_uid(sequence: str) -> str:
    return f"uprot_{sha1_text(sequence)[:16]}"


def reaction_uid(smiles: str) -> str:
    return f"rxn_{sha1_text(smiles.strip())[:16]}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = clean_sequence("".join(chunks))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records[current_id] = clean_sequence("".join(chunks))
    return records


def read_sabio_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        for row in reader:
            accession = (row.get("accession") or "").strip()
            sequence = clean_sequence(row.get("sequence") or "")
            if accession and sequence:
                sequences[accession] = sequence
    return sequences


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
            count += 1
    return count


def write_ids(path: Path, ids: Iterable[str]) -> int:
    ids_list = list(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids_list) + ("\n" if ids_list else ""), encoding="utf-8")
    return len(ids_list)


def reaction_smiles_by_id(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    for row in read_csv(path):
        reaction_id = (row.get("reaction_id") or "").strip()
        smiles = (row.get("reaction_smiles") or "").strip()
        if reaction_id and smiles:
            out[reaction_id] = smiles
    return out


def train_sources() -> list[TrainSource]:
    reactzyme = PROJECT_ROOT / "data/paper/reactzyme/eval"
    return [
        TrainSource(
            name="horizyn_train",
            dataset="horizyn",
            split="train",
            pairs_path=PROJECT_ROOT / "data/sota/train_pairs.csv",
            reactions_path=PROJECT_ROOT / "data/sota/train_rxns.csv",
            sequence_source="horizyn",
        ),
        TrainSource(
            name="reactzyme_time_train",
            dataset="reactzyme",
            split="time_train",
            pairs_path=reactzyme / "time/train_pairs.csv",
            reactions_path=reactzyme / "time/reactions.csv",
            sequence_source="reactzyme_pair_sequence",
        ),
        TrainSource(
            name="reactzyme_enzyme_smi_train",
            dataset="reactzyme",
            split="enzyme_smi_train",
            pairs_path=reactzyme / "enzyme_smi/train_pairs.csv",
            reactions_path=reactzyme / "enzyme_smi/reactions.csv",
            sequence_source="reactzyme_pair_sequence",
        ),
        TrainSource(
            name="reactzyme_reaction_smi_train",
            dataset="reactzyme",
            split="reaction_smi_train",
            pairs_path=reactzyme / "reaction_smi/train_pairs.csv",
            reactions_path=reactzyme / "reaction_smi/reactions.csv",
            sequence_source="reactzyme_pair_sequence",
        ),
        TrainSource(
            name="clipzyme_train",
            dataset="clipzyme",
            split="upstream_train",
            pairs_path=PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/pairs.csv",
            reactions_path=PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/reactions.csv",
            sequence_source="clipzyme",
        ),
        TrainSource(
            name="sabio_rk_novelty90_train",
            dataset="sabio_rk",
            split="novelty90",
            pairs_path=PROJECT_ROOT / "data/paper/sabio_rk/eval/novelty90/pairs.csv",
            reactions_path=PROJECT_ROOT / "data/paper/sabio_rk/eval/novelty90/reactions.csv",
            sequence_source="sabio",
        ),
    ]


def load_sequence_maps() -> dict[str, dict[str, str]]:
    return {
        "horizyn": read_fasta(PROJECT_ROOT / "data/sota/prots.fasta"),
        "clipzyme": read_fasta(PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta"),
        "sabio": read_sabio_sequences(
            PROJECT_ROOT / "data/paper/sabio_rk/eval/novelty90/sequences.tsv"
        ),
    }


def collect_training_pairs() -> tuple[list[PairRecord], dict[str, object]]:
    sequences_by_source = load_sequence_maps()
    collected: list[PairRecord] = []
    stats: dict[str, object] = {"sources": []}

    for source in train_sources():
        rows = read_csv(source.pairs_path)
        smiles_by_id = reaction_smiles_by_id(source.reactions_path)
        source_stats = {
            "name": source.name,
            "dataset": source.dataset,
            "split": source.split,
            "pairs_path": str(source.pairs_path),
            "reactions_path": "" if source.reactions_path is None else str(source.reactions_path),
            "input_pairs": len(rows),
            "kept_pairs_before_collapse": 0,
            "missing_reaction_id": 0,
            "missing_reaction_smiles": 0,
            "missing_protein_id": 0,
            "missing_sequence": 0,
            "empty_sequence": 0,
        }
        for idx, row in enumerate(rows):
            source_reaction_id = (row.get("reaction_id") or "").strip()
            source_protein_id = (row.get("protein_id") or "").strip()
            if not source_reaction_id:
                source_stats["missing_reaction_id"] += 1
                continue
            if not source_protein_id:
                source_stats["missing_protein_id"] += 1
                continue
            reaction_smiles = (row.get("reaction_smiles") or smiles_by_id.get(source_reaction_id) or "").strip()
            if not reaction_smiles:
                source_stats["missing_reaction_smiles"] += 1
                continue

            if source.sequence_source == "reactzyme_pair_sequence":
                sequence = clean_sequence(row.get("protein_sequence") or "")
            else:
                sequence = sequences_by_source[source.sequence_source].get(source_protein_id, "")
            if not sequence:
                if source.sequence_source == "reactzyme_pair_sequence":
                    source_stats["empty_sequence"] += 1
                else:
                    source_stats["missing_sequence"] += 1
                continue

            collected.append(
                PairRecord(
                    reaction_id=reaction_uid(reaction_smiles),
                    reaction_smiles=reaction_smiles,
                    protein_uid=protein_uid(sequence),
                    protein_sequence=sequence,
                    source_name=source.name,
                    source_dataset=source.dataset,
                    source_split=source.split,
                    source_reaction_id=source_reaction_id,
                    source_protein_id=source_protein_id,
                    source_pair_index=idx,
                )
            )
            source_stats["kept_pairs_before_collapse"] += 1
        stats["sources"].append(source_stats)  # type: ignore[index]
    stats["kept_pairs_before_collapse"] = len(collected)
    return collected, stats


def collapse_pairs(
    records: list[PairRecord],
    protein_id_by_uid: dict[str, str],
    protein_sequence_by_id: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    pair_groups: OrderedDict[tuple[str, str], list[PairRecord]] = OrderedDict()
    reaction_rows: OrderedDict[str, dict[str, object]] = OrderedDict()
    protein_sequences: dict[str, str] = dict(protein_sequence_by_id or {})

    for record in records:
        protein_id = protein_id_by_uid[record.protein_uid]
        pair_groups.setdefault((record.reaction_id, protein_id), []).append(record)
        if record.reaction_id not in reaction_rows:
            reaction_rows[record.reaction_id] = {
                "reaction_id": record.reaction_id,
                "reaction_smiles": record.reaction_smiles,
                "source_entries": [],
                "source_reaction_ids": [],
            }
        reaction_rows[record.reaction_id]["source_entries"].append(record.source_name)  # type: ignore[index]
        reaction_rows[record.reaction_id]["source_reaction_ids"].append(record.source_reaction_id)  # type: ignore[index]
        protein_sequences.setdefault(protein_id, record.protein_sequence)

    pair_rows: list[dict[str, object]] = []
    for idx, ((reaction_id, protein_id), members) in enumerate(pair_groups.items()):
        source_entries = [
            f"{member.source_name}:{member.source_reaction_id}:{member.source_protein_id}:{member.source_pair_index}"
            for member in members
        ]
        pair_rows.append(
            {
                "pr_id": idx,
                "reaction_id": reaction_id,
                "protein_id": protein_id,
                "member_count": len(members),
                "protein_uid": members[0].protein_uid,
                "source_datasets": "|".join(sorted({member.source_dataset for member in members})),
                "source_splits": "|".join(sorted({member.source_split for member in members})),
                "source_entries": "|".join(source_entries),
            }
        )

    finalized_reactions = []
    for row in reaction_rows.values():
        row = dict(row)
        row["source_entries"] = "|".join(sorted(set(row["source_entries"])))  # type: ignore[index]
        row["source_reaction_ids"] = "|".join(sorted(set(row["source_reaction_ids"])))  # type: ignore[index]
        finalized_reactions.append(row)
    return pair_rows, finalized_reactions, protein_sequences


def write_retrieval_split(
    output_dir: Path,
    records: list[PairRecord],
    protein_id_by_uid: dict[str, str],
    metadata: dict[str, object],
    protein_sequence_by_id: dict[str, str] | None = None,
) -> dict[str, object]:
    pair_rows, reaction_rows, protein_sequences = collapse_pairs(
        records,
        protein_id_by_uid,
        protein_sequence_by_id=protein_sequence_by_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "train_pairs.csv",
        [
            "pr_id",
            "reaction_id",
            "protein_id",
            "member_count",
            "protein_uid",
            "source_datasets",
            "source_splits",
            "source_entries",
        ],
        pair_rows,
    )
    write_csv(
        output_dir / "train_rxns.csv",
        ["reaction_id", "reaction_smiles", "source_entries", "source_reaction_ids"],
        reaction_rows,
    )
    protein_items = sorted(protein_sequences.items())
    write_fasta(output_dir / "train_proteins.fasta", protein_items)
    write_fasta(output_dir / "proteins.fasta", protein_items)
    write_ids(output_dir / "candidate_ids.txt", [protein_id for protein_id, _seq in protein_items])
    split_metadata = {
        **metadata,
        "train_pairs": len(pair_rows),
        "train_reactions": len(reaction_rows),
        "train_proteins": len(protein_items),
        "collapsed_duplicate_members": sum(max(int(row["member_count"]) - 1, 0) for row in pair_rows),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(split_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return split_metadata


def find_mmseqs(path: Path) -> Path:
    if path.exists():
        return path
    found = shutil.which("mmseqs")
    if found:
        return Path(found)
    raise FileNotFoundError("Could not find MMseqs; pass --mmseqs-bin")


def run_linclust(
    *,
    mmseqs: Path,
    input_fasta: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    reuse: bool,
) -> Path:
    cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
    if reuse and cluster_tsv.exists():
        return cluster_tsv
    cluster_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(mmseqs),
        "easy-linclust",
        str(input_fasta),
        str(cluster_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    print("running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs did not create expected cluster TSV: {cluster_tsv}")
    return cluster_tsv


def read_cluster_tsv(cluster_tsv: Path, all_uids: set[str]) -> dict[str, list[str]]:
    clusters: dict[str, set[str]] = defaultdict(set)
    with cluster_tsv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            representative, member = row[0], row[1]
            if representative in all_uids and member in all_uids:
                clusters[representative].add(member)
    assigned = set()
    for representative, members in clusters.items():
        members.add(representative)
        assigned.update(members)
    for uid in all_uids - assigned:
        clusters[uid].add(uid)
    return {representative: sorted(members) for representative, members in sorted(clusters.items())}


def write_cluster_member_table(path: Path, clusters: dict[str, list[str]], cluster_id_by_rep: dict[str, str]) -> None:
    rows = []
    for representative, members in clusters.items():
        cluster_id = cluster_id_by_rep[representative]
        for member in members:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "representative_protein_uid": representative,
                    "member_protein_uid": member,
                }
            )
    write_csv(path, ["cluster_id", "representative_protein_uid", "member_protein_uid"], rows)


def make_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


def write_eval_indices(output_root: Path) -> dict[str, object]:
    eval_specs = {
        "validation/clipzyme_eval": {
            "pairs.csv": PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/pairs.csv",
            "reactions.csv": PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/reactions.csv",
            "proteins.fasta": PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta",
            "candidate_ids.txt": PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/candidate_ids.txt",
        },
        "test/horizyn": {
            "test_pairs.csv": PROJECT_ROOT / "data/sota/test_pairs.csv",
            "test_rxns.csv": PROJECT_ROOT / "data/sota/test_rxns.csv",
            "proteins.fasta": PROJECT_ROOT / "data/sota/prots.fasta",
        },
        "test/reactzyme_time": {
            "test_pairs.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/time/test_pairs.csv",
            "reactions.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/time/reactions.csv",
            "proteins.fasta": PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta",
            "candidate_ids.txt": PROJECT_ROOT / "data/paper/reactzyme/eval/time/candidate_ids.txt",
        },
        "test/reactzyme_enzyme_smi": {
            "test_pairs.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/test_pairs.csv",
            "reactions.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/reactions.csv",
            "proteins.fasta": PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta",
            "candidate_ids.txt": PROJECT_ROOT / "data/paper/reactzyme/eval/enzyme_smi/candidate_ids.txt",
        },
        "test/reactzyme_reaction_smi": {
            "test_pairs.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/test_pairs.csv",
            "reactions.csv": PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/reactions.csv",
            "proteins.fasta": PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta",
            "candidate_ids.txt": PROJECT_ROOT / "data/paper/reactzyme/eval/reaction_smi/candidate_ids.txt",
        },
    }
    manifest: dict[str, object] = {}
    for folder, files in eval_specs.items():
        folder_manifest = {}
        for name, target in files.items():
            if target.exists():
                link = output_root / folder / name
                make_symlink(link, target)
                folder_manifest[name] = str(link)
            else:
                folder_manifest[name] = {"missing": str(target)}
        manifest[folder] = folder_manifest
    return manifest


def write_ai4protein_aux(output_root: Path) -> dict[str, object]:
    src = PROJECT_ROOT / "data/huggingface/AI4Protein_EC/train_with_ec.csv"
    rows = read_csv(src)
    out_dir = output_root / "auxiliary/ai4protein_ec"
    out_rows = []
    fasta_records: OrderedDict[str, str] = OrderedDict()
    for idx, row in enumerate(rows):
        sequence = clean_sequence(row.get("aa_seq") or "")
        if not sequence:
            continue
        pid = f"ai4protein_{sha1_text(sequence)[:16]}"
        fasta_records.setdefault(pid, sequence)
        out_rows.append(
            {
                "row_id": idx,
                "protein_id": pid,
                "name": row.get("name", ""),
                "uniprot_accession": row.get("uniprot_accession", ""),
                "ec_number": row.get("ec_number", ""),
                "complete_ec_numbers": row.get("complete_ec_numbers", ""),
                "has_complete_ec": row.get("has_complete_ec", ""),
                "label": row.get("label", ""),
            }
        )
    write_csv(
        out_dir / "train_with_ec_auxiliary.csv",
        [
            "row_id",
            "protein_id",
            "name",
            "uniprot_accession",
            "ec_number",
            "complete_ec_numbers",
            "has_complete_ec",
            "label",
        ],
        out_rows,
    )
    write_fasta(out_dir / "train_sequences.fasta", fasta_records.items())
    readme = [
        "# AI4Protein/EC Auxiliary Data",
        "",
        "AI4Protein/EC contains enzyme sequences and EC labels but no reaction",
        "SMILES or reaction identifiers. It is therefore not written into the",
        "Horizyn MLNCE retrieval `train_pairs.csv` files.",
        "",
        "Use this folder for enzyme-only EC or hyperbolic pretraining, or for",
        "future EC-to-reaction mapping experiments.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return {
        "source": str(src),
        "rows": len(rows),
        "written_rows": len(out_rows),
        "unique_sequences": len(fasta_records),
        "csv": str(out_dir / "train_with_ec_auxiliary.csv"),
        "fasta": str(out_dir / "train_sequences.fasta"),
        "retrieval_pair_status": "excluded: no reaction_smiles column",
    }


def write_readme(output_root: Path, summary: dict[str, object]) -> None:
    variants = summary["train_variants"]  # type: ignore[index]
    lines = [
        "# Retrieval Training Source Collapse",
        "",
        "This artifact follows the requested dataset plan:",
        "",
        "- training sources are collapsed into one retrieval training dataset;",
        "- exact duplicate protein-reaction pairs are removed;",
        "- `nr90` and `nr50` variants additionally collapse proteins by MMseqs",
        "  sequence-similarity clusters;",
        "- test data remains the original Horizyn and ReactZyme test splits;",
        "- validation data remains the CLIPZyme/EnzymeMap eval split.",
        "",
        "AI4Protein/EC is included as an auxiliary enzyme/EC catalog because it has",
        "no reaction SMILES and cannot form Horizyn MLNCE retrieval pairs directly.",
        "",
        "## Training Variants",
        "",
        "| Variant | Train pairs | Train reactions | Train proteins | Duplicate members collapsed |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("exact", "nr90", "nr50"):
        row = variants[variant]
        lines.append(
            "| `{}` | {:,} | {:,} | {:,} | {:,} |".format(
                variant,
                int(row["train_pairs"]),
                int(row["train_reactions"]),
                int(row["train_proteins"]),
                int(row["collapsed_duplicate_members"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "```text",
            "train_exact/train_pairs.csv",
            "train_exact/train_rxns.csv",
            "train_exact/train_proteins.fasta",
            "train_nr90/train_pairs.csv",
            "train_nr90/train_rxns.csv",
            "train_nr90/train_proteins.fasta",
            "train_nr50/train_pairs.csv",
            "train_nr50/train_rxns.csv",
            "train_nr50/train_proteins.fasta",
            "validation/clipzyme_eval/",
            "test/horizyn/",
            "test/reactzyme_time/",
            "test/reactzyme_enzyme_smi/",
            "test/reactzyme_reaction_smi/",
            "auxiliary/ai4protein_ec/",
            "manifest.json",
            "```",
            "",
            "The pair files are Horizyn-compatible: they contain at least",
            "`pr_id,reaction_id,protein_id`. Extra source-provenance columns are",
            "kept for auditing.",
            "",
            "Use `train_<variant>/train_proteins.fasta` for embedding extraction.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mmseqs-bin", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--reuse-mmseqs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records, source_stats = collect_training_pairs()
    exact_sequences = {
        record.protein_uid: record.protein_sequence
        for record in records
    }
    exact_protein_id_by_uid = {uid: uid for uid in exact_sequences}
    exact_summary = write_retrieval_split(
        output_root / "train_exact",
        records,
        exact_protein_id_by_uid,
        {
            "variant": "exact",
            "protein_collapse": "identical amino-acid sequence",
            "source_stats": source_stats,
        },
    )

    exact_fasta = output_root / "train_exact/train_proteins.fasta"
    all_uids = set(exact_sequences)
    mmseqs = find_mmseqs(args.mmseqs_bin)
    train_variants = {"exact": exact_summary}
    cluster_manifests = {}
    for threshold in (90, 50):
        variant = f"nr{threshold}"
        cluster_prefix = output_root / f"train_{variant}/mmseqs/{variant}"
        tmp_dir = output_root / f"train_{variant}/mmseqs/tmp"
        cluster_tsv = run_linclust(
            mmseqs=mmseqs,
            input_fasta=exact_fasta,
            cluster_prefix=cluster_prefix,
            tmp_dir=tmp_dir,
            min_seq_id=threshold / 100.0,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
            reuse=args.reuse_mmseqs,
        )
        clusters = read_cluster_tsv(cluster_tsv, all_uids)
        cluster_id_by_rep = {
            representative: f"{variant}_{representative.removeprefix('uprot_')}"
            for representative in clusters
        }
        uid_to_cluster = {}
        for representative, members in clusters.items():
            cluster_id = cluster_id_by_rep[representative]
            for member in members:
                uid_to_cluster[member] = cluster_id
        summary = write_retrieval_split(
            output_root / f"train_{variant}",
            records,
            uid_to_cluster,
            {
                "variant": variant,
                "protein_collapse": f"MMseqs min sequence identity {threshold}% with coverage {args.coverage}",
                "mmseqs_cluster_tsv": str(cluster_tsv),
                "coverage": args.coverage,
                "cov_mode": args.cov_mode,
                "source_stats": source_stats,
            },
            protein_sequence_by_id={
                cluster_id_by_rep[representative]: exact_sequences[representative]
                for representative in clusters
            },
        )
        write_cluster_member_table(
            output_root / f"train_{variant}/cluster_members.tsv",
            clusters,
            cluster_id_by_rep,
        )
        train_variants[variant] = summary
        cluster_manifests[variant] = {
            "cluster_tsv": str(cluster_tsv),
            "clusters": len(clusters),
            "members": sum(len(members) for members in clusters.values()),
        }

    eval_manifest = write_eval_indices(output_root)
    ai4protein_manifest = write_ai4protein_aux(output_root)
    manifest = {
        "output_root": str(output_root),
        "training_sources": [
            {
                "name": source.name,
                "dataset": source.dataset,
                "split": source.split,
                "pairs_path": str(source.pairs_path),
                "reactions_path": "" if source.reactions_path is None else str(source.reactions_path),
            }
            for source in train_sources()
        ],
        "train_variants": train_variants,
        "clusters": cluster_manifests,
        "validation_and_test_indices": eval_manifest,
        "ai4protein_ec_auxiliary": ai4protein_manifest,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output_root, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
