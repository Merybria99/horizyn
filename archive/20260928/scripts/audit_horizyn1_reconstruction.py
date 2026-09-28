#!/usr/bin/env python3
"""Read-only, streaming integrity/count audit of completed reconstruction outputs.

Uses only the Python standard library. Memory scales with protein identifiers
and collapsed edges, not sequences or raw edges. Paper count differences are
reported without implying that this reconstruction is the authors' dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Iterator

PAPER_TARGETS = {
    "raw_reactions": 31_101,
    "raw_proteins": 27_099_893,
    "raw_pairs": 34_385_290,
    "clustered_proteins": 7_063_237,
    "clustered_pairs": 8_897_870,
}
ARTIFACTS = {
    "raw_manifest": "raw/raw_manifest.json",
    "raw_proteins": "raw/raw_proteins.fasta",
    "raw_reactions": "raw/raw_reactions.tsv",
    "raw_pairs": "raw/raw_pairs.tsv",
    "clustered_manifest": "clustered/clustered_manifest.json",
    "clustered_proteins": "clustered/proteins.fasta",
    "clusters": "clustered/clusters.tsv",
    "clustered_pairs": "clustered/pairs.tsv",
}
PROGRESS_INTERVAL = 1_000_000


def progress(message: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}",
        file=sys.stderr,
        flush=True,
    )


def signature(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class Audit:
    def __init__(self, run_root: Path, stage: str):
        self.stage = stage
        self.paths = {name: run_root.resolve() / relative for name, relative in ARTIFACTS.items()}
        self.checks: dict[str, dict] = {}
        self.counts: dict[str, int] = {}
        self.source_bits: dict[str, int] = {}
        self.artifact_signatures: dict[str, list[int]] = {}
        self.incomplete = False

    def check(self, name: str) -> None:
        self.checks.setdefault(name, {"status": "passed", "error_count": 0, "examples": []})

    def error(self, name: str, message: str) -> None:
        self.check(name)
        result = self.checks[name]
        result["status"] = "failed"
        result["error_count"] += 1
        if len(result["examples"]) < 5:
            result["examples"].append(message)

    def rows(
        self, path: Path, check: str, header: list[str] | None, columns: int
    ) -> Iterator[tuple[int, list[str]]]:
        self.check(check)
        with path.open(encoding="utf-8", newline="") as handle:
            if header is not None:
                actual = handle.readline().rstrip("\r\n").split("\t")
                if actual != header:
                    self.error(check, f"{path.name}: expected header {header!r}, got {actual!r}")
                    return
            for index, line in enumerate(handle, start=2 if header is not None else 1):
                row = line.rstrip("\r\n").split("\t")
                if len(row) != columns:
                    self.error(check, f"{path.name}:{index}: expected {columns} columns")
                    continue
                if index % PROGRESS_INTERVAL == 0:
                    progress(f"{path.name}: {index:,} lines read")
                yield index, row

    def fasta(
        self,
        path: Path,
        check: str,
        *,
        retain_hashes: bool = False,
        expected_hashes: dict[str, bytes] | None = None,
    ) -> tuple[dict[str, int | None], dict[str, bytes]]:
        progress(f"Reading {path}")
        self.check(check)
        identifiers: dict[str, int | None] = {}
        hashes: dict[str, bytes] = {}
        protein: str | None = None
        length = records = residues = 0
        digest = None

        def finish() -> None:
            if protein is None:
                return
            if length == 0:
                self.error(check, f"Empty sequence for {protein!r}")
            if digest is not None:
                value = digest.digest()
                if retain_hashes:
                    hashes[protein] = value
                if expected_hashes is not None and protein in expected_hashes:
                    if value != expected_hashes[protein]:
                        self.error(
                            "representative_sequences", f"Raw sequence differs for {protein!r}"
                        )

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                sequence = line.strip()
                if line.startswith(">"):
                    finish()
                    tokens = line[1:].split()
                    protein = tokens[0] if tokens else ""
                    records += 1
                    if not protein:
                        self.error(check, f"Empty FASTA identifier at line {line_number}")
                    if protein in identifiers:
                        self.error(check, f"Duplicate FASTA identifier {protein!r}")
                    identifiers[protein] = None
                    length = 0
                    digest = (
                        hashlib.sha256()
                        if (
                            retain_hashes
                            or (expected_hashes is not None and protein in expected_hashes)
                        )
                        else None
                    )
                    if records % PROGRESS_INTERVAL == 0:
                        progress(f"{path.name}: {records:,} records read")
                elif sequence:
                    if protein is None:
                        self.error(
                            check, f"Sequence before first FASTA header at line {line_number}"
                        )
                        continue
                    if not sequence.isascii() or not sequence.isalpha():
                        self.error(
                            check,
                            f"Invalid amino-acid characters for {protein!r}, line {line_number}",
                        )
                    length += len(sequence)
                    residues += len(sequence)
                    if digest is not None:
                        digest.update(sequence.upper().encode("utf-8"))
            finish()
        if not records:
            self.error(check, "No FASTA records")
        self.counts[check] = records
        self.counts[f"{check}_unique_ids"] = len(identifiers)
        self.counts[f"{check}_residues"] = residues
        return identifiers, hashes

    def reactions(self) -> dict[str, int]:
        result: dict[str, int] = {}
        count = 0
        for line, (reaction, chemistry) in self.rows(
            self.paths["raw_reactions"],
            "raw_reactions",
            ["reaction_id", "reaction_smiles"],
            2,
        ):
            count += 1
            if not reaction or any(c.isspace() for c in reaction):
                self.error("raw_reactions", f"Invalid reaction identifier at line {line}")
            if not chemistry.strip():
                self.error("raw_reactions", f"Empty reaction chemistry for {reaction!r}")
            if reaction in result:
                self.error("raw_reactions", f"Duplicate reaction identifier {reaction!r}")
            else:
                result[reaction] = len(result)
        if not count:
            self.error("raw_reactions", "No reaction rows")
        self.counts["raw_reactions"] = count
        self.counts["raw_reactions_unique_ids"] = len(result)
        return result

    def clusters(self, proteins: dict[str, int | None], representatives: dict[str, int]) -> None:
        progress("Checking complete, unique cluster membership")
        rows = assigned = 0
        for line, (representative, member) in self.rows(
            self.paths["clusters"],
            "cluster_membership",
            None,
            2,
        ):
            rows += 1
            if representative not in representatives:
                self.error(
                    "cluster_membership", f"Unknown representative {representative!r}, line {line}"
                )
            if member not in proteins:
                self.error("cluster_membership", f"Unknown raw member {member!r}, line {line}")
            elif proteins[member] is not None:
                self.error(
                    "cluster_membership", f"Duplicate membership for {member!r}, line {line}"
                )
            elif representative in representatives:
                proteins[member] = representatives[representative]
                assigned += 1
        if assigned != len(proteins):
            missing = list(islice((key for key, value in proteins.items() if value is None), 5))
            self.error(
                "cluster_membership",
                f"{len(proteins) - assigned:,} raw proteins unmapped; examples {missing!r}",
            )
        for representative, index in representatives.items():
            if proteins.get(representative) != index:
                self.error(
                    "cluster_membership",
                    f"Representative lacks self-membership: {representative!r}",
                )
        self.counts["clustered_members"] = rows
        self.counts["clustered_unique_members"] = assigned

    def source_mask(self, value: str, check: str, line: int) -> int:
        mask = 0
        for source in value.split(","):
            source = source.strip()
            if not source:
                self.error(check, f"Empty provenance token at line {line}")
                continue
            if source not in self.source_bits:
                self.source_bits[source] = 1 << len(self.source_bits)
            mask |= self.source_bits[source]
        return mask

    def pairs(
        self,
        name: str,
        proteins: dict[str, int | None],
        reactions: dict[str, int],
        representative_count: int = 0,
        expected: dict[int, int] | None = None,
        build_expected: bool = False,
    ) -> dict[int, int]:
        progress(f"Checking {self.paths[name]}")
        result = {} if expected is None else expected
        previous: tuple[str, str] | None = None
        count = 0
        for line, (reaction, protein, sources) in self.rows(
            self.paths[name],
            name,
            ["reaction_id", "protein_id", "sources"],
            3,
        ):
            count += 1
            key = reaction, protein
            if previous is not None and key <= previous:
                self.error(name, f"Edges not strictly sorted/unique at line {line}: {key!r}")
            previous = key
            mask = self.source_mask(sources, name, line)
            if reaction not in reactions:
                self.error(name, f"Unknown reaction {reaction!r}, line {line}")
            if protein not in proteins:
                self.error(name, f"Unknown protein {protein!r}, line {line}")
            if reaction not in reactions or protein not in proteins:
                continue
            if expected is None and not build_expected:
                continue
            representative = proteins[protein]
            if representative is None:
                self.error(
                    "collapsed_edge_union", f"Raw edge references unmapped member {protein!r}"
                )
                continue
            edge = reactions[reaction] * representative_count + representative
            if build_expected:
                result[edge] = result.get(edge, 0) | mask
            else:
                wanted = result.pop(edge, None)
                if wanted is None:
                    self.error("collapsed_edge_union", f"Extra or duplicate final edge {key!r}")
                elif wanted != mask:
                    self.error(
                        "collapsed_source_union", f"Provenance union differs for final edge {key!r}"
                    )
        if not count:
            self.error(name, "No pair rows")
        self.counts[name] = count
        return result

    def compare_manifest(self, name: str, manifest: dict, keys: tuple[str, ...]) -> None:
        self.check(name)
        for key in keys:
            value = manifest.get(key)
            if type(value) is not int or value != self.counts[key]:
                self.error(name, f"{key}: manifest {value!r}, observed {self.counts[key]:,}")

    def run(self) -> dict:
        required = {
            name: path
            for name, path in self.paths.items()
            if self.stage == "clustered" or name.startswith("raw_")
        }
        self.check("completed_inputs")
        snapshots = {}
        for name, path in required.items():
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    self.error("completed_inputs", f"Missing or empty completed artifact: {path}")
                    self.incomplete = True
                else:
                    snapshots[name] = signature(path)
                    self.artifact_signatures[str(path.resolve())] = list(snapshots[name])
                if path.with_name(path.name + ".partial").exists():
                    self.error("completed_inputs", f"Partial artifact exists alongside {path}")
                    self.incomplete = True
            except OSError as exc:
                self.error("completed_inputs", f"Cannot inspect {path}: {exc}")
                self.incomplete = True
        if self.incomplete:
            return self.report()
        try:
            manifests = {}
            for name in ("raw_manifest", "clustered_manifest"):
                if name in required:
                    with required[name].open(encoding="utf-8") as handle:
                        manifests[name] = json.load(handle)
                    if not isinstance(manifests[name], dict):
                        raise ValueError(f"{name} must contain a JSON object")
            representatives = {}
            representative_hashes = None
            if self.stage == "clustered":
                rep_ids, representative_hashes = self.fasta(
                    self.paths["clustered_proteins"],
                    "clustered_proteins",
                    retain_hashes=True,
                )
                representatives = {protein: index for index, protein in enumerate(rep_ids)}
                del rep_ids
                self.check("representative_sequences")
            proteins, _ = self.fasta(
                self.paths["raw_proteins"],
                "raw_proteins",
                expected_hashes=representative_hashes,
            )
            del representative_hashes
            reactions = self.reactions()
            if self.stage == "clustered":
                for representative in representatives:
                    if representative not in proteins:
                        self.error(
                            "representative_sequences",
                            f"Representative absent from raw FASTA: {representative!r}",
                        )
                self.clusters(proteins, representatives)
                self.check("collapsed_edge_union")
                self.check("collapsed_source_union")
            expected = self.pairs(
                "raw_pairs",
                proteins,
                reactions,
                len(representatives),
                build_expected=self.stage == "clustered",
            )
            del proteins
            self.compare_manifest(
                "raw_manifest",
                manifests["raw_manifest"],
                ("raw_reactions", "raw_proteins", "raw_pairs"),
            )
            if self.stage == "clustered":
                self.counts["expected_collapsed_pairs"] = len(expected)
                self.pairs(
                    "clustered_pairs", representatives, reactions, len(representatives), expected
                )
                if expected:
                    self.error(
                        "collapsed_edge_union",
                        f"{len(expected):,} collapsed raw edges missing from final pairs",
                    )
                self.counts["missing_collapsed_pairs"] = len(expected)
                self.compare_manifest(
                    "clustered_manifest",
                    manifests["clustered_manifest"],
                    ("clustered_proteins", "clustered_pairs", "clustered_members"),
                )
        except (OSError, ValueError, UnicodeError) as exc:
            self.error("read_error", f"{type(exc).__name__}: {exc}")
        self.check("input_stability")
        for name, before in snapshots.items():
            path = required[name]
            try:
                changed = (
                    signature(path) != before or path.with_name(path.name + ".partial").exists()
                )
            except OSError:
                changed = True
            if changed:
                self.error("input_stability", f"Artifact changed during audit: {path}")
                self.incomplete = True
        return self.report()

    def report(self) -> dict:
        errors = [
            {"check": name, "count": value["error_count"], "examples": value["examples"]}
            for name, value in self.checks.items()
            if value["status"] == "failed"
        ]
        return {
            "status": "incomplete" if self.incomplete else ("failed" if errors else "passed"),
            "stage": self.stage,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "paths": {name: str(path) for name, path in self.paths.items()},
            "artifact_signatures": self.artifact_signatures,
            "checks": self.checks,
            "errors": errors,
            "observed_counts": self.counts,
            "paper_targets": PAPER_TARGETS,
            "paper_deltas": {
                key: self.counts[key] - target
                for key, target in PAPER_TARGETS.items()
                if key in self.counts
            },
            "provenance_sources": sorted(self.source_bits),
            "limitations": [
                "Paper-count differences are informative, not integrity failures or proof of exact reproduction.",
                "Chemistry is checked for presence, not chemical correctness or biological equivalence.",
                "This audit does not prove source annotation correctness, split leakage freedom, or MMseqs identity thresholds.",
                "Representative sequences are compared case-insensitively with streaming SHA-256 hashes.",
            ],
        }


def protect_output(output: Path, paths: dict[str, Path]) -> None:
    for original in paths.values():
        for protected in (original, original.with_name(original.name + ".partial")):
            if output.resolve() == protected.resolve() or (
                output.exists() and protected.exists() and output.samefile(protected)
            ):
                raise ValueError(f"Report output aliases an input artifact: {protected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("raw", "clustered"), default="clustered")
    parser.add_argument(
        "--output", type=Path, help="Optional JSON report (inputs are never modified)"
    )
    args = parser.parse_args(argv)
    audit = Audit(args.run_root, args.stage)
    if args.output is not None:
        try:
            protect_output(args.output, audit.paths)
        except (OSError, ValueError) as exc:
            audit.error("report_output", str(exc))
            print(json.dumps(audit.report(), indent=2, sort_keys=True))
            return 2
    report = audit.run()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        temporary = None
        try:
            # Recheck immediately before writing, including symlinks and hard links.
            protect_output(args.output, audit.paths)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=args.output.parent,
                prefix=args.output.name + ".",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            os.replace(temporary, args.output)
        except (OSError, ValueError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            print(f"Could not write report: {exc}", file=sys.stderr)
            audit.error("report_output", str(exc))
            print(json.dumps(audit.report(), indent=2, sort_keys=True))
            return 2
    print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
