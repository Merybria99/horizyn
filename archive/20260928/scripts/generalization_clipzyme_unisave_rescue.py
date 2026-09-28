#!/usr/bin/env python3
"""Recover released CLIPZyme empty sequences from archived UniProt versions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RELEASE = "2022_01"
HEADER = re.compile(r"^>[^|]+\|([^|]+)\|Release ([^/|]+)(?:/[^|]+)?\|(.+)$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(accession: str) -> tuple[str, bytes]:
    url = f"https://rest.uniprot.org/unisave/{accession}?format=fasta"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EnzymeDiscovery-research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"UniSave returned HTTP {response.status}")
                return url, response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError


def parse(accession: str, data: bytes):
    entries = []
    header = None
    sequence = []
    for line in data.decode("utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(sequence)))
            header = line
            sequence = []
        else:
            sequence.append(line.strip())
    if header is not None:
        entries.append((header, "".join(sequence)))
    parsed = []
    for header, sequence in entries:
        match = HEADER.match(header)
        if not match or match.group(1) != accession or not re.fullmatch(r"[A-Z]+", sequence):
            raise ValueError(f"Unexpected UniSave record for {accession}: {header}")
        parsed.append({"release": match.group(2), "date": match.group(3),
                       "sequence": sequence})
    exact = [item for item in parsed if item["release"] == RELEASE]
    if exact:
        sequences = {item["sequence"] for item in exact}
        if len(sequences) != 1:
            raise ValueError(f"Conflicting archived {RELEASE} sequences for {accession}")
        return next(iter(sequences)), parsed, "exact_release"
    dated = [item for item in parsed if re.fullmatch(r"20\d\d_\d\d", item["release"])]
    before = [item for item in dated if item["release"] < RELEASE]
    after = [item for item in dated if item["release"] > RELEASE]
    if before and after:
        older = max(before, key=lambda item: item["release"])
        newer = min(after, key=lambda item: item["release"])
        if older["sequence"] == newer["sequence"]:
            return older["sequence"], parsed, "bracketed_identical_sequence"
    return None, parsed, "no_exact_or_stable_bracket"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.candidates.open(newline="") as handle:
        missing = [row["protein_id"] for row in csv.DictReader(handle) if not row["sequence"]]
    if len(missing) != 72 or len(set(missing)) != 72:
        raise ValueError("Unexpected official CLIPZyme missing-sequence inventory")
    args.output.mkdir(parents=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir()
    receipt = {}
    rescue = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = {executor.submit(fetch, accession): accession for accession in missing}
        for job in as_completed(jobs):
            accession = jobs[job]
            try:
                url, data = job.result()
                sequence, versions, status = parse(accession, data)
                (raw_dir / f"{accession}.fasta").write_bytes(data)
                if sequence:
                    rescue[accession] = sequence
                receipt[accession] = {"url": url, "response_sha256": digest(data),
                                      "versions": len(versions), "status": status,
                                      "selected_length": len(sequence) if sequence else None,
                                      "selected_sha256": digest(sequence.encode()) if sequence else None}
            except Exception as error:
                receipt[accession] = {"status": "request_or_parse_error", "error": str(error)}
    fasta = args.output / "rescued_2022_01.fasta"
    with fasta.open("w") as handle:
        for accession, sequence in sorted(rescue.items()):
            handle.write(f">{accession}\n{sequence}\n")
    output = {"schema": "clipzyme_unisave_sequence_rescue_v1",
              "archive_release": RELEASE, "input_sha256": digest(args.candidates.read_bytes()),
              "missing_ids": len(missing), "rescued_ids": len(rescue),
              "exact_release_rescued": sum(v["status"] == "exact_release" for v in receipt.values()),
              "bracketed_identical_rescued": sum(v["status"] == "bracketed_identical_sequence" for v in receipt.values()),
              "unresolved_ids": sorted(set(missing) - set(rescue)),
              "fasta_sha256": digest(fasta.read_bytes()),
              "source_sha256": digest(Path(__file__).read_bytes()),
              "entries": receipt}
    (args.output / "receipt.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"missing_ids": len(missing), "rescued": len(rescue),
                      "unresolved_ids": output["unresolved_ids"]}), flush=True)


if __name__ == "__main__":
    main()
