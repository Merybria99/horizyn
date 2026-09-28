#!/usr/bin/env python3
"""Verify and extract the official CLIPZyme v4 data release for a matched FGW audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import pickletools
from zipfile import ZipFile


DATA_MD5 = "1ebd955e83fa480aea198c20c1a66381"
SPLIT_MD5 = "fb26cfa9bd7a93c724672cd8fd3ace67"
MEMBERS = {"clipzyme_screening_set.p", "cached_enzymemap.p",
           "uniprot2sequence.p", "enzymemap.json", "ec2uniprot.p"}
SOURCE_URL = "https://zenodo.org/records/15161343/files/clipzyme_data.zip?download=1"
SPLIT_URL = "https://zenodo.org/records/15161343/files/reaction_rule_split.p?download=1"


def hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8*2**20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def safe_split(path: Path) -> dict:
    blob = path.read_bytes()
    ops = Counter(op.name for op, _, _ in pickletools.genops(blob))
    forbidden = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX",
                 "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
    if forbidden & set(ops):
        raise ValueError("Split pickle uses executable opcodes")
    value = pickle.loads(blob)
    if not isinstance(value, dict) or len(value) != 394 or not all(
            isinstance(k, int) and v in ("train", "dev", "test") for k, v in value.items()):
        raise ValueError("Unexpected official reaction-rule split")
    return {"rule_ids": len(value), "rules_per_split": dict(Counter(value.values())),
            "min_rule_id": min(value), "max_rule_id": max(value)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    if hashes(args.zip)[0] != DATA_MD5 or hashes(args.split)[0] != SPLIT_MD5:
        raise ValueError("Downloaded files fail Zenodo v4 checksums")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Release directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    with ZipFile(args.zip) as archive:
        infos = archive.infolist()
        if {x.filename for x in infos} != MEMBERS or len(infos) != len(MEMBERS):
            raise ValueError("Unexpected archive member set")
        for info in infos:
            name = info.filename
            if Path(name).name != name or info.is_dir():
                raise ValueError("Unsafe or unexpected archive member")
            target = args.output / name
            tmp = args.output / (name + ".partial")
            sha = hashlib.sha256()
            count = 0
            with archive.open(info) as reader, tmp.open("xb") as writer:
                for chunk in iter(lambda: reader.read(8*2**20), b""):
                    writer.write(chunk)
                    sha.update(chunk)
                    count += len(chunk)
            if count != info.file_size:
                raise ValueError(f"Wrong uncompressed size: {name}")
            tmp.rename(target)
            records[name] = {"path": str(target.resolve()), "bytes": count,
                             "sha256": sha.hexdigest(), "archive_crc32": f"{info.CRC:08x}"}
            print(f"Extracted {name}: {count} bytes", flush=True)
    split_target = args.output / "reaction_rule_split.p"
    split_target.write_bytes(args.split.read_bytes())
    _, split_sha = hashes(split_target)
    receipt = {"schema": "official_clipzyme_v4_data_acquisition_v1",
               "release_record": "https://zenodo.org/records/15161343",
               "zip_source_url": SOURCE_URL, "split_source_url": SPLIT_URL,
               "zip": {"md5": DATA_MD5, "sha256": hashes(args.zip)[1],
                       "bytes": args.zip.stat().st_size},
               "reaction_rule_split": {"path": str(split_target.resolve()),
                                       "md5": SPLIT_MD5, "sha256": split_sha,
                                       **safe_split(split_target)},
               "files": records, "source_code_sha256": hashes(Path(__file__))[1],
               "labels_used_to_select_model": False,
               "paper_split_entry_counts_verified": False,
               "screening_library_count_verified": False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"complete": True, "members": len(records),
                      "rules": receipt["reaction_rule_split"]["rules_per_split"]}), flush=True)


if __name__ == "__main__":
    main()
