#!/usr/bin/env python3
"""Authenticate the original frozen phase2 campaign and its 30 prediction jobs."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for nested in value.values():
            yield from records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from records(nested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    phase = root / "phase2"
    digests, verified = {}, []

    def digest(path):
        path = path.resolve()
        if path not in digests:
            value = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 << 20), b""):
                    value.update(block)
            digests[path] = value.hexdigest()
        return digests[path]

    def verify(record, parent):
        path = Path(record["path"])
        if not path.is_absolute():
            path = parent / path
        actual = digest(path)
        if actual != record["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        verified.append(dict(path=str(path.resolve()), sha256=actual))

    for path, expected in [
        (root / "frozen_recipe.json", "5b15a59024c1e32c8bf43b9b11f7e58cd853d13d3bbe21762204aaf6e19d149b"),
        (phase / "frozen_recipe.json", "624084b2535afdeb030e8f34adc1ce6cc3a340e6d5136858f408f41d94e09d76"),
    ]:
        verify(dict(path=str(path), sha256=expected), path.parent)
    freeze = json.loads((phase / "frozen_recipe.json").read_text())
    for record in records(freeze):
        verify(record, phase)
    index = json.loads((phase / "prediction_index.json").read_text())
    jobs = index["jobs"]
    if len(jobs) != 30 or any(job["status"] != "complete" for job in jobs):
        raise ValueError("Original phase2 must have all30 completed prediction jobs")
    for job in jobs:
        out = Path(job["output"])
        receipt = json.loads((out / "complete.json").read_text())
        if receipt["labels_used"] is not False:
            raise ValueError("Prediction job used evaluation labels")
        if receipt["phase2_frozen_recipe"]["sha256"] != digest(phase / "frozen_recipe.json"):
            raise ValueError("Prediction belongs to another freeze")
        for record in records(receipt):
            verify(record, out)
        for filename, key in [("scores.npz", "output_sha256"), ("support_diagnostics.npz", "support_diagnostics_sha256")]:
            verify(dict(path=str(out / filename), sha256=receipt[key]), out)
        if digest(out / "catalog.json") != receipt["inputs"]["catalog"]["sha256"]:
            raise ValueError("Copied prediction catalog changed")
    output = dict(schema="frozen_phase2_full_integrity_audit_v1", passed=True,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prediction_jobs=len(jobs), unique_files_hashed=len(digests), record_checks=len(verified),
        frozen_implementation_sources=len(freeze["implementation_sources"]),
        scope="Both freeze identities; all phase2 frozen path/hash references; original30 completed prediction jobs and their input/output hashes; copied catalogs. Does not replace scientific validation or recursively reinterpret historical JSON receipts.",
        verified_files=[dict(path=str(path), sha256=value) for path,value in sorted(digests.items())],
        script_sha256=digest(Path(__file__)))
    (phase / "integrity_audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("verified_files", "scope")}))


if __name__ == "__main__":
    main()
