#!/usr/bin/env python3
"""Download reviewed UniProt EC labels for Horizyn protein IDs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    import h5py
except ImportError:  # pragma: no cover - only needed for --embeddings-path.
    h5py = None


UNIPROT_REST_URL = "https://rest.uniprot.org"


def _decode_id(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def collect_hdf5_ids(path: str | Path) -> list[str]:
    if h5py is None:
        raise ImportError("h5py is required for --embeddings-path input")
    with h5py.File(path, "r") as h5_file:
        if "ids" not in h5_file:
            raise KeyError(f"HDF5 file {path} does not contain required 'ids' dataset")
        return [_decode_id(value) for value in h5_file["ids"][:]]


def collect_csv_ids(path: str | Path, id_columns: Iterable[str]) -> list[str]:
    columns = list(id_columns)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in columns if column not in header]
        if missing:
            raise ValueError(f"Missing ID columns {missing} in {path}; available={header}")
        ids: list[str] = []
        for row in reader:
            for column in columns:
                value = row.get(column, "").strip()
                if value:
                    ids.append(value)
        return ids


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _request_json(url: str, data: bytes | None = None) -> dict:
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - user-requested data API
        return json.loads(response.read().decode("utf-8"))


def submit_mapping_job(ids: list[str], api_url: str = UNIPROT_REST_URL) -> str:
    payload = urllib.parse.urlencode(
        {
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ",".join(ids),
        }
    ).encode("utf-8")
    response = _request_json(f"{api_url}/idmapping/run", data=payload)
    job_id = response.get("jobId")
    if not job_id:
        raise RuntimeError(f"UniProt ID mapping response did not include jobId: {response}")
    return str(job_id)


def wait_for_mapping_job(
    job_id: str,
    api_url: str = UNIPROT_REST_URL,
    poll_interval: float = 3.0,
    timeout: float = 600.0,
) -> str:
    deadline = time.time() + timeout
    status_url = f"{api_url}/idmapping/status/{job_id}"
    while True:
        response = _request_json(status_url)
        status = response.get("jobStatus")
        if status == "RUNNING":
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for UniProt ID mapping job {job_id}")
            time.sleep(poll_interval)
            continue
        if status in {"FAILED", "ERROR"}:
            raise RuntimeError(f"UniProt ID mapping job {job_id} failed: {response}")
        redirect = response.get("redirectURL")
        if redirect:
            redirect = str(redirect)
            return redirect if redirect.startswith("http") else f"{api_url}{redirect}"
        return f"{api_url}/idmapping/uniprotkb/results/{job_id}"


def _next_link(headers: list[tuple[str, str]]) -> str | None:
    for key, value in headers:
        if key.lower() != "link":
            continue
        match = re.search(r"<([^>]+)>;\s*rel=\"next\"", value)
        if match:
            return match.group(1)
    return None


def fetch_mapping_tsv(
    results_url: str,
    fields: str = "accession,id,reviewed,ec",
    size: int = 500,
) -> str:
    separator = "&" if "?" in results_url else "?"
    url = f"{results_url}{separator}{urllib.parse.urlencode({'format': 'tsv', 'fields': fields, 'size': size})}"
    chunks: list[str] = []
    while url:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - user-requested data API
            chunks.append(response.read().decode("utf-8"))
            url = _next_link(response.getheaders())
    if not chunks:
        return ""
    header = chunks[0].splitlines()[0] if chunks[0].splitlines() else ""
    rows: list[str] = []
    for idx, chunk in enumerate(chunks):
        lines = chunk.splitlines()
        if idx == 0:
            rows.extend(lines)
        else:
            rows.extend(line for line in lines if line != header)
    return "\n".join(rows) + "\n"


def _pick(row: dict[str, str], *names: str) -> str:
    normalized = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.lower().strip())
        if value is not None:
            return value
    return ""


def normalize_mapping_rows(tsv_text: str, reviewed_only: bool = True) -> list[dict[str, str]]:
    if not tsv_text.strip():
        return []
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    for row in reader:
        reviewed = _pick(row, "Reviewed")
        reviewed_value = reviewed.strip().lower()
        if reviewed_only and reviewed_value not in {"reviewed", "true", "yes"}:
            continue
        protein_id = _pick(row, "From", "Entry", "accession").strip()
        accession = _pick(row, "Entry", "accession").strip()
        ec_number = _pick(row, "EC number", "ec").strip()
        if not protein_id or not accession:
            continue
        rows.append(
            {
                "protein_id": protein_id,
                "uniprot_accession": accession,
                "entry_name": _pick(row, "Entry Name", "id").strip(),
                "reviewed": reviewed,
                "ec_number": ec_number,
            }
        )
    return rows


def write_label_cache(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protein_id", "uniprot_accession", "entry_name", "reviewed", "ec_number"]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-path", default=None, help="HDF5 with an 'ids' dataset")
    parser.add_argument("--csv-path", action="append", default=[], help="CSV containing protein IDs")
    parser.add_argument(
        "--csv-id-columns",
        default="protein_id",
        help="Comma-separated ID columns for --csv-path files",
    )
    parser.add_argument(
        "--output-path",
        default="data/sota/uniprot_reviewed_ec_labels.csv",
        help="Output CSV cache path",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-url", default=UNIPROT_REST_URL)
    parser.add_argument(
        "--reviewed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only reviewed Swiss-Prot entries",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collected: list[str] = []
    if args.embeddings_path:
        collected.extend(collect_hdf5_ids(args.embeddings_path))
    csv_id_columns = [column.strip() for column in args.csv_id_columns.split(",") if column.strip()]
    for csv_path in args.csv_path:
        collected.extend(collect_csv_ids(csv_path, csv_id_columns))

    ids = unique_preserve_order(collected)
    if not ids:
        raise SystemExit("No protein IDs found. Provide --embeddings-path or --csv-path.")

    all_rows: list[dict[str, str]] = []
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start : start + args.batch_size]
        print(f"Submitting UniProt mapping batch {start // args.batch_size + 1}: {len(batch)} IDs")
        job_id = submit_mapping_job(batch, api_url=args.api_url)
        results_url = wait_for_mapping_job(
            job_id,
            api_url=args.api_url,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        tsv_text = fetch_mapping_tsv(results_url)
        all_rows.extend(normalize_mapping_rows(tsv_text, reviewed_only=args.reviewed_only))

    write_label_cache(all_rows, args.output_path)
    mapped = {row["protein_id"] for row in all_rows}
    with_ec = [row for row in all_rows if row["ec_number"]]
    complete_ec = [
        row
        for row in with_ec
        if any(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ec.strip()) for ec in row["ec_number"].split(";"))
    ]
    print(f"Requested IDs: {len(ids)}")
    print(f"Rows written: {len(all_rows)}")
    label = "Mapped reviewed IDs" if args.reviewed_only else "Mapped UniProt IDs"
    print(f"{label}: {len(mapped)}")
    print(f"Rows with EC labels: {len(with_ec)}")
    print(f"Rows with complete four-level EC labels: {len(complete_ec)}")
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
