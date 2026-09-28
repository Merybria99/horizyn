#!/usr/bin/env python3
"""Build a ChEBI cofactor-role dictionary for reaction annotation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.cofactors import cofactor_label_for_chebi_term

CHEBI_PAGE_URL = "https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}"
OLS_TERM_URL = (
    "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/"
    "http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{obo_id}"
)


def _fetch_text(url: str, timeout: int) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_chebi_role_fillers(html_text: str, role_chebi_id: str) -> list[tuple[str, str]]:
    """Parse incoming ``has role`` rows from a ChEBI role page."""

    start = html_text.find("Incoming Relation")
    segment = html_text[start:] if start >= 0 else html_text
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", segment, flags=re.S)
    terms: list[tuple[str, str]] = []
    for row in rows:
        plain = html.unescape(re.sub(r"\s+", " ", re.sub("<.*?>", " ", row))).strip()
        if "has role" not in plain or role_chebi_id not in plain:
            continue
        match = re.search(r"<span[^>]*>(.*?)</span>\s*\(<a href=\"/chebi/(CHEBI:\d+)\"", row, flags=re.S)
        if match is None:
            continue
        name = html.unescape(re.sub("<.*?>", "", match.group(1))).strip()
        chebi_id = match.group(2)
        if chebi_id != role_chebi_id:
            terms.append((chebi_id, name))
    return sorted(set(terms), key=lambda item: (item[1].casefold(), item[0]))


def fetch_term_annotation(chebi_id: str, timeout: int) -> dict[str, list[str]]:
    obo_id = urllib.parse.quote(chebi_id.replace(":", "_"), safe="")
    data = _fetch_json(OLS_TERM_URL.format(obo_id=obo_id), timeout=timeout)
    annotations = data.get("annotation", {})
    return {
        "synonyms": [str(x) for x in data.get("synonyms", [])],
        "smiles": [str(x) for x in annotations.get("smiles_string", [])],
        "inchi_key": [str(x) for x in annotations.get("inchi_key_string", [])],
    }


def build_dictionary(role_chebi_id: str, timeout: int) -> list[dict[str, str]]:
    page = _fetch_text(CHEBI_PAGE_URL.format(chebi_id=urllib.parse.quote(role_chebi_id)), timeout=timeout)
    rows: list[dict[str, str]] = []
    for chebi_id, name in parse_chebi_role_fillers(page, role_chebi_id):
        annotation = fetch_term_annotation(chebi_id, timeout=timeout)
        aliases = sorted({name, *annotation["synonyms"]})
        rows.append(
            {
                "chebi_id": chebi_id,
                "name": name,
                "label": cofactor_label_for_chebi_term(chebi_id, name),
                "aliases": "|".join(alias for alias in aliases if alias),
                "smiles": "|".join(annotation["smiles"]),
                "inchi_key": "|".join(annotation["inchi_key"]),
                "source_role": role_chebi_id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-chebi-id", default="CHEBI:23357")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    rows = build_dictionary(args.role_chebi_id, timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chebi_id", "name", "label", "aliases", "smiles", "inchi_key", "source_role"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ChEBI cofactor rows to {out_path}")


if __name__ == "__main__":
    main()
