import csv
import json

import pytest

from horizyn.chemistry.isomeric_smiles import (
    SMILES_MODE,
    canonicalize_isomeric_reaction_smiles,
    materialize_isomeric_reaction_csv,
)


def test_canonicalization_preserves_stereo_atom_maps_and_component_order():
    source = "[CH3:8][C@@H:7](O)C(=O)O.[13CH3:2][NH3+:1]>>[CH3:8][C@H:7](O)C(=O)O"
    result = canonicalize_isomeric_reaction_smiles(source)
    left, right = result.split(">>")
    first, second = left.split(".")
    assert "@@" in first or "@" in first
    assert ":7" in first and ":8" in first
    assert "13" in second and ":1" in second and ":2" in second
    assert "@" in right


def test_missing_stereo_is_not_inferred():
    result = canonicalize_isomeric_reaction_smiles("CC(O)C(=O)O>>CC(O)C(=O)O")
    assert "@" not in result


def test_invalid_component_fails_instead_of_being_dropped():
    with pytest.raises(ValueError, match="invalid molecular SMILES"):
        canonicalize_isomeric_reaction_smiles("CCO.not_a_smiles>>CCO")


def test_materialization_preserves_rows_ids_and_writes_provenance(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reaction_id", "reaction_smiles", "extra"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"reaction_id": "r2", "reaction_smiles": "OCC>>CCO", "extra": 2},
                {
                    "reaction_id": "r1",
                    "reaction_smiles": "F[C@H](Cl)Br>>F[C@@H](Cl)Br",
                    "extra": 1,
                },
            ]
        )
    report = materialize_isomeric_reaction_csv(source, output)
    with output.open("r", encoding="utf-8", newline="") as handle:
        materialized = list(csv.DictReader(handle))
    assert [row["reaction_id"] for row in materialized] == ["r2", "r1"]
    assert [row["extra"] for row in materialized] == ["2", "1"]
    assert report["smiles_mode"] == SMILES_MODE
    assert report["coverage"] == 1.0
    manifest = json.loads(output.with_suffix(".csv.provenance.json").read_text())
    assert manifest["source_csv_sha256"] == report["source_csv_sha256"]
