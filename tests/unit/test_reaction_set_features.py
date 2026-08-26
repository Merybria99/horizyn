import csv
import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from horizyn.capability.reaction_set_features import (
    COFACTOR_CAPACITY,
    build_reaction_set_feature_splits,
    molecule_set_components,
)


def _write_reactions(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "reaction_smiles"])
        writer.writerows(rows)


def test_molecule_set_components_are_self_reaction_and_permutation_invariant():
    first = sorted(molecule_set_components("CCO.CC(=O)O>>CC(=O)O.CCO"))
    second = sorted(molecule_set_components("CC(=O)O.CCO"))
    assert first == second
    assert len(first) == 2


def test_reaction_set_builder_uses_train_schema_and_fixed_capacity(tmp_path):
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    test = tmp_path / "test.csv"
    dictionary = tmp_path / "cofactors.csv"
    out = tmp_path / "features"
    _write_reactions(train, [("r1", "CCO.CC(=O)O>>CC(=O)O.CCO")])
    _write_reactions(validation, [("r2", "CCN.CCO")])
    _write_reactions(test, [("r3", "CCO.CCN")])
    dictionary.write_text("label,alias\nNAD,NAD\n", encoding="utf-8")

    report = build_reaction_set_feature_splits(
        train_reactions_path=train,
        validation_reactions_path=validation,
        test_reactions_path=test,
        cofactor_dictionary_path=dictionary,
        out_dir=out,
        morgan_bits=32,
    )

    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    payload = np.load(out / "validation_reaction_set_features.npz", allow_pickle=True)
    assert schema["fit_split"] == "train"
    assert schema["core_cofactor_capacity"] == COFACTOR_CAPACITY
    assert payload["vectors"].shape == (1, 32 + 64 + 4 + COFACTOR_CAPACITY)
    assert report["splits"]["validation"]["num_valid"] == 1
