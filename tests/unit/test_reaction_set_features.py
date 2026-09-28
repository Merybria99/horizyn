import csv
import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from horizyn.capability.reaction_set_features import (
    COFACTOR_CAPACITY,
    REACTION_SET_BASE_DIM,
    REACTION_SET_FULL_DIM,
    build_reaction_set_feature_splits,
    materialize_reaction_set_features_without_cofactors,
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


def test_no_cofactor_variant_removes_only_final_indicator_block(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    ids = np.asarray(["r1", "r2"], dtype=object)
    base = np.arange(2 * REACTION_SET_BASE_DIM, dtype=np.float32).reshape(
        2, REACTION_SET_BASE_DIM
    )
    cofactors = np.zeros((2, COFACTOR_CAPACITY), dtype=np.float32)
    cofactors[1, 3] = 1.0
    vectors = np.concatenate([base, cofactors], axis=1)
    assert vectors.shape[1] == REACTION_SET_FULL_DIM
    np.savez_compressed(
        source,
        ids=ids,
        vectors=vectors,
        mask=np.asarray([True, False]),
    )

    report = materialize_reaction_set_features_without_cofactors(
        source_path=source,
        output_path=output,
    )

    with np.load(output, allow_pickle=True) as payload:
        np.testing.assert_array_equal(payload["ids"], ids)
        np.testing.assert_array_equal(payload["vectors"], base)
        np.testing.assert_array_equal(payload["mask"], [True, False])
    assert report["source_dimension"] == REACTION_SET_FULL_DIM
    assert report["output_dimension"] == REACTION_SET_BASE_DIM
    assert report["removed_dimension"] == COFACTOR_CAPACITY
    assert report["num_with_cofactor_indicator"] == 1


def test_no_cofactor_variant_rejects_non_binary_trailing_columns(tmp_path):
    source = tmp_path / "source.npz"
    vectors = np.zeros((1, REACTION_SET_FULL_DIM), dtype=np.float32)
    vectors[0, -1] = 0.5
    np.savez_compressed(
        source,
        ids=np.asarray(["r1"], dtype=object),
        vectors=vectors,
        mask=np.asarray([True]),
    )

    with pytest.raises(ValueError, match="not binary"):
        materialize_reaction_set_features_without_cofactors(
            source_path=source,
            output_path=tmp_path / "output.npz",
        )
