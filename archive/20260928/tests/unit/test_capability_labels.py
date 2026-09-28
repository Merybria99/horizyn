import pytest

pd = pytest.importorskip("pandas")

from horizyn.capability.enzyme_labels import build_enzyme_capability_labels


def _write_reaction_features(path):
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "canonical_reaction_smiles": "CCO>>CC=O",
                "ec_numbers": ["1.1.1.1"],
                "cofactor_labels": ["NAD"],
                "core_cofactor_labels": ["NAD"],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
                "reaction_center_coarse_labels": ["c_o_change"],
                "substrate_class_labels": ["alcohol"],
                "product_class_labels": ["ketone"],
                "substrate_product_transition_labels": ["alcohol_to_ketone"],
                "reaction_type_labels": ["oxidoreduction"],
                "quality_flags": [],
            },
            {
                "reaction_id": "r2",
                "canonical_reaction_smiles": "CC(=O)O>>CCO",
                "ec_numbers": ["1.1.1.2"],
                "cofactor_labels": [],
                "core_cofactor_labels": [],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
                "reaction_center_coarse_labels": [],
                "substrate_class_labels": ["carboxylate"],
                "product_class_labels": ["alcohol"],
                "substrate_product_transition_labels": [],
                "reaction_type_labels": ["reduction"],
                "quality_flags": [],
            },
        ]
    ).to_parquet(path, index=False)


def test_enzyme_labels_use_train_only_pairs(tmp_path):
    train = tmp_path / "train_pairs.csv"
    train.write_text("pr_id,reaction_id,protein_id\n1,r1,e1\n2,r2,e1\n", encoding="utf-8")
    features = tmp_path / "reaction_features.parquet"
    _write_reaction_features(features)
    labels, pairs = build_enzyme_capability_labels(
        train_pairs_path=train,
        reaction_features_path=features,
        out_dir=tmp_path / "out",
    )
    row = labels.iloc[0]
    assert row["enzyme_id"] == "e1"
    assert row["train_reaction_ids"] == ["r1", "r2"]
    assert "NAD" in row["cofactor_labels_train"]
    assert "NAD" in row["core_cofactor_labels_train"]
    assert "alcohol_to_ketone" in row["substrate_product_transition_labels_train"]
    assert set(pairs["source_split"]) == {"train"}
    assert "core_cofactor_overlap" in pairs.columns
    assert "substrate_product_transition_overlap" in pairs.columns


def test_no_validation_or_test_pairs_in_enzyme_labels(tmp_path):
    train = tmp_path / "train_pairs.csv"
    valid = tmp_path / "valid_pairs.csv"
    train.write_text("pr_id,reaction_id,protein_id\n1,r1,e1\n", encoding="utf-8")
    valid.write_text("pr_id,reaction_id,protein_id\n1,r1,e1\n", encoding="utf-8")
    features = tmp_path / "reaction_features.parquet"
    _write_reaction_features(features)
    with pytest.raises(AssertionError):
        build_enzyme_capability_labels(
            train_pairs_path=train,
            reaction_features_path=features,
            out_dir=tmp_path / "out",
            validation_pairs_path=valid,
        )
