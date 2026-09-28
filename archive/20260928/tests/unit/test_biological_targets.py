import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from horizyn.capability.biological_targets import (
    COFACTOR_LABELS,
    MECHANISM_LABELS,
    build_minimal_targets,
    canonical_protein_key,
)


def test_minimal_targets_use_row_specific_members_and_protein_hash_aliases(tmp_path):
    train_pairs = tmp_path / "train_pairs.csv"
    pd.DataFrame(
        [
            {
                "reaction_id": "source_rxn",
                "protein_id": "prot_abc123",
            }
        ]
    ).to_csv(train_pairs, index=False)
    members = tmp_path / "matched_members.csv"
    pd.DataFrame(
        [
            {
                "source_reaction_id": "source_rxn",
                "source_protein_id": "uprot_abc123",
                "reaction_id": "directional_rxn",
                "match_confidence": 1.0,
            },
            {
                "source_reaction_id": "not_in_train",
                "source_protein_id": "uprot_abc123",
                "reaction_id": "unrelated_rxn",
                "match_confidence": 1.0,
            },
        ]
    ).to_csv(members, index=False)
    features = tmp_path / "directional_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "directional_rxn",
                "reaction_center_coarse_labels": "redox_like",
                "substrate_product_transition_labels": "alcohol_to_aldehyde",
                "reaction_type_labels": "oxidation",
                "core_cofactor_labels": "NAD",
                "cofactor_labels": "NAD",
                "reaction_center_mapping_confidence": 0.8,
            },
            {
                "reaction_id": "unrelated_rxn",
                "reaction_center_coarse_labels": "phosphate_transfer_like",
                "substrate_product_transition_labels": "phosphorylation_like",
                "reaction_type_labels": "phosphorylation",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "reaction_center_mapping_confidence": 1.0,
            },
        ]
    ).to_parquet(features)
    cofactors = tmp_path / "cofactors.csv"
    pd.DataFrame(
        [
            {
                "enzyme_id": "uprot_abc123",
                "enzyme_derived_core_cofactor_labels_train": "['PLP']",
                "uniprot_core_cofactor_labels_train": "[]",
                "enzyme_uniprotkb_cofactor_labels_train": "[]",
            }
        ]
    ).to_csv(cofactors, index=False)

    ids, arrays, metadata = build_minimal_targets(
        train_pairs_path=train_pairs,
        matched_members_path=members,
        directional_features_path=features,
        enzyme_cofactor_labels_path=cofactors,
    )

    assert ids == ["prot_abc123"]
    redox = MECHANISM_LABELS.index("redox_carbonyl_interconversion")
    phosphate = MECHANISM_LABELS.index("phosphate_transfer")
    assert arrays["mechanism_targets"][0, redox] == pytest.approx(1.0)
    assert arrays["mechanism_targets"][0, phosphate] == pytest.approx(0.0)
    assert arrays["mechanism_mask"].shape == (1, len(MECHANISM_LABELS))
    assert arrays["mechanism_confidence"][0, redox] == pytest.approx(0.8)
    # The curated PLP positive is softened by contradictory reaction-level
    # evidence instead of becoming an unconditional OR label.
    assert arrays["cofactor_targets"][0, COFACTOR_LABELS.index("PLP")] > 0.8
    assert arrays["cofactor_targets"][0, COFACTOR_LABELS.index("NAD_NADP")] > 0.0
    assert metadata["num_skipped_members"] == 1
    assert metadata["schema_version"] == "biofp_minimal_v1"
    assert canonical_protein_key("uprot_abc123") == canonical_protein_key("prot_abc123")


def test_minimal_target_builder_rejects_held_out_paths(tmp_path):
    validation = tmp_path / "validation" / "pairs.csv"
    validation.parent.mkdir()
    validation.write_text("reaction_id,protein_id\nr,p\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        build_minimal_targets(
            train_pairs_path=validation,
            matched_members_path=tmp_path / "matches.csv",
            directional_features_path=tmp_path / "features.parquet",
        )


def test_minimal_target_builder_rejects_held_out_filenames(tmp_path):
    validation = tmp_path / "validation_pairs.csv"
    validation.write_text("reaction_id,protein_id\nr,p\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        build_minimal_targets(
            train_pairs_path=validation,
            matched_members_path=tmp_path / "matches.csv",
            directional_features_path=tmp_path / "features.parquet",
        )
