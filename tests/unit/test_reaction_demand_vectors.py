import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from horizyn.capability.reaction_demand import build_reaction_demand_vectors


def test_reaction_demand_vector_dimensions_match_vocab(tmp_path):
    features = pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "ec_numbers": ["1.1.1.1"],
                "cofactor_labels": ["NAD"],
                "reaction_center_coarse_labels": ["c_o_change"],
                "substrate_class_labels": ["alcohol"],
                "product_class_labels": ["ketone"],
                "reaction_type_labels": ["oxidoreduction"],
                "quality_flags": [],
            }
        ]
    )
    features_path = tmp_path / "reaction_features.parquet"
    features.to_parquet(features_path, index=False)
    drfp_path = tmp_path / "reaction_drfp.npz"
    np.savez_compressed(drfp_path, ids=np.array(["r1"]), vectors=np.zeros((1, 2048), dtype=np.uint8))
    vectors, metadata = build_reaction_demand_vectors(
        reaction_features_path=features_path,
        reaction_drfp_path=drfp_path,
        out_dir=tmp_path / "out",
    )
    assert metadata.iloc[0]["reaction_id"] == "r1"
    assert vectors.shape[0] == 1
    assert vectors.shape[1] == 2048 + 5


def test_reaction_demand_vector_includes_revised_focus_labels(tmp_path):
    features = pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "ec_numbers": ["1.1.1.1"],
                "cofactor_labels": ["NAD", "Mg2+"],
                "core_cofactor_labels": ["NAD"],
                "metal_ion_labels": ["Mg2+"],
                "auxiliary_participant_labels": [],
                "reaction_center_coarse_labels": ["c_o_change"],
                "substrate_class_labels": ["alcohol"],
                "product_class_labels": ["ketone"],
                "substrate_product_transition_labels": ["alcohol_to_ketone"],
                "reaction_type_labels": ["oxidoreduction"],
                "quality_flags": [],
            }
        ]
    )
    features_path = tmp_path / "reaction_features.parquet"
    features.to_parquet(features_path, index=False)
    drfp_path = tmp_path / "reaction_drfp.npz"
    np.savez_compressed(drfp_path, ids=np.array(["r1"]), vectors=np.zeros((1, 2048), dtype=np.uint8))

    vectors, metadata = build_reaction_demand_vectors(
        reaction_features_path=features_path,
        reaction_drfp_path=drfp_path,
        out_dir=tmp_path / "out_revised",
    )

    assert vectors.shape == (1, 2048 + 9)
    assert metadata.iloc[0]["core_cofactor_labels"] == ["NAD"]
    assert metadata.iloc[0]["substrate_product_transition_labels"] == ["alcohol_to_ketone"]
