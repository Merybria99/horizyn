"""Tests for enzyme BioFP soft target construction."""

import csv

import numpy as np
import pytest


def test_builder_keeps_known_all_zero_targets_and_filters_allowed_reactions(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from scripts.build_enzyme_biofp_soft_targets import build_soft_targets

    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pr_id", "reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerow({"pr_id": "p1", "reaction_id": "rxn_known_zero", "protein_id": "enz_zero"})
        writer.writerow({"pr_id": "p2", "reaction_id": "rxn_labeled", "protein_id": "enz_labeled"})
        writer.writerow({"pr_id": "p3", "reaction_id": "rxn_filtered", "protein_id": "enz_filtered"})

    features_path = tmp_path / "reaction_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "rxn_known_zero",
                "reaction_center_coarse_labels": "charge_change",
                "core_cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "",
            },
            {
                "reaction_id": "rxn_labeled",
                "reaction_center_coarse_labels": "redox_like",
                "core_cofactor_labels": "NAD",
                "substrate_product_transition_labels": "alcohol_to_aldehyde",
                "reaction_type_labels": "oxidation",
            },
            {
                "reaction_id": "rxn_filtered",
                "reaction_center_coarse_labels": "redox_like",
                "core_cofactor_labels": "NAD",
                "substrate_product_transition_labels": "alcohol_to_aldehyde",
                "reaction_type_labels": "oxidation",
            },
        ]
    ).to_parquet(features_path)

    ids, arrays, metadata = build_soft_targets(
        pairs_path,
        features_path,
        allowed_reactions={"rxn_known_zero", "rxn_labeled"},
    )

    assert ids == ["enz_labeled", "enz_zero"]
    zero_idx = ids.index("enz_zero")
    labeled_idx = ids.index("enz_labeled")
    assert arrays["center_mask"][zero_idx]
    assert arrays["center_denominator"][zero_idx] == pytest.approx(1.0)
    assert arrays["center_targets"][zero_idx].sum() == pytest.approx(0.0)
    assert arrays["center_targets"][labeled_idx].sum() > 0.0
    assert metadata["num_train_pairs_filtered_by_allowed_reactions"] == 1
    assert metadata["zero_positive_known_targets"]["center"] == 1


def test_builder_can_use_enhanced_cofactors_and_glycosyl_rescue(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from scripts.build_enzyme_biofp_soft_targets import (
        COFACTOR_LABELS,
        build_soft_targets,
    )

    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerow({"reaction_id": "rxn_no_core", "protein_id": "enz_plp"})
        writer.writerow({"reaction_id": "rxn_glycosyl", "protein_id": "enz_gly"})
        writer.writerow({"reaction_id": "rxn_biotin", "protein_id": "enz_biotin"})

    features_path = tmp_path / "reaction_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "rxn_no_core",
                "reaction_center_coarse_labels": "c_n_change",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "",
            },
            {
                "reaction_id": "rxn_glycosyl",
                "reaction_center_coarse_labels": "c_o_change",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "glycosyl_transfer",
            },
            {
                "reaction_id": "rxn_biotin",
                "reaction_center_coarse_labels": "c_c_change",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "",
            },
        ]
    ).to_parquet(features_path)

    enhanced_path = tmp_path / "enhanced.csv"
    with enhanced_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "enzyme_id",
                "combined_core_cofactor_labels_train",
                "cofactor_architecture_bins_train",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "enzyme_id": "enz_plp",
                "combined_core_cofactor_labels_train": "['PLP']",
                "cofactor_architecture_bins_train": "['PLP_dependent']",
            }
        )
        writer.writerow(
            {
                "enzyme_id": "enz_biotin",
                "combined_core_cofactor_labels_train": "['biotin']",
                "cofactor_architecture_bins_train": "['other_organic_group_transfer']",
            }
        )

    ids, arrays, metadata = build_soft_targets(
        pairs_path,
        features_path,
        enzyme_cofactor_labels=enhanced_path,
        glycosyl_transfer_cofactor_rule="type",
    )

    idx_plp = ids.index("enz_plp")
    idx_gly = ids.index("enz_gly")
    idx_biotin = ids.index("enz_biotin")
    plp_col = COFACTOR_LABELS.index("PLP_like")
    sugar_col = COFACTOR_LABELS.index("sugar_nucleotide_glycosyl")

    assert arrays["cofactor_mask"][idx_plp]
    assert arrays["cofactor_targets"][idx_plp, plp_col] > 0.0
    assert arrays["cofactor_mask"][idx_gly]
    assert arrays["cofactor_targets"][idx_gly, sugar_col] == pytest.approx(1.0)
    assert arrays["cofactor_mask"][idx_biotin]
    assert arrays["cofactor_targets"][idx_biotin].sum() == pytest.approx(0.0)
    assert metadata["enzyme_cofactor_label_stats"]["known"] == 2
    assert metadata["enzyme_cofactor_label_stats"]["positive"] == 1
    assert metadata["enzyme_cofactor_label_stats"]["zero_positive_known"] == 1


def test_builder_expanded_cofactor_label_set_promotes_clean_core_families(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from scripts.build_enzyme_biofp_soft_targets import (
        EXPANDED_COFACTOR_LABELS,
        build_soft_targets,
    )

    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerow({"reaction_id": "rxn_biotin", "protein_id": "enz_biotin"})
        writer.writerow({"reaction_id": "rxn_quinone", "protein_id": "enz_quinone"})

    features_path = tmp_path / "reaction_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "rxn_biotin",
                "reaction_center_coarse_labels": "c_c_change",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "",
            },
            {
                "reaction_id": "rxn_quinone",
                "reaction_center_coarse_labels": "redox_like",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "",
            },
        ]
    ).to_parquet(features_path)

    enhanced_path = tmp_path / "enhanced.csv"
    with enhanced_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "enzyme_id",
                "combined_core_cofactor_labels_train",
                "cofactor_architecture_bins_train",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "enzyme_id": "enz_biotin",
                "combined_core_cofactor_labels_train": "['biotin']",
                "cofactor_architecture_bins_train": "['biotin_carboxylation']",
            }
        )
        writer.writerow(
            {
                "enzyme_id": "enz_quinone",
                "combined_core_cofactor_labels_train": "['quinone']",
                "cofactor_architecture_bins_train": "['quinone_PQQ_redox']",
            }
        )

    ids, arrays, metadata = build_soft_targets(
        pairs_path,
        features_path,
        enzyme_cofactor_labels=enhanced_path,
        cofactor_label_set="expanded",
    )

    biotin_col = EXPANDED_COFACTOR_LABELS.index("biotin_carboxylation")
    quinone_col = EXPANDED_COFACTOR_LABELS.index("quinone_PQQ_redox")
    assert arrays["cofactor_targets"][ids.index("enz_biotin"), biotin_col] == pytest.approx(1.0)
    assert arrays["cofactor_targets"][ids.index("enz_quinone"), quinone_col] == pytest.approx(1.0)
    assert metadata["cofactor_label_set"] == "expanded"
    assert metadata["families"]["cofactor"] == EXPANDED_COFACTOR_LABELS


def test_builder_keeps_glycosyl_cofactor_rescue_opt_in(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from scripts.build_enzyme_biofp_soft_targets import build_soft_targets

    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerow({"reaction_id": "rxn_glycosyl", "protein_id": "enz_gly"})

    features_path = tmp_path / "reaction_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "rxn_glycosyl",
                "reaction_center_coarse_labels": "c_o_change",
                "core_cofactor_labels": "",
                "cofactor_labels": "",
                "substrate_product_transition_labels": "",
                "reaction_type_labels": "glycosyl_transfer",
            }
        ]
    ).to_parquet(features_path)

    ids, arrays, metadata = build_soft_targets(pairs_path, features_path)

    idx = ids.index("enz_gly")
    assert arrays["cofactor_mask"][idx] == np.False_
    assert metadata["glycosyl_transfer_cofactor_rule"] == "none"
