import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("rdkit")

from horizyn.capability.cofactors import (
    CHEBI_COFACTOR_ROLE_TERMS,
    cofactor_labels_for_substrate_filter,
    extract_cofactor_labels,
    load_cofactor_aliases,
    split_cofactor_label_tiers,
)
from horizyn.capability.reaction_center import (
    coarse_reaction_center_labels,
    extract_reaction_center_raw_labels,
)
from horizyn.capability.reaction_features import (
    canonicalize_reaction_smiles,
    compute_drfp_active_bits,
    load_participant_texts,
)
from horizyn.capability.reaction_types import reaction_type_labels
from horizyn.capability.substrate_classes import (
    classify_smiles_list,
    substrate_product_transition_labels,
)


def test_canonicalize_reaction_smiles_preserves_stereo():
    canonical = canonicalize_reaction_smiles("F[C@H](Cl)Br>>F[C@@H](Cl)Br")
    assert canonical is not None
    assert "@" in canonical


def test_canonicalize_supports_reactants_reagents_products():
    canonical = canonicalize_reaction_smiles("CCO>O>CC=O")
    assert canonical is not None
    assert canonical.endswith(">>CC=O")
    assert "CCO" in canonical.split(">>")[0]


def test_invalid_smiles_sets_none():
    assert canonicalize_reaction_smiles("not_smiles>>CCO") is None


def test_drfp_vector_shape_2048():
    pytest.importorskip("drfp")
    arr, active = compute_drfp_active_bits("CCO>>CC=O", 2048)
    assert arr.shape == (2048,)
    assert arr.dtype == np.uint8
    assert all(0 <= idx < 2048 for idx in active)


def test_reaction_center_detects_bond_order_change():
    labels = extract_reaction_center_raw_labels("[CH3:1][OH:2]>>[CH2:1]=[O:2]")
    assert "bond_order_change_C_O" in labels
    coarse = coarse_reaction_center_labels(labels)
    assert "bond_order_change" in coarse


def test_reaction_center_detects_charge_change():
    labels = extract_reaction_center_raw_labels("[NH3+:1]>>[NH2:1]")
    assert "charge_change_N" in labels


def test_reaction_center_coarse_dictionary_detects_halogen_and_sulfur_changes():
    coarse = coarse_reaction_center_labels(["bond_formed_C_Cl", "bond_broken_S_S"])
    assert "c_halogen_change" in coarse
    assert "halogenation_like" in coarse
    assert "s_s_change" in coarse
    assert "disulfide_like" in coarse


def test_cofactor_alias_maps_nadh_to_nad():
    labels, flags, status = extract_cofactor_labels(["NADH"])
    assert labels == ["NAD"]
    assert flags == ["cofactor_from_name_match"]
    assert status == "weak_name_match"


def test_cofactor_alias_maps_nadph_to_nadp():
    labels, _, _ = extract_cofactor_labels(["NADPH"])
    assert labels == ["NADP"]


def test_cofactor_dictionary_contains_chebi_role_fillers():
    assert len(CHEBI_COFACTOR_ROLE_TERMS) >= 160
    aliases = load_cofactor_aliases()
    assert "CHEBI:83088" in aliases["lipoate"]
    assert "CHEBI:18315" in aliases["PQQ"]
    assert "CHEBI:58937" in aliases["TPP"]
    assert "CHEBI:25372" in aliases["molybdopterin"]


def test_cofactor_chebi_id_maps_to_broad_label():
    labels, flags, status = extract_cofactor_labels(["CHEBI:18315"])
    assert labels == ["PQQ"]
    assert "cofactor_from_name_match" in flags
    assert "cofactor_from_chebi_role_match" in flags
    assert status == "weak_name_match"


def test_cofactor_chebi_role_exact_fallback_label():
    labels, _, _ = extract_cofactor_labels(["CHEBI:61338"])
    assert labels == ["bacillithiol"]


def test_cofactor_vanadium_does_not_match_nad():
    labels, _, _ = extract_cofactor_labels(["vanadium cation"])
    assert "NAD" not in labels
    assert "V" in labels


def test_external_cofactor_dictionary_matches_smiles(tmp_path):
    dictionary = tmp_path / "cofactor_dictionary.tsv"
    dictionary.write_text(
        "chebi_id\tname\tlabel\taliases\tsmiles\n"
        "CHEBI:18315\tpyrroloquinoline quinone\tPQQ\tPQQ\t"
        "O=C(O)c1cc(C(=O)O)c2c(n1)C(=O)C(=O)c1cc(C(=O)O)nc1-2\n",
        encoding="utf-8",
    )
    aliases = load_cofactor_aliases(dictionary)
    labels, flags, status = extract_cofactor_labels(
        ["O=C(O)c1cc(C(=O)O)c2c(n1)C(=O)C(=O)c1cc(C(=O)O)nc1-2"],
        cofactor_aliases=aliases,
    )
    assert "PQQ" in labels
    assert "cofactor_from_chebi_smiles_match" in flags
    assert status == "chebi_smiles_match"


def test_cofactor_unicode_name_normalization():
    labels, _, _ = extract_cofactor_labels(["β-carotene"])
    assert labels == ["carotenoid"]


def test_cofactor_structure_detects_nad_smiles():
    nad = (
        "NC(=O)C1=CC=C[N+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "OC[C@H]3O[C@@H](N4C=NC5=C4N=CN=C5N)[C@H](O)[C@@H]3O)"
        "[C@@H](O)[C@H]2O)=C1"
    )
    labels, flags, status = extract_cofactor_labels([nad])
    assert labels == ["NAD"]
    assert flags == ["cofactor_from_structure_match"]
    assert status == "structure_match"


def test_cofactor_labels_split_into_biological_tiers():
    tiers = split_cofactor_label_tiers(["NAD", "Mg2+", "hydrogenphosphate"])
    assert tiers["core_cofactor_labels"] == ["NAD"]
    assert tiers["metal_ion_labels"] == ["Mg2+"]
    assert tiers["auxiliary_participant_labels"] == ["hydrogenphosphate"]


def test_participant_texts_accepts_rhea_id_column(tmp_path):
    path = tmp_path / "rhea_molecules.tsv"
    path.write_text(
        "Rhea ID\tsubstrate\tproduct\n"
        "RHEA:12345\tNADH\tNAD+\n",
        encoding="utf-8",
    )
    texts = load_participant_texts(path)
    assert texts["RHEA:12345"] == ["NADH", "NAD+"]
    assert texts["Rh_12345"] == ["NADH", "NAD+"]


def test_substrate_class_smarts_detects_carboxylate():
    labels, flags = classify_smiles_list(["CC(=O)O"])
    assert "carboxylate" in labels
    assert "substrate_class_from_smarts" in flags


def test_substrate_class_smarts_detects_alcohol():
    labels, _ = classify_smiles_list(["CCO"])
    assert "alcohol" in labels


def test_substrate_class_expanded_dictionary_detects_esters_and_nitriles():
    labels, _ = classify_smiles_list(["CC(=O)OCC", "CC#N"])
    assert "ester" in labels
    assert "nitrile" in labels


def test_substrate_class_expanded_dictionary_detects_nucleotide_like():
    labels, _ = classify_smiles_list(
        ["Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O"]
    )
    assert "nucleotide_like" in labels


def test_substrate_class_can_remove_detected_core_cofactor():
    nad = (
        "NC(=O)C1=CC=C[N+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "OC[C@H]3O[C@@H](N4C=NC5=C4N=CN=C5N)[C@H](O)[C@@H]3O)"
        "[C@@H](O)[C@H]2O)=C1"
    )
    labels, flags = classify_smiles_list(
        [nad, "CCO"],
        skip_cofactor_labels=cofactor_labels_for_substrate_filter(["NAD"]),
    )
    assert "alcohol" in labels
    assert "phosphate_containing" not in labels
    assert "substrate_class_removed_cofactor_molecule" in flags


def test_substrate_product_transition_labels_are_directional():
    labels = substrate_product_transition_labels(["alcohol"], ["ketone"])
    assert "alcohol_to_ketone" in labels
    assert "loss_alcohol" in labels
    assert "gain_ketone" in labels


def test_reaction_type_detects_nad_redox():
    labels, _, _ = reaction_type_labels([], ["NAD"], [], [], [])
    assert "oxidoreduction" in labels
    assert "hydride_transfer" in labels


def test_reaction_type_detects_plp_transamination():
    labels, _, _ = reaction_type_labels(["c_n_change"], ["PLP"], [], [], [])
    assert "transamination" in labels
