"""Regression cases for explicit chemical names and unknown-only export semantics."""

import pytest

from horizyn.capability.biological_targets import COFACTOR_LABELS as V1_LABELS
from horizyn.capability.cofactor_vocabulary_v2 import (
    COFACTOR_LABELS,
    COFACTOR_VOCABULARY_VERSION,
    KNOWN_COFACTOR_LABELS,
    UNKNOWN_INDEX,
    UNKNOWN_LABEL,
    cofactor_name_groups,
)


def test_versioned_order_and_unknown_contract():
    assert COFACTOR_VOCABULARY_VERSION == "circe_cofactor_v2"
    assert KNOWN_COFACTOR_LABELS[:10] == V1_LABELS
    assert len(V1_LABELS) == 10
    assert len(KNOWN_COFACTOR_LABELS) == len(set(KNOWN_COFACTOR_LABELS)) == 31
    assert COFACTOR_LABELS == KNOWN_COFACTOR_LABELS + ("unknown",)
    assert UNKNOWN_LABEL == "unknown"
    assert UNKNOWN_INDEX == 31
    assert COFACTOR_LABELS[UNKNOWN_INDEX] == UNKNOWN_LABEL


@pytest.mark.parametrize(
    "label,names",
    [
        ("NAD_NADP", ["NAD", "NAD+", "NADH", "NADP", "NADPH", "NAD(P)H",
                      "nicotinamide adenine dinucleotide", "NAD_NADP"]),
        ("FAD_FMN", ["FAD", "FADH2", "FMN", "FMNH2", "FMNH₂", "reduced flavin",
                     "flavin adenine dinucleotide", "6-hydroxy-FAD(3-)"]),
        ("PLP", ["PLP", "pyridoxal 5'-phosphate", "pyridoxamine phosphate"]),
        ("TPP", ["TPP", "ThDP", "thiamine diphosphate", "thiamin pyrophosphate"]),
        ("CoA", ["CoA", "coenzyme A", "acetyl-CoA"]),
        ("SAM", ["SAM", "AdoMet", "S-adenosyl-L-methionine", "S-adenosylmethionine",
                 "S_adenosyl_L_methionine"]),
        ("FeS", ["FeS", "FeS_cluster", "Fe-S", "iron-sulfur", "iron–sulphur",
                 "[2Fe-2S] cluster", "[3Fe-4S] cluster", "[4Fe-4S] cluster",
                 "[8Fe-7S] cluster", "[8Fe–7S] cluster"]),
        ("heme", ["heme", "haem b", "siroheme", "ferriheme a", "ferroheme b", "hemin"]),
        ("quinone", ["PQQ", "pyrroloquinoline quinone", "ubiquinone", "menaquinol"]),
        ("thiol_lipoate", ["CoB", "CoM", "coenzyme B", "coenzyme M", "glutathione",
                           "thioredoxin", "(R)-lipoate", "lipoic acid", "dihydrolipoamide",
                           "2-mercaptoethanesulfonate"]),
        ("metal_Mg", ["Mg", "Mg2+", "Mg(2+)", "Mg²⁺", "magnesium", "magnesium ion"]),
        ("metal_Mn", ["Mn2+", "Mn(2+)", "manganese", "manganese(II) ion"]),
        ("metal_Zn", ["Zn2+", "Zn(2+)", "zinc", "zinc cation"]),
        ("metal_Fe", ["Fe2+", "Fe(2+)", "Fe(3+)", "iron cation", "ferrous", "ferric ion"]),
        ("metal_Cu", ["Cu+", "Cu(1+)", "Cu(2+)", "Cu cation", "copper", "cupric ion"]),
        ("metal_Co", ["Co2+", "Co(2+)", "cobalt", "cobalt(III)"]),
        ("metal_Ni", ["Ni2+", "Ni(2+)", "nickel", "nickel(II)"]),
        ("metal_Ca", ["Ca2+", "Ca(2+)", "calcium", "calcium ion"]),
        ("metal_K", ["K+", "K(+)", "potassium", "potassium ion"]),
        ("metal_Na", ["Na+", "Na(+)", "sodium", "sodium ion"]),
        ("metal_generic", ["metal", "metal cation", "A metal ion"]),
        ("metal_divalent", ["A divalent metal cation", "divalent metal ion", "divalent cation"]),
        ("cobalamin", ["cobalamin", "coenzyme B12", "vitamin B12", "cob(I)alamin",
                       "cob(II)alamin", "cob(II)alamin(1-)", "Adenosylcob(III)alamin",
                       "Methylcob(III)alamin", "adenosylcobalamin", "cobamamide",
                       "corrinoid", "5-hydroxybenzimidazolylcob(I)amide"]),
        ("biotin", ["biotin", "biotinate", "carboxybiotin"]),
        ("molybdopterin_tungsten", ["molybdopterin", "Mo-molybdopterin",
                                    "Mo-bis(molybdopterin guanine dinucleotide)",
                                    "W-bis(molybdopterin guanine dinucleotide)",
                                    "bis(molybdopterin)tungsten cofactor"]),
        ("F420", ["F420", "coenzyme F420", "F420H2", "8-hydroxy-5-deazaflavin"]),
        ("F430", ["F430", "coenzyme F430", "coenzyme F430(4-)"]),
        ("folate", ["folate", "tetrahydrofolate", "THF", "5-methyltetrahydrofolate",
                    "(6R)-5,10-methylenetetrahydrofolate(2-)", "tetrahydrofolic acid"]),
        ("pyruvoyl", ["pyruvoyl", "pyruvoyl group", "pyruvoyl cofactor"]),
        ("dipyrromethane", ["Dipyrromethane", "dipyrromethane cofactor(4-)"]),
        ("phosphopantetheine", ["4'-phosphopantetheine", "4′-phosphopantetheine",
                                "phosphopantetheine", "pantetheine 4'-phosphate",
                                "pantetheine_4_phosphate_2"]),
    ],
)
def test_supported_aliases_keep_chemical_group_identity(label, names):
    for name in names:
        assert cofactor_name_groups([name]) == {label}, name


@pytest.mark.parametrize(
    "name",
    ["", "unknown", "none", "no_cofactor", "uncharacterized cofactor", "vanadium",
     "dihydrogenvanadate", "CO", "FAD2", "FMN2", "cobalt transporter", "zinc finger",
     "iron-dependent protein", "sodium chloride", "K12", "K_m", "B120", "CoB12",
     "Pyruvate", "pyruvic acid", "ATP", "GTP", "phosphate", "pantetheine",
     "a monovalent cation"],
)
def test_unsupported_names_never_become_positive_or_unknown(name):
    assert cofactor_name_groups([name]) == set()


def test_empty_or_unrecognized_names_do_not_materialize_unknown():
    assert cofactor_name_groups([]) == set()
    assert cofactor_name_groups([None, float("nan"), "unknown"]) == set()
    assert cofactor_name_groups(["unknown", "Mg(2+)", "vanadium"]) == {"metal_Mg"}


def test_multiple_names_union_without_losing_raw_oxidation_names():
    names = ["Fe(2+)", "Fe(3+)", "[8Fe-7S] cluster", "cob(II)alamin", "CoB", "unknown"]
    original = names.copy()
    assert cofactor_name_groups(names) == {"metal_Fe", "FeS", "cobalamin", "thiol_lipoate"}
    assert names == original


def test_metal_clusters_and_organometallic_cofactors_do_not_imply_simple_ions():
    assert cofactor_name_groups(["[4Fe-4S] cluster", "iron-sulfur", "heme", "cobalamin",
                                "coenzyme F430"]) == {"FeS", "heme", "cobalamin", "F430"}


def test_generic_and_specific_metals_stay_independent():
    assert cofactor_name_groups(["A divalent metal cation", "metal", "Zn(2+)"]) == {
        "metal_divalent", "metal_generic", "metal_Zn"
    }


def test_strings_are_single_names_and_mutable_results_are_not_shared():
    result = cofactor_name_groups("cob(II)alamin")
    assert result == {"cobalamin"}
    result.add("unknown")
    assert cofactor_name_groups(iter(["cob(II)alamin"])) == {"cobalamin"}
