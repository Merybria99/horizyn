"""Export the controlled dictionaries used by capability annotations."""

from __future__ import annotations

from typing import Any

from horizyn.capability import cofactors
from horizyn.capability import reaction_center
from horizyn.capability import substrate_classes


def annotation_dictionaries_payload() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of active annotation dictionaries."""

    cofactor_aliases = cofactors.load_cofactor_aliases(None)
    return {
        "version": "expanded_20260701",
        "cofactors": {
            "num_alias_labels": len(cofactor_aliases),
            "core_cofactor_labels": sorted(cofactors.CORE_COFACTOR_LABELS),
            "metal_ion_labels": sorted(cofactors.METAL_ION_LABELS),
            "common_auxiliary_participant_labels": sorted(
                cofactors.COMMON_AUXILIARY_PARTICIPANTS
            ),
            "curated_alias_labels": sorted(cofactors.CURATED_COFACTOR_ALIASES),
            "chebi_cofactor_role_terms": len(cofactors.CHEBI_COFACTOR_ROLE_TERMS),
        },
        "reaction_centers": {
            "atom_pair_groups": {
                "halogens": sorted(reaction_center.HALOGENS),
                "metals": sorted(reaction_center.METALS),
            },
            "coarse_label_families": [
                "bond_formation",
                "bond_cleavage",
                "bond_order_change",
                "charge_change",
                "chirality_change",
                "c_o_change",
                "c_n_change",
                "c_c_change",
                "c_s_change",
                "p_o_change",
                "p_s_change",
                "c_p_change",
                "c_b_change",
                "n_o_change",
                "s_o_change",
                "s_s_change",
                "c_halogen_change",
                "c_metal_change",
                "metal_ligand_change",
                "phosphate_transfer_like",
                "thiophosphate_transfer_like",
                "organophosphorus_change",
                "organoboron_change",
                "disulfide_like",
                "halogenation_like",
                "dehalogenation_like",
                "organometallic_change",
                "oxygen_transfer_like",
                "sulfur_transfer_like",
                "redox_like",
            ],
        },
        "substrate_product_classes": {
            "smarts_labels": sorted(substrate_classes.SMARTS),
            "transition_rule_labels": sorted(
                transition for _, _, transition in substrate_classes.TRANSITION_RULES
            ),
            "derived_label_families": [
                "amino_acid_like",
                "keto_acid_like",
                "hydroxy_acid_like",
                "sugar_like",
                "nucleobase_like",
                "nucleoside_like",
                "nucleotide_like",
                "peptide_like",
                "organic_acid",
                "large_hydrophobic",
                "fatty_acid_like",
                "lipid_like",
                "terpene_like",
                "steroid_like",
                "small_aliphatic",
                "heteroaromatic",
                "halogenated",
                "glycoside",
            ],
        },
    }
