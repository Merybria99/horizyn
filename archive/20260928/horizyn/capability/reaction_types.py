"""Conservative reaction-type labels from extracted chemistry features."""

from __future__ import annotations


def reaction_type_labels(
    center_labels: list[str],
    cofactors: list[str],
    substrate_labels: list[str],
    product_labels: list[str],
    ec_numbers: list[str],
) -> tuple[list[str], list[str], str]:
    c = set(center_labels)
    cof = set(cofactors)
    sub = set(substrate_labels)
    prod = set(product_labels)
    out: set[str] = set()
    flags: set[str] = set()
    non_ec_evidence = bool(c or cof or sub or prod)

    for ec in ec_numbers:
        ec = str(ec)
        if ec.startswith("1."):
            out.add("oxidoreduction")
        elif ec.startswith("2."):
            out.add("group_transfer")
        elif ec.startswith("3."):
            out.add("hydrolysis")
        elif ec.startswith("4."):
            out.add("bond_cleavage")
        elif ec.startswith("5."):
            out.add("isomerization")
        elif ec.startswith("6."):
            out.add("ligation")
        elif ec.startswith("7."):
            out.add("translocation")

    if out and not non_ec_evidence:
        flags.add("reaction_type_from_ec_only")

    if cof & {"NAD", "NADP", "FAD", "FMN", "quinone", "ferredoxin"}:
        out.add("oxidoreduction")
    if cof & {"NAD", "NADP"}:
        out.add("hydride_transfer")
    if cof & {"FAD", "FMN", "quinone", "ferredoxin"}:
        out.add("electron_transfer")
    if "PLP" in cof and (
        "c_n_change" in c or "amino_acid_like" in sub or "amino_acid_like" in prod
    ):
        out.add("transamination")
    if "ATP" in cof and (
        "phosphate_transfer_like" in c
        or "phosphate_containing" in sub
        or "phosphate_containing" in prod
    ):
        out.add("phosphorylation")
    if "SAM" in cof:
        out.add("methyl_transfer")
    if "CoA" in cof or "coa_thioester" in sub or "coa_thioester" in prod:
        out.add("acyl_transfer")
    if "biotin" in cof or "carboxylation_like" in c:
        out.add("carboxylation")
    if "bond_cleavage" in c:
        out.add("bond_cleavage")
    if "bond_formation" in c:
        out.add("bond_formation")
    if "oxygen_transfer_like" in c:
        out.add("oxygenation")
    if "sulfur_transfer_like" in c:
        out.add("sulfur_transfer")
    if "halogenation_like" in c:
        out.add("halogenation")
    if "dehalogenation_like" in c:
        out.add("dehalogenation")
    if "disulfide_like" in c:
        out.add("disulfide_exchange")
    if "metal_ligand_change" in c or "organometallic_change" in c:
        out.add("metal_dependent_transformation")
    if "chirality_change" in c:
        out.add("stereochemical_inversion")
    if "charge_change" in c:
        out.add("acid_base_or_charge_transfer")
    if "sugar_like" in sub or "sugar_like" in prod:
        if "c_o_change" in c:
            out.add("glycosyl_transfer")
    if "hydroxy_acid_like" in sub and "keto_acid_like" in prod:
        out.add("oxidation")
    if "keto_acid_like" in sub and "hydroxy_acid_like" in prod:
        out.add("reduction")
    if "ester" in sub and "carboxylate" in prod:
        out.add("ester_hydrolysis")
    if "amide" in sub and "carboxylate" in prod:
        out.add("amide_hydrolysis")
    if "nitrile" in sub and ("amide" in prod or "carboxylate" in prod):
        out.add("nitrile_hydrolysis")
    if "phosphate_ester" in sub and "alcohol" in prod:
        out.add("dephosphorylation")
    if "alcohol" in sub and "phosphate_ester" in prod:
        out.add("phosphorylation")
    if "organohalide" in sub and "organohalide" not in prod:
        out.add("dehalogenation")

    if out:
        status = "weak_rule" if flags else "ok"
    else:
        status = "unknown"
    return sorted(out), sorted(flags), status
