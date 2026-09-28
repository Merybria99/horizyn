"""Reaction-center extraction from atom-mapped reaction SMILES."""

from __future__ import annotations

from rdkit import Chem

HALOGENS = frozenset({"F", "Cl", "Br", "I"})
METALS = frozenset(
    {
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Mo",
        "W",
        "V",
        "Cd",
        "La",
    }
)


def split_mapped_reaction(mapped_rxn: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    if ">>" not in mapped_rxn:
        raise ValueError("Mapped reaction SMILES must contain '>>'")
    left, right = mapped_rxn.split(">>", 1)
    reactants = [Chem.MolFromSmiles(x) for x in left.split(".") if x]
    products = [Chem.MolFromSmiles(x) for x in right.split(".") if x]
    if any(mol is None for mol in reactants + products):
        raise ValueError("Could not parse mapped reaction molecule")
    return reactants, products


def atom_state(mol: Chem.Mol) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for atom in mol.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap == 0:
            continue
        out[amap] = {
            "symbol": atom.GetSymbol(),
            "charge": atom.GetFormalCharge(),
            "chiral": str(atom.GetChiralTag()),
        }
    return out


def bond_state(mol: Chem.Mol) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetAtomMapNum()
        a2 = bond.GetEndAtom().GetAtomMapNum()
        if a1 == 0 or a2 == 0:
            continue
        out[tuple(sorted((a1, a2)))] = str(bond.GetBondType())
    return out


def merge_states(mols: list[Chem.Mol], fn) -> dict:
    merged = {}
    for mol in mols:
        merged.update(fn(mol))
    return merged


def extract_reaction_center_raw_labels(mapped_rxn: str) -> list[str]:
    reactants, products = split_mapped_reaction(mapped_rxn)
    r_atoms = merge_states(reactants, atom_state)
    p_atoms = merge_states(products, atom_state)
    r_bonds = merge_states(reactants, bond_state)
    p_bonds = merge_states(products, bond_state)
    labels: set[str] = set()

    for bond_key in set(r_bonds) | set(p_bonds):
        r_order = r_bonds.get(bond_key)
        p_order = p_bonds.get(bond_key)
        symbols = []
        for amap in bond_key:
            state = r_atoms.get(amap) or p_atoms.get(amap)
            symbols.append(str(state["symbol"]) if state else "X")
        pair = "_".join(sorted(symbols))
        if r_order is None and p_order is not None:
            labels.add(f"bond_formed_{pair}")
        elif r_order is not None and p_order is None:
            labels.add(f"bond_broken_{pair}")
        elif r_order != p_order:
            labels.add(f"bond_order_change_{pair}")

    for amap in set(r_atoms) | set(p_atoms):
        r_atom = r_atoms.get(amap)
        p_atom = p_atoms.get(amap)
        if r_atom is None or p_atom is None:
            continue
        symbol = str(r_atom["symbol"])
        if r_atom["charge"] != p_atom["charge"]:
            labels.add(f"charge_change_{symbol}")
        if r_atom["chiral"] != p_atom["chiral"]:
            labels.add(f"chirality_change_{symbol}")
    return sorted(labels)


def _bond_change(label: str) -> tuple[str, tuple[str, str]] | None:
    for prefix, change in (
        ("bond_formed_", "formed"),
        ("bond_broken_", "broken"),
        ("bond_order_change_", "order"),
    ):
        if label.startswith(prefix):
            symbols = tuple(label[len(prefix) :].split("_"))
            if len(symbols) == 2:
                return change, (symbols[0], symbols[1])
    return None


def _has_pair(
    bond_changes: list[tuple[str, tuple[str, str]]],
    left: set[str],
    right: set[str],
    *,
    change: str | None = None,
) -> bool:
    for observed_change, pair in bond_changes:
        if change is not None and observed_change != change:
            continue
        a, b = pair
        if (a in left and b in right) or (a in right and b in left):
            return True
    return False


def coarse_reaction_center_labels(raw_labels: list[str]) -> list[str]:
    raw = set(raw_labels)
    out: set[str] = set()
    bond_changes = [value for label in raw if (value := _bond_change(label)) is not None]
    if _has_pair(bond_changes, {"C"}, {"O"}):
        out.add("c_o_change")
    if _has_pair(bond_changes, {"C"}, {"N"}):
        out.add("c_n_change")
    if _has_pair(bond_changes, {"C"}, {"C"}):
        out.add("c_c_change")
    if _has_pair(bond_changes, {"C"}, {"S", "Se"}):
        out.add("c_s_change")
    if _has_pair(bond_changes, {"P"}, {"O"}):
        out.add("p_o_change")
        out.add("phosphate_transfer_like")
    if _has_pair(bond_changes, {"P"}, {"S", "Se"}):
        out.add("p_s_change")
        out.add("thiophosphate_transfer_like")
    if _has_pair(bond_changes, {"C"}, {"P"}):
        out.add("c_p_change")
        out.add("organophosphorus_change")
    if _has_pair(bond_changes, {"C"}, {"B"}):
        out.add("c_b_change")
        out.add("organoboron_change")
    if _has_pair(bond_changes, {"N"}, {"O"}):
        out.add("n_o_change")
    if _has_pair(bond_changes, {"S", "Se"}, {"O"}):
        out.add("s_o_change")
    if _has_pair(bond_changes, {"S", "Se"}, {"S", "Se"}):
        out.add("s_s_change")
        out.add("disulfide_like")
    if _has_pair(bond_changes, {"C"}, set(HALOGENS)):
        out.add("c_halogen_change")
        if _has_pair(bond_changes, {"C"}, set(HALOGENS), change="formed"):
            out.add("halogenation_like")
        if _has_pair(bond_changes, {"C"}, set(HALOGENS), change="broken"):
            out.add("dehalogenation_like")
    if _has_pair(bond_changes, {"C"}, set(METALS)):
        out.add("c_metal_change")
        out.add("organometallic_change")
    if _has_pair(bond_changes, {"O", "N", "S", "Se"}, set(METALS)):
        out.add("metal_ligand_change")
    if _has_pair(bond_changes, {"C", "N", "S", "P"}, {"O"}, change="formed"):
        out.add("oxygen_transfer_like")
    if _has_pair(bond_changes, {"C", "P"}, {"S", "Se"}, change="formed"):
        out.add("sulfur_transfer_like")
    if any(x.startswith("bond_formed") for x in raw):
        out.add("bond_formation")
    if any(x.startswith("bond_broken") for x in raw):
        out.add("bond_cleavage")
    if any(x.startswith("bond_order_change") for x in raw):
        out.add("bond_order_change")
    if any(x.startswith("charge_change") for x in raw):
        out.add("charge_change")
    if any(x.startswith("chirality_change") for x in raw):
        out.add("chirality_change")
    if "bond_order_change_C_O" in raw or "charge_change_O" in raw:
        out.add("redox_like")
    return sorted(out)
