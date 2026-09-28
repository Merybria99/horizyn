import csv

from rdkit import Chem

from scripts.normalize_enzymecage_f3_reactions import normalize
from horizyn.chemistry.standardizer import Standardizer


def test_charge_sensitive_reaction_is_preserved(tmp_path):
    original = "O=c1[nH]c(=O)[c-](O)c(=O)[nH]1>>O=c1[nH]c(=O)[c-](O)c(=O)[nH]1"
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("reaction_id", "reaction_smiles"))
        writer.writerow(("r1", original))
    target = tmp_path / "normalized.csv"
    report = normalize(source, target, tmp_path / "audit.csv",
                       Standardizer(standardize_uncharge=False))
    assert report["reactions"] == 1
    product = next(csv.DictReader(target.open()))["reaction_smiles"].split(">>")[1]
    assert Chem.MolFromSmiles(product) is not None
    assert "[c-]" in product
