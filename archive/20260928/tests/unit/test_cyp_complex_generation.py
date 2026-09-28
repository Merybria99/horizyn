import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import generate_cyp_complexes as gen


def test_ligand_check_ignores_maps_but_preserves_stereochemistry():
    assert gen.ligand_key("[CH3:1][OH:2]") == gen.ligand_key("CO")
    assert gen.ligand_key("C[C@H](O)N") != gen.ligand_key("C[C@@H](O)N")


def test_msa_query_must_match(tmp_path):
    msa = tmp_path / "p.a3m"
    msa.write_text(">query\nACD\nEF\n>hit\nAC-EF\n")
    gen.validate_msa(msa, "ACDEF")
    with pytest.raises(ValueError, match="mismatch"):
        gen.validate_msa(msa, "ACDE")


def test_gpu_guard_never_kills(monkeypatch):
    calls = []
    def query(cmd, **kwargs):
        calls.append(cmd); return "1234\n"
    monkeypatch.setattr(gen.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="occupied"):
        gen.check_gpu("2")
    assert len(calls) == 1 and calls[0][0] == "nvidia-smi"


@pytest.fixture
def planning(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(gen, "RUN", tmp_path / "run")
    pair = dict(query_id="r_q", protein_id="P", sequence="ACD", reaction="O.CC>>CCC")
    monkeypatch.setattr(gen, "inputs", lambda: ([pair], {"P": "ACD"}))
    monkeypatch.setattr(gen, "rows", lambda path: [pair])
    asset = gen.ASSETS / "boltzcyp_generation_inputs"
    asset.mkdir(parents=True)
    (asset / "complete.json").write_text("{}")
    (asset / "q_0.fasta").write_text(">A|protein|empty\nACD\n>B|ccd\nHEM\n>C|smiles\nCC\n")
    (asset / "P.a3m").write_text(">P\nACD\n")
    report = gen.RUN / "recovery/enzymecage/report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"failures": [dict(query_id="r_q", protein_id="P",
                       error_type="FileNotFoundError", missing_complex=True)]}))
    return asset, report


def test_plan_preserves_original_pair_and_chains(planning):
    specs = gen.plan()
    assert len(specs) == 1
    assert specs[0]["sequence"] == "ACD"
    assert specs[0]["substrate"] == "CC"
    assert specs[0]["name"] == "q_0"
    assert specs[0]["protocol"]["boltz"] == "0.4.1"


def test_no_silent_single_sequence_fallback(planning):
    asset, _ = planning
    (asset / "P.a3m").unlink()
    spec = gen.plan()[0]
    assert spec["msa_source"] == gen.MSA_SERVER
    with pytest.raises(RuntimeError, match="Missing released MSA"):
        gen.ensure_msa(spec, False)


def test_wrong_ligand_rejected(planning):
    asset, _ = planning
    (asset / "q_0.fasta").write_text(">A|protein|empty\nACD\n>B|ccd\nHEM\n>C|smiles\nN\n")
    with pytest.raises(ValueError, match="matching released"):
        gen.plan()


def test_other_feature_errors_not_treated_as_missing_complexes(planning):
    _, report = planning
    data = json.loads(report.read_text())
    data["failures"][0]["missing_complex"] = False
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Not a missing-complex"):
        gen.plan()


def test_resume_verifies_hash_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "OUTPUT", tmp_path)
    directory = tmp_path / "q_0"; directory.mkdir()
    cif = directory / "q_0_model_0.cif"; cif.write_text("structure")
    spec = dict(sequence="ACD", query_id="q", protein_id="P", substrate="CC", protocol=gen.PROTOCOL)
    (directory / "input.json").write_text(json.dumps(spec))
    receipt = dict(input=spec, cif_sha256=gen.digest(cif),
                   checkpoint_sha256=gen.WEIGHTS["boltz1_conf.ckpt"], ccd_sha256=gen.WEIGHTS["ccd.pkl"])
    (directory / "complete.json").write_text(json.dumps(receipt))
    assert gen.verified_complex("q_0", "ACD", "q", "P", "CC") == cif
    with pytest.raises(ValueError, match="input/protocol"):
        gen.verified_complex("q_0", "AAA", "q", "P", "CC")
    with pytest.raises(ValueError, match="input/protocol"):
        gen.verified_complex("q_0", "ACD", "q", "P", "N")
    cif.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        gen.verified_complex("q_0", "ACD", "q", "P", "CC")
