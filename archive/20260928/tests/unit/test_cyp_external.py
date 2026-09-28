import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cyp_external_common as common
from prepare_cyp_external_assets import choose


def test_pinned_archive_hashes_are_sha256():
    import re
    from prepare_cyp_external_assets import ARCHIVES
    assert all(re.fullmatch(r"[0-9a-f]{64}", value[1]) for value in ARCHIVES.values())


def test_missing_scores_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path)
    monkeypatch.setattr(common, "inputs", lambda: ([dict(query_id="q", protein_id="p")], {}))
    with pytest.raises(ValueError, match="Incomplete"):
        common.finish("horizyn1_dev", {}, "unused", "scope")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_rejected(tmp_path, monkeypatch, score):
    monkeypatch.setattr(common, "RUN", tmp_path)
    monkeypatch.setattr(common, "inputs", lambda: ([dict(query_id="q", protein_id="p")], {}))
    with pytest.raises(ValueError): common.finish("horizyn1_dev", {("q", "p"): score}, "unused", "scope")


def test_structure_selection_excludes_training():
    from pathlib import PurePosixPath
    assert choose("clipzyme_structures", PurePosixPath("data_dir/eval/organism/a.cif")) is not None
    assert choose("clipzyme_structures", PurePosixPath("data_dir/train/a.cif")) is None
    assert choose("clipzyme_structures", PurePosixPath("data_dir/eval/predictions.csv")) is None


def test_missing_structure_rejected(tmp_path, monkeypatch):
    import cyp_native_clipzyme as clip
    monkeypatch.setattr(clip, "ASSETS", tmp_path)
    monkeypatch.setattr(clip, "RUN", tmp_path)
    with pytest.raises(ValueError, match="pool cannot be shrunk"):
        clip.structures([dict(cif="/old/AF-X.cif", protein_id="X")])


def test_restricted_pickle_rejects_code():
    import io
    from cyp_native_enzymecage import feature_load
    import pickle
    with pytest.raises(pickle.UnpicklingError): feature_load(io.BytesIO(b"cos\nsystem\n."))


def test_reaction_subset_accepts_equivalent_stereo_graph():
    from cyp_native_enzymecage import reaction_component_subset
    released = "CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)([O-])OP(=O)([O-])[O-]>>C/C(=C\\CC/C(C)=C/CC/C(C)=C/COP(=O)([O-])OP(=O)([O-])[O-])CO"
    full = "CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)([O-])OP(=O)([O-])[O-].O=O>>C/C(=C\\COP(=O)([O-])OP(=O)([O-])[O-])CC/C=C(\\C)CC/C=C(\\C)CO.O"
    assert reaction_component_subset(released, full) == (True, True)


def test_reaction_subset_rejects_different_product():
    from cyp_native_enzymecage import reaction_component_subset
    assert reaction_component_subset("CCO>>CC=O", "CCO.O=O>>CCO.O") == (False, False)
    assert reaction_component_subset("CC>>C/C=C/C", "CC>>C/C=C\\C") == (False, False)


def test_recovery_extracts_eval_complexes_only():
    from pathlib import PurePosixPath as P
    assert choose("enzymecage_complexes", P("dataset/boltz_outputs/eval/p_model_0.cif"))
    assert choose("enzymecage_complexes", P("dataset/boltz_outputs/train_pos/p_model_0.cif")) is None
    assert choose("boltzcyp_recovery", P("data_dir_backup/eval/structures/p.cif"))
    assert choose("boltzcyp_recovery", P("data_dir_backup/train/structures/p.cif")) is None
    assert choose("boltzcyp_recovery", P("data_dir_backup/eval/structures/._p.cif")) is None


def test_fold_unknown_residue_round_trip(tmp_path):
    import torch
    from recover_cyp_structures import fold_ca_pdb, sequence, validate_fold_sequence
    validate_fold_sequence("AXG")
    frames = torch.zeros(1, 1, 3, 7)
    frames[0, 0, :, 0] = 1
    frames[0, 0, :, 4:] = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    pdb = tmp_path / "unknown.pdb"
    pdb.write_text(fold_ca_pdb({"frames": frames}, "AXG"))
    assert sequence(pdb) == "AXG"
    from Bio.PDB import PDBParser
    residues = list(PDBParser(QUIET=True).get_structure("test", pdb).get_residues())
    assert [r.id[1] for r in residues] == [1, 2, 3]
    assert residues[1].resname == "UNK"
    assert list(residues[1]["CA"].coord) == [4., 5., 6.]


@pytest.mark.parametrize("seq", ["", "ACU", "ACB", "acx", "A*G"])
def test_fold_rejects_implicit_substitution(seq):
    from recover_cyp_structures import validate_fold_sequence
    with pytest.raises(ValueError): validate_fold_sequence(seq)


def test_fold_rejects_invalid_frames():
    import torch
    from recover_cyp_structures import fold_ca_pdb
    frames = torch.zeros(1, 1, 1, 7)
    with pytest.raises(ValueError, match="length mismatch"):
        fold_ca_pdb({"frames": frames}, "AX")
    frames[0, 0, 0, 4] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        fold_ca_pdb({"frames": frames}, "X")
