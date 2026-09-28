import hashlib
import json

import pytest

from scripts.generalization_test_export import record_freeze


def test_export_requires_real_nonempty_freeze_before_creating_output(tmp_path):
    output = tmp_path / "heldout"
    with pytest.raises(FileNotFoundError):
        record_freeze(tmp_path / "missing.json", output, "reaction_smi")
    assert not output.exists()
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    with pytest.raises(ValueError, match="nonempty"):
        record_freeze(freeze, output, "reaction_smi")
    assert not output.exists()


def test_frozen_recipe_identity_is_recorded_and_cannot_change(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"recipe": "validation-selected"}))
    output = tmp_path / "heldout"
    receipt = record_freeze(freeze, output, "reaction_smi")
    assert receipt["freeze_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()
    assert json.loads((output / "freeze_receipt.json").read_text()) == receipt
    assert record_freeze(freeze, output, "reaction_smi") == receipt
    freeze.write_text(json.dumps({"recipe": "changed-after-test"}))
    with pytest.raises(ValueError, match="another frozen recipe"):
        record_freeze(freeze, output, "reaction_smi")
    assert json.loads((output / "freeze_receipt.json").read_text()) == receipt


def test_synthetic_official_export_preserves_catalog_and_rejects_missing_candidate(tmp_path, monkeypatch):
    """Exercise the pipeline with invented arrays, never the real test split."""
    import sys

    import h5py
    import numpy as np
    import yaml

    from scripts import generalization_test_export as export
    from scripts.generalization_export import identity

    monkeypatch.setattr(export, "ROOT", tmp_path)
    monkeypatch.setitem(export.EXPECTED, "reaction_smi", (2, 3))
    helper = tmp_path / "scripts/generalization_export.py"
    helper.parent.mkdir()
    helper.write_text("# synthetic provenance fixture\n")
    source = tmp_path / "source"
    source.mkdir()
    pairs = source / "pairs.csv"
    pairs.write_text("reaction_id,protein_id\nq1,p2\nq1,p1\nq2,p3\n")
    reactions = source / "reactions.csv"
    reactions.write_text("reaction_id,reaction_smiles\nq2,CO\nq1,CC\n")
    candidates = source / "candidates.txt"
    candidates.write_text("p3\np1\np2\n")

    def vectors(path, ids, values):
        with h5py.File(path, "w") as h:
            h["ids"] = np.asarray(ids, dtype=h5py.string_dtype())
            h["vectors"] = values

    residues = source / "residues.h5"
    vectors(residues, ["p3", "p1", "p2"], np.arange(4096, dtype=np.float16).reshape(4, 1024))
    with h5py.File(residues, "a") as h:
        h["offsets"] = [0, 1, 3, 4]
    features = {}
    vectors(source / "t5v2.h5", ["q2_f", "q1_f"], np.ones((2, 768), np.float16))
    features["reaction_t5v2_embeds_path"] = source / "t5v2.h5"
    for modality, dim in (("unimol2", 768), ("chiro", 256)):
        path = source / f"{modality}.h5"
        with h5py.File(path, "w") as h:
            h["ids"] = np.asarray(["q2_f", "q1_f"], dtype=h5py.string_dtype())
            for side in ("reactant", "product"):
                h[f"{side}_offsets"] = [0, 1, 2]
                h[f"{side}_vectors"] = np.ones((2, dim), np.float16)
        features[f"reaction_{modality}_embeds_path"] = path
    chemistry = source / "chemistry.npz"
    np.savez(chemistry, ids=np.asarray(["q2", "q1"]), vectors=np.ones((2, 617), np.float32), mask=np.ones(2, bool))
    (source / "schema.json").write_text(json.dumps({"fit_split": "train"}))
    features["reaction_chemistry_vectors_path"] = chemistry
    checkpoint = tmp_path / "runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic frozen weights")
    cache = tmp_path / "runs/biological_residual_reaction_smi/cache/reactzyme_test"
    cache.mkdir(parents=True)
    vectors(cache / "enzyme_base.h5", ["p3", "p1", "p2"], np.ones((3, 512), np.float32))
    vectors(cache / "validation_reaction_base.h5", ["q2_f", "q1_f"], np.ones((2, 512), np.float32))
    manifest = dict(dtype="float32", signature=dict(precision="32", inputs=dict(
        checkpoint=identity(checkpoint), validation_pairs=identity(pairs, True),
        validation_reactions=identity(reactions, True), protein_residues=identity(residues),
        reaction_feature_inputs={key: identity(path) for key, path in features.items()})))
    (cache / "manifest.json").write_text(json.dumps(manifest))
    config = tmp_path / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"data": dict(test_pairs_path=str(pairs), test_reactions_path=str(reactions),
        validation_retrieval_candidate_ids_path=str(candidates), protein_residue_embeds_path=str(residues),
        **{key: str(value) for key, value in features.items()})}))
    freeze = tmp_path / "frozen.json"
    freeze.write_text(json.dumps({"recipe": "synthetic fixed model"}))
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["export", "--freeze", str(freeze), "--split", "reaction_smi", "--output", str(output)])
    export.main()
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["proteins"] == ["p1", "p2", "p3"]
    assert catalog["reactions"] == ["q1", "q2"]
    with np.load(output / "pairs.npz") as z:
        np.testing.assert_array_equal(z["test"], [[0, 0], [0, 1], [1, 2]])
    with h5py.File(output / "protein_mean.h5") as h:
        assert h["complete"][:].all()
        np.testing.assert_array_equal(h["vectors"][0], np.arange(4096, dtype=np.float16).reshape(4, 1024)[1:3].mean(0, dtype=np.float32))
    assert json.loads((output / "complete.json").read_text())["selection_allowed"] is False
    candidates.write_text("p1\np2\n")
    monkeypatch.setattr(sys, "argv", ["export", "--freeze", str(freeze), "--split", "reaction_smi", "--output", str(tmp_path / "invalid")])
    with pytest.raises(ValueError, match="complete held-out positive enzyme universe"):
        export.main()
