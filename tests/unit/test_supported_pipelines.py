"""Regression contracts for the supported pipeline API and frozen V4 recipe."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from horizyn.config import load_config
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.pipelines.checkpoints import load_head
from horizyn.pipelines.evaluation import evaluate_embeddings
from horizyn.pipelines.fusion import adjusted
from horizyn.pipelines.inference import RetrievalPipeline
from horizyn.pipelines.refinement import train_refinement
from horizyn.pipelines.registry import validate_pipeline
from horizyn.pipelines.scoring import canonical_dot, refine

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["v4", "f3", "circev2"])
def test_maintained_profiles_and_wrong_family(name):
    paths = list((ROOT / "configs/pipelines" / name).glob("*.yaml"))
    config = load_config(next(p for p in paths if p.name != "phase2.yaml"))
    assert validate_pipeline(config, name).value == name
    if name != "v4":
        with pytest.raises(ValueError, match="enzyme_input_mode"):
            validate_pipeline(config, "v4")
    config.model.e2r_adapter = {"enabled": True}
    with pytest.raises(ValueError, match="e2r_adapter"):
        validate_pipeline(config, name)


def test_v4_score_recipe_and_single_calibration():
    torch.manual_seed(7)
    g, f = F.normalize(torch.randn(5, 8), dim=-1), F.normalize(torch.randn(5, 8), dim=-1)
    scale = 0.03
    original = F.normalize(g + scale * f, dim=-1, eps=1e-6)
    calibrated = adjusted(g, f, scale, 3, original)
    assert torch.equal(adjusted(g, f, scale * 3, 1, calibrated), calibrated)
    head = FrozenGeometryResidual(8, 16, 0.2)
    with torch.no_grad():
        head.enzyme[-1].weight.normal_(0, 0.02)
    expected = F.normalize(calibrated + 0.1 * head.enzyme(calibrated), dim=-1)
    assert torch.equal(refine(calibrated, head, "enzyme", 0.5), expected)
    r = F.normalize(torch.randn(3, 8), dim=-1)
    assert torch.equal(canonical_dot(r, expected), (r.double() @ expected.double().T).float())


def test_changed_manifest_artifact_fails_before_model_loading(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("changed config")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"pipeline": "v4", "config": "config.yaml",
                                    "config_sha256": "incorrect", "checkpoint": "missing.ckpt"}))
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        RetrievalPipeline.from_manifest(manifest)


def test_dictionary_protocol_is_rejected(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"model": {}, "recipe": {"semantic_alpha": 0.4}}))
    with pytest.raises(ValueError, match="dictionary"):
        RetrievalPipeline.from_manifest(path)


def test_reactzyme_averages_all_partners_not_only_best():
    r = torch.tensor([[1.0, 0.0]])
    e = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    result = evaluate_embeddings(r, e, np.array([[0, 0], [0, 1]]))
    metrics = result["reaction_to_enzyme"]["metrics"]
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert result["reaction_to_enzyme"]["candidate_count"] == 3
    assert result["enzyme_to_reaction"]["num_queries"] == 2


def test_screening_uses_whole_bank_and_early_retrieval_metrics():
    r = torch.tensor([[1.0, 0.0]])
    e = torch.stack([torch.tensor([1 - i / 20, i / 20]) for i in range(20)])
    result = evaluate_embeddings(r, e, np.array([[0, 0]]), protocol="screening")
    assert set(result) == {"reaction_to_enzyme"}
    metrics = result["reaction_to_enzyme"]["metrics"]
    assert not any("mrr" in key for key in metrics)
    assert metrics["ef_0_05"] == pytest.approx(20)
    assert metrics["ef_0_1"] == pytest.approx(10)


def feature_fixture(path):
    path.mkdir()
    rng = np.random.default_rng(3)
    (path / "manifest.json").write_text(json.dumps({"test_used": False, "training_edges_only": True}))
    (path / "catalog.json").write_text(json.dumps({"reactions": ["r0", "r1"],
                                                  "train_reactions": ["r0", "r1"],
                                                  "proteins": ["e0", "e1", "e2"]}))
    np.savez(path / "pairs.npz", train=np.array([[0, 0], [0, 1], [1, 1], [1, 2]]),
             test=np.array([[999, 999]]))  # Must never be accessed by refinement.
    np.savez(path / "f3_features.npz", proteins=rng.normal(size=(3, 8)).astype(np.float32),
             train_reactions=rng.normal(size=(2, 8)).astype(np.float32))


def test_full_graph_refinement_updates_heads_and_preserves_checkpoint_schema(tmp_path):
    features, output = tmp_path / "features", tmp_path / "fit"
    feature_fixture(features)
    model = train_refinement(features, output, steps=3, hidden=16)
    assert model.enzyme[-1].weight.abs().sum() > 0
    registry = json.loads((output / "registry.json").read_text())
    loaded = load_head(output / "step0003.pt", registry["feature_manifest_sha256"], "cpu")
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="lineage"):
        load_head(output / "step0003.pt", "wrong training split", "cpu")
    with pytest.raises(ValueError, match="fresh"):
        train_refinement(features, output, steps=1)


def test_refinement_rejects_test_features_and_duplicate_edges(tmp_path):
    features = tmp_path / "features"
    feature_fixture(features)
    np.savez(features / "pairs.npz", train=np.array([[0, 0], [0, 0]]))
    with pytest.raises(ValueError, match="unique"):
        train_refinement(features, tmp_path / "fit", steps=1)
    (features / "manifest.json").write_text(json.dumps({"test_used": True, "training_edges_only": True}))
    with pytest.raises(ValueError, match="training-only"):
        train_refinement(features, tmp_path / "fit", steps=1)


def test_training_export_to_refinement_end_to_end(tmp_path):
    """Exercise the real checkpoint, feature readers, V4 encoders, and exporter."""
    import importlib.util
    import lightning.pytorch as pl
    import yaml

    from horizyn.model import FunctionalResidueScorer
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    from horizyn.pipelines.export import export_refinement

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        fixture_path = ROOT / "tests/fixtures/multimodal_training.py"
        spec = importlib.util.spec_from_file_location("export_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.make_multimodal_fixture(tmp_path, False)
        scorer = tmp_path / "sleec.pt"
        torch.save(FunctionalResidueScorer(8, 6).scorer.state_dict(), scorer)
        model = ProteinPooledLitModule(
            query_encoder_dims=[8, 8], target_encoder_dims=[8, 8], embedding_dim=8,
            residue_dim=8, pooling="sleec_guided_attention", sleec_scorer_hidden_dim=6,
            sleec_checkpoint_path=str(scorer), sleec_freeze_scorer=True,
            enzyme_input_mode="raw_mean_sleec_multiview",
            enzyme_multiview={"hidden_dim": 8, "num_slots": 2, "dropout": 0.0},
            query_encoder_type="multimodal_reaction_attention", reaction_model_dim=6,
            reaction_unimol_dim=5, reaction_chienn_dim=3, reaction_chemistry_dim=617,
            reaction_use_chemistry=True, reaction_modality_encoder_widths=[8],
            loss_name="DecoupledAllPositiveInfoNCELoss",
        ).eval()
        checkpoint = tmp_path / "base.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, checkpoint)
        config = {
            "model": {"name": "ProteinPooledDualModel", "query_encoder_type": "multimodal_reaction_attention",
                      "enzyme_input_mode": "raw_mean_sleec_multiview"},
            "data": {"train_pairs_path": str(tmp_path / "pairs.csv"),
                     "train_reactions_path": str(tmp_path / "reactions.csv"),
                     "protein_residue_embeds_path": str(tmp_path / "proteins.h5"),
                     "max_protein_tokens": 8, "protein_truncation": "ends_center",
                     "reaction_representation": "multimodal_reaction_attention",
                     "train_reaction_t5v2_embeds_path": str(tmp_path / "t5.h5"),
                     "train_reaction_unimol2_embeds_path": str(tmp_path / "unimol.h5"),
                     "train_reaction_chiro_embeds_path": str(tmp_path / "chiro.h5"),
                     "train_reaction_chemistry_vectors_path": str(tmp_path / "chem.npz"),
                     "reaction_model_dim": 6, "reaction_unimol_dim": 5,
                     "reaction_chiro_dim": 3, "reaction_chemistry_dim": 617,
                     "reaction_use_chemistry": True, "standardize_reactions": False}}
        config_path = tmp_path / "config.yaml"
        defaults = yaml.safe_load((ROOT / "configs/pipelines/v4/reactzyme_reaction_smi.yaml").read_text())
        defaults["model"].update(config["model"])
        defaults["model"].update({"query_encoder_dims": [8, 8], "target_encoder_dims": [8, 8],
                                  "embedding_dim": 8, "enzyme_multiview": {"hidden_dim": 8, "num_slots": 2}})
        defaults["model"]["sleec_pooling"].update({"checkpoint_path": str(scorer), "scorer_hidden_dim": 6})
        defaults["data"].update(config["data"])
        defaults["data"]["residue_dim"] = 8
        defaults["training"]["validation_enabled"] = False
        defaults["training"]["validation_retrieval_metrics"] = False
        config = defaults
        config_path.write_text(yaml.safe_dump(config))
        features = tmp_path / "export"
        export_refinement(config_path, checkpoint, features, batch_size=2)
        with np.load(features / "f3_features.npz") as data:
            assert data["proteins"].shape == data["train_reactions"].shape == (4, 8)
            np.testing.assert_allclose(np.linalg.norm(data["proteins"], axis=1), 1, atol=1e-6)
        manifest = json.loads((features / "manifest.json").read_text())
        assert manifest["test_used"] is False and manifest["fusion_multiplier"] == 1
        train_refinement(features, tmp_path / "heads", steps=2, hidden=16)
        assert (tmp_path / "heads/step0002.pt").exists()
    finally:
        torch.set_num_threads(previous_threads)
