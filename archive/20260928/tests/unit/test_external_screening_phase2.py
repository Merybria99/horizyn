import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


def test_external_screening_trainer_never_computes_mrr_or_uses_validation_edges(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("screen_phase2_trainer", root / "scripts/generalization_full_graph.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    def forbidden(*args, **kwargs):
        raise AssertionError("External screening selection must not use small-pool validation")

    monkeypatch.setattr(trainer, "validation_data", forbidden)
    monkeypatch.setattr(trainer, "evaluate_scores", forbidden)
    features = tmp_path / "features"
    features.mkdir()
    catalog = dict(proteins=["e0", "e1", "e2", "e3"], reactions=["r0", "r1", "r2"],
                   train_reactions=["r0", "r1", "r2"])
    (features / "catalog.json").write_text(json.dumps(catalog))
    (features / "manifest.json").write_text("{}")
    np.savez(features / "pairs.npz", train=np.array([[0, 0], [0, 1], [1, 2], [2, 3]]))
    rng = np.random.default_rng(42)
    reactions = rng.normal(size=(3, 8)).astype("float32")
    np.savez(features / "f3_features.npz", proteins=rng.normal(size=(4, 8)).astype("float32"),
             reactions=reactions, train_reactions=reactions)
    args = SimpleNamespace(features=features, output=tmp_path / "result", seed=42,
        cpu_threads=1, device="cpu", hidden=8, scale=.2, learning_rate=.001,
        weight_decay=.001, selection_method="external_screening", steps=5,
        snapshot_every=5, validate_every=1, contrastive_objective="decoupled",
        temperature=.1, ranking_weight=0., identity_weight=10., enzyme_weighting="uniform")
    trainer.run(args)
    complete = json.loads((args.output / "complete.json").read_text())
    assert complete["validation_selection_pending"] and complete["best_step"] is None
    assert not (args.output / "validation.json").exists()
    assert not (args.output / "selected.pt").exists()
    saved = torch.load(args.output / "step0005.pt", weights_only=False)
    assert saved["fixed_step"] == 5
    assert saved["state_dict"]["reaction.3.weight"].abs().sum() > 0
    # A zero-learning-rate continuation must preserve every loaded parameter,
    # while accepting an identity target at that initialization.
    args.warm_start = args.output / "step0005.pt"
    args.output = tmp_path / "warm_result"
    args.identity_reference = "initial"
    args.learning_rate = 0.
    trainer.run(args)
    continued = torch.load(args.output / "step0005.pt", weights_only=False)
    assert continued["registry"]["warm_start"]["sha256"] == trainer.sha(args.warm_start)
    for key, value in saved["state_dict"].items():
        torch.testing.assert_close(continued["state_dict"][key], value, rtol=0, atol=0)
