from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from horizyn.checkpoint_utils import (
    extract_query_encoder_state_dict,
    load_query_encoder_checkpoint,
)
from horizyn.datasets.base import BaseDataset
from horizyn.hyperbolic_enzyme import HierarchyRankingLoss
from scripts.pretrain_hyperbolic_reaction import (
    ReactionECFingerprintDataset,
    base_reaction_id,
    build_reaction_encoder,
    load_reaction_ec_label_sets,
    reaction_ec_collate_fn,
)
from horizyn.lorentz import expmap0_lorentz


def test_reaction_ec_labels_are_derived_from_train_pairs(tmp_path: Path) -> None:
    pairs_path = tmp_path / "train_pairs.csv"
    labels_path = tmp_path / "ec_labels.csv"

    with open(pairs_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pr_id", "reaction_id", "protein_id"])
        writer.writerows(
            [
                ["0", "r0", "p0"],
                ["1", "r0", "p1"],
                ["2", "r1", "p2"],
                ["3", "r2", "missing"],
            ]
        )

    with open(labels_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number"])
        writer.writerows(
            [
                ["p0", "1.1.1.1"],
                ["p1", "2.7.7.7; 2.7.7.8"],
                ["p2", "1.1.1.-"],
            ]
        )

    label_sets = load_reaction_ec_label_sets(pairs_path, labels_path)

    assert label_sets["r0"] == {"1.1.1.1", "2.7.7.7", "2.7.7.8"}
    assert label_sets["r1"] == {"1.1.1.-"}
    assert "r2" not in label_sets


def test_reaction_ec_fingerprint_dataset_and_collate() -> None:
    features = BaseDataset(
        keys=["r0_f", "r0_r", "r1_f", "r2_f"],
        array_data=torch.arange(16, dtype=torch.float32).reshape(4, 4),
    )
    dataset = ReactionECFingerprintDataset(
        reaction_features=features,
        reaction_ec_labels={
            "r0": {"1.1.1.1", "1.1.1.2"},
            "r1": {"2.7.7.7"},
        },
    )

    assert base_reaction_id("r0_f") == "r0"
    assert base_reaction_id("r0_r") == "r0"
    assert len(dataset) == 5

    batch = reaction_ec_collate_fn([dataset[idx] for idx in range(len(dataset))])
    assert batch["reaction_vec"].shape == (5, 4)
    assert batch["ec_depth_matrix"].shape == (5, 5)
    assert torch.equal(
        torch.diag(batch["ec_depth_matrix"]),
        torch.tensor([4, 4, 4, 4, 4]),
    )


def test_reaction_encoder_hierarchy_ranking_backward() -> None:
    torch.manual_seed(6)
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.1",
        "2.7.7.7",
        "2.7.7.8",
        "3.5.4.4",
    ]
    features = torch.randn(len(labels), 16)
    encoder = build_reaction_encoder([16, 8, 4])
    tangent = encoder(features)
    z_hyp = expmap0_lorentz(tangent, kappa=0.25)
    depths = reaction_ec_collate_fn(
        [
            {"reaction_id": f"r{idx}", "reaction_vec": features[idx], "ec_label": label}
            for idx, label in enumerate(labels)
        ]
    )["ec_depth_matrix"]

    loss = HierarchyRankingLoss(curvature=0.25)(z_hyp, depths).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert encoder.main_nn[0].weight.grad is not None
    assert torch.isfinite(encoder.main_nn[0].weight.grad).all()


def test_query_encoder_checkpoint_loader_supports_reaction_stage1(tmp_path: Path) -> None:
    source = build_reaction_encoder([4, 8, 3])
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}

    checkpoint_path = tmp_path / "stage1.ckpt"
    torch.save({"query_encoder_state_dict": source_state}, checkpoint_path)

    target = build_reaction_encoder([4, 8, 3])
    for parameter in target.parameters():
        parameter.data.zero_()

    load_query_encoder_checkpoint(target, checkpoint_path)
    for key, value in target.state_dict().items():
        assert torch.allclose(value, source_state[key])

    prefixed = {f"model.query_encoder.{key}": value for key, value in source_state.items()}
    extracted = extract_query_encoder_state_dict({"state_dict": prefixed})
    assert set(extracted) == set(source_state)
    for key, value in extracted.items():
        assert torch.allclose(value, source_state[key])


def test_tiny_reaction_pretraining_script_smoke_test(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pairs_path = tmp_path / "train_pairs.csv"
    reactions_path = tmp_path / "train_rxns.csv"
    labels_path = tmp_path / "ec_labels.csv"
    checkpoint_path = tmp_path / "reaction_stage1.ckpt"

    with open(reactions_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "reaction_smiles"])
        writer.writerows(
            [
                ["r0", "CCO>>CC=O"],
                ["r1", "CCCO>>CCC=O"],
                ["r2", "CCN>>CC=O"],
                ["r3", "CCCl>>CCBr"],
                ["r4", "CCO.CC(=O)O>>CCOC(C)=O"],
                ["r5", "CCO>>CCCl"],
            ]
        )

    with open(pairs_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pr_id", "reaction_id", "protein_id"])
        writer.writerows(
            [
                ["0", "r0", "p0"],
                ["1", "r1", "p1"],
                ["2", "r2", "p2"],
                ["3", "r3", "p3"],
                ["4", "r4", "p4"],
                ["5", "r5", "p5"],
            ]
        )

    with open(labels_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number"])
        writer.writerows(
            [
                ["p0", "1.1.1.1"],
                ["p1", "1.1.1.2"],
                ["p2", "2.7.7.7"],
                ["p3", "2.7.7.8"],
                ["p4", "3.5.4.4"],
                ["p5", "3.5.4.5"],
            ]
        )

    command = [
        sys.executable,
        str(repo_root / "scripts" / "pretrain_hyperbolic_reaction.py"),
        "--train-pairs-path",
        str(pairs_path),
        "--train-reactions-path",
        str(reactions_path),
        "--ec-labels-path",
        str(labels_path),
        "--rdkit-fp-dim",
        "8",
        "--drfp-dim",
        "8",
        "--reaction-encoder-dims",
        "16",
        "8",
        "4",
        "--hyp-dim",
        "4",
        "--batch-size",
        "12",
        "--epochs",
        "1",
        "--max-steps",
        "1",
        "--output-checkpoint",
        str(checkpoint_path),
        "--device",
        "cpu",
        "--log-every",
        "1",
        "--no-standardize-reactions",
    ]
    subprocess.run(command, cwd=repo_root, check=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_path.exists()
    assert checkpoint["input_dim"] == 16
    assert checkpoint["reaction_embedding_dim"] == 4
    assert "query_encoder_state_dict" in checkpoint
    assert "hyperbolic_projector_state_dict" in checkpoint
