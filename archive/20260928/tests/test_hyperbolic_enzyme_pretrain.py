from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from horizyn.hyperbolic_enzyme import (
    ECPrefixEntailmentLoss,
    HierarchyRankingLoss,
    LorentzEnzymeProjector,
    SLEECGuidedAttentionPool,
    build_hierarchy_triplets,
    compute_ec_shared_depth,
    known_ec_depth,
    parse_ec_prefixes,
)
from horizyn.lorentz import expmap0_lorentz
from scripts.evaluate_hyperbolic_enzyme import (
    compute_pair_distance_diagnostics,
    compute_triplet_diagnostics,
)


def test_ec_shared_depth_matrix() -> None:
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.1",
        "1.2.1.1",
        "2.1.1.1",
    ]

    depths = compute_ec_shared_depth(labels)
    assert depths.shape == (5, 5)
    assert int(depths[0, 1].item()) == 3
    assert int(depths[0, 2].item()) == 2
    assert int(depths[0, 3].item()) == 1
    assert int(depths[0, 4].item()) == 0
    assert torch.equal(torch.diag(depths), torch.full((5,), 4, dtype=torch.long))


def test_incomplete_ec_shared_depth_uses_only_known_prefixes() -> None:
    labels = [
        "1.1.1.1",
        "1.1.1.-",
        "1.1.-.-",
        "1.2.-.-",
        "2.-.-.-",
    ]

    assert parse_ec_prefixes("1.1.1.-") == ("1", "1.1", "1.1.1", None)
    assert parse_ec_prefixes("1.1") == ("1", "1.1", None, None)
    assert known_ec_depth("1.1.-.-") == 2

    depths = compute_ec_shared_depth(labels)
    assert depths.shape == (5, 5)
    assert torch.equal(torch.diag(depths), torch.tensor([4, 3, 2, 2, 1]))
    assert int(depths[0, 1].item()) == 3
    assert int(depths[0, 2].item()) == 2
    assert int(depths[0, 3].item()) == 1
    assert int(depths[0, 4].item()) == 0
    assert int(depths[1, 2].item()) == 2
    assert int(depths[2, 3].item()) == 1

    triplets = build_hierarchy_triplets(depths, max_triplets_per_anchor=8)
    assert triplets[0].numel() > 0


def test_sleec_guided_attention_pool_shapes_masks_and_prior() -> None:
    torch.manual_seed(2)
    pooler = SLEECGuidedAttentionPool(
        input_dim=1280,
        scorer_hidden_dim=16,
        p0=0.34,
        sleec_checkpoint_path=None,
        freeze_sleec=True,
    )
    residues = torch.randn(4, 128, 1280)
    residue_mask = torch.ones(4, 128, dtype=torch.bool)
    residue_mask[1, 96:] = False
    residue_mask[2, 64:] = False
    residue_mask[3, 32:] = False
    sleec_logits = torch.randn(4, 128)

    pooled, details = pooler(
        residues,
        residue_mask=residue_mask,
        sleec_logits=sleec_logits,
        return_details=True,
    )

    assert pooled.shape == (4, 1280)
    assert details["attention_weights"].shape == (4, 128)
    assert details["attention_logits"].shape == (4, 128)
    assert details["sleec_logits"].shape == (4, 128)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(details["attention_weights"]).all()
    assert torch.allclose(
        details["attention_weights"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.equal(
        details["attention_weights"][~residue_mask],
        torch.zeros_like(details["attention_weights"][~residue_mask]),
    )
    assert float(details["prior_strength"].item()) >= 0.0


def test_hierarchy_ranking_loss_backward() -> None:
    torch.manual_seed(3)
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.1",
        "1.2.1.1",
        "2.1.1.1",
        "2.1.1.2",
        "2.1.2.1",
        "3.1.1.1",
    ]
    depths = compute_ec_shared_depth(labels)
    triplets = build_hierarchy_triplets(depths, max_triplets_per_anchor=8)
    assert triplets[0].numel() > 0

    tangent = (torch.randn(8, 512) * 0.1).requires_grad_()
    z_hyp = expmap0_lorentz(tangent, kappa=0.25)
    loss_output = HierarchyRankingLoss(curvature=0.25)(z_hyp, depths, triplets=triplets)
    assert loss_output.loss.ndim == 0
    assert torch.isfinite(loss_output.loss)

    loss_output.loss.backward()
    assert tangent.grad is not None
    assert torch.isfinite(tangent.grad).all()


def test_hyperbolic_enzyme_evaluator_depth_metrics() -> None:
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.1",
        "1.2.1.1",
        "2.1.1.1",
        "2.1.1.2",
        "2.1.2.1",
        "3.1.1.1",
    ]
    tangent = torch.tensor(
        [
            [0.10, 0.00, 0.00],
            [0.12, 0.00, 0.00],
            [0.25, 0.00, 0.00],
            [0.45, 0.00, 0.00],
            [-0.10, 0.00, 0.00],
            [-0.12, 0.00, 0.00],
            [-0.25, 0.00, 0.00],
            [0.00, 0.60, 0.00],
        ],
        dtype=torch.float32,
    )
    z_hyp = expmap0_lorentz(tangent, kappa=0.25)
    distance_fn = lambda left, right: torch.linalg.norm(left - right, dim=-1)

    pair_metrics, distance_rows, _ = compute_pair_distance_diagnostics(
        ec_labels=labels,
        representations={"tangent_euclidean": (tangent, distance_fn)},
        max_pairs=0,
        seed=1,
        pair_batch_size=16,
    )
    assert pair_metrics["num_pair_samples"] == 28
    assert "tangent_euclidean_spearman_depth_neg_distance" in pair_metrics
    assert len(distance_rows) == 5

    triplet_metrics, triplet_rows = compute_triplet_diagnostics(
        ec_labels=labels,
        representations={"lorentz": (z_hyp, lambda left, right: torch.linalg.norm(left - right, dim=-1))},
        base_margin=0.05,
        max_triplet_anchors=8,
        max_triplets_per_anchor=4,
        seed=1,
        pair_batch_size=16,
    )
    assert triplet_metrics["num_triplet_samples"] > 0
    assert 0.0 <= triplet_metrics["lorentz_triplet_accuracy"] <= 1.0
    assert triplet_rows


def test_ec_prefix_entailment_loss_backward_with_incomplete_ecs() -> None:
    torch.manual_seed(7)
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.-",
        "1.1.-.-",
        "2.1.1.1",
        "2.1.1.2",
        "2.1.-.-",
        "2.-.-.-",
    ]
    tangent = (torch.randn(8, 16) * 0.1).requires_grad_()
    z_hyp = expmap0_lorentz(tangent, kappa=0.25)
    loss_fn = ECPrefixEntailmentLoss(
        curvature=0.25,
        min_radius=0.1,
        eta=1.0,
        min_group_size=2,
    )

    loss, logs = loss_fn(z_hyp, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(logs["num_ec_entailment_terms"].item()) > 0.0
    assert torch.isfinite(logs["ec_entailment_violation_rate"])
    assert torch.isfinite(logs["mean_ec_entailment_angle"])
    assert torch.isfinite(logs["mean_ec_entailment_aperture"])

    loss.backward()
    assert tangent.grad is not None
    assert torch.isfinite(tangent.grad).all()


def test_full_pretraining_forward_backward() -> None:
    torch.manual_seed(4)
    labels = [
        "1.1.1.1",
        "1.1.1.2",
        "1.1.2.1",
        "1.2.1.1",
        "2.1.1.1",
        "2.1.1.2",
        "2.1.2.1",
        "3.1.1.1",
    ]
    residues = torch.randn(8, 128, 1280)
    residue_mask = torch.ones(8, 128, dtype=torch.bool)
    residue_mask[0, 96:] = False
    residue_mask[3, 64:] = False
    sleec_logits = torch.randn(8, 128)
    depths = compute_ec_shared_depth(labels)

    pooler = SLEECGuidedAttentionPool(
        input_dim=1280,
        scorer_hidden_dim=16,
        p0=0.34,
        sleec_checkpoint_path=None,
        freeze_sleec=True,
    )
    projector = LorentzEnzymeProjector(input_dim=1280, hyp_dim=512, curvature=0.25)
    loss_fn = HierarchyRankingLoss(curvature=0.25)

    pooled, _ = pooler(
        residues,
        residue_mask=residue_mask,
        sleec_logits=sleec_logits,
        return_details=True,
    )
    z_hyp, z_tangent = projector(pooled)
    assert z_tangent.shape == (8, 512)
    loss_output = loss_fn(z_hyp, depths)
    assert torch.isfinite(loss_output.loss)

    loss_output.loss.backward()
    assert pooler.attention.weight.grad is not None
    assert pooler.raw_prior_strength.grad is not None
    assert projector.projection.weight.grad is not None
    assert torch.isfinite(pooler.attention.weight.grad).all()
    assert torch.isfinite(projector.projection.weight.grad).all()


def test_tiny_training_script_smoke_test(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    residues_path = tmp_path / "residues.h5"
    logits_path = tmp_path / "sleec_logits.h5"
    labels_path = tmp_path / "ec_labels.csv"
    checkpoint_path = tmp_path / "last.ckpt"

    rng = np.random.default_rng(0)
    protein_ids = [f"p{i}" for i in range(8)]
    lengths = [5, 6, 4, 7, 5, 6, 4, 5]
    offsets = np.array([0, *np.cumsum(lengths)], dtype="int64")
    vectors = rng.normal(size=(int(offsets[-1]), 6)).astype("float32")
    logits = rng.normal(size=(int(offsets[-1]),)).astype("float32")
    encoded_ids = np.array([protein_id.encode("utf-8") for protein_id in protein_ids])

    with h5py.File(residues_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=encoded_ids)
        h5_file.create_dataset("vectors", data=vectors)
        h5_file.create_dataset("offsets", data=offsets)
    with h5py.File(logits_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=encoded_ids)
        h5_file.create_dataset("logits", data=logits)
        h5_file.create_dataset("offsets", data=offsets)

    ec_labels = [
        ("p0", "1.1.1.1"),
        ("p1", "1.1.1.2"),
        ("p2", "1.1.2.-"),
        ("p3", "1.2.-.-"),
        ("p4", "2.-.-.-"),
        ("p5", "2.1.1.2"),
        ("p6", "2.1.2.1"),
        ("p7", "3.1.1.1"),
    ]
    with open(labels_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number"])
        writer.writerows(ec_labels)

    command = [
        sys.executable,
        str(repo_root / "scripts" / "pretrain_hyperbolic_enzyme.py"),
        "--residue-embeddings-path",
        str(residues_path),
        "--sleec-logits-path",
        str(logits_path),
        "--data-path",
        str(labels_path),
        "--input-dim",
        "6",
        "--hyp-dim",
        "4",
        "--batch-size",
        "8",
        "--epochs",
        "1",
        "--max-steps",
        "2",
        "--output-checkpoint",
        str(checkpoint_path),
        "--device",
        "cpu",
        "--log-every",
        "1",
        "--num-workers",
        "0",
        "--use-ec-entailment",
        "--alpha-ec-entailment",
        "0.1",
        "--ec-entailment-warmup-epochs",
        "0",
        "--ec-entailment-min-group-size",
        "2",
    ]
    subprocess.run(command, cwd=repo_root, check=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary_path = checkpoint_path.with_name("pretrain_summary.json")
    assert checkpoint_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_epochs"] == 1
    assert summary["requested_epochs"] == 1
    assert checkpoint["input_dim"] == 6
    assert checkpoint["hyp_dim"] == 4
    assert checkpoint["curvature"] == 0.25
    assert checkpoint["config"]["use_ec_entailment"] is True
    assert "attention_pooler_state_dict" in checkpoint
    assert "hyperbolic_projector_state_dict" in checkpoint
    assert "ec_node_embedding_state_dict" not in checkpoint

    pooler = SLEECGuidedAttentionPool(input_dim=6, scorer_hidden_dim=256, p0=0.34)
    projector = LorentzEnzymeProjector(input_dim=6, hyp_dim=4, curvature=0.25)
    pooler.load_state_dict(checkpoint["attention_pooler_state_dict"])
    projector.load_state_dict(checkpoint["hyperbolic_projector_state_dict"])
    fake_residues = torch.randn(1, 5, 6)
    fake_mask = torch.ones(1, 5, dtype=torch.bool)
    fake_logits = torch.randn(1, 5)
    pooled = pooler(fake_residues, residue_mask=fake_mask, sleec_logits=fake_logits)
    z_hyp, z_tangent = projector(pooled)
    assert z_hyp.shape == (1, 5)
    assert z_tangent.shape == (1, 4)
    assert torch.isfinite(z_hyp).all()

    resume_command = command.copy()
    resume_command[resume_command.index("--epochs") + 1] = "2"
    resume_command.extend(["--resume-checkpoint", str(checkpoint_path)])
    subprocess.run(resume_command, cwd=repo_root, check=True)

    resumed_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    resumed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert resumed_checkpoint["epoch"] == 2
    assert resumed_summary["completed_epochs"] == 2
    assert resumed_summary["requested_epochs"] == 2


def test_hyperbolic_enzyme_evaluator_script_smoke_test(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    residues_path = tmp_path / "residues.h5"
    logits_path = tmp_path / "sleec_logits.h5"
    labels_path = tmp_path / "ec_labels.csv"
    checkpoint_path = tmp_path / "model.ckpt"
    output_dir = tmp_path / "eval"

    rng = np.random.default_rng(1)
    protein_ids = [f"p{i}" for i in range(10)]
    lengths = [5, 6, 4, 7, 5, 6, 4, 5, 6, 5]
    offsets = np.array([0, *np.cumsum(lengths)], dtype="int64")
    vectors = rng.normal(size=(int(offsets[-1]), 6)).astype("float32")
    logits = rng.normal(size=(int(offsets[-1]),)).astype("float32")
    encoded_ids = np.array([protein_id.encode("utf-8") for protein_id in protein_ids])

    with h5py.File(residues_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=encoded_ids)
        h5_file.create_dataset("vectors", data=vectors)
        h5_file.create_dataset("offsets", data=offsets)
    with h5py.File(logits_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=encoded_ids)
        h5_file.create_dataset("logits", data=logits)
        h5_file.create_dataset("offsets", data=offsets)

    with open(labels_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number"])
        writer.writerows(
            [
                ("p0", "1.1.1.1"),
                ("p1", "1.1.1.2"),
                ("p2", "1.1.2.1"),
                ("p3", "1.2.1.1"),
                ("p4", "2.1.1.1"),
                ("p5", "2.1.1.2"),
                ("p6", "2.1.2.1"),
                ("p7", "3.1.1.1"),
                ("p8", "3.1.1.2"),
                ("p9", "3.1.2.1"),
            ]
        )

    pooler = SLEECGuidedAttentionPool(input_dim=6, scorer_hidden_dim=8, p0=0.34)
    projector = LorentzEnzymeProjector(input_dim=6, hyp_dim=4, curvature=0.25)
    torch.save(
        {
            "attention_pooler_state_dict": pooler.state_dict(),
            "hyperbolic_projector_state_dict": projector.state_dict(),
            "config": {
                "residue_embeddings_path": str(residues_path),
                "sleec_logits_path": str(logits_path),
                "ec_labels_path": str(labels_path),
                "input_dim": 6,
                "hyp_dim": 4,
                "curvature": 0.25,
                "p0_sleec_threshold": 0.34,
                "sleec_scorer_hidden_dim": 8,
                "batch_size": 10,
            },
            "input_dim": 6,
            "hyp_dim": 4,
            "curvature": 0.25,
            "epoch": 0,
            "global_step": 0,
            "best_epoch_loss": 0.0,
        },
        checkpoint_path,
    )

    command = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_hyperbolic_enzyme.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--batch-size",
        "10",
        "--max-samples",
        "10",
        "--max-pairs",
        "100",
        "--max-triplet-anchors",
        "10",
        "--max-triplets-per-anchor",
        "4",
        "--max-knn-samples",
        "10",
        "--k-values",
        "1",
        "3",
        "--max-visualization-points",
        "10",
        "--max-pcoa-points",
        "10",
        "--no-include-negative-controls",
    ]
    subprocess.run(command, cwd=repo_root, check=True)

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "distance_by_ec_depth.csv").exists()
    assert (output_dir / "triplet_metrics_by_gap.csv").exists()
    assert (output_dir / "knn_purity.csv").exists()
    assert (output_dir / "raw_pooled_pca.csv").exists()
    assert (output_dir / "tangent_pca.csv").exists()
    assert (output_dir / "lorentz_pcoa.csv").exists()
