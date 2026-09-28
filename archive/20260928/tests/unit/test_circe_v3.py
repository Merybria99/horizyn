"""Train-only preparation and loss-reduction contracts for CIRCE-v3."""
import csv
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from horizyn.circe_v3 import (
    MONITOR, RankingConstraints, association_holdout, audited_targets, configure, prepare_run,
    read_pairs, verify_run, write_pairs,
)
from horizyn.losses import DecoupledAllPositiveInfoNCELoss
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule


def evidence_file(tmp_path, rows):
    path = tmp_path / "evidence.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow("protein_id,family,label_index,value,confidence,source,evidence_kind,support_reaction_id".split(","))
        writer.writerows(rows)
    return path


def test_holdout_preserves_endpoints_and_is_deterministic():
    edges = [(f"r{i}", f"p{j}") for i in range(4) for j in range(4)]
    train, hidden = association_holdout(edges, .25, 42)
    assert (train, hidden) == association_holdout(list(reversed(edges)), .25, 42)
    assert len(hidden) == 4
    assert not set(train) & set(hidden)
    assert set(train + hidden) == set(edges)
    assert {q for q, _ in train} == {q for q, _ in edges}
    assert {p for _, p in train} == {p for _, p in edges}
    assert association_holdout([("r", "p")], .9, 42)[1] == []


def test_audit_masks_unknown_conflicting_and_one_class_labels(tmp_path):
    path = evidence_file(tmp_path, [
        ("p1", "cofactor", 0, 1, .8, "assay", "curated_positive", ""),
        ("p2", "cofactor", 0, 0, .4, "assay", "measured_negative", ""),
        ("p1", "cofactor", 1, 1, 1, "assay", "curated_positive", ""),
        ("p2", "cofactor", 1, "unknown", "", "", "", ""),
        ("p1", "mechanism", 0, 1, 1, "assay", "curated_positive", ""),
        ("p1", "mechanism", 0, 0, 1, "assay", "measured_negative", ""),
        ("p2", "mechanism", 0, 1, 1, "heldout", "training_association", "hidden"),
    ])
    payload, audit, weights = audited_targets(path, [("r", "p1"), ("r", "p2")], {"cofactor": 2, "mechanism": 1})
    assert weights == {"cofactor": 1.0}
    assert payload["cofactor_mask"].tolist() == [[True, False], [True, False]]
    assert payload["cofactor_confidence"][:, 0].tolist() == pytest.approx([.8, .4])
    assert not payload["mechanism_mask"].any()
    assert audit["evidence"]["unretained_support"] == 1
    assert audit["families"]["mechanism"]["conflicting_cells"] == 1


def test_missing_labels_do_not_enable_heads():
    payload, audit, weights = audited_targets(None, [("r", "p")], {"mechanism": 8, "cofactor": 10})
    assert not weights
    assert not payload["cofactor_mask"].any()
    assert not audit["families"]["mechanism"]["enabled"]


def test_association_absence_cannot_become_negative(tmp_path):
    path = evidence_file(tmp_path, [("p", "cofactor", 0, 0, 1, "rhea", "training_association", "r")])
    with pytest.raises(ValueError, match="not a negative"):
        audited_targets(path, [("r", "p")], {"cofactor": 1})


@pytest.mark.parametrize("split", ["time", "enzyme_smi", "reaction_smi"])
def test_prepare_each_split_and_hash_contract(tmp_path, split):
    base = {"model": {"enzyme_input_mode": "raw_mean_sleec_biological_factorized",
                      "biofp": {"family_dims": {"mechanism": 8, "cofactor": 10}}},
            "data": {"train_reaction_t5v2_embeds_path": str(tmp_path / "train.h5"),
                     "validation_reaction_t5v2_embeds_path": str(tmp_path / "val.h5")},
            "training": {}, "logging": {}, "ablation": {"split": split}}
    test = copy.deepcopy(base)
    edges = [(f"r{i}", f"p{j}") for i in range(3) for j in range(3)]
    for name, pairs in (("train", edges), ("validation", [("val", "p0")]), ("test", [("test", "p0")])):
        write_pairs(tmp_path / f"{name}.csv", pairs)
        (tmp_path / f"{name}_rxns.csv").write_text("reaction_id,reaction_smiles\n")
        for cfg in (base, test):
            cfg["data"][f"{name}_pairs_path"] = str(tmp_path / f"{name}.csv")
            cfg["data"][f"{name}_reactions_path"] = str(tmp_path / f"{name}_rxns.csv")
    candidate = tmp_path / "candidates.txt"
    candidate.write_text("p0\np1\np2\n")
    for cfg in (base, test):
        cfg["data"]["validation_retrieval_candidate_ids_path"] = str(candidate)
    source, test_source = tmp_path / "base.yaml", tmp_path / "test.yaml"
    source.write_text(yaml.safe_dump(base))
    test_source.write_text(yaml.safe_dump(test))
    output = tmp_path / "v3"
    manifest = prepare_run(source, output, test_template=test_source, holdout_fraction=.3)
    assert verify_run(output) == manifest
    cfg = yaml.safe_load((output / "configs/train.yaml").read_text())
    assert cfg["ablation"]["split"] == split
    assert cfg["training"]["loss"]["positive_pair_source"] == "all_known_in_batch"
    assert cfg["training"]["loss"]["biofp_aux_weight"] == 0
    assert cfg["logging"]["checkpoint_monitor"] == MONITOR
    assert set(manifest["configs"]) == {"train", "validation", "hidden", "test"}
    hidden = yaml.safe_load((output / "configs/hidden.yaml").read_text())
    assert hidden["data"]["reaction_t5v2_embeds_path"] == cfg["data"]["train_reaction_t5v2_embeds_path"]
    assert hidden["training"]["validation_retrieval_candidate_ids_path"] == str(output / "data/hidden_candidate_ids.txt")
    with pytest.raises(FileExistsError):
        prepare_run(source, output)
    (output / "data/train_pairs.csv").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_run(output)


def test_fresh_towers_and_low_weight(tmp_path):
    base = {"model": {"enzyme_input_mode": "raw_mean_sleec_biological_factorized", "query_encoder_checkpoint_path": "old"},
            "training": {"init_from_checkpoint": "old", "loss": {}}, "data": {}, "logging": {}}
    cfg = configure(base, tmp_path, tmp_path / "pairs.csv", 42, 4, {"cofactor": 1.0})
    assert "init_from_checkpoint" not in cfg["training"]
    assert "query_encoder_checkpoint_path" not in cfg["model"]
    assert cfg["training"]["loss"]["biofp_aux_weight"] == .02
    assert base["training"]["init_from_checkpoint"] == "old"


@pytest.mark.parametrize("epoch,expected", [(0, 0), (1, .02/3), (3, .02), (30, .02)])
def test_auxiliary_ramp(epoch, expected):
    stub = SimpleNamespace(biofp_aux_weight=.02, biofp_aux_warmup_epochs=3, current_epoch=epoch)
    assert ProteinPooledLitModule._effective_biofp_weight(stub) == pytest.approx(expected)


def test_vectorized_loss_matches_reference_value_and_gradient():
    torch.manual_seed(4)
    scores = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    positives = torch.rand(5, 7) > .6
    positives[0] = True
    positives[1] = False
    beta = torch.tensor(3., requires_grad=True, dtype=torch.float64)
    loss = DecoupledAllPositiveInfoNCELoss(unknown_negative_weight=.5)
    actual = loss._decoupled_anchor_loss(scores, positives, beta)
    terms = []
    for row, mask in zip(scores, positives):
        if mask.any() and (~mask).any():
            logits = -beta * row
            terms.append(torch.nn.functional.softplus(torch.logsumexp(logits[~mask] + np.log(.5), 0) - logits[mask]).mean())
    expected = torch.stack(terms).mean()
    assert torch.allclose(actual, expected)
    actual_grads = torch.autograd.grad(actual, (scores, beta), retain_graph=True)
    reference_grads = torch.autograd.grad(expected, (scores, beta))
    for a, b in zip(actual_grads, reference_grads):
        assert torch.allclose(a, b)


def test_auxiliary_normalization_and_unknown_gradients():
    stub = SimpleNamespace(biofp_aux_weight=.02, biofp_normalize_active_families=True,
                           biofp_family_weights={"mechanism": 1., "cofactor": 1.}, biofp_confidence_cap=1.)
    logits = torch.tensor([[0., 3.], [1., -2.]], requires_grad=True)
    absent = torch.randn(2, 1, requires_grad=True)
    targets = {"biofp_mechanism_targets": torch.tensor([[1., 0.], [0., 1.]]),
               "biofp_mechanism_mask": torch.tensor([[True, False], [True, False]]),
               "biofp_mechanism_denominator": torch.ones(2, 2),
               "biofp_mechanism_confidence": torch.tensor([[1., 1.], [.5, 1.]])}
    loss, _ = ProteinPooledLitModule._biofp_auxiliary_loss(stub,
        pooling_details={"biofp_logits_mechanism": logits, "biofp_logits_cofactor": absent}, biofp_targets=targets)
    expected = (torch.nn.functional.softplus(-logits[0, 0]) + .5 * torch.nn.functional.softplus(logits[1, 0])) / 1.5
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert torch.equal(absent.grad, torch.zeros_like(absent))


def test_gradient_diagnostic_does_not_populate_parameter_grad():
    parameter = torch.tensor([1., 2.], requires_grad=True)
    shared = parameter * 2
    retrieval = shared.square().mean()
    auxiliary = shared.sum()
    stub = SimpleNamespace(biofp_family_weights={"cofactor": 1.})
    result = ProteinPooledLitModule._biofp_gradient_diagnostics(
        stub, retrieval, auxiliary, {"biofp_shared_cofactor": shared}, .02)
    assert parameter.grad is None
    assert result["biofp_cofactor_gradient_ratio"] > 0
    (retrieval + .02 * auxiliary).backward()
    assert torch.isfinite(parameter.grad).all()


def test_hidden_recovery_filters_train_positives_both_directions(tmp_path):
    path = tmp_path / "known.csv"
    write_pairs(path, [("r0", "p0"), ("r1", "p1")])
    constraints = RankingConstraints(path)
    scores = torch.tensor([10., 3., 2.])
    remaining, positives = constraints.restrict(scores, torch.tensor([1]), "r0_f", ["p0", "p1", "p2"], "reaction_to_enzyme")
    assert remaining.tolist() == [3., 2.]
    assert positives.tolist() == [0]
    remaining, positives = constraints.restrict(scores, torch.tensor([2]), "p1", ["r0_f", "r1_f", "r2_f"], "enzyme_to_reaction")
    assert remaining.tolist() == [10., 2.]
    assert positives.tolist() == [1]


def test_family_pool_requires_full_coverage_and_keeps_positives(tmp_path):
    path = tmp_path / "families.csv"
    path.write_text("protein_id,family_id\np0,A\np1,B\np2,A\n")
    constraints = RankingConstraints(family_path=path)
    remaining, positive = constraints.restrict(torch.tensor([1., 3., 2.]), torch.tensor([2]), "r_f", ["p0", "p1", "p2"], "reaction_to_enzyme")
    assert remaining.tolist() == [1., 2.]
    assert positive.tolist() == [1]
    with pytest.raises(ValueError, match="missing 1"):
        constraints.restrict(torch.ones(2), torch.tensor([0]), "r_f", ["p0", "unknown"], "reaction_to_enzyme")


def _distributed_aux_worker(rank, directory):
    from pathlib import Path
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + directory + "/rendezvous", rank=rank, world_size=2)
    try:
        weight = torch.tensor([[.4]], requires_grad=True)
        shared = (rank + 1) * weight
        stub = SimpleNamespace(biofp_aux_weight=.02, biofp_normalize_active_families=True,
                               biofp_family_weights={"cofactor": 1.}, biofp_confidence_cap=1.)
        loss, _ = ProteinPooledLitModule._biofp_auxiliary_loss(stub,
            pooling_details={"biofp_logits_cofactor": shared}, biofp_targets={
                "biofp_cofactor_targets": torch.ones(1, 1),
                "biofp_cofactor_mask": torch.tensor([[rank == 1]]),
                "biofp_cofactor_denominator": torch.ones(1, 1)})
        loss.backward()
        # DDP's gradient mean must equal a single global evidence-weighted loss.
        dist.all_reduce(weight.grad)
        weight.grad /= 2
        Path(directory, f"gradient_{rank}.json").write_text(json.dumps(weight.grad.item()))
    finally:
        dist.destroy_process_group()


def test_sparse_labels_across_two_cpu_ranks(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_aux_worker, args=(str(tmp_path),), nprocs=2, join=True)
    weight = torch.tensor(.4, requires_grad=True)
    torch.nn.functional.softplus(-2 * weight).backward()
    for rank in range(2):
        actual = json.loads((tmp_path / f"gradient_{rank}.json").read_text())
        assert actual == pytest.approx(weight.grad.item(), rel=1e-6)


def test_f3_training_step_with_v3_loss_and_auxiliary_backward():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 6], target_encoder_dims=[6, 6], embedding_dim=6,
        residue_dim=4, pooling="sleec_guided_attention", sleec_scorer_hidden_dim=8,
        sleec_freeze_scorer=True, enzyme_input_mode="raw_mean_sleec_biological_factorized",
        hyperbolic_hyp_dim=2,
        enzyme_block_dims={"core": 2, "site": 1, "mechanism": 1, "cofactor": 1, "ec": 1},
        enzyme_block_weights={"core": .4, "site": .2, "mechanism": .15, "cofactor": .15, "ec": .1},
        biofp_family_dims={"mechanism": 2, "cofactor": 1}, biofp_hidden_dim=8,
        biofp_family_weights={"mechanism": 1., "cofactor": 1.}, biofp_aux_weight=.02,
        biofp_aux_warmup_epochs=3, biofp_normalize_active_families=True,
        loss_name="DecoupledAllPositiveInfoNCELoss", beta=10., unknown_negative_weight=.5,
        positive_pair_source="all_known_in_batch", validation_retrieval_metrics=False)
    module.trainer = SimpleNamespace(current_epoch=3, global_step=100,
        datamodule=SimpleNamespace(_train_query_to_targets={"q0": ["p0", "p1"], "q1": ["p1"], "q2": ["p2"]}))
    batch = {"query_vec": torch.randn(3, 4), "query_id": ["q0", "q1", "q2"],
             "target_id": ["p0", "p1", "p2"], "residue_embeddings": torch.randn(3, 5, 4),
             "residue_padding_mask": torch.zeros(3, 5, dtype=torch.bool),
             "biofp_mechanism_targets": torch.tensor([[1., 0.], [0., 1.], [0., 0.]]),
             "biofp_mechanism_mask": torch.tensor([[True, False], [True, False], [False, False]]),
             "biofp_mechanism_denominator": torch.ones(3, 2)}
    loss, _, _, components = module._compute_full_batch_loss(batch)
    assert torch.isfinite(loss)
    assert components["biofp_effective_weight"].item() == pytest.approx(.02)
    assert "biofp_mechanism_gradient_ratio" in components
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("occupied_uuid,blocked", [("GPU-other", False), ("GPU-chosen", True)])
def test_launcher_checks_only_selected_gpu_uuids(monkeypatch, occupied_uuid, blocked):
    from scripts import run_circe_v3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    def query(command, **kwargs):
        if "--query-gpu=uuid" in command:
            return SimpleNamespace(stdout="GPU-chosen\nGPU-second\n")
        return SimpleNamespace(stdout=f"{occupied_uuid},123\n")
    monkeypatch.setattr(run_circe_v3.subprocess, "run", query)
    if blocked:
        with pytest.raises(RuntimeError, match="compute jobs"):
            run_circe_v3.check_gpus(2)
    else:
        run_circe_v3.check_gpus(2)
