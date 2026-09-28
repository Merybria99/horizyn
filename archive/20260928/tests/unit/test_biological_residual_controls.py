from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
import csv
import json

import h5py
import numpy as np
import pytest
import torch
import yaml

from horizyn.biological_residual import (
    SLEECFunctionalTokenEncoder,
    select_functional_token_indices,
)
from scripts.cache_sleec_functional_tokens import (
    _cache_metadata,
    random_selection_scores,
    write_shard,
)
from scripts.evaluate_biological_residual import (
    evaluate_score_matrix,
    fixed_test_alpha,
    rank_metrics,
    select_alpha,
)
from scripts.evaluate_protein_pooling import append_retrieval_metrics
from scripts.report_biological_residual_controls import bootstrap_difference, report
from scripts.run_biological_residual_controls import (
    ROOT,
    VARIANTS,
    build_configs,
    cache_random_tokens,
    check_gpus,
    prepare,
)
import scripts.run_biological_residual_controls as campaign


def test_configs_isolate_pretraining_and_residual_guidance(tmp_path):
    configs = build_configs(tmp_path)
    original = yaml.safe_load(
        (ROOT / "configs/reactzyme_reaction_smi_biological_residual.yaml").read_text()
    )
    r2 = configs["R2_no_source_pretrain_finetune"]
    r3 = configs["R3_unguided_finetune"]
    assert r2["model"] == original["model"]
    assert (
        "F3_set_chemistry/protein-pooling-epoch=29.ckpt" in r2["training"]["init_from_checkpoint"]
    )
    assert "R3_unguided/checkpoints/source/last.ckpt" in r3["training"]["init_from_checkpoint"]
    for config in (r2, r3):
        assert config["training"]["loss"] == original["training"]["loss"]
        assert config["data"]["source_replay"]["fraction"] == 0.15
        assert config["data"]["train_sampler"] == original["data"]["train_sampler"]
        assert config["training"]["max_epochs"] == 3
        assert config["training"]["devices"] == 4
        assert config["seed"] == 42
    assert r3["model"]["sleec_pooling"] == original["model"]["sleec_pooling"]
    for stage in ("source", "finetune"):
        assert configs[f"R3_unguided_{stage}"]["model"]["biological_residual"]["sleec_bias"] == 0
        assert (
            configs[f"R3_unguided_{stage}"]["model"]["biological_residual"]["sleec_pool_scale"] == 0
        )
    assert configs["R3_unguided_source"]["training"]["max_epochs"] == 5


def test_preparation_is_idempotent_and_refuses_changed_recipe(tmp_path):
    prepare(tmp_path)
    prepare(tmp_path)
    path = tmp_path / "configs/R2_no_source_pretrain_finetune.yaml"
    path.write_text("different recipe")
    with pytest.raises(ValueError, match="differs"):
        prepare(tmp_path)


def test_controller_schedules_only_two_controls_and_selects_before_test(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("sys.argv", ["campaign", "run", "--run-root", str(tmp_path)])
    monkeypatch.setattr(campaign, "check_gpus", lambda _: None)
    monkeypatch.setattr(campaign, "preflight", lambda _: None)
    monkeypatch.setattr(campaign, "train", lambda path: events.append(("train", path.name)))
    monkeypatch.setattr(
        campaign, "evaluate", lambda _, variant, subset: events.append((subset, variant))
    )
    monkeypatch.setattr(
        campaign, "cache_random_tokens", lambda _: events.append(("cache", "random"))
    )
    monkeypatch.setattr(
        "scripts.report_biological_residual_controls.report",
        lambda *_: events.append(("report", "all")),
    )
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    campaign.main()
    assert [item for item in events if item[0] == "train"] == [
        ("train", "R2_no_source_pretrain_finetune.yaml"),
        ("train", "R3_unguided_source.yaml"),
        ("train", "R3_unguided_finetune.yaml"),
    ]
    assert max(i for i, item in enumerate(events) if item[0] == "validation") < min(
        i for i, item in enumerate(events) if item[0] == "test"
    )
    assert events[-1] == ("report", "all")


def test_test_command_uses_validation_selection_file(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(campaign, "child", commands.append)
    campaign.evaluate(tmp_path, "R2_no_source_pretrain", "test")
    command = commands[0]
    assert command[command.index("--evaluation-split") + 1] == "test"
    assert command[command.index("--selection-from") + 1].endswith("results/validation.json")
    assert "--alphas" not in command


def progress_checkpoint(completed, *, post_fit=False, last_batch=True, ready=2, done=2):
    started = completed if post_fit else completed + 1
    return {
        "epoch": completed,
        "loops": {
            "fit_loop": {
                "epoch_progress": {
                    "current": dict(
                        ready=started, started=started, processed=completed, completed=completed
                    )
                },
                "epoch_loop.batch_progress": {
                    "current": dict(ready=ready, started=ready, processed=done, completed=done),
                    "is_last_batch": last_batch,
                },
            }
        },
    }


@pytest.mark.parametrize(
    "saved,expected_calls",
    [
        (progress_checkpoint(2), 0),  # Three epochs, saved at validation end.
        (progress_checkpoint(3, post_fit=True), 0),
        (progress_checkpoint(0), 1),
        (progress_checkpoint(2, post_fit=True), 1),
        (progress_checkpoint(2, last_batch=False), 1),
        (progress_checkpoint(2, done=1), 1),  # Final batch has not completed.
        (progress_checkpoint(2, ready=0, done=0), 1),
    ],
)
def test_training_resume_never_silently_restarts(tmp_path, monkeypatch, saved, expected_calls):
    prepare(tmp_path)
    config_path = tmp_path / "configs/R2_no_source_pretrain_finetune.yaml"
    directory = tmp_path / "R2_no_source_pretrain/checkpoints/finetune"
    directory.mkdir(parents=True)
    checkpoint = directory / "last.ckpt"
    torch.save(saved, checkpoint)
    commands = []

    def complete(command):
        commands.append(command)
        torch.save(progress_checkpoint(3, post_fit=True), checkpoint)

    monkeypatch.setattr(campaign, "child", complete)
    campaign.train(config_path)
    assert len(commands) == expected_calls
    if commands:
        assert commands[0][commands[0].index("--resume") + 1] == str(checkpoint)


@pytest.mark.parametrize(
    "saved,message",
    [
        ({"epoch": 2}, "lacks"),
        (progress_checkpoint(-1), "inconsistent"),
        (progress_checkpoint(2, done=3), "inconsistent"),
        (progress_checkpoint(2, last_batch="true"), "inconsistent"),
    ],
)
def test_completion_check_refuses_unreliable_progress(saved, message):
    with pytest.raises(ValueError, match=message):
        campaign.completed_training_epochs(saved)


def test_training_refuses_over_budget_checkpoint(tmp_path, monkeypatch):
    prepare(tmp_path)
    config_path = tmp_path / "configs/R2_no_source_pretrain_finetune.yaml"
    checkpoint = tmp_path / "R2_no_source_pretrain/checkpoints/finetune/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(progress_checkpoint(4, post_fit=True), checkpoint)
    monkeypatch.setattr(campaign, "child", lambda _: pytest.fail("Training must not launch"))
    with pytest.raises(ValueError, match="exceeds"):
        campaign.train(config_path)


def test_training_rejects_incomplete_successful_child(tmp_path, monkeypatch):
    prepare(tmp_path)
    config_path = tmp_path / "configs/R2_no_source_pretrain_finetune.yaml"
    checkpoint = tmp_path / "R2_no_source_pretrain/checkpoints/finetune/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    monkeypatch.setattr(
        campaign,
        "child",
        lambda _: torch.save(progress_checkpoint(2, last_batch=False), checkpoint),
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        campaign.train(config_path)


@pytest.mark.parametrize("validation,post_fit", [(False, False), (False, True), (True, False)])
def test_training_accepts_real_lightning_checkpoint_formats(
    tmp_path, monkeypatch, validation, post_fit
):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, TensorDataset

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return (self.weight * batch[0]).square().mean()

        def validation_step(self, batch, batch_idx):
            self.log("val/loss", (self.weight * batch[0]).square().mean())

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    checkpoint = tmp_path / "checkpoints/last.ckpt"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "logging": {"checkpoint_dir": str(checkpoint.parent)},
                "training": {"max_epochs": 2},
            }
        )
    )

    def train_on_cpu(command):
        assert "--resume" not in command
        options = dict(save_top_k=0, every_n_epochs=1)
        if validation:
            options.update(save_top_k=1, monitor="val/loss", save_on_train_epoch_end=False)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint.parent,
                    save_last=True,
                    **options,
                )
            ],
        )
        loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)
        trainer.fit(Model(), loader, loader if validation else None)
        if post_fit:
            trainer.save_checkpoint(checkpoint)

    monkeypatch.setattr(campaign, "child", train_on_cpu)
    campaign.train(config_path)  # Fresh training must pass the post-run check.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["epoch"] == (2 if post_fit else 1)
    assert campaign.completed_training_epochs(saved) == 2
    monkeypatch.setattr(campaign, "child", lambda _: pytest.fail("Must reuse completed stage"))
    campaign.train(config_path)  # A subsequent launch must skip it.


def test_random_cache_commands_use_full_residues_not_guided_tokens(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr("scripts.run_biological_residual_controls.child", commands.append)
    cache_random_tokens(build_configs(tmp_path))
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--selection") + 1] == "random"
        assert "--sleec-checkpoint" not in command
        assert command[command.index("--top-k") + 1] == "48"
        assert command[command.index("--context-k") + 1] == "16"
    assert commands[1].count("--pairs") == 3


def test_random_selection_is_id_stable_and_padding_independent():
    valid = torch.tensor([[True] * 80 + [False] * 8, [True] * 88])
    a = random_selection_scores(["p1", "p2"], valid, 42)
    b = random_selection_scores(["p2", "p1"], valid.flip(0), 42).flip(0)
    assert torch.equal(a, b)
    alone = random_selection_scores(["p1"], valid[:1, :80], 42)
    assert torch.equal(a[:1, :80], alone)
    indices, mask = select_functional_token_indices(a, valid, top_k=48, context_k=16)
    assert mask.sum(dim=1).tolist() == [64, 64]
    assert all(len(set(row.tolist())) == 64 for row in indices)
    assert not torch.equal(a, random_selection_scores(["p1", "p2"], valid, 7))


def test_random_cache_does_not_require_or_call_sleec(tmp_path):
    class Dataset:
        vec_dim = 4

        def __getitem__(self, key):
            return {"residue_embeddings": torch.arange(320).view(80, 4).float()}

    args = Namespace(
        top_k=48, context_k=16, batch_size=2, bf16=False, selection="random", selection_seed=42
    )
    output = tmp_path / "random.h5"
    write_shard(output, ["p1"], Dataset(), None, args, torch.device("cpu"), 0)
    with h5py.File(output) as handle:
        assert handle["token_mask"][:].sum() == 64
        assert not handle["sleec_scores"][:].any()
        assert handle["token_vectors"].shape == (1, 64, 4)


def test_random_cache_end_to_end_and_idempotent_reuse(tmp_path, monkeypatch):
    import scripts.cache_sleec_functional_tokens as cache

    residues = tmp_path / "residues.h5"
    with h5py.File(residues, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(["p1", "p2"], dtype="S2"))
        handle.create_dataset("vectors", data=np.arange(400, dtype=np.float32).reshape(100, 4))
        handle.create_dataset("offsets", data=np.asarray([0, 80, 100], dtype=np.int64))
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("reaction_id,protein_id\nr1,p1\nr2,p2\n")
    args = Namespace(
        residue_h5=residues,
        sleec_checkpoint=None,
        pairs=[pairs],
        selection="random",
        selection_seed=42,
        top_k=48,
        context_k=16,
        max_tokens=1022,
        scorer_hidden_dim=256,
        bf16=False,
        force=False,
        batch_size=2,
        output=tmp_path / "tokens.h5",
    )
    monkeypatch.setattr(cache, "parse_args", lambda: args)
    monkeypatch.setattr(cache, "distributed_context", lambda: (0, 1, torch.device("cpu")))
    cache.main()
    before = args.output.stat().st_mtime_ns
    cache.main()
    assert args.output.stat().st_mtime_ns == before
    with h5py.File(args.output) as handle:
        assert handle["token_mask"][:].sum(axis=1).tolist() == [64, 20]
        assert not handle["sleec_scores"][:].any()
        assert json.loads(handle.attrs["metadata_json"])["selection"] == "random"
    args.selection_seed = 7
    with pytest.raises(FileExistsError, match="incompatible"):
        cache.main()


def test_unguided_pooling_is_invariant_to_sleec_values():
    encoder = SLEECFunctionalTokenEncoder(
        input_dim=4, token_dim=8, heads=2, layers=1, dropout=0, sleec_pool_scale=0
    ).eval()
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    a = encoder(x, mask, torch.rand(2, 5))
    b = encoder(x, mask, torch.rand(2, 5))
    assert all(torch.equal(i, j) for i, j in zip(a, b))


def test_cache_metadata_distinguishes_guidance_and_seed(tmp_path):
    residues = tmp_path / "residues.h5"
    residues.write_bytes(b"fixture")
    args = Namespace(
        residue_h5=residues,
        sleec_checkpoint=None,
        pairs=[],
        selection="random",
        selection_seed=42,
        top_k=48,
        context_k=16,
        max_tokens=1022,
        scorer_hidden_dim=256,
        bf16=True,
    )
    metadata = _cache_metadata(args, requested_count=1, resolved_count=1, missing_count=0)
    assert metadata["selection"] == "random" and metadata["sleec_checkpoint"] is None
    args.selection_seed = 7
    assert metadata != _cache_metadata(args, requested_count=1, resolved_count=1, missing_count=0)


def test_multi_positive_metrics_match_legacy_when_scores_do_not_tie():
    scores = torch.tensor([0.9, 0.6, 0.8, 0.3, 0.1])
    old = defaultdict(list)
    append_retrieval_metrics(old, scores, torch.tensor([1, 4]))
    new = rank_metrics(scores, [1, 4])
    for key, values in old.items():
        assert new[key] == pytest.approx(values[0], abs=1e-7)
    assert new["mean_rank"] == 4
    assert new["precision_at_2"] == 0
    assert new["precision_at_5"] == 0.4
    assert new["precision_at_10"] == 0.2


def test_metric_ties_use_candidate_order_and_export_both_directions():
    assert rank_metrics(torch.ones(3), [1, 2])["first_positive_rank"] == 2
    rows = []
    metrics = evaluate_score_matrix(
        torch.tensor([[0.8, 0.7], [0.3, 0.9]]),
        ["r1", "r2"],
        ["e1", "e2"],
        {"r1": ["e1", "e2"], "r2": ["e2"]},
        per_query=rows,
    )
    assert len(rows) == 4
    assert metrics["reaction_to_enzyme/num_queries"] == 2
    assert "enzyme_to_reaction/precision_at_20" in metrics


def test_alpha_selection_uses_requested_metric_and_both_direction_guards():
    def values(e, r, first):
        return {
            "enzyme_to_reaction/reactzyme_mrr": e,
            "reaction_to_enzyme/reactzyme_mrr": r,
            "balanced_reactzyme_mrr": (e + r) / 2,
            "balanced_first_positive_mrr": first,
            "enzyme_to_reaction/first_positive_mrr": first,
            "reaction_to_enzyme/first_positive_mrr": first,
        }

    results = {
        "0": values(0.5, 0.5, 0.7),
        "0.025": values(0.51, 0.50, 0.71),
        "0.05": values(0.6, 0.49, 0.75),
    }
    assert select_alpha(results, "reactzyme_mrr", 0.005)[0] == 0.025
    assert select_alpha(results, "first_positive_mrr", 0.005)[0] == 0.05
    assert (
        select_alpha(
            {"0.1": values(0.5, 0.5, 0.7), "0": values(0.5, 0.5, 0.7)}, "reactzyme_mrr", 0.005
        )[0]
        == 0
    )


def test_test_alpha_is_fixed_by_validation_not_test_performance(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    path = tmp_path / "validation.json"
    selection = {
        "evaluation_split": "validation",
        "selection_metric": "balanced_reactzyme_mrr",
        "checkpoint": str(checkpoint),
        "best_alpha": 0.075,
        "alpha_results": {"0": {}, "0.075": {}},
    }
    path.write_text(json.dumps(selection))
    assert fixed_test_alpha(path, checkpoint, "reactzyme_mrr") == 0.075
    with pytest.raises(ValueError, match="checkpoints disagree"):
        fixed_test_alpha(path, tmp_path / "different.ckpt", "reactzyme_mrr")
    selection["evaluation_split"] = "test"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="validation evaluation"):
        fixed_test_alpha(path, checkpoint, "reactzyme_mrr")


def test_gpu_check_uses_uuid_not_unfiltered_process_list(monkeypatch):
    inventory = "\n".join(f"{i}, GPU-{i}, 140000" for i in range(5))

    def response(command, **kwargs):
        return inventory if "--query-gpu=index,uuid,memory.free" in command else "GPU-4, 12345\n"

    monkeypatch.setattr("subprocess.check_output", response)
    check_gpus("0,1,2,3")
    with pytest.raises(RuntimeError, match="occupied"):
        check_gpus("1,2,3,4")
    with pytest.raises(ValueError, match="four distinct"):
        check_gpus("0,1")


def test_paired_report_and_missing_metadata_are_explicit(tmp_path):
    for variant in VARIANTS:
        directory = tmp_path / variant / "results"
        directory.mkdir(parents=True)
        selection = {"checkpoint": variant, "best_alpha": 0.1}
        (directory / "validation.json").write_text(json.dumps(selection))
        (directory / "test.json").write_text(
            json.dumps(
                {
                    **selection,
                    "evaluation_split": "test",
                    "selection_metric": "balanced_reactzyme_mrr",
                }
            )
        )
        rows = []
        for alpha in (0, 0.1):
            for direction in ("enzyme_to_reaction", "reaction_to_enzyme"):
                for i in range(3):
                    rows.append(
                        {
                            "alpha": alpha,
                            "query_id": f"q{i}",
                            "direction": direction,
                            "known_positive_count": i + 1,
                            "train_known_association_count": i,
                            "has_unimol2": True,
                            "has_chiro": False,
                            **{
                                key: 0.5 + alpha
                                for key in (
                                    "reactzyme_mrr",
                                    "first_positive_mrr",
                                    "top_1",
                                    "precision_at_10",
                                    "mean_rank",
                                )
                            },
                        }
                    )
        with (directory / "test.per_query.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = report(tmp_path)
    assert len(payload["table"]) == 8
    assert payload["missing_analysis"] == ["training_similarity_bins", "annotation_coverage_bins"]
    assert payload["paired_query_bootstrap"]["R1_full_minus_R0_F3"]["enzyme_to_reaction"][
        "delta"
    ] == pytest.approx(0.1)
    assert (tmp_path / "reports/stratified_metrics.csv").is_file()
    assert bootstrap_difference(np.zeros(3))["ci95"] == [0, 0]
