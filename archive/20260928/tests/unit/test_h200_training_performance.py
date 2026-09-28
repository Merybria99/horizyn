"""CPU AMP contract tests; CUDA execution is explicitly opt-in, never automatic."""
import importlib.util
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from horizyn.config import load_config, validate_config
from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector
from horizyn.protein_pooling_lightning_module import (
    ProteinPooledLitModule, _install_fp32_sensitive_hooks,
)
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule

ROOT = Path(__file__).parents[2]


def _module(**kwargs):
    return ProteinPooledLitModule(
        query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4,
        residue_dim=4, pooling="mean", loss_name="BidirectionalSampledMultiPositiveInfoNCELoss",
        sampled_require_both_directions=True, **kwargs,
    )


@pytest.mark.parametrize("similarity", ["cosine", "dot"])
def test_cpu_autocast_score_and_loss_are_really_fp32(similarity):
    module = _module(contrastive_fp32=True, embedding_similarity=similarity)
    q = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    p = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    observed = []
    def check_amp(_module, args):
        observed.append((args[0].dtype, torch.is_autocast_enabled("cpu")))
    module.loss_fn.register_forward_pre_hook(check_amp)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores, details = module._compute_embedding_distances_with_details(q, p)
        reverse = module._compute_enzyme_to_reaction_distances(p, q)
        retrieval = module._compute_retrieval_scores(q, p)
        assert scores.dtype == reverse.dtype == retrieval.dtype == torch.float32
        loss, components = module._loss_with_components(
            scores, torch.arange(3), torch.arange(3), q, p,
            biological_negative_mask=torch.tensor([[False, True, False], [False, False, False], [False, False, False]]),
            random_negative_mask=torch.zeros(3, 3, dtype=torch.bool), prototype_details=details,
        )
        assert torch.is_autocast_enabled("cpu")
    assert loss.dtype == torch.float32
    assert observed == [(torch.float32, False)]
    loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(p.grad).all()
    assert components["r2e_valid_anchors"] == components["e2r_valid_anchors"] == 1
    expected = module._compute_embedding_distances(q.float(), p.float())
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)


def test_fp32_islands_are_opt_in_and_keep_checkpoint_keys():
    legacy = _module()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert legacy._compute_dot_distances(torch.randn(2, 4), torch.randn(2, 4)).dtype == torch.bfloat16
    projector = LorentzEnzymeProjector(input_dim=4, hyp_dim=3)
    keys = set(projector.state_dict())
    _install_fp32_sensitive_hooks(projector)
    assert set(projector.state_dict()) == keys
    source = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hyp, tangent = projector(source)
        assert hyp.dtype == tangent.dtype == torch.float32
        assert torch.isfinite(hyp).all() and torch.isfinite(tangent).all()
        assert torch.is_autocast_enabled("cpu")
        with pytest.raises(ValueError):
            projector(torch.randn(2, 5))
        assert torch.is_autocast_enabled("cpu")  # exception path restores outer AMP
    tangent.sum().backward()
    assert torch.isfinite(source.grad).all()
    assert not projector._circe_fp32_contexts
    assert set(pickle.loads(pickle.dumps(projector)).state_dict()) == keys


def test_sensitive_sleec_pool_retains_fp32_under_cpu_amp():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4,
        residue_dim=4, pooling="sleec_guided_attention", sleec_scorer_hidden_dim=4,
        fp32_sensitive_modules=True, contrastive_fp32=True,
    )
    values = torch.randn(2, 8, 4, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pooled, details = module.model.pooling(values, return_details=True)
    assert pooled.dtype == details["logits"].dtype == torch.float32
    assert torch.isfinite(pooled).all() and torch.isfinite(details["weights"]).all()
    pooled.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_fused_adamw_opt_in_falls_back_on_cpu_without_changing_lr():
    module = _module(fused_adamw=True, learning_rate=.0001, weight_decay=.01)
    optimizer = module.configure_optimizers()
    assert not module._using_fused_adamw
    assert optimizer.defaults.get("fused") is False
    assert optimizer.param_groups[0]["lr"] == .0001
    assert optimizer.param_groups[0]["weight_decay"] == .01
    sum(parameter.sum() for parameter in module.parameters() if parameter.requires_grad).backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
    assert _module().configure_optimizers().defaults.get("fused") is None


def test_unsupported_fused_optimizer_constructor_falls_back_without_cuda(monkeypatch):
    module = _module(fused_adamw=True)
    # Simulate CUDA eligibility and an older/unsupported fused optimizer API;
    # do not allocate a CUDA tensor or execute a GPU kernel in this CPU suite.
    parameter = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(module, "parameters", lambda: iter([parameter]))
    calls = []
    def optimizer(parameters, *, lr, weight_decay, fused=None):
        calls.append(fused)
        if fused:
            raise RuntimeError("fused unsupported")
        return SimpleNamespace(parameters=parameters, lr=lr)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    with pytest.warns(RuntimeWarning, match="ordinary AdamW"):
        result = module.configure_optimizers()
    assert calls == [True, None]
    assert result.parameters == [parameter] and result.lr == module.learning_rate
    assert not module._using_fused_adamw


@pytest.mark.parametrize("workers", [0, 4])
def test_worker_options_are_bounded_and_zero_worker_safe(workers):
    data = SimpleNamespace(num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, worker_num_threads=1)
    training = ReactionConditionedDataModule._loader_worker_kwargs(data, training=True)
    validation = ReactionConditionedDataModule._loader_worker_kwargs(data)
    assert training["num_workers"] == workers
    if workers:
        assert training["persistent_workers"] and not validation["persistent_workers"]
        assert training["prefetch_factor"] == 2
        assert pickle.loads(pickle.dumps(training["worker_init_fn"])).keywords == {"num_threads": 1}
    else:
        assert "persistent_workers" not in training
        assert "prefetch_factor" not in training
        assert "worker_init_fn" not in training


def test_h200_profile_preserves_batch_lr_sampler_and_legacy_defaults():
    base = load_config(str(ROOT / "configs/horizyn1_circe_v2.yaml"))
    h200 = load_config(str(ROOT / "configs/horizyn1_circe_v2_h200.yaml"))
    assert base.training.precision == "32-true"
    assert h200.training.precision == "bf16-mixed"
    assert h200.training.contrastive_fp32 and h200.training.fp32_sensitive_modules
    assert h200.training.fused_adamw and not h200.training.log_attention_stats
    assert h200.data.train_batch_size * h200.training.devices * h200.training.accumulate_grad_batches == 400
    assert h200.data.typed_negative_positive_fraction == base.data.typed_negative_positive_fraction == .85
    assert h200.training.learning_rate == base.training.learning_rate
    assert h200.training.loss == base.training.loss
    assert h200.training.strategy == "ddp_find_unused_parameters_true"
    assert h200.data.num_workers == 4 and h200.data.worker_num_threads == 1


@pytest.mark.parametrize("section,key,value", [
    ("training", "contrastive_fp32", "yes"), ("training", "fused_adamw", 1),
    ("training", "fp32_sensitive_modules", "true"), ("training", "ddp_gradient_as_bucket_view", "true"),
    ("training", "cpu_num_threads", 0), ("data", "persistent_workers", "true"),
    ("data", "prefetch_factor", 0), ("data", "worker_num_threads", -1),
    ("data", "num_workers", True), ("training", "precision", "bf16-true"),
])
def test_performance_config_rejects_invalid_values(section, key, value):
    config = load_config(str(ROOT / "configs/horizyn1_circe_v2_h200.yaml"))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_ddp_bucket_views_preserve_find_unused_policy():
    path = ROOT / "scripts/train_protein_pooling.py"
    spec = importlib.util.spec_from_file_location("h200_train_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    strategy = script._performance_ddp_strategy("ddp_find_unused_parameters_true", gradient_as_bucket_view=True)
    assert strategy._ddp_kwargs["find_unused_parameters"] is True
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] is True
    assert script._performance_ddp_strategy("ddp_find_unused_parameters_true") == "ddp_find_unused_parameters_true"
    assert script._performance_ddp_strategy("auto", gradient_as_bucket_view=True) == "auto"


def test_real_indexed_multimodal_two_steps_with_cpu_bf16(tmp_path, monkeypatch):
    # Reuse the existing actual HDF5/NPZ multimodal integration fixture; only
    # substitute this opt-in numeric/optimizer policy and Lightning precision.
    path = ROOT / "tests/unit/test_horizyn1_multimodal_training_smoke.py"
    spec = importlib.util.spec_from_file_location("h200_amp_smoke_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    original_module, original_trainer = fixture.ProteinPooledLitModule, fixture.pl.Trainer
    def module(**kwargs):
        return original_module(**kwargs, contrastive_fp32=True, fp32_sensitive_modules=True, fused_adamw=True)
    def trainer(**kwargs):
        return original_trainer(**kwargs, precision="bf16-mixed")
    monkeypatch.setattr(fixture, "ProteinPooledLitModule", module)
    monkeypatch.setattr(fixture.pl, "Trainer", trainer)
    fixture.test_indexed_multimodal_factorized_training_two_steps(tmp_path, materialized_directions=True)


@pytest.mark.skipif(os.environ.get("HORIZYN_RUN_CUDA_TESTS") != "1", reason="CUDA tests require explicit HORIZYN_RUN_CUDA_TESTS=1")
def test_optional_cuda_bf16_scores_and_fused_adamw():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    module = _module(contrastive_fp32=True, fused_adamw=True).cuda()
    optimizer = module.configure_optimizers()
    assert module._using_fused_adamw
    q = torch.randn(2, 4, device="cuda", requires_grad=True)
    p = torch.randn(2, 4, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dists = module._compute_embedding_distances(q, p)
        loss, _ = module._loss_with_components(dists, torch.arange(2, device="cuda"), torch.arange(2, device="cuda"), q, p, biological_negative_mask=torch.tensor([[False, True], [False, False]], device="cuda"), random_negative_mask=torch.zeros(2, 2, dtype=torch.bool, device="cuda"))
    assert dists.dtype == loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(p.grad).all()
    sum(parameter.sum() for parameter in module.parameters() if parameter.requires_grad).backward()
    optimizer.step()
