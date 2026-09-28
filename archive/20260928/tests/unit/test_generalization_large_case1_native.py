from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.generalization_large_case1_native import SharedNativeEncoders, validate_block


def test_native_block_guards_are_strict():
    raw = np.ones((3, 1024), np.float16)
    assert validate_block(raw, [0, 1, 3], ["a", "b"]).tolist() == [0, 1, 3]
    for offsets in ([1, 2, 3], [0, 3, 3], [0, 3, 2], [0., 1., 3.]):
        with pytest.raises(ValueError):
            validate_block(raw, offsets, ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        validate_block(raw, [0, 1, 3], ["a", "a"])
    with pytest.raises(ValueError, match="FP16"):
        validate_block(raw.astype(np.float32), [0, 1, 3], ["a", "b"])
    with pytest.raises(ValueError, match="batch_size"):
        validate_block(raw, [0, 1, 3], ["a", "b"], 16)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_block(raw, [0, 1, 3], ["a", "b"])


def test_native_encoders_share_identical_tensor_and_preserve_order():
    calls = []
    class Target:
        def encode_targets(self, residues, **kwargs):
            mask = kwargs["residue_padding_mask"]
            calls.append((residues.data_ptr(), mask.data_ptr(), tuple(residues.shape), torch.is_inference_mode_enabled()))
            assert residues.dtype == torch.float32
            assert kwargs["retrieval_direction"] == "reaction_to_enzyme"
            assert kwargs["score_residue_embeddings"] is None
            return residues[:, 0, :512].clone()
    encoder = SharedNativeEncoders.__new__(SharedNativeEncoders)
    encoder.device = torch.device("cpu")
    encoder.models = {name: SimpleNamespace(model=Target()) for name in ["f3", "circev2"]}
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    lengths = [1024] + [1] * 32
    offsets = np.r_[0, np.cumsum(lengths)]
    raw = np.concatenate([np.full((length, 1024), i + 1, np.float16) for i, length in enumerate(lengths)])
    before = raw.copy()
    out = encoder.encode_block(raw, offsets, [f"p{i}" for i in range(33)])
    assert np.array_equal(out["f3"], out["circev2"])
    assert np.array_equal(out["f3"][:, 0], np.arange(1, 34))
    assert np.array_equal(raw, before)
    assert calls[0] == calls[1] and calls[2] == calls[3]
    assert calls[0][2] == (32, 1022, 1024) and calls[2][2] == (1, 1, 1024)
    assert all(row[3] for row in calls)
    assert out["timing"]["proteins"] == 33


def test_model_output_and_device_guards():
    class BadTarget:
        def encode_targets(self, residues, **kwargs):
            return torch.zeros(len(residues), 512)
    encoder = SharedNativeEncoders.__new__(SharedNativeEncoders)
    encoder.device = torch.device("cpu")
    encoder.models = {"f3": SimpleNamespace(model=BadTarget())}
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = np.ones((1, 1024), np.float16)
    with pytest.raises(ValueError, match="device"):
        encoder.encode_block(raw, [0, 1], ["a"], device="cuda:0")
    with pytest.raises(ValueError, match="Invalid native"):
        encoder.encode_block(raw, [0, 1], ["a"])
