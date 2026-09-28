import h5py
import numpy as np
import pytest

from scripts.align_enzymecage_f3_features import align


def test_forward_alignment_preserves_vectors_and_source_and_rejects_stale_cache(tmp_path):
    source, target = tmp_path / "raw.h5", tmp_path / "forward.h5"
    values = np.arange(12, dtype=np.float16).reshape(2, 6)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("ids", data=["r1", "r2"], dtype=h5py.string_dtype())
        handle.create_dataset("vectors", data=values)
        handle.attrs["embedding_dim"] = 6
    before = source.read_bytes()
    assert align(source, target, {"r1", "r2"}) == 2
    assert align(source, target, {"r1", "r2"}) == 2
    assert source.read_bytes() == before
    with h5py.File(target, "r") as handle:
        assert handle["ids"].asstr()[:].tolist() == ["r1_f", "r2_f"]
        np.testing.assert_array_equal(handle["vectors"][:], values)
        assert handle.attrs["embedding_dim"] == 6
    with h5py.File(source, "r+") as handle:
        handle["vectors"][0, 0] = 42
    with pytest.raises(ValueError, match="Stale aligned cache"):
        align(source, target, {"r1", "r2"})


def test_alignment_rejects_unexpected_reactions(tmp_path):
    source = tmp_path / "raw.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("ids", data=["r1_r"], dtype=h5py.string_dtype())
    with pytest.raises(ValueError, match="Unexpected"):
        align(source, tmp_path / "forward.h5", {"r1"})
