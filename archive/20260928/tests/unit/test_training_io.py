import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from horizyn.datasets.indexed_pairs import (
    ARRAY_NAMES, IndexedPairs, IndexedTypedNegativeBatchSampler, PAIR_TYPES,
    SEMANTICS, write_indexed_pairs,
)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.training_io import (
    FastIODataModule, ResidueTransportDataModule, ResidentIndexedPairs, StoragePrecisionResidues,
    restore_residue_precision, training_io_adapters,
)
from horizyn.utils.collate import residue_collate_fn

ROOT = Path(__file__).resolve().parents[2]


def test_residue_only_adapter_preserves_original_sampler_and_validation_module():
    import horizyn.reaction_conditioned_data_module as data
    original_index, original_reader = data.IndexedPairs, data.ResidueEmbedDataset
    marker = object()
    entrypoint = SimpleNamespace(ReactionConditionedDataModule=data.ReactionConditionedDataModule,
                                ProteinPooledLitModule=marker)
    original_data = entrypoint.ReactionConditionedDataModule
    with training_io_adapters(entrypoint, residue_only=True):
        assert data.IndexedPairs is original_index
        assert data.ResidueEmbedDataset is StoragePrecisionResidues
        assert entrypoint.ProteinPooledLitModule is marker
        assert entrypoint.ReactionConditionedDataModule is ResidueTransportDataModule
    assert data.ResidueEmbedDataset is original_reader
    assert entrypoint.ReactionConditionedDataModule is original_data


@pytest.fixture
def index_dir(tmp_path):
    path = tmp_path / "index"
    write_indexed_pairs(
        path, query_ids=np.array([f"q{i}" for i in range(4)]),
        protein_ids=np.array([f"p{i}" for i in range(4)]),
        pairs=np.column_stack((np.arange(4), np.arange(4))),
        protein_ec=np.arange(4), ec_prefix=np.array([0, 0, 1, 1]),
        query_ec_indptr=np.arange(5), query_ec=np.arange(4),
        mechanism_bits=np.ones(4, dtype=int), native_cofactor_bits=np.ones(4, dtype=int),
        reaction_cofactor_bits=np.zeros(4, dtype=int),
        ec_eligible=np.ones(4, dtype=bool), biological_eligible=np.ones(4, dtype=bool),
        query_signature=np.arange(4),
        provenance={"pair_scope": "train", "annotation_semantics": SEMANTICS},
    )
    return path


def test_resident_arrays_immutable_and_worker_pickle_compact(index_dir):
    original, resident = IndexedPairs(index_dir), ResidentIndexedPairs(index_dir)
    for name in ARRAY_NAMES:
        a, b = getattr(original, name), getattr(resident, name)
        np.testing.assert_array_equal(a, b, strict=True)
        assert not isinstance(b, np.memmap) and not b.flags.writeable
    assert resident.query_to_targets.ids is resident.query_ids
    assert resident.target_to_queries.ids is resident.protein_ids
    payload = pickle.dumps(resident)
    assert len(payload) < 512
    restored = pickle.loads(payload)
    assert type(restored) is IndexedPairs and isinstance(restored.pairs, np.memmap)
    with pytest.raises(ValueError, match="limit"):
        ResidentIndexedPairs(index_dir, max_bytes=1)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("epoch", [0, 1, 17])
def test_same_sampler_rows_rng_categories_and_masks(index_dir, rank, epoch):
    baseline, fast = IndexedPairs(index_dir), ResidentIndexedPairs(index_dir)
    samplers = [IndexedTypedNegativeBatchSampler(index, 100, rank=rank, world_size=4)
                for index in (baseline, fast)]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    a, b = [list(sampler) for sampler in samplers]
    assert a == b
    assert samplers[0].last_counts == samplers[1].last_counts
    for rows in a:
        queries = [str(baseline.query_ids[q]) for q, p, k in rows]
        proteins = [str(baseline.protein_ids[p]) for q, p, k in rows]
        kinds = [PAIR_TYPES[k] for q, p, k in rows]
        assert kinds.count("positive") == 85
        assert baseline.query_to_targets.batch_positive_indices(queries, proteins) == fast.query_to_targets.batch_positive_indices(queries, proteins)
        expected = baseline.batch_negative_masks(queries, proteins, queries, proteins, kinds)
        actual = fast.batch_negative_masks(queries, proteins, queries, proteins, kinds)
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x, y)


def _residues(path, dtype):
    values = np.random.default_rng(42).normal(size=(17, 4)).astype(dtype)
    with h5py.File(path, "w") as handle:
        handle["ids"] = np.array(["p0", "p1"], dtype="S")
        handle["vectors"] = values
        handle["offsets"] = [0, 12, 17]
    return values


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_transport_restores_exact_values_masks_and_model_gradients(tmp_path, dtype):
    path = tmp_path / "residues.h5"
    _residues(path, dtype)
    baseline = ResidueEmbedDataset(str(path), max_tokens=8)
    fast = StoragePrecisionResidues(str(path), max_tokens=8)
    rows = ["p0", "p1", "p0"]
    def collate(dataset):
        return residue_collate_fn([dict(dataset[pid], target_id=pid) for pid in rows])
    expected, transported = collate(baseline), collate(fast)
    assert transported["residue_embeddings"].dtype == (torch.float16 if dtype == "float16" else torch.float32)
    if dtype == "float16":
        assert transported["residue_embeddings"].nbytes * 2 == expected["residue_embeddings"].nbytes
    actual = restore_residue_precision(transported)
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value
    # Reduction and backward see precisely the same FP32 input, not FP16 sums.
    linear = torch.nn.Linear(4, 2)
    gradients = []
    for batch in (expected, actual):
        linear.zero_grad(set_to_none=True)
        linear(batch["residue_embeddings"].mean(1)).square().sum().backward()
        gradients.append([p.grad.clone() for p in linear.parameters()])
    for x, y in zip(*gradients):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(fast))  # Open HDF5 handles must not be pickled.
    assert restored._vectors is None and restored.file is None
    assert restored["p0"]["residue_embeddings"].dtype == fast.dtype
    restored.close()
    fast.close()
    baseline.close()


def test_finite_guard_not_disabled(tmp_path):
    path = tmp_path / "residues.h5"
    _residues(path, "float16")
    with h5py.File(path, "r+") as handle:
        handle["vectors"][0, 0] = np.nan
    dataset = StoragePrecisionResidues(str(path))
    with pytest.raises(ValueError, match="non-finite"):
        dataset["p0"]
    dataset.close()


def test_virtual_dataset_handle_retained_and_fork_reopened(tmp_path, monkeypatch):
    values = np.arange(68, dtype=np.float16).reshape(17, 4)
    layout = h5py.VirtualLayout(shape=values.shape, dtype="float16")
    for name, start, end in (("a", 0, 11), ("b", 11, 17)):
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("vectors", data=values[start:end], chunks=(2, 4))
        layout[start:end] = h5py.VirtualSource(str(path), "vectors", shape=(end - start, 4))
    path = tmp_path / "virtual.h5"
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_virtual_dataset("vectors", layout)
        handle["ids"] = np.array(["p0", "p1"], dtype="S")
        handle["offsets"] = [0, 12, 17]
    reader = StoragePrecisionResidues(str(path), max_tokens=8)
    baseline = ResidueEmbedDataset(str(path), max_tokens=8)
    for pid in ("p0", "p1"):
        torch.testing.assert_close(reader[pid]["residue_embeddings"].float(), baseline[pid]["residue_embeddings"], rtol=0, atol=0)
    vector_handle = reader._vectors
    reader["p0"]
    assert reader._vectors is vector_handle
    monkeypatch.setattr("horizyn.datasets.residue_hdf5.os.getpid", lambda: -1234)
    reader["p1"]
    assert reader._vectors is not vector_handle and not vector_handle.id.valid
    baseline.close()
    reader.close()


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_residue_loader_multiprocessing_after_parent_read(tmp_path, context):
    from torch.utils.data import DataLoader
    path = tmp_path / "residues.h5"
    _residues(path, "float16")
    reader = StoragePrecisionResidues(str(path), max_tokens=8)
    expected = residue_collate_fn([reader[0], reader[1]])
    loader = DataLoader(reader, batch_size=2, num_workers=2,
                        multiprocessing_context=context, collate_fn=residue_collate_fn)
    actual = list(loader)[0]
    torch.testing.assert_close(expected["residue_embeddings"], actual["residue_embeddings"], rtol=0, atol=0)
    reader.close()


def test_nested_transfer_only_changes_residue_precision():
    tensor = torch.ones(2, 3, dtype=torch.float16)
    mask = torch.ones(2, dtype=torch.bool)
    batch = [{"residue_embeddings": tensor, "query_vec": tensor, "mask": mask},
             ({"score_residue_embeddings": tensor, "id": "p0"},)]
    result = FastIODataModule.transfer_batch_to_device(object.__new__(FastIODataModule), batch, torch.device("cpu"), 0)
    assert result[0]["residue_embeddings"].dtype == torch.float32
    assert result[1][0]["score_residue_embeddings"].dtype == torch.float32
    assert result[0]["query_vec"].dtype == torch.float16
    assert result[0]["mask"].dtype == torch.bool and result[1][0]["id"] == "p0"


def test_adapter_scope_restores_original_factories_on_error():
    import horizyn.reaction_conditioned_data_module as data
    old = (data.IndexedPairs, data.ResidueEmbedDataset, data.ReactionConditionedDataModule)
    entrypoint = SimpleNamespace(ReactionConditionedDataModule=data.ReactionConditionedDataModule)
    with pytest.raises(RuntimeError, match="test error"):
        with training_io_adapters(entrypoint):
            assert data.IndexedPairs is ResidentIndexedPairs
            assert data.ResidueEmbedDataset is StoragePrecisionResidues
            assert entrypoint.ReactionConditionedDataModule is FastIODataModule
            raise RuntimeError("test error")
    assert old == (data.IndexedPairs, data.ResidueEmbedDataset, entrypoint.ReactionConditionedDataModule)


def test_launcher_preserves_training_contract_and_isolates_output(tmp_path):
    spec = importlib.util.spec_from_file_location("fast_io_launcher", ROOT / "scripts/train_protein_pooling_fast_io.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source = launcher.load_config(ROOT / "configs/horizyn1_circe_v2_h200.yaml")
    source.data.indexed_pairs_dir = str(tmp_path / "index")
    args = SimpleNamespace(io_mode="fast", io_output_dir=str(tmp_path / "new"), io_prefetch=4,
                           io_benchmark_steps=0, resume=None)
    overrides = launcher.output_overrides(args, source)
    assert set(overrides) == {"logging.log_dir", "logging.checkpoint_dir", "data.prefetch_factor"}
    args.io_output_dir = source.logging.log_dir
    with pytest.raises(ValueError, match="overlaps"):
        launcher.output_overrides(args, source)
    args.io_output_dir = str(tmp_path / "new")
    args.io_benchmark_steps, args.resume = 60, "/checkpoint.ckpt"
    with pytest.raises(ValueError, match="not --resume"):
        launcher.output_overrides(args, source)


@pytest.mark.parametrize("materialized_directions", [False, True])
def test_full_multimodal_training_with_fast_io(tmp_path, materialized_directions):
    spec = importlib.util.spec_from_file_location(
        "fast_io_multimodal_fixture", ROOT / "tests/unit/test_horizyn1_multimodal_training_smoke.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with training_io_adapters(fixture):
        fixture.test_indexed_multimodal_factorized_training_two_steps(tmp_path, materialized_directions)
