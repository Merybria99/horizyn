import csv
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from horizyn.capability.cofactor_vocabulary_v2 import COFACTOR_LABELS
from horizyn.config import DotDict, validate_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.indexed_pairs import (
    IndexedPairDataset, IndexedPairs, IndexedTypedNegativeBatchSampler,
    PAIR_TYPES, SEMANTICS, write_indexed_pairs,
)


def index_kwargs():
    return dict(
        query_ids=np.array(["q0", "q1", "q2", "q3"]),
        protein_ids=np.array([f"p{i}" for i in range(6)]),
        pairs=np.array([[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5]]),
        protein_ec=np.array([0, 0, 1, 2, 3, -1]), ec_prefix=np.array([0, 0, 1, 0]),
        query_ec_indptr=np.arange(5), query_ec=np.arange(4),
        mechanism_bits=np.array([1, 1, 1, 2, 1, 1]),
        native_cofactor_bits=np.array([1, 1, 1, 1, 2, 0]),
        reaction_cofactor_bits=np.zeros(6, dtype=np.uint32),
        ec_eligible=np.array([1, 1, 1, 1, 1, 0]),
        biological_eligible=np.array([1, 1, 1, 1, 1, 0]),
        query_signature=np.arange(4),
        provenance={"pair_scope": "train", "annotation_semantics": SEMANTICS},
        excluded_pairs=np.array([[0, 4]]), min_biofp_similarity=0.85,
    )


@pytest.fixture
def index(tmp_path):
    write_indexed_pairs(tmp_path / "index", **index_kwargs())
    return IndexedPairs(tmp_path / "index")


def test_compact_maps_lookup_batch_matches_all_known_positives(index):
    assert isinstance(index.pairs, np.memmap)
    assert list(index.query_to_targets["q0"]) == ["p0", "p1"]
    assert list(index.target_to_queries["p2"]) == ["q1"]
    assert index.query_to_targets.get("absent", ()) == ()
    assert index.query_to_targets.batch_positive_indices(["q0", "absent", "q1"], ["p2", "p0", "absent", "p1"]) == ([0, 0, 2], [1, 3, 0])
    assert len(pickle.dumps(index)) < 512
    assert isinstance(pickle.loads(pickle.dumps(index)).pairs, np.memmap)


def test_annotation_policy_roles_unknown_and_exclusions(index):
    assert index.negative_kind(0, 0) is None  # positive
    assert index.negative_kind(1, 2) is None
    assert index.negative_kind(0, 5) is None  # missing EC eligibility
    assert index.negative_kind(0, 4) is None  # prohibition-only edge
    assert index.negative_kind(0, 2) == 1
    assert index.negative_kind(0, 3) == 2
    assert index.negative_kind(1, 4) == 2  # .75 < .85; cofactors really affect score
    assert index.similarity((1, 1, 0), (1, 1, 0)) == pytest.approx(.9)
    assert index.similarity((1, 1, 0), (1, 0, 1)) == pytest.approx(.75)
    assert index.similarity((0, 0, 0), (0, 0, 0)) == 0


def test_shared_candidates_expand_only_checked_sampled_rows_and_columns(index):
    queries = ["q0", "q1", "q2", "q3"]
    proteins = [f"p{i}" for i in range(6)]
    biological, random = index.batch_negative_masks(
        queries, proteins, ["q0", "q1", "q2", "q0"], ["p0", "p2", "p3", "p2"],
        ["positive", "positive", "positive", "biological_negative"],
    )
    assert biological[0, 2]  # original sampled edge
    assert random[2, 2]  # previously inactive q2 gains checked sampled p2
    assert random[0, 3]  # sampled q0 may be reused with another batch protein
    assert not (biological[1, 2] or random[1, 2])  # true positive
    assert not (biological[0, 1] or random[0, 1])  # matching EC / true positive
    assert not (biological[:, 5] | random[:, 5]).any()  # unknown EC candidate
    assert not (biological[0, 4] or random[0, 4])  # prohibition-only edge
    assert not (biological[2, 0] or random[2, 0])  # neither endpoint sampled negative
    assert not np.any(biological & random)
    empty = index.batch_negative_masks(queries, proteins, ["q0"], ["p0"], ["positive"])
    assert not any(mask.any() for mask in empty)
    with pytest.raises(ValueError, match="does not match"):
        index.batch_negative_masks(queries, proteins, ["q0"], ["p0"], ["random_negative"])


def test_vectorized_shared_policy_preserves_scalar_threshold_rounding(tmp_path):
    kwargs = index_kwargs()
    kwargs["mechanism_bits"][:3] = [107, 107, 104]
    kwargs["native_cofactor_bits"][:3] = [249448455, 249448455, 833483208]
    kwargs["reaction_cofactor_bits"][:3] = [961982622, 961982622, 1856543212]
    kwargs["min_biofp_similarity"] = .5
    write_indexed_pairs(tmp_path / "index", **kwargs)
    index = IndexedPairs(tmp_path / "index")
    assert index.negative_kind(0, 2) == 2
    biological, random = index.batch_negative_masks(["q0", "q1"], ["p0", "p2"], ["q0"], ["p2"], ["random_negative"])
    assert random[0, 1] and not biological[0, 1]


def test_signature_and_ineligible_positive_ec_still_exclude(tmp_path):
    kwargs = index_kwargs()
    kwargs["query_signature"] = np.array([0, 0, 2, 3])
    kwargs["query_ec_indptr"] = np.array([0, 2, 3, 4, 5])
    kwargs["query_ec"] = np.array([0, 2, 1, 2, 3])
    write_indexed_pairs(tmp_path / "index", **kwargs)
    index = IndexedPairs(tmp_path / "index")
    assert index.negative_kind(0, 2) is None  # equivalent observed chemistry
    assert index.negative_kind(0, 3) is None  # extra complete EC from conflicted positive evidence


@pytest.mark.parametrize("field,value", [
    ("native_cofactor_bits", [1 << 31] * 6),
    ("reaction_cofactor_bits", [1 << 31] * 6),
    ("mechanism_bits", [256] * 6),
    ("ec_eligible", [1] * 6),
    ("biological_eligible", [1] * 6),
    ("query_ec", [1, 1, 2, 3]),
])
def test_writer_rejects_invalid_semantics(tmp_path, field, value):
    kwargs = index_kwargs()
    kwargs[field] = np.asarray(value)
    with pytest.raises(ValueError):
        write_indexed_pairs(tmp_path / "bad", **kwargs)


def test_rejects_unsplit_metadata_and_overwrite(tmp_path, index):
    kwargs = index_kwargs()
    kwargs["provenance"]["pair_scope"] = "unsplit_inventory"
    with pytest.raises(ValueError, match="train-only"):
        write_indexed_pairs(tmp_path / "bad", **kwargs)
    with pytest.raises(ValueError, match="overwrite"):
        write_indexed_pairs(index.directory, **index_kwargs())
    manifest_path = index.directory / "manifest.json"
    meta = json.loads(manifest_path.read_text())
    meta["pair_scope"] = "unsplit_inventory"
    manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="non-training"):
        IndexedPairs(index.directory)


def _assert_supported_batch(index, rows):
    assert len(rows) == 20
    positives = {(q, p) for q, p, kind in rows if kind == 0}
    assert sum(kind == 0 for _, _, kind in rows) == 17
    assert sum(kind != 0 for _, _, kind in rows) == 3
    for q, p, kind in rows:
        if kind:
            assert index.negative_kind(q, p) == kind
            assert any(a == q for a, _ in positives)
            assert any(b == p for _, b in positives)
        else:
            assert index.is_positive(q, p)


def test_exact_quota_determinism_epochs_and_ddp_support(index):
    batches = []
    for rank in range(2):
        sampler = IndexedTypedNegativeBatchSampler(index, 20, rank=rank, world_size=2)
        first = list(sampler)
        sampler.set_epoch(0)
        assert list(sampler) == first
        sampler.set_epoch(1)
        assert list(sampler) != first
        assert len(first) == len(sampler)
        for rows in first:
            _assert_supported_batch(index, rows)
        batches.extend(first)
    # All true train edges are visited via the affine base traversal, even when
    # the total graph is smaller than one global batch and needs padding.
    observed = {(q, p) for rows in batches for q, p, kind in rows if kind == 0}
    assert observed == set(map(tuple, index.pairs.tolist()))


def test_lightning_epoch_hook_reaches_indexed_batch_sampler(index):
    from lightning.fabric.utilities.data import _set_sampler_epoch
    sampler = IndexedTypedNegativeBatchSampler(index, 20)
    _set_sampler_epoch(SimpleNamespace(batch_sampler=sampler), 7)
    assert sampler.epoch == 7
    actual = next(iter(sampler))
    reference = IndexedTypedNegativeBatchSampler(index, 20)
    reference.set_epoch(7)
    assert actual == next(iter(reference))


@pytest.mark.parametrize("batch_size", [1, 19, 21, 64, 0, True])
def test_exact_quota_rejects_nonintegral_batches(index, batch_size):
    with pytest.raises(ValueError, match="divisible by 20"):
        IndexedTypedNegativeBatchSampler(index, batch_size)


def test_bounded_sampler_failure_is_not_blanket_negatives(tmp_path):
    kwargs = index_kwargs()
    kwargs["query_ec_indptr"] = np.arange(0, 17, 4)
    kwargs["query_ec"] = np.tile(np.arange(4), 4)
    write_indexed_pairs(tmp_path / "none", **kwargs)
    sampler = IndexedTypedNegativeBatchSampler(IndexedPairs(tmp_path / "none"), 20, max_negative_attempts=4)
    with pytest.raises(RuntimeError, match="bounded proposal budget"):
        next(iter(sampler))


def test_dataset_feature_suffix_and_missing_are_explicit(index):
    query = BaseDataset(keys=[f"q{i}_f" for i in range(4)], array_data=torch.randn(4, 3))
    protein = BaseDataset(keys=[f"p{i}" for i in range(6)], array_data=torch.randn(6, 3))
    dataset = IndexedPairDataset(index, query, protein)
    sample = dataset[(0, 2, 1)]
    assert (sample["query_id"], sample["target_id"], sample["pair_type"]) == ("q0", "p2", "biological_negative")
    torch.testing.assert_close(sample["query_vec"], query["q0_f"])
    with pytest.raises(ValueError, match="feature missing"):
        IndexedPairDataset(index, query, BaseDataset(keys=["p0"], array_data=torch.randn(1, 3)))


@pytest.mark.parametrize("dictionary", [False, True])
def test_batch_fetch_reads_each_endpoint_once_without_dropping_rows(index, dictionary):
    class CountedDataset:
        def __init__(self, keys, field):
            self.keys = keys
            self.field = field
            self.calls = []

        def __getitem__(self, key):
            self.calls.append(key)
            vector = torch.tensor([float(self.keys.index(key))])
            return {self.field: vector} if dictionary else vector

    query = CountedDataset([f"q{i}" for i in range(4)], "query_vec")
    protein = CountedDataset([f"p{i}" for i in range(6)], "target_vec")
    dataset = IndexedPairDataset(index, query, protein)
    rows = [(0, 0, 0), (1, 2, 0), (0, 2, 1), (0, 0, 0)]
    expected = [dataset[row] for row in rows]
    query.calls.clear()
    protein.calls.clear()
    actual = dataset.__getitems__(rows)
    assert query.calls == ["q0", "q1"]
    assert protein.calls == ["p0", "p2"]
    assert len(actual) == len(rows)
    for first, second in zip(actual, expected):
        assert first.keys() == second.keys()
        for key in first:
            if torch.is_tensor(first[key]):
                torch.testing.assert_close(first[key], second[key])
            else:
                assert first[key] == second[key]
    dataset.__getitems__([rows[0]])
    assert query.calls == ["q0", "q1", "q0"]  # No persistent unbounded feature cache.


def test_config_new_defaults_and_legacy_gate():
    config = DotDict(dict(
        data=dict(train_pairs_path="pairs.csv", test_pairs_path="test.csv", train_reactions_path="rxn.csv", test_reactions_path="test_rxn.csv", protein_embeds_path="emb.h5", indexed_pairs_dir="index", reaction_direction_mode="forward_only", train_batch_size=20),
        model=dict(query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4),
        training=dict(max_epochs=1, loss=dict(name="BidirectionalSampledMultiPositiveInfoNCELoss", positive_pair_source="all_known_in_batch")),
    ))
    validate_config(config)
    config.data.typed_negative_pools_path = "pools.json"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(config)
    del config.data["indexed_pairs_dir"]
    config.data.typed_negative_positive_fraction = .85
    with pytest.raises(ValueError, match="exactly 0.5"):
        validate_config(config)


def test_validation_query_panels_are_bounded_and_fail_closed(tmp_path):
    from horizyn.reaction_conditioned_data_module import _directional_validation_queries
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"reaction_to_enzyme": ["q0"], "enzyme_to_reaction": ["p1"]}))
    assert _directional_validation_queries(path, ["q0_f", "q1_f"], ["p0", "p1"]) == (["q0_f"], ["p1"])
    with pytest.raises(ValueError, match="lacks gold/features"):
        _directional_validation_queries(path, ["q1_f"], ["p0", "p1"])


def test_indexed_validation_keeps_full_reaction_catalog_and_directional_panels(index, tmp_path, monkeypatch):
    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    _write_csv(tmp_path / "validation.csv", ["pr_id", "reaction_id", "protein_id"], [(0, "q0", "p0"), (1, "q1", "p2")])
    _write_csv(tmp_path / "reactions.csv", ["reaction_id", "reaction_smiles"], [(f"q{i}", "A>>B") for i in range(5)])
    query_path = tmp_path / "panels.json"
    query_path.write_text(json.dumps({"reaction_to_enzyme": ["q0"], "enzyme_to_reaction": ["p2"]}))
    query = BaseDataset(keys=[f"q{i}" for i in range(5)], array_data=torch.randn(5, 3))
    protein = BaseDataset(keys=[f"p{i}" for i in range(6)], array_data=torch.randn(6, 3))
    module = ReactionConditionedDataModule(
        train_pairs_path="unused.csv", test_pairs_path=str(tmp_path / "validation.csv"),
        train_reactions_path=str(tmp_path / "reactions.csv"), test_reactions_path=str(tmp_path / "reactions.csv"),
        protein_residue_embeds_path="unused.h5", indexed_pairs_dir=str(index.directory),
        train_batch_size=20, reaction_direction_mode="forward_only", residue_dim=3,
        validation_retrieval_metrics=True, validation_retrieval_query_ids_path=str(query_path),
    )
    monkeypatch.setattr(module, "_create_query_dataset", lambda *args, **kwargs: query)
    monkeypatch.setattr(module, "_create_target_dataset", lambda: protein)
    module.setup("fit")
    assert module._val_retrieval_query_candidate_ids == [f"q{i}" for i in range(5)]
    assert module._val_metric_query_lookup_data.keys == [f"q{i}" for i in range(5)]
    assert module._val_metric_query_data.keys == ["q0"]
    assert module._val_metric_target_query_data.keys == ["p2"]
    assert set(module._query_to_targets) == {"q0", "q1"}  # full panel gold unchanged
    assert "q4" not in module._query_to_targets  # still a real retrieval candidate
    query_path.write_text(json.dumps({"reaction_to_enzyme": ["q4"], "enzyme_to_reaction": ["p2"]}))
    with pytest.raises(ValueError, match="lacks gold/features"):
        module._setup_validation_data()
    monkeypatch.setattr(module, "_create_query_dataset", lambda *args, **kwargs: BaseDataset(keys=[f"q{i}" for i in range(4)], array_data=torch.randn(4, 3)))
    with pytest.raises(ValueError, match="candidate reactions lack features"):
        module._setup_validation_data()


def _write_csv(path, headers, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(values)


def _builder_fixture(tmp_path):
    script = Path(__file__).parents[2] / "scripts/build_indexed_training_pairs.py"
    spec = importlib.util.spec_from_file_location("indexed_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _write_csv(tmp_path / "pairs.csv", ["reaction_id", "protein_id"], [("q0", "prot_a"), ("q1", "prot_b"), ("q2", "prot_c")])
    _write_csv(tmp_path / "reactions.csv", ["reaction_id", "reaction_smiles"], [("q0", "A>>B"), ("q1", "C>>D"), ("q2", "E>>F")])
    _write_csv(tmp_path / "ec.csv", ["protein_id", "ec_number", "known_depth"], [("a", "1.1.1.1", 4), ("b", "1.1.1.2", 4), ("c", "2.1.1.1", 4), ("heldout", "9.9.9.9", 4)])
    _write_csv(tmp_path / "eligibility.csv", ["protein_id", "ec_negative_candidate_eligible", "biological_negative_candidate_eligible"], [(p, 1, 1) for p in ("a", "b", "c", "heldout")])
    payload = dict(ids=np.array(["a", "b", "c", "heldout"]), annotation_semantics=np.array(SEMANTICS), cofactor_vocabulary_version=np.array("circe_cofactor_v2"), cofactor_labels=np.array(COFACTOR_LABELS), cofactor_unknown_index=np.array(31, dtype=np.int64), pair_scope=np.array("train"))
    for family in ("mechanism", "cofactor", "native_cofactor", "reaction_cofactor"):
        values = np.zeros((4, 8 if family == "mechanism" else 32), dtype=np.float32)
        values[:, 0 if family == "mechanism" else 31] = 1
        confidence = values * np.float32(.4)
        if family != "mechanism":
            confidence[:, 31] = 0
        payload[f"{family}_targets"] = values
        payload[f"{family}_mask"] = values.astype(bool)
        payload[f"{family}_confidence"] = confidence
    np.savez(tmp_path / "biofp.npz", **payload)
    return module, payload, dict(train_pairs=tmp_path / "pairs.csv", train_reactions=tmp_path / "reactions.csv", ec_labels=tmp_path / "ec.csv", biofp_targets=tmp_path / "biofp.npz", candidate_eligibility=tmp_path / "eligibility.csv", output_dir=tmp_path / "output")


def test_offline_builder_filters_train_entities_and_preserves_prohibition_only(tmp_path):
    module, _, kwargs = _builder_fixture(tmp_path)
    _write_csv(tmp_path / "excluded.csv", ["reaction_id", "protein_id"], [("q0", "prot_b"), ("q0", "heldout")])
    manifest = module.build(**kwargs, exclude_pairs=[tmp_path / "excluded.csv"])
    assert manifest["num_proteins"] == 3 and manifest["num_pairs"] == 3
    index = IndexedPairs(kwargs["output_dir"])
    assert index.negative_kind(0, 1) is None
    assert not index.is_positive(0, 1)
    assert np.all(index.native_cofactor_bits == 0)


@pytest.mark.parametrize("corruption", ["scope", "confidence", "metadata"])
def test_offline_builder_rejects_malformed_new_annotations(tmp_path, corruption):
    module, payload, kwargs = _builder_fixture(tmp_path)
    if corruption == "scope":
        payload["pair_scope"] = np.array("unsplit_inventory")
    elif corruption == "confidence":
        payload["mechanism_confidence"][:] = 0
    else:
        del payload["annotation_semantics"]
    np.savez(kwargs["biofp_targets"], **payload)
    with pytest.raises(ValueError):
        module.build(**kwargs)


def _loss_module(index):
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    module = ProteinPooledLitModule(query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4, residue_dim=4, pooling="mean", loss_name="BidirectionalSampledMultiPositiveInfoNCELoss", sampled_require_both_directions=True, positive_pair_source="all_known_in_batch")
    module.trainer = SimpleNamespace(datamodule=SimpleNamespace(_train_query_to_targets=index.query_to_targets, _train_typed_negative_targets={}))
    return module


def _batch_loss(module, index, model, rows):
    qids = list(dict.fromkeys(str(index.query_ids[q]) for q, _, _ in rows))
    pids = list(dict.fromkeys(str(index.protein_ids[p]) for _, p, _ in rows))
    coords = torch.tensor([int(q[1:]) for q in qids] + [int(p[1:]) + len(index.query_ids) for p in pids])
    embedding = model(coords)
    return module._global_full_batch_loss(
        embedding[:len(qids)], embedding[len(qids):], None, None, qids, pids,
        [str(index.query_ids[q]) for q, _, _ in rows], [str(index.protein_ids[p]) for _, p, _ in rows],
        [PAIR_TYPES[kind] for _, _, kind in rows], structure_terms_enabled=False,
    )


def _ddp_worker(rank, directory, rendezvous, output):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.nn.functional import all_gather
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        index = IndexedPairs(directory)
        module = _loss_module(index)
        def gather(tensor, group=None, sync_grads=False):
            if sync_grads:
                return torch.stack(all_gather(tensor))
            values = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(values, tensor)
            return torch.stack(values)
        module.all_gather = gather
        torch.manual_seed(77)
        model = DistributedDataParallel(torch.nn.Embedding(10, 4))
        rows = next(iter(IndexedTypedNegativeBatchSampler(index, 20, rank=rank, world_size=2, seed=19)))
        loss, components = _batch_loss(module, index, model, rows)
        loss.backward()
        assert components["r2e_valid_anchors"] > 0 and components["e2r_valid_anchors"] > 0
        if rank == 0:
            torch.save({"loss": loss.detach(), "gradient": model.module.weight.grad}, output)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_global_masks_and_ddp_gradients_equal_single_process(index, tmp_path):
    import torch.multiprocessing as mp
    output = tmp_path / "ddp.pt"
    mp.spawn(_ddp_worker, args=(str(index.directory), str(tmp_path / "rendezvous"), str(output)), nprocs=2, join=True)
    rows = []
    for rank in range(2):
        rows.extend(next(iter(IndexedTypedNegativeBatchSampler(index, 20, rank=rank, world_size=2, seed=19))))
    torch.manual_seed(77)
    model = torch.nn.Embedding(10, 4)
    loss, _ = _batch_loss(_loss_module(index), index, model, rows)
    loss.backward()
    result = torch.load(output, weights_only=True)
    torch.testing.assert_close(result["loss"], loss)
    torch.testing.assert_close(result["gradient"], model.weight.grad, rtol=2e-5, atol=2e-6)
