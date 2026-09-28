"""Integration checks against the released weights and upstream preprocessing."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "VenusRXN")]
from horizyn.pretrained_graphormer import (
    GRAPHORMER_REVISION, UPSTREAM_REVISION, PretrainedGraphBuilder,
    PretrainedMolecularGraphormer, collate_pretrained_graphs,
)
from horizyn.token_retrieval import TokenModelConfig, TokenRetrievalModel
from horizyn.token_retrieval_data import MoleculeStore
from scripts.train_token_retrieval import build_optimizer, configure_graph_training, check_checkpoint_architecture

CHECKPOINT = ROOT / ".deps/graphormer-base-pcqm4mv2" / GRAPHORMER_REVISION
UPSTREAM = ROOT / ".deps/graphormer_upstream" / UPSTREAM_REVISION / "graphormer/data"
pytestmark = pytest.mark.skipif(not (CHECKPOINT / "pytorch_model.bin").is_file(),
                                reason="Download the pinned Graphormer integration checkpoint")


def small_pool_config():
    return TokenModelConfig(residue_dim=12, width=16, heads=4, ffn_dim=32,
                            graph_layers=12, latent_layers=1, protein_tokens=4,
                            reaction_tokens=3, local_dim=8, global_dim=12,
                            molecule_chunk=2, protein_chunk=2, dropout=0.0,
                            graph_backbone="pcqm4mv2_pretrained", graph_checkpoint_dir=str(CHECKPOINT))


@pytest.fixture(scope="module")
def encoder():
    return PretrainedMolecularGraphormer(CHECKPOINT).eval()


def test_released_encoder_is_loaded_bitwise(encoder):
    state = torch.load(CHECKPOINT / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert encoder.provenance["encoder_tensors_loaded"] == 202
    assert encoder.provenance["encoder_parameters"] == 47084608
    for name, tensor in encoder.encoder.state_dict().items():
        assert torch.equal(tensor, state["encoder.graph_encoder." + name])


def test_missing_or_altered_checkpoint_fails_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        PretrainedMolecularGraphormer(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected pretrained"):
        PretrainedMolecularGraphormer(tmp_path)


def upstream_preprocessor():
    if not UPSTREAM.is_dir():
        pytest.skip("Pinned Microsoft preprocessing sources required for parity audit")
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=2)
    sys.path.insert(0, str(UPSTREAM))
    import algos
    # Execute only the two original functions, avoiding unrelated fairseq/dataset
    # imports. Removing JIT decoration does not change their tensor operations.
    tree = ast.parse((UPSTREAM / "wrapper.py").read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"convert_to_single_emb", "preprocess_item"}]
    for node in funcs:
        node.decorator_list = []
    namespace = {"torch": torch, "np": np, "algos": algos}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(UPSTREAM / "wrapper.py"), "exec"), namespace)
    spec = importlib.util.spec_from_file_location("graphormer_original_collator", UPSTREAM / "collator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return namespace["preprocess_item"], module.collator


def test_features_and_padding_match_microsoft_pipeline(encoder, tmp_path):
    from ogb.utils import smiles2graph
    preprocess, collate = upstream_preprocessor()
    store = MoleculeStore(PretrainedGraphBuilder(encoder.config, tmp_path))
    graphs = []
    originals = []
    # Rings exercise alternative shortest paths; chain exercises hop truncation;
    # stereochemistry, charged ions and unequal sizes exercise feature/pad IDs.
    for smiles in ["[Na+]", "CCO", "c1ccccc1", "C[C@H](O)C(=O)O", "CCCCCCCCCCCC"]:
        graph, = store.reaction(smiles)
        graphs.append(graph)
        raw = smiles2graph(graph.canonical_smiles)
        item = SimpleNamespace(idx=len(originals), x=torch.from_numpy(raw["node_feat"]),
                               edge_index=torch.from_numpy(raw["edge_index"]),
                               edge_attr=torch.from_numpy(raw["edge_feat"]), y=torch.zeros(1))
        originals.append(preprocess(item))
    expected = collate(originals, max_node=encoder.config.max_nodes,
                       multi_hop_max_dist=encoder.config.multi_hop_max_dist,
                       spatial_pos_max=encoder.config.spatial_pos_max)
    actual = collate_pretrained_graphs(graphs, encoder.config)
    aliases = {"input_nodes": "x", "input_edges": "edge_input"}
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[aliases.get(name, name)], atol=0, rtol=0)
    cached = MoleculeStore(PretrainedGraphBuilder(encoder.config, tmp_path))
    again = collate_pretrained_graphs([cached.reaction(g.canonical_smiles)[0] for g in graphs], encoder.config)
    for key in actual:
        torch.testing.assert_close(actual[key], again[key], atol=0, rtol=0)


def test_isolated_atoms_and_batch_padding_have_finite_invariant_outputs(encoder):
    store = MoleculeStore(PretrainedGraphBuilder(encoder.config))
    ion, = store.reaction("[Na+]")
    molecule, = store.reaction("CCCO")
    with torch.no_grad():
        single, _ = encoder(collate_pretrained_graphs([ion], encoder.config))
        mixed, _ = encoder(collate_pretrained_graphs([ion, molecule], encoder.config))
    assert single.shape == (1, 1, 768) and torch.isfinite(mixed).all()
    torch.testing.assert_close(single[0, 0], mixed[0, 0], atol=2e-5, rtol=2e-5)


def test_oversized_components_fail_explicitly(encoder):
    fake = SimpleNamespace(GetNumAtoms=lambda: encoder.config.max_nodes + 1)
    with pytest.raises(ValueError, match="atoms per component"):
        PretrainedGraphBuilder(encoder.config)(fake, "oversized")


def test_pretrained_b1_b2_start_identically():
    torch.manual_seed(42)
    first = TokenRetrievalModel(small_pool_config(), "B1")
    torch.manual_seed(42)
    second = TokenRetrievalModel(small_pool_config(), "B2")
    assert first.reaction_pool.input_projection[1].in_features == 768
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_warmup_preserves_pretraining_then_retrieval_gradients_update_encoder(variant):
    model = TokenRetrievalModel(small_pool_config(), variant)
    store = MoleculeStore(PretrainedGraphBuilder(model.graph_encoder.config))
    training = {"learning_rate": 1e-4, "graph_learning_rate": 1e-5,
                "weight_decay": 0.01, "graph_warmup_epochs": 2}
    optimizer = build_optimizer(model, training)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 1e-5]
    before = next(model.graph_encoder.parameters()).detach().clone()
    for epoch in [0, 2]:
        active = configure_graph_training(model, training, epoch)
        model.train()
        assert model.graph_encoder.training is active
        optimizer.zero_grad(set_to_none=True)
        reaction, enzyme = model([torch.randn(6, 12), torch.randn(8, 12)],
                                 [store.reaction("CCO"), store.reaction("CC(=O)O")])
        scores = model.score_pairs(reaction, enzyme, torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        loss = torch.nn.functional.cross_entropy(10 * scores.reshape(2, 2), torch.arange(2))
        loss.backward()
        gradients = [p.grad for p in model.graph_encoder.parameters() if p.grad is not None]
        assert bool(gradients) is active
        if active:
            assert all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        optimizer.step()
        assert torch.equal(before, next(model.graph_encoder.parameters())) is (not active)


def test_scratch_checkpoint_cannot_resume_pretrained_architecture():
    with pytest.raises(ValueError, match="Checkpoint architecture"):
        check_checkpoint_architecture({"config": {"model": {}}}, {"model": small_pool_config().__dict__})
