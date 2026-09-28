"""Dependency-light contracts for pooled-model initialization and dispatch.

These tests execute the real encoding-dispatch method with recording towers.
They do not claim to test tensor arithmetic, gradients or distributed training.
"""

import ast
import copy
import itertools
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@lru_cache
def methods(filename, class_name):
    tree = ast.parse((ROOT / "horizyn" / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def load_dispatch():
    method = copy.deepcopy(
        methods("protein_pooling_lightning_module.py", "ProteinPooledLitModule")[
            "_encode_training_inputs"
        ]
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "encoding_dispatch", "exec"), namespace
    )
    return namespace[method.name]


class RecordingTowers:
    def __init__(self, pooling, mode, query_returns_details):
        self.pooling_name = pooling
        self.enzyme_input_mode = mode
        self.query_returns_details = query_returns_details
        self.calls = []

    def encode_queries(self, value, **kwargs):
        self.calls.append(("query", value, kwargs))
        if kwargs.get("return_attention") and self.query_returns_details:
            return "query_embedding", "query_details"
        return "query_embedding"

    def encode_targets(self, value, **kwargs):
        self.calls.append(("target", value, kwargs))
        if kwargs.get("return_pooling_details") or kwargs.get("return_attention"):
            return "target_embedding", "target_details"
        return "target_embedding"


def report(name):
    return lambda *args: {name: args}


class CoreStructureTests(unittest.TestCase):
    def test_common_constructor_defaults_remain_compatible(self):
        defaults = []
        for filename, cls in (
            ("protein_pooling_lightning_module.py", "ProteinPooledLitModule"),
            ("models/dual.py", "ProteinPooledDualModel"),
        ):
            init = methods(filename, cls)["__init__"]
            defaults.append(
                {
                    arg.arg: ast.unparse(default)
                    for arg, default in zip(
                        init.args.args[-len(init.args.defaults) :], init.args.defaults
                    )
                }
            )
        training, model = defaults
        for name in training.keys() & model.keys():
            self.assertEqual(training[name], model[name], name)
        self.assertEqual(training["biofp_aux_weight"], "0.0")
        self.assertEqual(training["training_stage"], "'joint'")
        self.assertEqual(training["pooling"], "'mean'")
        for name in ("capability_vector_dim", "text_vector_dim", "validation_similarity"):
            self.assertEqual(training[name], "None")

    def test_initializers_keep_explicit_signatures_and_capture_original_hparams(self):
        for filename, cls in (
            ("protein_pooling_lightning_module.py", "ProteinPooledLitModule"),
            ("models/dual.py", "ProteinPooledDualModel"),
        ):
            init = methods(filename, cls)["__init__"]
            names = {arg.arg for arg in init.args.args}
            self.assertTrue({"pooling", "enzyme_input_mode", "sleec_mode"} <= names)
        init = methods("protein_pooling_lightning_module.py", "ProteinPooledLitModule")["__init__"]
        self.assertEqual(ast.unparse(init.body[0]), "super().__init__()")
        self.assertEqual(ast.unparse(init.body[1]), "self.save_hyperparameters()")
        parameter_names = {arg.arg for arg in init.args.args}
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in parameter_names
                for target in node.targets
            ):
                self.assertGreater(node.lineno, init.body[1].lineno)

    def test_dispatch_matrix_encodes_each_tower_once_and_preserves_detail_contracts(self):
        dispatch = load_dispatch()
        pools = ("mean", "attention", "sleec", "sleec_guided_attention")
        modes = ("standard", "raw_mean_sleec_biofp_split", "raw_mean_sleec_biological_factorized")
        stages = ("joint", "e2r_adapter", "r2e_adapter", "bidirectional_adapters")
        flags = tuple(itertools.product((False, True), repeat=7))
        for pool, mode, stage, bits, query_tuple in itertools.product(
            pools, modes, stages, flags, (False, True)
        ):
            stats, reaction, residue, biofp, alignment, block_kl, adapter = bits
            towers = RecordingTowers(pool, mode, query_tuple)
            state = SimpleNamespace(
                model=towers,
                training_stage=stage,
                lambda_residue=float(residue),
                biofp_aux_weight=float(biofp),
                cross_tower_alignment_weight=float(alignment),
                enzyme_block_weight_kl_weight=float(block_kl),
                r2e_identity_weight=float(adapter),
                reaction_attention_entropy_weight=float(reaction),
                reaction_chemistry_consistency_weight=0.0,
                reaction_residual_identity_weight=0.0,
                _reaction_fingerprint_attention_stats=report("fingerprint"),
                _reaction_multimodal_attention_stats=report("multimodal"),
                _query_attention_stats=report("query"),
                _sleec_stats=report("sleec"),
                _sleec_guided_attention_stats=report("sleec_guided_attention"),
                _attention_stats=report("attention"),
            )
            query = {"query": "features"} if alignment else "query_features"
            direction = "enzyme_to_reaction" if stage == "e2r_adapter" else "reaction_to_enzyme"
            target = {
                "residue_padding_mask": "mask",
                "score_residue_embeddings": "scores",
                "score_residue_padding_mask": "score_mask",
                "capability_vectors": "capabilities",
                "capability_mask": None,
                "factorized_capability_vectors": {"mechanism": "value"},
                "factorized_capability_masks": None,
                "text_vectors": None,
                "text_mask": None,
                "retrieval_direction": (
                    "enzyme_to_reaction"
                    if stage in {"e2r_adapter", "bidirectional_adapters"}
                    else "reaction_to_enzyme"
                ),
            }
            original_target = copy.deepcopy(target)
            result = dispatch(
                state,
                query,
                "residues",
                target,
                query_retrieval_direction=direction,
                return_attention_stats=stats,
            )
            self.assertEqual(target, original_target)
            self.assertEqual([call[0] for call in towers.calls], ["query", "target"])
            wants_query = stats or reaction
            expected_query = {"retrieval_direction": direction}
            if wants_query:
                expected_query["return_attention"] = True
            self.assertEqual(towers.calls[0], ("query", query, expected_query))
            if wants_query:
                wants_pool = pool in {"sleec", "sleec_guided_attention"}
                wants_attention = pool == "attention"
            else:
                wants_pool = (
                    (pool == "sleec" and residue)
                    or (pool == "sleec_guided_attention" and (alignment or block_kl))
                    or (biofp and mode != "standard")
                    or (adapter and stage in {"r2e_adapter", "bidirectional_adapters"})
                )
                wants_attention = False
            expected_target = dict(target)
            if wants_query and pool in {"mean", "attention"}:
                del expected_target["score_residue_embeddings"]
                del expected_target["score_residue_padding_mask"]
            if wants_pool:
                expected_target["return_pooling_details"] = True
            elif wants_attention:
                expected_target["return_attention"] = True
            self.assertEqual(towers.calls[1], ("target", "residues", expected_target))
            self.assertEqual(
                result[:4],
                (
                    "query_embedding",
                    "target_embedding",
                    "query_details" if wants_query and query_tuple else None,
                    "target_details" if wants_pool else None,
                ),
            )
            expected_stats = {}
            if stats and query_tuple:
                expected_stats.update(fingerprint=("query_details",), multimodal=("query_details",))
                if isinstance(query, dict):
                    expected_stats["query"] = ("query_details", query)
            if wants_query and pool != "mean":
                expected_stats[pool] = ("target_details", "mask")
            self.assertEqual(result[4], expected_stats)

    def test_each_reaction_regularizer_requests_attention_without_logging_query_stats(self):
        dispatch = load_dispatch()
        for field in (
            "reaction_attention_entropy_weight",
            "reaction_chemistry_consistency_weight",
            "reaction_residual_identity_weight",
        ):
            state = SimpleNamespace(
                model=RecordingTowers("mean", "standard", True),
                training_stage="joint",
                r2e_identity_weight=0,
                biofp_aux_weight=0,
                lambda_residue=0,
                cross_tower_alignment_weight=0,
                enzyme_block_weight_kl_weight=0,
                reaction_attention_entropy_weight=0,
                reaction_chemistry_consistency_weight=0,
                reaction_residual_identity_weight=0,
            )
            setattr(state, field, 0.2)
            result = dispatch(
                state,
                "queries",
                "residues",
                {
                    "residue_padding_mask": None,
                    "score_residue_embeddings": None,
                    "score_residue_padding_mask": None,
                },
                query_retrieval_direction="reaction_to_enzyme",
                return_attention_stats=False,
            )
            self.assertEqual(result[2], "query_details")
            self.assertEqual(result[4], {})


if __name__ == "__main__":
    unittest.main()
