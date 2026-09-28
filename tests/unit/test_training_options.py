"""CPU-only regression checks for training configuration wiring.

These tests intentionally import neither PyTorch nor the training entry point.
They can also run directly with ``python tests/unit/test_training_options.py``.
"""

import ast
import contextlib
import copy
import inspect
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from horizyn.config import DotDict
from horizyn.training_options import (
    _with_defaults,
    protein_pooling_model_kwargs as model_options,
    reaction_data_module_kwargs as data_options,
    resolve_reaction_chirality,
)


def minimal_config():
    return DotDict(
        seed=17,
        data={
            "train_pairs_path": "train.csv",
            "train_reactions_path": "reactions.csv",
            "protein_residue_embeds_path": "proteins.h5",
            "train_batch_size": 200,
        },
        model={
            "query_encoder_dims": [16, 8],
            "target_encoder_dims": [32, 8],
            "embedding_dim": 8,
        },
        training={
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "loss": {"beta": 10.0},
        },
    )


class TrainingOptionsTests(unittest.TestCase):
    def test_neighborhood_regularization_defaults_off_and_weights_are_independent(self):
        config = minimal_config()
        options = model_options(config)
        self.assertEqual(options["protein_geometry_weight"], 0.0)
        self.assertEqual(options["reaction_geometry_weight"], 0.0)
        config.training.protein_geometry = {"weight": 0.2}
        config.training.reaction_geometry = {"weight": 0.1}
        options = model_options(config)
        self.assertEqual(options["protein_geometry_weight"], 0.2)
        self.assertEqual(options["reaction_geometry_weight"], 0.1)

    def test_model_builder_needs_only_config(self):
        self.assertEqual(list(inspect.signature(model_options).parameters), ["config"])

    def test_forwarding_keeps_null_false_zero_and_ignores_unlisted_fields(self):
        self.assertEqual(
            _with_defaults({"a": None, "b": False, "c": 0, "unused": 9}, a=1, b=True, c=2, d=3),
            {"a": None, "b": False, "c": 0, "d": 3},
        )

    def test_data_defaults_and_indexed_negative_fraction(self):
        config = minimal_config()
        options = data_options(config)
        self.assertEqual(options["test_pairs_path"], "train.csv")
        self.assertEqual(options["test_reactions_path"], "reactions.csv")
        self.assertEqual(options["typed_negative_positive_fraction"], 0.5)
        self.assertEqual(options["typed_negative_seed"], 17)
        config.data.indexed_pairs_dir = "index"
        self.assertEqual(data_options(config)["typed_negative_positive_fraction"], 0.85)
        config.data.typed_negative_positive_fraction = 0.9
        self.assertEqual(data_options(config)["typed_negative_positive_fraction"], 0.9)

    def test_prefixed_forwarding_uses_unprefixed_inputs_and_preserves_values(self):
        section = {"enabled": False, "dim": None, "dropout": 0, "head_enabled": True, "unused": 9}
        before = copy.deepcopy(section)
        result = _with_defaults(section, prefix="head_", enabled=True, dim=8, dropout=0.1, layers=2)
        self.assertEqual(
            list(result.items()),
            [
                ("head_enabled", False),
                ("head_dim", None),
                ("head_dropout", 0),
                ("head_layers", 2),
            ],
        )
        self.assertEqual(section, before)

    def test_cli_dimension_and_similarity_defaults_differ_from_direct_constructors(self):
        options = model_options(minimal_config())
        self.assertEqual(options["capability_vector_dim"], 256)
        self.assertEqual(options["text_vector_dim"], 768)
        self.assertEqual(options["validation_similarity"], "cosine")

    def test_validation_path_precedence_and_explicit_none(self):
        config = minimal_config()
        config.data.test_pairs_path = "test.csv"
        self.assertEqual(data_options(config)["test_pairs_path"], "test.csv")
        config.data.validation_pairs_path = "validation.csv"
        self.assertEqual(data_options(config)["test_pairs_path"], "validation.csv")
        config.data.validation_pairs_path = None
        self.assertIsNone(data_options(config)["test_pairs_path"])

    def test_resolved_chirality_aliases_and_source_replay(self):
        config = minimal_config()
        config.data.update(
            reaction_chiro_embeds_path="chiro.h5",
            reaction_chiro_dim=64,
            reaction_use_chiro=False,
        )
        config.training.validation_enabled = False
        options = data_options(
            config,
            replay_config={"train_pairs_path": "source.csv"},
            replay_fraction=0.15,
            replay_seed=123,
        )
        self.assertEqual(options["reaction_chienn_embeds_path"], "chiro.h5")
        self.assertEqual(options["reaction_chienn_dim"], 64)
        self.assertFalse(options["reaction_use_chienn"])
        self.assertEqual(options["replay_config"], {"train_pairs_path": "source.csv"})
        self.assertEqual(options["replay_fraction"], 0.15)
        self.assertEqual(options["replay_seed"], 123)
        self.assertFalse(options["validation_enabled"])

    def test_chirality_alias_precedence(self):
        config = minimal_config()
        config.data.update(
            reaction_chiro_embeds_path=None,
            reaction_chirality_embeds_path="preferred.h5",
            reaction_chienn_embeds_path="legacy.h5",
            reaction_chiro_dim=None,
            reaction_chirality_dim=64,
            reaction_chienn_dim=32,
            reaction_use_chiro=True,
            reaction_allow_missing_chiro=False,
            reaction_allow_missing_chienn=True,
        )
        config.model.update(reaction_use_chiro=None, reaction_use_chirality=False)
        chirality = resolve_reaction_chirality(config)
        self.assertEqual(chirality, ("preferred.h5", 64, False, False))
        self.assertEqual(data_options(config)["reaction_chienn_dim"], 64)
        self.assertEqual(model_options(config)["reaction_chienn_dim"], 64)

    def test_default_replay_seed_uses_config(self):
        config = minimal_config()
        self.assertEqual(data_options(config)["replay_seed"], 17)
        self.assertEqual(data_options(config, replay_seed=0)["replay_seed"], 0)
        del config["seed"]
        self.assertEqual(data_options(config)["replay_seed"], 42)

    def test_sampler_selection(self):
        config = minimal_config()
        switches = {
            "reaction_degree_balanced": "reaction_balanced_sampling",
            "enzyme_grouped": "enzyme_grouped_sampling",
            "hypergraph_grouped": "hypergraph_sampling",
        }
        for name in ["shuffle", *switches]:
            with self.subTest(sampler=name):
                config.data.train_sampler = {"name": name}
                options = data_options(config)
                for sampler, key in switches.items():
                    self.assertEqual(options[key], sampler == name)

    def test_model_optimizer_and_loss_defaults(self):
        options = model_options(minimal_config())
        self.assertEqual(options["query_encoder_dims"], [16, 8])
        self.assertEqual(options["target_encoder_dims"], [32, 8])
        self.assertEqual(options["learning_rate"], 1e-4)
        self.assertEqual(options["weight_decay"], 0.01)
        self.assertEqual(options["beta"], 10.0)
        self.assertEqual(options["loss_name"], "FullBatchMLNCELoss")
        self.assertEqual(options["lambda_r2e"], 0.5)
        self.assertEqual(options["lambda_e2r"], 0.5)
        self.assertEqual(options["biofp_aux_weight"], 0.0)
        self.assertEqual(options["beta_min"], -float("inf"))
        self.assertEqual(options["beta_max"], float("inf"))
        self.assertIsNone(options["query_encoder_checkpoint_path"])

    def test_loss_and_performance_overrides(self):
        config = minimal_config()
        config.training.loss.update(
            name="SampledMultiPositiveInfoNCELoss",
            beta_r2e=4.0,
            beta_e2r=8.0,
            lambda_r2e=0.7,
            lambda_e2r=0.3,
            sampled_require_both_directions=True,
            biofp_aux_weight=0.2,
        )
        config.training.fused_adamw = True
        config.training.contrastive_fp32 = True
        options = model_options(config)
        for name, value in config.training.loss.items():
            self.assertEqual(options["loss_name" if name == "name" else name], value)
        self.assertTrue(options["fused_adamw"])
        self.assertTrue(options["contrastive_fp32"])

    def test_symmetric_reaction_blocks(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = minimal_config()
                config.model.enzyme_block_fusion = {"dims": [2, 3, 3], "weights": [0.2, 0.3, 0.5]}
                config.model.reaction_multimodal_attention = {
                    "symmetric_output_blocks": {"enabled": enabled}
                }
                options = model_options(config)
                self.assertEqual(
                    options["reaction_output_block_dims"], [2, 3, 3] if enabled else None
                )
                self.assertEqual(
                    options["reaction_output_block_weights"],
                    [0.2, 0.3, 0.5] if enabled else None,
                )

    def test_builders_do_not_mutate_config(self):
        config = minimal_config()
        original = copy.deepcopy(config)
        data_options(config)
        model_options(config)
        self.assertEqual(config, original)

    def test_experimental_heads_retain_inactive_defaults_and_overrides(self):
        config = minimal_config()
        defaults = model_options(config)
        self.assertFalse(defaults["e2r_adapter_enabled"])
        self.assertFalse(defaults["r2e_adapter_enabled"])
        self.assertEqual(defaults["enzyme_prototype_count"], 1)
        self.assertFalse(defaults["biological_residual_enabled"])
        config.model.e2r_adapter = {"enabled": True, "hidden_dim": 128}
        config.model.r2e_adapter = {"enabled": True, "gate_init": 0.3}
        config.model.enzyme_prototypes = {"count": 4}
        config.model.biological_residual = {"enabled": True, "fusion_alpha": 0.2}
        configured = model_options(config)
        self.assertEqual(list(configured), list(defaults))
        for key, value in {
            "e2r_adapter_enabled": True,
            "e2r_adapter_hidden_dim": 128,
            "r2e_adapter_enabled": True,
            "r2e_adapter_gate_init": 0.3,
            "enzyme_prototype_count": 4,
            "biological_residual_enabled": True,
            "biological_residual_fusion_alpha": 0.2,
        }.items():
            self.assertEqual(configured[key], value)

    def test_console_report_has_no_dependency_on_main_locals(self):
        # Execute the actual reporter without importing the ML entry point.
        source = ast.parse((ROOT / "horizyn/pipelines/training.py").read_text())
        reporter = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "_print_training_config"
        )
        namespace = {
            "resolve_reaction_chirality": resolve_reaction_chirality,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        }
        exec(compile(ast.Module(body=[reporter], type_ignores=[]), "reporter", "exec"), namespace)
        config = minimal_config()
        config.training.max_epochs = 30
        config.model.name = "ProteinPooledDualModel"
        config.logging = {"log_dir": "logs", "checkpoint_dir": "checkpoints"}
        for enabled in (True, False):
            with self.subTest(validation=enabled):
                config.training.validation_enabled = enabled
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    namespace["_print_training_config"](
                        config, {"enabled": False, "mode": "disabled"}
                    )
                self.assertIn("Epochs: 30", output.getvalue())
                self.assertIn(f"Validation enabled: {enabled}", output.getvalue())

    def test_constructor_keywords_match_actual_apis(self):
        # Check all keyword names without importing GPU/model dependencies.
        for filename, class_name, options in (
            (
                "reaction_conditioned_data_module.py",
                "ReactionConditionedDataModule",
                data_options(minimal_config()),
            ),
            (
                "protein_pooling_lightning_module.py",
                "ProteinPooledLitModule",
                model_options(minimal_config()),
            ),
        ):
            with self.subTest(constructor=class_name):
                tree = ast.parse((ROOT / "horizyn" / filename).read_text())
                cls = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                init = next(
                    node
                    for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                )
                accepted = {arg.arg for arg in [*init.args.args, *init.args.kwonlyargs]}
                self.assertFalse(set(options) - accepted)


if __name__ == "__main__":
    unittest.main()
