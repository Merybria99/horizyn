"""Guard error precedence and legacy edge cases when sharing validation rules."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from horizyn.config import DotDict, apply_overrides, parse_overrides, validate_config


def config():
    return DotDict(
        data={
            "train_pairs_path": "train.csv",
            "train_reactions_path": "reactions.csv",
            "test_pairs_path": "validation.csv",
            "test_reactions_path": "validation_reactions.csv",
            "protein_residue_embeds_path": "residues.h5",
        },
        model={
            "name": "ProteinPooledDualModel",
            "query_encoder_dims": [16, 8],
            "target_encoder_dims": [32, 8],
            "embedding_dim": 8,
        },
        training={"max_epochs": 5},
    )


class ConfigValidationSectionsTests(unittest.TestCase):
    def assert_error(self, value, expected):
        with self.assertRaises(ValueError) as caught:
            validate_config(value)
        self.assertEqual(str(caught.exception), expected)

    def test_minimal_config_remains_valid(self):
        validate_config(config())

    def test_numeric_cli_overrides_pass_strict_count_validation(self):
        value = config()
        apply_overrides(value, parse_overrides([
            "--data.num_workers", "0", "--data.prefetch_factor=1",
            "--data.worker_num_threads", "1", "--training.cpu_num_threads=1",
            "--training.max_epochs", "1", "--data.pin_memory", "false",
        ]))
        validate_config(value)
        self.assertEqual(value.data.num_workers, 0)
        self.assertIs(type(value.data.num_workers), int)
        self.assertIs(type(value.data.prefetch_factor), int)
        self.assertIs(value.data.pin_memory, False)

    def test_data_errors_precede_model_and_experimental_errors(self):
        value = config()
        value.data.protein_capability_vectors_path = []
        value.model.query_encoder_dims = None
        value.model.e2r_adapter = "invalid"
        self.assert_error(
            value, "'data.protein_capability_vectors_path' must be a string path or null"
        )
        del value.data["protein_capability_vectors_path"]
        self.assert_error(value, "'model.query_encoder_dims' must be a list, got NoneType")

    def test_e2r_errors_precede_r2e_and_prototype_errors(self):
        value = config()
        value.model.e2r_adapter = {"enabled": "yes"}
        value.model.r2e_adapter = {"enabled": "yes"}
        value.model.enzyme_prototypes = {"count": 0}
        self.assert_error(value, "'model.e2r_adapter.enabled' must be a boolean")
        value.model.e2r_adapter.enabled = False
        self.assert_error(value, "'model.r2e_adapter.enabled' must be a boolean")

    def test_inactive_adapters_still_validate_scalar_options(self):
        for name in ("e2r_adapter", "r2e_adapter"):
            for key, bad, message in (
                ("enabled", None, "must be a boolean"),
                ("hidden_dim", 0, "must be a positive integer"),
                ("dropout", True, "must be in [0, 1]"),
                ("dropout", float("nan"), "must be in [0, 1]"),
                ("gate_init", 0.0, "must be in (0, 1)"),
                ("gate_init", 1.0, "must be in (0, 1)"),
            ):
                with self.subTest(adapter=name, field=key, value=bad):
                    value = config()
                    value.model[name] = {"enabled": False, key: bad}
                    self.assert_error(value, f"'model.{name}.{key}' {message}")

    def test_adapter_integer_bool_behavior_is_not_silently_changed(self):
        # The existing rule uses isinstance(value, int), which accepts True.
        # Tightening validation would be a separate behavioral change.
        for name in ("e2r_adapter", "r2e_adapter"):
            value = config()
            value.model[name] = {"hidden_dim": True}
            validate_config(value)

    def test_optional_paths_keep_explicit_null(self):
        value = config()
        for key in (
            "protein_capability_vectors_path",
            "protein_text_vectors_path",
            "protein_biofp_targets_path",
            "reaction_chemistry_vectors_path",
        ):
            value.data[key] = None
        validate_config(value)

    def test_missing_policy_validation_keeps_order(self):
        value = config()
        value.data.capability_missing_policy = None
        value.data.biofp_missing_policy = "invalid"
        self.assert_error(value, "'data.capability_missing_policy' must be 'zero_with_mask'")
        del value.data["capability_missing_policy"]
        self.assert_error(value, "'data.biofp_missing_policy' must be 'zero_with_mask'")


if __name__ == "__main__":
    unittest.main()
