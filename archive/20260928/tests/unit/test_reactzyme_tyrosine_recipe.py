import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_reactzyme_tyrosine_recipe.py"
SPEC = importlib.util.spec_from_file_location("reactzyme_recipe", SCRIPT)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class RecipeTests(unittest.TestCase):
    def test_source_aggregate_is_excluded_if_any_edge_is_not_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            source.write_text("reaction_id,protein_id\na,prot_one\nb,prot_one\na,prot_two\n")
            safe, unsafe = recipe.safe_label_proteins(source, {("a", "prot_one"), ("a", "prot_two")})
            self.assertEqual(safe, {"prot_two"})
            self.assertEqual(unsafe, {"prot_one"})

    def test_alias_csv_preserves_only_requested_fields_and_original_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source.csv", Path(directory) / "dest.csv"
            original = "reaction_id,protein_id,protein_sequence\na,prot_one,ABCD\n"
            source.write_text(original)
            recipe.alias_csv(source, dest, ("reaction_id", "protein_id"))
            with dest.open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"reaction_id": "a_f", "protein_id": "prot_one"}])
            self.assertEqual(source.read_text(), original)
            with self.assertRaises(FileExistsError):
                recipe.alias_csv(source, dest, ("reaction_id", "protein_id"))

    def test_double_alias_rejected(self):
        for value in ("a_f", "a_r", ""):
            with self.assertRaises(ValueError):
                recipe.forward_id(value)

    def test_training_pair_alias_preserves_original_unique_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source.csv", Path(directory) / "dest.csv"
            original = ("pr_id,reaction_id,protein_id,protein_sequence\n"
                        "2,a,prot_one,ABCD\n17,a,prot_two,EFGH\n")
            source.write_text(original)
            recipe.alias_csv(source, dest, recipe.PAIR_COLUMNS)
            with dest.open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [
                    {"pr_id": "2", "reaction_id": "a_f", "protein_id": "prot_one"},
                    {"pr_id": "17", "reaction_id": "a_f", "protein_id": "prot_two"},
                ])
            self.assertEqual(source.read_text(), original)

    def test_missing_empty_or_duplicate_pair_keys_rejected(self):
        for original in (
            "reaction_id,protein_id\na,prot_one\n",
            "pr_id,reaction_id,protein_id\n,a,prot_one\n",
            "pr_id,reaction_id,protein_id\n2,a,prot_one\n2,b,prot_two\n",
        ):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                source, dest = Path(directory) / "source.csv", Path(directory) / "dest.csv"
                source.write_text(original)
                with self.assertRaises(ValueError):
                    recipe.alias_csv(source, dest, recipe.PAIR_COLUMNS)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("h5py"),
                         "Actual CSV loader requires the ML environment")
    def test_actual_validation_csv_loader_accepts_prepared_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source.csv", Path(directory) / "dest.csv"
            source.write_text("pr_id,reaction_id,protein_id\n2,a,prot_one\n17,a,prot_two\n")
            recipe.alias_csv(source, dest, recipe.PAIR_COLUMNS)
            self.assertEqual(recipe.check_pair_loader(dest), 2)
            from horizyn.datasets.csv import CSVDataset
            pairs = CSVDataset(str(dest), key_column="pr_id",
                               columns=["reaction_id", "protein_id"],
                               rename_map={"reaction_id": "query_id", "protein_id": "target_id"})
            self.assertEqual(pairs["17"], {"query_id": "a_f", "target_id": "prot_two"})
            broken = Path(directory) / "previously_broken.csv"
            broken.write_text("reaction_id,protein_id\na_f,prot_one\n")
            with self.assertRaisesRegex(ValueError, "pr_id"):
                recipe.check_pair_loader(broken)

    def test_config_copies_recipe_without_horizyn_data_or_warmstart(self):
        base = {"data": {"train_pairs_path": "old/train.csv", "validation_retrieval_candidate_ids_path": "candidates.txt",
                         "protein_residue_embeds_path": "cache.h5"},
                "model": {"sleec_pooling": {"checkpoint_path": "sleec.ckpt"},
                          "reaction_multimodal_attention": {"side_composition": "molecule_set"}}}
        source = {"training": {"max_epochs": 5, "learning_rate": 0.0001, "weight_decay": 0.01,
                               "loss": {"name": "BidirectionalSampledMultiPositiveInfoNCELoss", "biofp_aux_weight": 0.0},
                               "devices": 4, "init_from_checkpoint": "do-not-load.ckpt",
                               "validation_retrieval_query_ids_path": "horizyn-queries.json"},
                  "logging": {"wandb": {"enabled": True}}}
        output = Path("/example/run")
        config = recipe.config_for_run(base, source, output)
        self.assertEqual(config["training"]["loss"], source["training"]["loss"])
        self.assertEqual(config["data"]["train_batch_size"] * config["training"]["devices"], 400)
        self.assertEqual(config["data"]["typed_negative_positive_fraction"], 0.85)
        self.assertIsNone(config["training"]["validation_interval_steps"])
        self.assertNotIn("init_from_checkpoint", config["training"])
        self.assertNotIn("validation_retrieval_query_ids_path", config["training"])
        self.assertEqual(config["model"]["reaction_multimodal_attention"]["side_composition"], "molecule_set")
        self.assertEqual(base["data"]["train_pairs_path"], "old/train.csv")
        self.assertEqual(source["training"]["devices"], 4)
        self.assertFalse(config["logging"]["wandb"]["enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
