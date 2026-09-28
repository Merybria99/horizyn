"""CPU-only policy/configuration tests; no CUDA or torch imports required."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "arginine_recipe", ROOT / "scripts/prepare_reactzyme_arginine_classification.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def policy_module():
    # Only the unused PyTorch base classes/distributed discovery are replaced.
    # Execute the actual NumPy index, policy and sampler implementation.
    path = ROOT / "horizyn/datasets/indexed_pairs.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(name.name.startswith("torch") for name in node.names)
        or isinstance(node, ast.ImportFrom) and node.module.startswith("torch"))]
    scope = dict(Dataset=object, BatchSampler=object,
                 dist=SimpleNamespace(is_available=lambda: False))
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


class ArginineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def index(self, no_negatives=False):
        module = policy_module()
        directory = self.output / "index"
        module.write_indexed_pairs(
            directory, query_ids=np.array([f"q{i}" for i in range(4)]),
            protein_ids=np.array([f"p{i}" for i in range(8)]),
            pairs=np.array([(i // 2, i) for i in range(8)]),
            protein_ec=np.repeat(np.arange(4), 2), ec_prefix=np.array([0, 0, 1, 1]),
            query_ec_indptr=np.arange(5) * (4 if no_negatives else 1),
            query_ec=np.tile(np.arange(4), 4) if no_negatives else np.arange(4),
            mechanism_bits=np.ones(8, dtype=np.uint32),
            native_cofactor_bits=np.ones(8, dtype=np.uint32),
            reaction_cofactor_bits=np.zeros(8, dtype=np.uint32),
            ec_eligible=np.ones(8, dtype=bool), biological_eligible=np.ones(8, dtype=bool),
            query_signature=np.arange(4),
            provenance={"pair_scope": "train", "annotation_semantics": module.SEMANTICS},
            min_biofp_similarity=0.5,
        )
        return module, module.IndexedPairs(directory)

    def test_both_ratios_are_exact_deterministic_and_supported(self):
        module, index = self.index()
        for fraction, positive_count in ((0.85, 17), (0.5, 10)):
            seen = set()
            for rank in range(4):
                sampler = module.IndexedTypedNegativeBatchSampler(
                    index, 20, positive_fraction=fraction, rank=rank, world_size=4)
                first = list(sampler)
                sampler.set_epoch(0)
                self.assertEqual(first, list(sampler))
                sampler.set_epoch(1)
                self.assertNotEqual(first, list(sampler))
                for rows in first:
                    self.assertEqual(len(rows), 20)
                    self.assertEqual(sum(k == 0 for _, _, k in rows), positive_count)
                    positive = {(q, p) for q, p, k in rows if k == 0}
                    seen.update(positive)
                    for q, p, k in rows:
                        if k:
                            self.assertEqual(index.negative_kind(q, p), k)
                            self.assertTrue(any(a == q for a, _ in positive))
                            self.assertTrue(any(b == p for _, b in positive))
                        else:
                            self.assertTrue(index.is_positive(q, p))
            self.assertEqual(seen, set(map(tuple, index.pairs)))

    def test_50_50_refuses_unlabelled_cross_pairs_as_blanket_negatives(self):
        module, index = self.index(no_negatives=True)
        sampler = module.IndexedTypedNegativeBatchSampler(index, 20, positive_fraction=0.5)
        with self.assertRaisesRegex(RuntimeError, "No eligible"):
            next(iter(sampler))

    def test_50_50_batch_bounds(self):
        module, index = self.index()
        for batch in (True, 0, 2, 3, 5):
            with self.assertRaises(ValueError):
                module.IndexedTypedNegativeBatchSampler(index, batch, positive_fraction=0.5)
        # Unlike 85/15, divisibility by 20 is not required.
        self.assertEqual(module.IndexedTypedNegativeBatchSampler(index, 64, positive_fraction=0.5).positive_count, 32)

    def test_policy_grid_matches_scalar_and_biological_fallback_stays_50_50(self):
        module, index = self.index()
        qids, pids = list(index.query_ids), list(index.protein_ids)
        masks = index._negative_masks(qids, pids, set(qids), set(pids))
        for q in range(len(qids)):
            for p in range(len(pids)):
                for kind, mask in enumerate(masks, 1):
                    self.assertEqual(bool(mask[q, p]), index.negative_kind(q, p) == kind)
        index.biological_eligible = np.zeros(len(pids), dtype=bool)
        sampler = module.IndexedTypedNegativeBatchSampler(index, 20, positive_fraction=0.5)
        rows = next(iter(sampler))
        self.assertEqual(sum(k == 0 for _, _, k in rows), 10)
        self.assertEqual(sum(k == 2 for _, _, k in rows), 10)
        self.assertEqual(sampler.last_counts["category_fallbacks"], 5)
        self.assertEqual(len({(q,p) for q,p,k in rows if k == 2}), 10)

    def test_labels_keep_masks_confidence_and_drop_unsafe_ids(self):
        payload = {"ids": np.array(["safe", "unsafe", "not_train"])}
        for family, width in (("mechanism", 8), ("cofactor", 10)):
            for field in ("targets", "denominator", "confidence"):
                payload[f"{family}_{field}"] = np.full((3, width), 0.15)
            payload[f"{family}_mask"] = np.ones((3, width), dtype=bool)
            payload[f"{family}_mask"][0, 0] = False
        targets, coverage = recipe.filtered_targets(payload, {"safe", "not_train"}, {"safe", "unsafe"})
        self.assertEqual(targets["ids"].tolist(), ["safe"])
        self.assertFalse(targets["cofactor_mask"][0, 0])
        self.assertEqual(targets["cofactor_confidence"][0, 1], 0.15)
        self.assertEqual(coverage["mechanism"]["cells"], 7)

    def test_config_changes_are_scoped_and_lambda_is_not_silent(self):
        source = dict(data={}, model={"biofp": {"family_dims": {"mechanism": 8, "cofactor": 10}}},
                      training={"loss": {"name": "original"}, "init_from_checkpoint": "old.ckpt"}, logging={})
        before = copy.deepcopy(source)
        config = recipe.run_config(source, self.output, 800)
        self.assertEqual(source, before)
        self.assertEqual(config["training"]["loss"], dict(name="original", biofp_aux_weight=0.2,
                         biofp_family_weights={"mechanism": 0.5, "cofactor": 0.5}, biofp_confidence_cap=1.0))
        self.assertEqual(config["training"]["devices"], 4)
        self.assertEqual(config["training"]["max_epochs"], 30)
        self.assertEqual(config["training"]["max_steps"], -1)
        self.assertFalse(config["training"]["early_stopping"]["enabled"])
        self.assertNotIn("init_from_checkpoint", config["training"])
        self.assertEqual(config["data"]["typed_negative_positive_fraction"], 0.5)
        self.assertTrue(config["data"]["protein_biofp_targets_path"])
        self.assertEqual(config["model"], source["model"])

    def test_batch_choice_uses_throughput_all_ranks_and_memory_not_largest(self):
        for batch, seconds, memory in ((400, 1, 20), (800, 1.5, 40), (1600, 10, 60), (3200, 1, 95)):
            directory = self.output / f"pilot_{batch}"
            directory.mkdir()
            (self.output / f"batch_{batch}.yaml").write_text(f"batch: {batch}\n")
            for rank in range(4):
                (directory / f"timing_rank{rank}.json").write_text(json.dumps(dict(
                    complete=True, measured_steps=10, world_size=4, rank=rank,
                    mean_step_seconds=seconds, peak_cuda_allocated_bytes=memory,
                    classification=dict(weighted_biofp=0.1, biofp_mechanism_active=20,
                                        biofp_cofactor_active=80))))
        recipe.select_batch(self.output, 100)
        self.assertEqual(json.loads((self.output / "batch_selection.json").read_text())["batch_per_gpu"], 800)
        for path in self.output.glob("pilot_*/timing_rank*.json"):
            value = json.loads(path.read_text())
            value["classification"] = {}
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "No pilot passed"):
            recipe.select_batch(self.output, 100)


if __name__ == "__main__":
    unittest.main()
