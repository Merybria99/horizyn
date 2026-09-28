"""Small fixtures for the read-only existing-cache screening command."""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_reactzyme_prott5_cache.py"
SPEC = importlib.util.spec_from_file_location("reactzyme_cache_screen", SCRIPT)
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidates = self.root / "candidates"
        for protocol in ("reaction_smi", "enzyme_smi", "time"):
            directory = self.candidates / protocol
            directory.mkdir(parents=True)
            for part, text in (("train", "p0\np1\n"), ("validation", "p1\n"), ("test", "p2\n")):
                (directory / f"{part}_candidate_ids.txt").write_text(text)

    def tearDown(self):
        self.temporary.cleanup()

    def test_union_and_split_counts(self):
        required, counts, identities = screen.required_ids(self.candidates, 3)
        self.assertEqual(required, {"p0", "p1", "p2"})
        self.assertEqual(len(counts), 9)
        self.assertEqual(counts["reaction_smi/train"], 2)
        self.assertEqual(len(identities), 9)

    def test_duplicate_candidates_rejected(self):
        (self.candidates / "time/test_candidate_ids.txt").write_text("p2\np2\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            screen.required_ids(self.candidates, 3)

    def test_missing_candidate_list_rejected(self):
        (self.candidates / "time/test_candidate_ids.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            screen.required_ids(self.candidates, 3)

    def test_wrong_expected_union_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            screen.required_ids(self.candidates, 4)


@unittest.skipUnless(importlib.util.find_spec("h5py") and importlib.util.find_spec("numpy"),
                     "HDF5 fixture tests require h5py and numpy")
class Hdf5Tests(CandidateTests):
    def setUp(self):
        super().setUp()
        import h5py
        import numpy as np
        self.h5py, self.np = h5py, np
        self.cache = self.root / "cache.h5"
        with h5py.File(self.cache, "w") as handle:
            handle.create_dataset("ids", data=["p0", "p1", "p2"], dtype=h5py.string_dtype())
            handle.create_dataset("offsets", data=[0, 2, 5, 6])
            handle.create_dataset("vectors", data=np.ones((6, 1024), dtype=np.float16))
            handle.attrs["model_name"] = "fixture"
        self.args = argparse.Namespace(cache=self.cache, candidate_root=self.candidates,
                                       expected_proteins=3, samples=2)

    def check(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            screen.check(self.args)
        return captured.getvalue()

    def test_valid_cache_screen_is_read_only_and_explicitly_partial(self):
        identity = screen.file_identity(self.cache)
        result = self.check()
        self.assertEqual(screen.file_identity(self.cache), identity)
        self.assertIn("SCREENING PASSED", result)
        self.assertIn('"full_vector_scan": false', result)
        self.assertIn('"representation_provenance_verified": false', result)
        self.assertIn('"sampled_proteins": 3', result)

    def test_missing_required_id_rejected(self):
        with self.h5py.File(self.cache, "r+") as handle:
            handle["ids"][2] = "unknown"
        with self.assertRaisesRegex(ValueError, "Missing 1 required"):
            self.check()

    def test_bad_offsets_rejected(self):
        with self.h5py.File(self.cache, "r+") as handle:
            handle["offsets"][2] = 1
        with self.assertRaisesRegex(ValueError, "offset"):
            self.check()

    def test_nan_sample_rejected(self):
        with self.h5py.File(self.cache, "r+") as handle:
            handle["vectors"][2, 0] = self.np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.check()

    def test_zero_sample_rejected(self):
        with self.h5py.File(self.cache, "r+") as handle:
            handle["vectors"][2:5] = 0
        with self.assertRaisesRegex(ValueError, "All-zero"):
            self.check()

    def test_missing_virtual_source_rejected(self):
        absent = self.root / "missing_source.h5"
        layout = self.h5py.VirtualLayout(shape=(6, 1024), dtype=self.np.float16)
        layout[:] = self.h5py.VirtualSource(str(absent), "vectors", shape=(6, 1024))
        with self.h5py.File(self.cache, "r+") as handle:
            del handle["vectors"]
            handle.create_virtual_dataset("vectors", layout)
        with self.assertRaises(FileNotFoundError):
            self.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
