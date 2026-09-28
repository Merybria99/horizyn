import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("read_ahead", [0, 1])
@pytest.mark.parametrize("identity_only", [False, True])
def test_contiguous_copy_preserves_selected_ragged_rows_and_published_receipt(tmp_path, read_ahead, identity_only):
    source = tmp_path / "source.h5"
    offsets = np.array([0, 2, 5, 9, 10, 16, 19])
    values = np.arange(19 * 1024, dtype=np.float32).reshape(19, 1024).astype(np.float16)
    with h5py.File(source, "w") as h:
        h.create_dataset("ids", data=np.array([f"p{i}" for i in range(6)], dtype=object), dtype=h5py.string_dtype())
        h.create_dataset("offsets", data=offsets)
        h.create_dataset("vectors", data=values)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"proteins": ["p5", "p0", "p4", "p1"]}))
    out, link = tmp_path / "out.h5", tmp_path / "published.h5"
    script = Path(__file__).resolve().parents[2] / "scripts/generalization_clipzyme_compact_residues.py"
    subprocess.run([sys.executable, str(script), "--source", str(source), "--catalog", str(catalog),
                    "--output", str(out), "--block-proteins", "4", "--purpose", "screening",
                    "--read-ahead-mib", str(read_ahead),
                    "--publish-link", str(link),
                    *(["--source-identity-only"] if identity_only else [])], check=True, capture_output=True)
    with h5py.File(link) as h:
        assert h["ids"].asstr()[:].tolist() == ["p0", "p1", "p4", "p5"]
        np.testing.assert_array_equal(h["vectors"][:], np.concatenate([values[:5], values[10:19]]))
        np.testing.assert_array_equal(h["offsets"][:], [0, 2, 5, 11, 14])
        assert h["vectors"].dtype == np.float16
    receipt = json.loads(link.with_suffix(".receipt.json").read_text())
    assert receipt["protein_count"] == 4 and receipt["purpose"] == "screening"
    assert receipt["output_mtime_ns"] == link.stat().st_mtime_ns
    assert (receipt["source_sha256"] is None) == identity_only
    assert receipt["source_identity"]["mtime_ns"] == source.stat().st_mtime_ns
