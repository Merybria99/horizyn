import importlib.util
import json
from pathlib import Path
import sys

import pytest


def selector():
    path = Path(__file__).resolve().parents[2] / "scripts/generalization_clipzyme_checkpoint_select.py"
    spec = importlib.util.spec_from_file_location("clipzyme_checkpoint_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation(checkpoint, bedroc85, mrr):
    return dict(schema="clipzyme_f3_full_library_validation_v1",
                validation_only=True, test_labels_read=False,
                selection_metric="table1.bedroc85", selection_value=bedroc85,
                association_manifest_sha256="same-train-dev", checkpoint_sha256=checkpoint,
                summary={table: dict(queries=queries, candidate_ids=candidates,
                    bedroc85=bedroc85, bedroc20=.7, **{"ef0.05": 15., "ef0.1": 8., "mrr": mrr})
                    for table, queries, candidates in (("table1", 2652, 261907), ("table2", 2216, 252113))})


def run_selector(tmp_path, monkeypatch, records):
    output = tmp_path / "selected.json"
    argv = ["checkpoint_select"]
    for index, record in enumerate(records):
        path = tmp_path / f"evaluation{index}.json"
        path.write_text(json.dumps(record))
        argv.extend(["--evaluation", str(path)])
    monkeypatch.setattr(sys, "argv", argv + ["--output", str(output)])
    selector().main()
    return json.loads(output.read_text())


def test_selection_uses_bedroc85_and_reports_all_four_metrics(tmp_path, monkeypatch):
    selected = run_selector(tmp_path, monkeypatch, [
        validation("high-mrr", .4, .99), validation("high-bedroc85", .5, .1)])
    assert selected["selected"]["checkpoint_sha256"] == "high-bedroc85"
    assert selected["reported_metrics"] == ["bedroc85", "bedroc20", "ef0.05", "ef0.1"]
    for row in selected["predeclared_grid"]:
        for values in row["summary"].values():
            assert set(values) == set(selected["reported_metrics"])


@pytest.mark.parametrize("invalid", ["stale-selection-value", "nonfinite"])
def test_invalid_screening_metrics_cannot_select_a_checkpoint(tmp_path, monkeypatch, invalid):
    record = validation("checkpoint", .5, .1)
    if invalid == "stale-selection-value":
        record["selection_value"] = .9
    else:
        record["summary"]["table2"]["ef0.1"] = float("nan")
    with pytest.raises(ValueError):
        run_selector(tmp_path, monkeypatch, [record])
    assert not (tmp_path / "selected.json").exists()
