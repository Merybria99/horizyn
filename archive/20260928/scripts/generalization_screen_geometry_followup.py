#!/usr/bin/env python3
"""Apply the fixed phase-2 study to the validation winner of two F3 studies."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

from generalization_screen_phase2_campaign import invoke, sha
from generalization_clipzyme_test_queue import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    plan = json.loads((out / "followup_plan.json").read_text())
    cross = out.parent
    grid = []
    for campaign in plan["campaigns"]:
        spec = json.loads((Path(campaign) / "protocol.json").read_text())
        for task in spec["tasks"]:
            for epoch in spec["validation_epochs"]:
                run = Path(task["run_root"])
                grid.append(dict(run=str(run), epoch=epoch,
                    config=task["train_config"],
                    summary=str(run / f"screen_epoch{epoch-1}/validation_evaluation/summary.json"),
                    checkpoint=str(run / f"checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt")))
    parent = cross / "sleec_multiview_phase2_v1"
    old = json.loads((parent / "protocol.json").read_text())
    grid.append(dict(run=str(Path(old["base_config"]).parents[1]), epoch=20,
        config=old["base_config"], summary=old["validation_summary"],
        checkpoint=old["base_checkpoint"], original_control=True))
    write_json(out / "frozen_selection_grid.json", dict(grid=grid,
        metric="full_library_validation.table1.bedroc85", test_used=False))
    while True:
        missing = [row["summary"] for row in grid if not Path(row["summary"]).exists()]
        write_json(out / "state.json", dict(stage="waiting_for_full_validation_grid",
            completed=len(grid)-len(missing), required=len(grid), missing=missing,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if not missing:
            break
        time.sleep(20)
    for row in grid:
        result = json.loads(Path(row["summary"]).read_text())
        if not result["validation_only"] or result["test_labels_read"]:
            raise ValueError("Selection requires validation-only evaluation")
        row["value"] = result["summary"]["table1"]["bedroc85"]
        row["summary_sha256"] = sha(row["summary"])
        if result["checkpoint_sha256"] != sha(row["checkpoint"]):
            raise ValueError("Validated checkpoint has changed")
    selected = max(grid, key=lambda row: row["value"])
    write_json(out / "base_selection.json", dict(grid=grid, selected=selected,
        selected_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False))
    if selected.get("original_control"):
        write_json(out / "complete.json", dict(original_control_retained=True,
            reason="The existing residue-view base leads the complete validation grid"))
        return
    write_json(out / "protocol.json", dict(base_checkpoint=selected["checkpoint"],
        base_checkpoint_sha256=sha(selected["checkpoint"]), base_config=selected["config"],
        base_config_sha256=sha(selected["config"]), validation_summary=selected["summary"],
        selected_by="Maximum full-library validation BEDROC85 over the complete frozen grid",
        training_pairs_sha256=old["training_pairs_sha256"], phase2_training_edges_only=True,
        retains_sleec=True, ensembles=False, hubness=False, test_used_for_selection=False,
        followup_plan_sha256=sha(out / "followup_plan.json")))
    features = out / "features"
    features.mkdir()
    source = parent / "features"
    for name in ("catalog.json", "pairs.npz", "reaction_features.npz"):
        (features / name).symlink_to((source / name).resolve())
    shutil.copyfile(source / "protein_mean.h5", features / "protein_mean.h5")
    manifest = json.loads((source / "manifest.json").read_text())
    for key in ("checkpoint", "f3_export_config", "f3_residue_cache"):
        manifest.pop(key, None)
    manifest["raw_feature_reuse"] = dict(source_manifest=str(source / "manifest.json"),
        sha256=sha(source / "manifest.json"), purpose="Same target raw inputs; regenerate learned F3 coordinates")
    manifest["sources"]["config"] = dict(path=selected["config"], sha256=sha(selected["config"]))
    write_json(features / "manifest.json", manifest)
    write_json(out / "state.json", dict(stage="selected_f3_feature_export", selected=selected))
    invoke("generalization_clipzyme_phase2_export.py", ["--stage", "f3",
        "--catalog", cross / "clipzyme_f3_catalog_v1", "--config", selected["config"],
        "--checkpoint", selected["checkpoint"], "--output", features, "--batch-size", 128,
        "--residue-cache", "/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5"],
        0, out / "features.log")
    write_json(out / "state.json", dict(stage="phase2_training_and_validation"))
    invoke("generalization_screen_phase2_campaign.py", ["--output", out], 0, out / "phase2.log")
    if json.loads((out / "complete.json").read_text())["base_retained"]:
        write_json(out / "state.json", dict(stage="complete", base_retained=True))
        return
    invoke("generalization_clipzyme_phase2_dictionary.py", ["--features", features,
        "--output", out / "anchors.pt"], 0, out / "dictionary.log")
    write_json(out / "state.json", dict(stage="semantic_composition"))
    invoke("generalization_screen_phase2_compose.py", ["--campaign", out], 0, out / "composition.log")
    write_json(out / "state.json", dict(stage="complete", base_retained=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
