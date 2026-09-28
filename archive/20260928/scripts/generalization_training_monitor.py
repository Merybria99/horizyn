#!/usr/bin/env python3
"""Record live F3 training health and finished screening metrics without changing runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--control-root", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8)
    a = p.parse_args()
    output = a.campaign / "monitor"
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    known_results = set()
    while time.monotonic() - started < a.hours * 3600:
        now = datetime.now(timezone.utc).isoformat()
        state = {"updated_utc": now, "training": [], "validation_results": []}
        for root, names in ((a.campaign, ("balanced_anchors", "chemistry_dropout", "prototype4")),
                            (a.control_root, ("mlnce_allknown", "anchor_allknown", "directional_anchor_allknown"))):
            for name in names:
                arm = root / name
                paths = sorted((arm / "logs").rglob("metrics.csv"))
                rows = list(csv.DictReader(paths[-1].open())) if paths else []
                rows = [r for r in rows if r.get("epoch") and r.get("step")]
                losses = [r for r in rows if r.get("train/loss_epoch")]
                diagnostic = [r for r in rows if r.get("val/mean_bidirectional_mrr")]
                text = (arm / "console.log").read_text(errors="replace") if (arm / "console.log").exists() else ""
                last_loss = float(losses[-1]["train/loss_epoch"]) if losses else None
                state["training"].append(dict(
                    arm=name, epoch_1_based=max([int(float(r["epoch"])) + 1 for r in rows], default=0),
                    last_completed_epoch_loss=last_loss,
                    small_pool_mrr_diagnostic=float(diagnostic[-1]["val/mean_bidirectional_mrr"]) if diagnostic else None,
                    completed="max_epochs=100" in text,
                    error_detected="Traceback" in text or "OutOfMemoryError" in text or
                                   (last_loss is not None and not math.isfinite(last_loss)),
                    metrics_file_modified_utc=datetime.fromtimestamp(paths[-1].stat().st_mtime, timezone.utc).isoformat() if paths else None))
        paths = list(a.campaign.rglob("summary.json"))
        paths += list((a.control_root / "directional_anchor_allknown").glob("screen_epoch*/validation_evaluation/summary.json"))
        for path in sorted(paths):
            try:
                result = json.loads(path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                # An evaluator can still be writing a newly discovered result.
                continue
            if result.get("schema") != "clipzyme_f3_full_library_validation_v1":
                continue
            item = dict(path=str(path), **result["summary"])
            state["validation_results"].append(item)
            if str(path) not in known_results:
                with (output / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(observed_utc=now, validation=item)) + "\n")
                known_results.add(str(path))
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
            state["gpus"] = [dict(zip(("index", "used_mib", "total_mib", "utilization_percent"),
                                      map(int, line.split(",")))) for line in result.stdout.splitlines()]
        except (subprocess.SubprocessError, ValueError) as exc:
            state["gpu_query_error"] = str(exc)
        temporary = output / "status.tmp.json"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        print(json.dumps(dict(time=now, epochs={r["arm"]: r["epoch_1_based"] for r in state["training"]},
                             completed_validations=len(state["validation_results"]))), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
