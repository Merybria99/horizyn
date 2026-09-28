#!/usr/bin/env python3
"""Finish the fresh EnzymeCAGE F3 P450 audit after training exits cleanly."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/generalization_20260919_2251/cross_paper_retraining"
RUN = CAMPAIGN / "enzymecage_f3_seed42_local"
EXPORT = CAMPAIGN / "enzymecage_f3_p450_scores"
EVALUATION = CAMPAIGN / "enzymecage_f3_p450_evaluation"
COMPARISON = CAMPAIGN / "enzymecage_f3_p450_comparison"
SESSION = "enzyme_matched_f3_20260920"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def session_active() -> bool:
    return subprocess.run(("tmux", "has-session", "-t", SESSION),
                          cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def run(command: list[str], log: Path) -> None:
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")


def main() -> None:
    status_path = CAMPAIGN / "enzymecage_f3_finalize_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    started = now()
    deadline = time.monotonic() + 4 * 3600
    while session_active() and time.monotonic() < deadline:
        time.sleep(30)
    status = {"schema": "matched_enzymecage_f3_finalize_watch_v1",
              "started_utc": started, "finished_utc": now(),
              "training_session": SESSION,
              "export_source_sha256": sha256(ROOT / "scripts/generalization_matched_enzymecage_p450_export.py"),
              "evaluator_source_sha256": sha256(ROOT / "scripts/generalization_matched_enzymecage_p450_evaluate.py"),
              "comparison_source_sha256": sha256(ROOT / "scripts/generalization_matched_enzymecage_p450_compare.py"),
              "test_scores_used_for_model_selection": False}
    try:
        if session_active():
            raise TimeoutError("Training still active after four-hour watcher deadline")
        if "PROTEIN-POOLING TRAINING COMPLETE" not in (RUN / "pipeline.log").read_text():
            raise RuntimeError("Training exited without its success marker")
        status["training_complete"] = True
        run((sys.executable,
             str(ROOT / "scripts/generalization_matched_enzymecage_p450_export.py"),
             "--run", str(RUN), "--output", str(EXPORT)),
            CAMPAIGN / "enzymecage_f3_p450_export.log")
        status["score_export_sha256"] = sha256(EXPORT / "export.json")
        run((str(ROOT / ".deps/cyp-external-env/bin/python"),
             str(ROOT / "scripts/generalization_matched_enzymecage_p450_evaluate.py"),
             "--export", str(EXPORT), "--output", str(EVALUATION)),
            CAMPAIGN / "enzymecage_f3_p450_evaluate.log")
        status["evaluation_sha256"] = sha256(EVALUATION / "summary.json")
        run((str(ROOT / ".deps/cyp-external-env/bin/python"),
             str(ROOT / "scripts/generalization_matched_enzymecage_p450_compare.py"),
             "--export", str(EXPORT), "--evaluation", str(EVALUATION),
             "--output", str(COMPARISON)),
            CAMPAIGN / "enzymecage_f3_p450_compare.log")
        status["comparison_sha256"] = sha256(COMPARISON / "summary.json")
        status["complete"] = True
        print(json.dumps({"complete": True, "evaluation": str(EVALUATION)}), flush=True)
    except Exception as error:
        status["complete"] = False
        status["error"] = str(error)
        print(json.dumps({"complete": False, "error": str(error)}), flush=True)
    finally:
        status["finished_utc"] = now()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
