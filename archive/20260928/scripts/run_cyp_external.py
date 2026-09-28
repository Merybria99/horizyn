"""Run native external CYP baselines; each model gets a separate Python process."""
import argparse
import fcntl
import importlib
import os
import subprocess
import sys

from cyp_external_common import ROOT, RUN, SOURCES

MODULES = {"horizyn1_dev": "cyp_native_horizyn", "clipzyme_pretrained": "cyp_native_clipzyme",
           "enzymecage_pretrained": "cyp_native_enzymecage"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["preflight", "run", "worker"])
    p.add_argument("--models", nargs="+", choices=list(MODULES), default=["horizyn1_dev"])
    p.add_argument("--gpus", default="0", help="One physical GPU ID; models run sequentially")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    if args.stage == "worker":
        if len(args.models) != 1: p.error("worker requires one model")
        importlib.import_module(MODULES[args.models[0]]).run(args.device)
        return
    if args.stage == "preflight":
        for method in args.models:
            subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import {MODULES[method]} as m; m.run('cpu', preflight=True)"], check=True)
        return
    ids = args.gpus.split(",")
    if len(ids) != 1 or not ids[0].isdigit(): p.error("Specify one physical GPU ID")
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "controller.lock").open("a") as lock:
        print("Waiting for the external-baseline controller lock...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for method in args.models:
            scores = RUN / method / "scores.csv"
            if not scores.exists():
                subprocess.run([sys.executable, __file__, "preflight", "--models", method], check=True, cwd=ROOT)
                output = subprocess.check_output(["nvidia-smi", "-i", ids[0], "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
                if output.strip(): raise RuntimeError(f"GPU {ids[0]} occupied; no processes were stopped")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[0], PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
                with (RUN / f"{method}.log").open("a") as log:
                    print(f"Starting {method}; log: {log.name}", flush=True)
                    subprocess.run([sys.executable, __file__, "worker", "--models", method], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            subprocess.run([str(ROOT.parent / ".capability-run-py/bin/python"), str(ROOT / "scripts/run_cyp_baselines.py"),
                "import", "--method", method, "--scores", str(scores), "--provenance", str(scores.with_suffix(".json"))], check=True, cwd=ROOT)
        print("All requested native baselines imported; report: runs/cyp_baselines_v1/report/summary.md", flush=True)


if __name__ == "__main__": main()
