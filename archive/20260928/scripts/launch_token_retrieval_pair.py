#!/usr/bin/env python3
"""Launch independent B1/B2 torchrun jobs on GPUs 0-1 and 2-3."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--python", default=str(ROOT.parent / ".b12-pretrained-py/bin/python"))
    parser.add_argument("--config-template", default="reactzyme_reaction_smi_{variant}_esmc_graphormer.yaml")
    parser.add_argument("--preflight-dir", default=None)
    parser.add_argument("--graph-warmup-epochs", type=int, default=None,
                        help="Smoke-only override to exercise pretrained encoder gradients")
    args = parser.parse_args()
    if args.graph_warmup_epochs is not None and not args.smoke_steps:
        raise ValueError("Warmup overrides are reserved for smoke runs")
    # Preserve the venv entry-point path: resolving its symlink selects the
    # system interpreter and loses the isolated package environment.
    python = Path(args.python).absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    gpu_rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    usage = {int(row.split(",")[0]): int(row.split(",")[1]) for row in gpu_rows}
    if any(i not in usage or usage[i] > 1024 for i in range(4)):
        raise RuntimeError(f"Expected four available GPUs; current memory usage MiB: {usage}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if args.smoke_steps else "train"
    campaign = Path(args.campaign_dir).resolve() if args.campaign_dir else ROOT / "runs" / f"b12_pretrained_reaction_smi_{suffix}_{stamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    snapshot = campaign / "source_snapshot"
    snapshot.mkdir()
    sources = [ROOT / "horizyn/token_retrieval.py", ROOT / "horizyn/token_retrieval_data.py",
               ROOT / "horizyn/pretrained_graphormer.py", ROOT / "horizyn/_graphormer_algos.pyx",
               ROOT / "horizyn/LICENSE_GRAPHORMER",
               ROOT / "horizyn/token_retrieval_evaluation.py", ROOT / "scripts/evaluate_protein_pooling.py",
               ROOT / "horizyn/protein_pooling_lightning_module.py", ROOT / "horizyn/data_module.py",
               ROOT / "scripts/train_token_retrieval.py", ROOT / "scripts/launch_token_retrieval_pair.py",
               ROOT / "scripts/preflight_pretrained_graphormer.py", ROOT / "scripts/smoke_pretrained_graphormer_ddp.py",
               ROOT / "tests/unit/test_pretrained_graphormer.py", ROOT / "requirements-b12-pretrained.txt",
               ROOT / "horizyn/losses.py", ROOT / "horizyn/reaction_conditioned_data_module.py"]
    sources += list((ROOT.parent / "VenusRXN/rxnzyme/models/modules").glob("*.py"))
    sources += [ROOT.parent / "VenusRXN/rxnzyme/data/reaction.py",
                ROOT.parent / "VenusRXN/rxnzyme/data/graph_utils/collator.py"]
    fingerprints = {}
    for source in sources:
        relative = source.relative_to(ROOT.parent)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        fingerprints[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (campaign / "source_sha256.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
    if args.preflight_dir:
        shutil.copytree(Path(args.preflight_dir).resolve(), campaign / "preflight")
    # Record installed versions without requiring pip in the isolated runtime.
    packages = subprocess.check_output([str(python), "-c",
        "import importlib.metadata as m; names={d.metadata['Name'] for d in m.distributions() if d.metadata['Name']}; print('\\n'.join(sorted(n+'=='+m.version(n) for n in names)))"], text=True)
    (campaign / "runtime_packages.txt").write_text(packages)
    launches = []
    for variant, gpus in [("B1", "0,1"), ("B2", "2,3")]:
        run_dir = campaign / variant.lower()
        run_dir.mkdir()
        config = ROOT / "configs" / args.config_template.format(variant=variant.lower())
        frozen_config = run_dir / "launch_config.yaml"
        values = yaml.safe_load(config.read_text())
        if args.graph_warmup_epochs is not None:
            values["training"]["graph_warmup_epochs"] = args.graph_warmup_epochs
        frozen_config.write_text(yaml.safe_dump(values, sort_keys=False))
        command = [str(python), "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   "--nproc-per-node=2", str(ROOT / "scripts/train_token_retrieval.py"),
                   "--config", str(frozen_config), "--run-dir", str(run_dir)]
        if args.smoke_steps:
            command += ["--max-steps", str(args.smoke_steps)]
        if args.batch_size:
            command += ["--batch-size", str(args.batch_size)]
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                           PYTHONUNBUFFERED="1", PYTHONHASHSEED="42", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", OUTDATED_IGNORE="1")
        log = run_dir / "train.log"
        with log.open("wb") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        (run_dir / "launcher.pid").write_text(str(process.pid) + "\n")
        launches.append({"variant": variant, "gpus": gpus, "launcher_pid": process.pid,
                         "run_dir": str(run_dir), "log": str(log), "command": command})
    manifest = {"campaign_dir": str(campaign), "created_utc": datetime.now(timezone.utc).isoformat(),
                "smoke_steps": args.smoke_steps, "launches": launches}
    (campaign / "launches.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
