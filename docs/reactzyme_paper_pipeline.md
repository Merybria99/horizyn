# ReactZyme Paper Pipeline

This is the active workflow for ReactZyme paper-comparable ablations. Older
`reactzyme_clean`, `reactzyme_official`, source-collapse, and directional scripts
remain available for reproducing historical runs, but they are not part of this
pipeline.

## Components

| Responsibility | Location |
| --- | --- |
| Protocol materialization and audits | `horizyn/benchmarks/reactzyme_protocol.py` |
| Ablation declaration validation | `horizyn/benchmarks/reactzyme_ablations.py` |
| Matrix generation | `horizyn/benchmarks/reactzyme_matrix.py` |
| Launcher runtime utilities | `horizyn/benchmarks/reactzyme_runtime.py` |
| Split declarations | `configs/benchmarks/reactzyme_paper/splits.yaml` |
| Latent A0-A7 matrix | `configs/benchmarks/reactzyme_paper/latent.yaml` |
| Representation A8-A15 matrix | `configs/benchmarks/reactzyme_paper/representation.yaml` |
| Training launcher | `scripts/run_reactzyme_paper_ablation_matrix.sh` |

The older `create_reactzyme_latent_ablation_matrix.py` and
`run_reactzyme_latent_ablation_matrix.sh` filenames are compatibility wrappers.

## Data Contract

Each of `time`, `enzyme_smi`, and `reaction_smi` is independent.

- Training: deterministic 90% sample of the released training pairs.
- Validation: remaining 10% of released training pairs.
- Test: released test CSVs copied byte-for-byte.
- Direction: canonical forward reaction only.
- Validation checkpoint: minimum `val/loss`, evaluated every epoch by default.
- Final evaluation: `paper_test_candidates`; the released test is not used for
  checkpoint selection.

ReactionT5v2 is required for every query. Missing UniMol2 and ChIRo records are
represented by zero values with explicit masks and are reported in each run
manifest.

## Commands

Materialize the paper split once:

```bash
../env/bin/python scripts/build_reactzyme_paper_protocols.py
```

Generate one matrix:

```bash
../env/bin/python scripts/create_reactzyme_paper_ablation_matrix.py --matrix latent
../env/bin/python scripts/create_reactzyme_paper_ablation_matrix.py --matrix representation
```

Launch a generated matrix:

```bash
outputs/reactzyme_latent_organization_paper_ablation_20260717/run_reactzyme_paper_ablation_matrix.sh --detach
```

The launcher redirects `HOME`, caches, W&B files, and temporary state into the
run directory. Each ablation wave starts three independent DDP jobs, one per
ReactZyme split, with every job using the configured four GPUs.

## Adding An Ablation

Add a variant to the appropriate matrix YAML. IDs and labels must be unique;
block dimensions must sum to 512; block weights must sum to 1. The generator
validates these rules before writing any run configuration.
