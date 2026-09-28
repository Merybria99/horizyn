# V4, F3, and CIRCEv2

Three maintained enzyme–reaction retrieval pipelines share feature readers, training infrastructure, and evaluation metrics. **V4 is the default**, using the frozen dictionary-free recipe from the recent case studies.

| Pipeline | Enzyme representation | Training/scoring |
|---|---|---|
| V4 | ProtT5 global, SLEEC site, four learned residue views | Anchor-balanced alignment; full-graph residual refinement; cosine score |
| F3 | Historical SLEEC-guided/factorized encoder | Original F3 loss and base dual-encoder score |
| CIRCEv2 | Historical ProtT5/SLEEC encoder | Annotation-derived negatives and original sampled multi-positive loss |

## Layout

```text
horizyn/pipelines/        maintained API and CLI; V4 is the default
horizyn/models/           common layers, enzyme, reaction, and dual encoder
horizyn/model.py          historical import path for checkpoint compatibility
horizyn/enzyme_multiview.py   V4 residue views and fusion
configs/pipelines/        explicit profiles for the three supported pipelines
scripts/                  feature extraction, preparation, compatibility launchers
tests/                    shared-core and supported-pipeline regression tests
docs/                     maintained implementation and reproduction guides
archive/20260928/         historical experiments, configs, tests, documentation
data/, runs/, checkpoints/   scientific artifacts at their existing paths
```

`horizyn/models/compat.py` contains inactive historical classes required by shared constructors. These are not selectable pipelines. Supported entry points reject unrelated encoder modes, directional adapters, and prototype scoring.

## Run

Run from this directory using the workspace environment:

```bash
../env/bin/python -m horizyn.pipelines list
../env/bin/python -m horizyn.pipelines train \
  --config configs/pipelines/v4/reactzyme_reaction_smi.yaml
```

Use `--model-preflight-only` to construct the model without training. F3 and CIRCEv2 require explicit family selection:

```bash
../env/bin/python -m horizyn.pipelines train --pipeline f3 \
  --config configs/pipelines/f3/published_retrieval.yaml
../env/bin/python -m horizyn.pipelines train --pipeline circev2 \
  --config configs/pipelines/circev2/reactzyme_reaction_smi.yaml
```

The F3 profile retains its original published-retrieval data, not a matched-data ReactZyme comparison. V4 profiles cover ReactZyme ReactionSim, EnzymeSim, Time, and EnzymeMap. Required local feature caches and pretrained scorer weights are explicit in the YAML.

V4 phase 2 is separate from encoder training:

```bash
../env/bin/python -m horizyn.pipelines export-refinement \
  --config configs/pipelines/v4/reactzyme_reaction_smi.yaml \
  --checkpoint PATH_TO_PHASE1.ckpt --output runs/v4/train_features --device cuda:0
../env/bin/python -m horizyn.pipelines refine \
  --features runs/v4/train_features --output runs/v4/refinement \
  --config configs/pipelines/v4/phase2.yaml --device cuda:0
```

The CLI also provides `encode` and `evaluate`. Use `--help` for each command. [PIPELINES.md](docs/PIPELINES.md) explains architecture, feature contracts, scoring coefficients, manifests, and dataset-specific evaluation. Editable installation exposes the same CLI as `circe`; module invocation needs no reinstall.

## Verify

```bash
../env/bin/python -m pytest tests -q -o addopts=''
../env/bin/python tools/organize_source.py --check
```

[Cleanup and compatibility evidence](docs/CLEANUP_20260928.md) records the source migration and checkpoint checks. Archived launchers are research records, not supported entry points.

[Output artifact cleanup](docs/ARTIFACT_CLEANUP_20260928.md) documents the removed scratch files and regenerable caches, retained scientific assets, and deletion audit trail.

[Checkpoint cleanup](docs/CHECKPOINT_CLEANUP_20260928.md) records the pruned historical intermediate epochs and protected model selections.

[Baseline sources and evidence](docs/BASELINES.md) distinguishes retained comparators from retired integrations. [Research documents](documents/README.md) indexes the archived campaigns and drafts.

## Attribution

This workspace extends Horizyn, originally developed by Dayhoff Labs. The original PolyForm Noncommercial license, copyright notices, and third-party licenses remain applicable; see [LICENSE](LICENSE).
