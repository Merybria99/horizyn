# Cached-feature reaction-smi training on Glutamine

Launch from Glutamine, not Tyrosine or node014:

```bash
cd /
/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
  /bin/bash --noprofile --norc \
  /datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/launch_reactzyme_glutamine_cached.sh
```

The launcher creates detached tmux session `reactzyme_circe_v2_glutamine` and
prints a local `/tmp/reactzyme_train.*/pipeline.log` path. It creates a fresh,
timestamped `runs/reactzyme_reaction_smi_tyrosine_recipe_glutamine_*` directory;
the same console output is also logged there after directory creation.
Training metrics and checkpoints are beneath that run's `training/` directory.

Only GPUs **1 and 2** are visible. Both must be idle A100 GPUs with at least
60,000 MiB free, checked before setup and again before training. No other job
is stopped, no existing output is overwritten, and no extraction is launched.

The stages are CPU HDF5-checker fixture tests, read-only cache screening,
CPU index/configuration preparation, and training. CPU stages have bounded
timeouts and a failure stops the chain. Screening checks all 178,327 required
protein IDs, all offsets, and sampled finite/nonzero vectors; it is not a full
finite-value certificate. Per-access finite checking remains enabled during
training. A failed setup requires investigation before rerunning; a new launch
creates a new output directory, not an automatic checkpoint resume.

## Recipe and differences

The active Tyrosine training YAML supplies the bidirectional sampled
multi-positive InfoNCE loss, beta 10, equal directional weights, auxiliary
classification weight zero, AdamW learning rate 1e-4/weight decay 0.01, bf16
with FP32-sensitive operations, five-epoch/100,000-step caps, and early stopping.
The same indexed sampler supplies 85% positive / 15% typed-negative rows,
including genuine positive support for both endpoints of each negative.
The batch is 200 rows per GPU, global 400, with no gradient accumulation.

Dataset-specific differences are deliberate and recorded in `preparation.json`:

- The original ReactZyme paper reaction-smi train/validation split and F3
  molecule-set chemistry are retained. These inputs are not replaced by
  Horizyn's directed chemistry. Forward ID aliases match existing feature IDs.
- Legacy eight-mechanism and ten-curated-native-cofactor positive annotations
  are reused only for proteins whose **every source annotation-training edge**
  also occurs in the current training set. Source hashes are checked. Unsafe
  aggregates are unknown, not negative labels. Other v2 cofactor groups and
  reaction-role cofactors remain unknown; expanded Horizyn label coverage is
  not claimed. Zero/unobserved legacy target entries are not negative evidence.
- The released test edges do not enter the training graph, label projection,
  or model selection. Reading test candidate IDs during frozen-cache screening
  checks availability only; it is not test-label supervision.
- Validation runs after every ReactZyme epoch. The experiment keeps the
  ReactZyme F3 tower configuration and existing pretrained SLEEC scorer; it
  does not load a fitted Horizyn/ReactZyme retrieval checkpoint.

This is a matched-loss/optimization computational check, not an assertion of
identical annotation coverage or input chemistry across the two datasets.
No automatic held-out test or enzyme-smi/time training is chained yet.

Local verification: configuration helper and train-scope filtering unit tests,
shell syntax, wrong-host refusal, actual generated-YAML schema, dependency-path
presence, and annotation source hashes. Full GPU execution cannot be verified
from Tyrosine: SSH authentication to Glutamine is unavailable, and ML-library
imports on Tyrosine have encountered storage waits. The launcher performs its
runtime cache tests and indexed-sampler checks on Glutamine before training.
