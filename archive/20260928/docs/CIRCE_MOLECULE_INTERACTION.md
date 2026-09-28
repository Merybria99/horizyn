# Single-encoder reaction experiment

This replaces the reaction tower, not the enzyme tower or retrieval objective.
Reaction inputs are the existing frozen UniMol2 molecular vectors. No ReactionT5,
ChIRo, fingerprint, descriptor, cofactor-indicator or directional-label branch is
loaded. UniMol2 itself is not fine-tuned; trainable layers operate on its cached
per-molecule features. No new feature extraction is required.

## Architecture and training

Each 768-dimensional molecular vector is normalized and projected to 256
dimensions. Two independently initialized Transformer layers (4 heads, feedforward
width 512, pre-layer normalization, no positional embeddings, dropout 0) allow
participants to interact. Learned masked attention pooling summarizes the set.
Small projection layers produce the final normalized 512-dimensional reaction
embedding. The existing encoder interface is reused with a single active modality;
there is no learned mixture of different chemical feature sources.

The enzyme tower, frozen SLEEC scorer, optimizer, precision and retrieval loss are
inherited from the no-classification CIRCE-v3 reaction-smi control. The trainable
towers are initialized fresh with seed 42, not resumed from trained checkpoints.
Classification remains disabled; known positives use `all_known_in_batch` and the
unknown-pair penalty stays 0.5. The global batch is 1,536 examples: 512 per GPU on
three GPUs, 384 on four, or 768 on two. Hardware changes are not bitwise-equivalent.

The default is a short, fixed-budget **864-update** screen (9 epochs for these
data). Early stopping is disabled for this screen. Validation runs every epoch;
the final fixed-step checkpoint is then evaluated in both directions. The script
does not select or tune using the test set. For a longer *separate* run, `--steps`
can be specified; only the default 864-step run is automatically compared with the
existing matched-budget control.

## Launch detached

Run on a machine with the shared repository/caches and the specified GPUs free:

```bash
env -u BASH_ENV -u ENV bash --noprofile --norc \
  /datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/launch_circe_molecule_interaction.sh \
  --gpus 0,1,3
```

For four free GPUs, use `--gpus 0,1,2,3`. The launcher refuses occupied devices and
never stops another job. It prepares a new unique run directory, prints coverage,
and prints the tmux session and exact `tail -f .../pipeline.log` command. Do not
relaunch to monitor: each invocation creates a new experiment. Closing the terminal
does not stop a successfully launched tmux session.

Per-epoch losses and validation metrics: `logs/protein_pooling_training/version_0/metrics.csv`.
Checkpoints: `checkpoints/`, including recovery snapshots every 50 updates.
Final outputs: `evaluation/metrics.json`, `evaluation/queries.json`,
`evaluation/generalization.json`, `evaluation/summaries.csv` and
`evaluation/paired_differences.csv`. The stratified comparison reuses the completed
CPU novelty diagnostics under `runs/circe_generalization_diagnostics_v1`.

If training finished but the final evaluation failed, evaluate that same output
directory with free GPU IDs; this does **not** retrain:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
../env/bin/python scripts/run_circe_molecule_interaction.py evaluate \
  --output runs/REPLACE_WITH_THE_PRINTED_RUN_DIRECTORY --gpus 2
```

## Missing features and interpretation

Preflight checks found 5/6,977 training reactions and 2/2,448 validation reactions
without UniMol2 features. These reactions and all their candidates remain in the
benchmark. Missing inputs are masked and map to a shared uninformative fallback;
the output's coverage manifest records the exact IDs. No missing feature is
replaced by a reaction identity embedding or another modality.

The tower learns relationships among unordered participants. It does **not**
recover verified substrate/product assignments, atom mappings or reaction centers.
An improvement would support this replacement architecture, but would not isolate
the effects of interaction, reduced parameter count and removed input modalities.
The experiment uses the same seed, not a guarantee of identical initial enzyme
weights: changing the architecture changes random-number consumption.

CPU checks cover permutation invariance, padding, all-missing sets, gradient flow,
configuration, two training steps, checkpoint reload and bidirectional evaluation
with deliberately missing molecular features. Multi-GPU speed/VRAM usage must be
measured on the launch machine. Keeping the contrastive batch fixed takes priority
over filling unused GPU memory.
