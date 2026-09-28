# Biological-residual controls: reaction-smi

This campaign reuses the frozen F3 parent (epoch 29) and the completed R1
biological residual. It does not retrain the CIRCE towers or extract backbone
embeddings. Original checkpoints, features, and evaluation JSONs are read-only.

| Variant | Initialization | Source stage | ReactZyme stage | Residual guidance |
|---|---|---|---|---|
| R0 F3 | Exact original F3 | None | None | Residual score disabled |
| R1 full | Existing trained residual | Reused | Reused | SLEEC |
| R2 no source pretraining | F3 + newly initialized residual | Skipped | 3 epochs, 15% replay | SLEEC |
| R3 unguided | F3 + newly initialized residual | 5 epochs | 3 epochs, 15% replay | Random 48 + context 16; no SLEEC score/pooling bias |

The same F3 base, effective batch, optimizer, loss terms, seed 42 and fixed final
stage checkpoints are used. R3 replaces only *residual* guidance; F3's own SLEEC
components remain unchanged and frozen. R3 random sampling is per protein,
without replacement, independent of rank/batch order, and recorded in separate
cache metadata. Its source and ReactZyme token caches are built from existing
full-residue HDF5 stores, not the guided token caches. No ProtT5 inference runs.

R1–R2 measures the pretraining stage under a fixed fine-tuning budget, not the
effect of removing all extra data (R2 retains source replay). R1–R3 measures
additional biological guidance at matched capacity/data/training duration.

## Launch

On a healthy node with four free GPUs and access to the shared project:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV bash --noprofile --norc scripts/launch_biological_residual_controls.sh
```

Defaults: GPU indices `0,1,2,3`, run directory
`runs/biological_residual_controls_reaction_smi_v1`. Override with
`CIRCE_CONTROL_GPUS` and `CIRCE_CONTROL_RUN_ROOT` if needed. Four devices are
required to preserve the recipe. No old processes are stopped. A tmux-session
check, shared controller lock, GPU UUID occupancy check and input preflight
prevent accidental duplicate/conflicting launches. This is not a Slurm resource
allocation; use an allocated/authorized node.

```bash
tail -f runs/biological_residual_controls_reaction_smi_v1/pipeline.log
```

The controller prepares immutable configurations; verifies existing inputs;
re-evaluates R1 on validation; trains/validates R2; caches random tokens and
trains/validates R3; evaluates all three on test with frozen validation-selected
alphas; then writes the report. Evaluation uses one GPU, training/cache creation
use four. Existing features may still require substantial shared-storage reads.

Re-running the same launcher after it exits resumes incomplete stages from
`last.ckpt`, skips completed stages, and reuses compatible token/evaluation
caches. Different configurations or code after execution starts require a new
run root. A corrupt checkpoint is reported, not silently discarded. Ctrl-C in
the attached controller requests shutdown of only its current subprocess group.

## Evaluation and outputs

- Primary selection: mean bidirectional **all-positive ReactZyme MRR**.
- Validation alpha grid: `0, .025, .05, .075, .1`; each direction may lose at
  most `.005` MRR relative to alpha zero. Ties prefer the smaller alpha.
- Test reads the alpha from the matching validation JSON. Test scores never
  determine alpha or invoke the validation preservation guard.
- Rank ties use stable candidate order. R1/R0 are re-evaluated consistently;
  older result files remain unchanged.
- Metrics: Hit@K, Precision@K, both MRR definitions, R-precision, AP, mean rank.
  Mean rank averages *all known positive ranks* within each query, then queries.
  Precision@K divides known positives in the top K by K.
- Per-query CSVs preserve all evaluated alphas and training-only association
  counts. Reports compare selected-alpha results with paired F3 queries.
- Built-in strata: known association count, training association count, reaction
  UniMol2/ChIRo availability. These are not complete promiscuity measurements.
- Bootstrap intervals are paired-query resampling conditional on seed 42, not
  repeated-training uncertainty; biologically related queries may be dependent.

Outputs under the new run directory:

```text
R*/results/validation.json
R*/results/test.json
R*/results/{validation,test}.per_query.csv
reports/summary.md
reports/test_summary.csv
reports/stratified_metrics.csv
reports/report.json
```

Independent similarity and annotation strata are optional, not guessed from
retrieval embeddings. Supply an audited CSV with unique `direction,query_id`
keys and `stratum_*` columns (for example `stratum_train_similarity` and
`stratum_annotation_coverage`). Reaction query IDs end in `_f`. Missing values
are reported as unknown. Similarity must be measured against training entities.

```bash
../env/bin/python scripts/run_biological_residual_controls.py report \
  --query-metadata /absolute/path/to/query_metadata.csv
```

Without such metadata the report explicitly lists these analyses as unavailable.
Further seeds/splits and a fresh external benchmark remain conditional follow-up
experiments; this launcher does not schedule them.

## Checks

```bash
CUDA_VISIBLE_DEVICES= ../env/bin/python -m pytest -q --no-cov \
  tests/unit/test_biological_residual_controls.py tests/unit/test_biological_residual.py
CUDA_VISIBLE_DEVICES= ../env/bin/python scripts/run_biological_residual_controls.py preflight
```
