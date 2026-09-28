# CYP within-family comparison

This adds an independent, retrospective comparison around the completed
`runs/cyp_specificity_v1` scores. It does **not** retrain models, extract CIRCE
features again, touch source receipts, or allocate GPUs.

**Completion update (2026-09-20):** the separate native-inference runner has
imported complete CLIPZyme, Horizyn-1 and EnzymeCAGE scores. The
[current report](../runs/cyp_baselines_v1/report/summary.md) contains ten
evaluated rows and no pending public-checkpoint rows. The import-only
description below refers to this baseline runner; native execution and
recovered-input provenance are documented in [cyp_external.md](cyp_external.md).

## What runs

| Method | Implementation | Inputs / supervision |
|---|---|---|
| Random | Exact expected rank statistics | Each released candidate pool |
| Sequence alignment | Maximum global BLOSUM62 score against CYP train references | Sequence only |
| Substrate + alignment | Nearest substrate fingerprint, then mean alignment to its training enzymes | Substrate + sequence |
| CIRCE F3 / biological residual | Reuse checksum-verified existing predictions | Full reaction; fixed checkpoints and residual alpha |
| FusionESP original | Actual CPU inference with released weights and cached embeddings | Substrate + ESM2-3B sequence embeddings |
| FusionESP CYP-adapted | Actual CPU inference with the separately released fine-tuned weights | Same inputs; separate adapted-model group |
| CLIPZyme, official Horizyn-1 dev, EnzymeCAGE original | Strict score import only; reported PENDING until provided | Must run their native inference separately |

The last three are **not implemented inference backends**. Import support is not
evidence that those models have been evaluated. The report marks them pending,
and its overall `complete` flag remains false until their scores are imported.

## Protocol

- Same 45 queries, 22 organisms, 4,135 protein IDs and 10,105 query–candidate
  pairs for every row. No missing-embedding filtering or per-method smaller pool.
- R→E only: the dataset does not define an equivalent E→R evaluation.
- Other candidates are unassayed/unknown, not experimental negatives. Metrics
  measure recovery of the designated catalyst, not activity classification.
- Exact equal scores are ties. Primary MRR is the mean of reciprocal ranks
  within the tied interval, **not** the reciprocal of its mean rank. Hit rates
  also integrate over uniform random tie ordering. Legacy alphabetical ranks
  remain diagnostic only.
- Report Hit@1/5/10, top 10/20/50%, MRR, screening depth and percentile.
  Top p% means `rank <= floor(p*N/100)`; on a six-candidate pool top 10% is empty,
  matching the authors' percentile convention. Percentile is `100*rank/N`.
- Query-macro means; paired bootstrap resamples **whole organisms**, with
  10,000 replicates and seed 42. Differences are residual minus each control or
  model. Confidence intervals are exploratory, unadjusted for multiple
  comparisons, and conditional on the fixed checkpoints.
- Protocol/config/code hashes freeze at first invocation. Changes require a
  new run root. This is explicitly **retrospective**, not a preregistered test:
  initial F3/residual results had already been inspected. Do not choose a new
  checkpoint/alpha/threshold using CYP scores.
- Exact-sequence overlap from the completed source audits accompanies the
  report. Unknown upstream/backbone/SLEEC exposure remains unknown. No
  leakage-free or architecture-only superiority claim is made.

## Important alignment reproduction caveat

The paper calls its controls “BLAST”, but its [released implementation](https://gitlab.com/cyp_pred_repos/boltzcyp/-/blob/a1a110a7bf850af7650f12219e8facf04039b659/src/boltzcyp/runscripts/evaluation_baseline.py)
uses Biopython **global** alignment, BLOSUM62, gap-open −11 and gap-extension −1.
The substrate fingerprint is **RDKFingerprint(maxPath=2)**, not Morgan radius 2.
The second baseline averages alignment scores across training records for the
first maximum-Tanimoto substrate. Duplicate training records therefore retain
their original weighting in the mean.

We reconstruct 1,539 positive records / 330 unique sequences from the pinned
FusionESP `train_pos` release; `val_pos` and negatives are excluded. The separate
BoltzCYP training DataFrame's exact membership and ordering have not been
independently verified. We sort reference sample IDs and record all
nearest-substrate ties in `controls/nearest_substrates.csv`. Thus these are
**paper-style controls, not exact reproductions of the published baseline numbers**.
The original reaction's last substrate component is used, before molecule sorting.

## Launch

From `/datastor2/deep-proteins/EnzymeDiscovery/horizyn`, the existing environments
provide NumPy, RDKit, PyYAML and an isolated torch 2.6 CPU installation.
Biopython 1.85 is installed separately under `.deps/cyp-baseline-deps`, which the
runner adds to its own import path. The training environment is not modified.
On a new copy, install that small scoped dependency with:

```bash
uv pip install --python ../.capability-run-py/bin/python --no-deps \
  --target .deps/cyp-baseline-deps biopython==1.85
```

The small checkpoint archive is
already downloaded in this workspace. On a new copy, obtain it with:

```bash
../.capability-run-py/bin/hf download lizmahood/cyp_pred_repos \
  fusionesp_model_ckpts.tar.gz --repo-type dataset \
  --revision a5c15469c6caae1328ef777ba141215a8c7c14b5 \
  --local-dir data/external/cyp_specificity_2026/release
```

The data/model archives are SHA256-checked before loading. The native FusionESP
adapter uses a narrow allowlist and `torch.load(weights_only=True)` with
torch≥2.6; it never falls back to unrestricted pickle loading. It reuses the
authors' ESM2/MolFormer embeddings, so it needs no extraction GPU.

Detached CPU run (four worker processes, no GPU changes):

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV bash --noprofile --norc scripts/launch_cyp_baselines.sh
tail -f runs/cyp_baselines_v1/pipeline.log
```

Set `CYP_CPU_WORKERS=8` only if eight CPU cores are available. This does not
change the scientific settings. Duplicate sessions/run-root controllers are
rejected; no existing process is killed.

Foreground / explicit-stage commands:

```bash
../.capability-run-py/bin/python scripts/run_cyp_baselines.py prepare
../.capability-run-py/bin/python scripts/run_cyp_baselines.py controls --workers 4
../.capability-run-py/bin/python scripts/run_cyp_baselines.py fusion --workers 4
../.capability-run-py/bin/python scripts/run_cyp_baselines.py report
```

Or use `all` instead of the individual stages. Completed, checksum-valid score
stages are reused. Interrupted alignments recompute; existing CIRCE caches and
checkpoints are never changed. A partially written score lacking its receipt
fails closed and needs inspection/a new run root.

Outputs: `runs/cyp_baselines_v1/report/summary.md`, `summary.json`, per-query
CSV files and deterministic diagnostic rankings. Each score file has a
provenance receipt. FusionESP's raw released prediction table lacks a checkpoint
receipt: it is compared numerically with the two newly generated tables in
`fusion_native/release_prediction_audit.json`, never silently labelled as one of them.

## External models

Run each model on the unchanged released pools. `inputs/mapped_inputs.csv`
preserves atom-mapped reactions, sequence, protein ID, query ID and the authors'
structure path. Those `cif` paths are **not verified local structures**. CLIPZyme
requires its native structure/feature setup; do not silently drop unavailable
proteins. The official Horizyn-1 dev checkpoint is distinct from our reconstructed
large-data CIRCE-v2 checkpoint. These are planned pretrained-model rows, not
CYP-fine-tuned rows; a new variant needs a separately frozen protocol/run.

Produce a CSV with precisely `query_id,protein_id,score`, with larger scores
meaning better matches. Accompany it with JSON containing:

```json
{
  "method": "clipzyme_pretrained",
  "variant": "pretrained",
  "checkpoint_sha256": "<64 lowercase hexadecimal characters>",
  "source_revision": "<exact upstream code revision>",
  "input_manifest_sha256": "<SHA256 of benchmark/manifest.json>",
  "input_scope": "reaction + sequence + structure; no oracle EC filtering",
  "upstream_exposure": "UNKNOWN unless independently audited",
  "checkpoint_selection": "fixed_without_cyp_selection",
  "generator_command": "<actual native inference command>",
  "scores_sha256": "<SHA256 of the submitted CSV>",
  "score_direction": "higher_is_better"
}
```

```bash
../.capability-run-py/bin/python scripts/run_cyp_baselines.py import \
  --method clipzyme_pretrained --scores /absolute/path/scores.csv \
  --provenance /absolute/path/provenance.json
```

Use method `horizyn1_dev` or `enzymecage_pretrained` for the other planned rows.
Import verifies hashes, variant declaration and exact candidate coverage. The
user-supplied provenance declaration is not an independent audit of upstream
training exposure.

## Tests

```bash
../.capability-run-py/bin/python -m pytest -o addopts='' \
  tests/unit/test_cyp_baselines.py tests/unit/test_cyp_specificity.py -q
```

Tests cover exact and partial ties, small-pool percentile boundaries, missing /
duplicate / non-finite scores, paired resampling, safe archive reading, train-only
references, alignment-record weighting, immutable receipts and strict external imports.
