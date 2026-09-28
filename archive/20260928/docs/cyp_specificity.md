# Within-family specificity retrieval: CYP

This is a frozen-checkpoint external **reaction → enzyme** evaluation, not a new
training run. It asks whether the model retrieves a published catalyst among
other cytochrome P450 enzymes from the same organism. It therefore tests a
finer distinction than separating broad enzyme families.

## Released data and labels

The benchmark comes from the 2026 preprint
[Rethinking Benchmarks and Models for Enzyme Specificity Prediction](https://arxiv.org/abs/2607.05084).
The implementation reads the published candidate-pool CSVs from the authors'
[Hugging Face release](https://huggingface.co/datasets/lizmahood/cyp_pred_repos/tree/a5c15469c6caae1328ef777ba141215a8c7c14b5):

- Revision: `a5c15469c6caae1328ef777ba141215a8c7c14b5`.
- Archive: `fusionesp_data_dir.tar.gz`, approximately 207 MB; checksum pinned in
  `scripts/run_cyp_specificity.py`. Only CSV members are read, without extracting
  arbitrary paths, structures, executable code, or pickle files.
- 45 queries, 22 source organisms, 4,135 distinct protein IDs, and 10,105
  query–candidate pairs. Published pools contain **6–500** candidates; the
  implementation retains these exact pools rather than reconstructing them
  from a generic CYP sequence collection.
- The designated positive is identified by the published query filename.
  **All other candidates are unannotated, not experimentally inactive.**
- There are 118 duplicate sequences across distinct protein IDs. The primary
  benchmark keeps the published IDs; sequence-deduplicated ranks are a separate
  diagnostic. Three query reactions repeat; the query-specific positive and
  organism-specific pool are retained.

Atom maps and participant ordering are canonicalized with RDKit; reaction sides,
stereochemistry, charges, and multiplicities are retained in the benchmark input.
Prepared CSV/FASTA files and individual release members have SHA-256 manifests.
The benchmark does not provide a comparably defined enzyme → reaction task;
the script deliberately does not manufacture one from sparse annotations.

## Models and controls

Defaults in `configs/benchmarks/cyp_specificity.yaml`:

| Method | Frozen checkpoint | Role |
|---|---|---|
| `f3` | ReactZyme reaction-smi F3, epoch 29 | Base model |
| `residual` | Its biological residual, final ReactZyme fine-tuning checkpoint | Main comparison |
| `homology_reactzyme` | No learned checkpoint | Reaction-conditioned sequence baseline |
| `homology_residual_source` | No learned checkpoint | Same baseline with the residual's additional reference source |

The residual's F3 weights must exactly match the parent checkpoint. Its score is
`cosine + 0.1 * tanh(biological_score)`: alpha comes from the existing ReactZyme
validation selection, **never from these CYP labels**. This evaluates the whole
residual training recipe, including its extra source pretraining; it is not an
architecture-only ablation with matched training data.

Optional `--models f3 residual circe_v2` adds the large-data CIRCE-v2 step-16k
checkpoint. Its different supervision/data make it a deployment comparison,
not a controlled architectural ablation. There is no retraining, alpha sweep,
classifier fitting, or biological-label input to inference.

Protein and reaction features are extracted once and shared between models.
ProtT5 is loaded from an isolated `safetensors` copy, verified tensor-by-tensor
against the SHA-256-pinned original snapshot. This avoids the Transformers
restriction on loading legacy `.bin` weights with PyTorch <2.6. The original HF
snapshot and the shared inference environments are unchanged. Conversion uses
the separate CPU-only `.deps/cyp-checkpoint-converter` environment:

```bash
CUDA_VISIBLE_DEVICES= .deps/cyp-checkpoint-converter/bin/python \
  scripts/prepare_cyp_prott5_safetensors.py
```

`data/external/cyp_specificity_2026/models/prott5_safetensors/conversion.json`
records the original checksum, output checksums, and exact tensor verification.
The preflight now rejects incompatible legacy checkpoints before CPU audits.

ProtT5 uses the existing 1,022-residue ends/center truncation recipe (14 released
proteins exceed this length); ReactionT5 uses the existing 512-token cap.
The chemistry schema is transformed, not refitted. UniMol2/ChIRo retain their
training-time molecule preprocessing, including omission of non-encodable small
molecules; these omissions are logged. No complete query or candidate may be
dropped. Missing reaction modalities and non-finite/incomplete scores fail the
evaluation. GPU functional-token extraction uses the original BF16 selection
recipe, 48 selected residues plus 16 context tokens. Final learned scoring is
FP32. CPU feature extraction is supported but is not a speed recommendation;
its SLEEC token selection uses FP32, recorded separately in feature provenance.

## Training exposure and homology baseline

For each configured training source, the audit uses **only its positive training
edges**. It checks sequence hashes across aliases, reaction identity, observed
pairs, and MMseqs2 hits with at least 80% coverage of both sequences. Per-protein
maximum reported identity is saved; no reported hit is **not** proof of low
identity. Untraced upstream SLEEC/backbone exposure remains **UNKNOWN**.

ReactZyme training reactions are participant sets without side assignments.
Those are compared to query participant sets and labeled as such: they cannot
establish exact transformation/pair exposure. Directed sources additionally
use reversal-invariant canonical exact matching. These are conservative string
comparisons, not equivalence under arbitrary cofactor/proton changes. Invalid
source chemistry is counted and exported; it cannot supply chemical references.

The baseline finds the most chemically similar training reaction(s), then ranks
each candidate by its maximum reported MMseqs bitscore against their known
enzymes. Chemical similarity is a chiral Morgan-radius-2 participant proxy
(2 × 2,048 bits, direction-aware for directed sources, symmetric for participant
sets), **not a mechanistic-equivalence oracle**. All tied nearest reactions are
included. No alignment hit gives score zero; its tie bounds are reported.
MMseqs uses sensitivity 7.5, E-value 0.001, and at most 10,000 reported targets.
This is an explicitly defined control, not a claim to reproduce a published
BLAST baseline exactly.

The main comparison always uses all 45 released queries and identical pools.
Exposure flags are reported separately, not used to give different models easier
subsets. This experiment alone does not demonstrate leakage-free generalization
or prospective activity.

## Run

From the project directory, use one **idle** GPU. The launcher never stops or
shares another job. For example, if physical GPU 1 is available:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
bash scripts/launch_cyp_specificity.sh 1
tail -f runs/cyp_specificity_v1/pipeline.log
```

The detached `cyp_specificity_v1` session runs preparation verification → input/
checkpoint preflight → CPU audits/baselines → feature extraction → frozen model
scoring → reports. GPU inactivity during the CPU audit is expected. Extraction
details are in `runs/cyp_specificity_v1/features/*.log`. Successful stages are
reused only when their recorded inputs/outputs match; interrupted MMseqs searches
are never accepted as complete. Changed inputs require a new run root.

ProtT5 extraction writes a resumable shard, then a separate CPU merge creates
the standalone `features/proteins.h5`. Completion is recorded only after both
steps succeed. The merge preserves length-sorted shard order; consumers index
embeddings by protein ID. A completed shard is validated and reused without
loading ProtT5 again. The `merge-proteins` stage runs only this CPU merge;
`--device` should retain the original extraction setting for provenance.
Recovery of the known missing-merge failure preserves the original input
receipt and records the migration in `features/prott5.merge_recovery.json`.

Individual stages and options are available through:

```bash
../.capability-run-py/bin/python scripts/run_cyp_specificity.py --help
../.capability-run-py/bin/python scripts/run_cyp_specificity.py preflight
```

`--skip-homology-audit` skips the broad training-sequence identity search, **not**
the smaller alignment needed for the baseline; it labels homology as not run.
Use the Python runner directly with `--run-root` for a separate experiment;
the detached launcher intentionally fixes its output directory/session.

## Outputs and additional baselines

`runs/cyp_specificity_v1/reports/summary.md` and `summary.json` contain Hit@1/5/10,
designated-positive MRR, screening depth, rank percentile, tie bounds, and
sequence-deduplicated diagnostics. The paired residual-minus-F3 MRR confidence
interval uses 2,000 organism-cluster bootstrap samples (seed 42), conditional on
these fixed checkpoints. It does not measure training-seed uncertainty.

**This MRR is not the all-positive ReactZyme MRR.** Per-query metrics and full
rankings are exported. `provenance.json` records model/checkpoint/code identities,
feature metadata, fixed alpha, training-source audits, and exposure counts.

An external method such as CLIPZyme can supply a CSV with exactly
`query_id,protein_id,score` for every published pair (larger is better):

```bash
../.capability-run-py/bin/python scripts/run_cyp_specificity.py report \
  --external-scores clipzyme=/absolute/path/clipzyme_scores.csv
```

This is a strict **score-file adapter**, not a native CLIPZyme inference launcher.
External model/training provenance remains the provider's responsibility.
No external baseline result is fabricated or copied from a different pool.

Regression tests:

```bash
../.capability-run-py/bin/python -m pytest -o addopts= -q \
  tests/unit/test_cyp_specificity.py tests/unit/test_biological_residual.py
```
