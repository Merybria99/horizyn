# Experiment 1: measured within-family activity comparison

Run from `/datastor2/deep-proteins/EnzymeDiscovery/horizyn` on tyrosine.

This is frozen-checkpoint evaluation. It does not train models, tune thresholds,
select checkpoints on assay labels, or modify the existing CYP benchmark.

## Panel and experimental meaning

The first implemented panel is the released nitrilase matrix from
[enzyme-datasets](https://github.com/samgoldman97/enzyme-datasets/tree/627556e265e2a52d39753e684c05b550d53a9be4),
associated with [Black et al., Chemical Communications 2015](https://doi.org/10.1039/C4CC06021K).
It contains 18 distinct protein sequences, 38 substrates, and 684 measured pairs:
85 active and 599 labelled inactive. Eight proteins and three substrates have no
active entries. All 684 pairs remain in the primary evaluation.

The archived curated workbook and processed labels are pinned by SHA256. The
preparer rejects missing/nonfinite raw measurements and checks the multiset of
activity profiles between workbook and CSV. The public parser otherwise maps
every value not greater than zero, including missing values, to zero. The audit
therefore matters even though this particular workbook has no missing cells.
Protein IDs use sequence hashes. Identical activity profiles are not used to
guess an accession-to-sequence mapping.

The binary threshold follows the released dataset: its positive curated signal
is active, zero is no detected activity under the assay conditions. This does
not establish universal inactivity, successful expression of every enzyme, or
zero catalytic rate. The original screen uses ammonia detection; this benchmark
does not have product-resolved or kinetic ground truth.

## Reaction inputs

Every substrate receives the same label-independent single-nitrile hydrolysis
template, including two waters and ammonia. Distinct canonical products from
different possible nitrile sites are enumerated, and each model's maximum score
across those fixed variants is the final substrate score. The panel happens to
produce 38 distinct variant entries for 38 substrates; the symmetric dinitrile
has one unique single-hydrolysis product after canonicalization.

Products are computational template hypotheses. They do not support a claim
about regioselectivity, stereoselectivity, or the experimentally predominant
product. No activity label is used to generate or choose a product.

## Frozen comparisons

`configs/benchmarks/activity_panels_nitrilase.yaml` names:

- CIRCE v2 trained on ReactZyme, epoch 23;
- CIRCE v2 large-corpus checkpoint selected in the existing validation-fixed run;
- F3 trained on ReactZyme, epoch 29;
- official Horizyn-1 development checkpoint, using its pinned native code;
- training-only chemistry-plus-sequence transfer, separately using ReactZyme
  and the reconstructed large-corpus training associations;
- exact random-ranking expectation.

The two CIRCE variants were explicitly requested as separate comparisons.
These models use different training corpora; this is a comparison of frozen
systems, not a controlled architectural ablation. Official Horizyn development
is not silently substituted for the paper's large inference-library experiment.

ProtT5 residue features, ReactionT5v2, UniMol2 and ChIRo features are generated
with existing extractors. Chemistry vectors use each model's fitted schema,
without refitting on this panel. Protein sequences are 259–363 residues, so none
are truncated at the 1,022-residue cap. Official Horizyn uses the full-sequence
mean of the shared ProtT5 cache, its native fingerprints, and forward scoring.
The protein cache stores FP16 vectors; final pooling/scoring uses FP32. CIRCE
uses each checkpoint's existing reaction aggregation and protein pooling.

No whole reaction or candidate is dropped because a model has a missing feature.
Existing extractor rules may omit non-encodable small molecules; omissions are
logged. Complete reaction-modality availability is checked before scoring.

## Metrics

For R→E, each substrate/reaction ranks all 18 enzymes. For E→R, each enzyme ranks
all 38 substrate/reaction queries. Results are macro-averaged over queries.

- Average precision: expected item-level AP over uniform random ordering within
  exactly tied scores. This differs from grouped-threshold PR-area conventions.
- Hit@1/3/5/8/24: probability of finding at least one measured active pair.
- Active hits@K, precision@K and recall@K.
- First-positive MRR, explicitly distinct from ReactZyme all-positive MRR.
- Requested K values exceeding the pool are capped; `k_used_*` records the cap.

No-positive queries receive AP/Hit/recall zero in the all-query summary. A
separate conditional summary includes only queries with an active candidate;
counts are always reported. Thus even a perfect ranking cannot reach an
all-query score of one on this panel.

An additional R→E diagnostic removes enzymes inactive on every tested substrate.
This uses assay labels to define a retrospective subset and is not a deployable
candidate-selection rule. It helps distinguish substrate preference from broad
panel reactivity. The unchanged 18-enzyme panel remains primary.

Random scores are tied exactly and integrated analytically, not approximated
using one lucky shuffle. All methods must supply precisely one finite score for
every measured pair. The tests compare tie-aware AP, MRR and Hit@K against
exhaustive permutations, and test missing/duplicate/nonfinite scores, tampering,
reaction stoichiometry, equivalent SMILES and all-inactive queries.

Paired AP differences use 2,000 query-bootstrap replicates, seed 42. These
intervals are exploratory: related substrates/proteins are correlated, multiple
comparisons are unadjusted, and one family is not independent family replication.

## Exposure audit and simple controls

Each internal training source is scanned for exact sequence and reaction/pair
exposure. Only positive training associations form the reference graph. The
large sequence catalog is streamed; evaluation labels never define references.

For each query, the chemistry/sequence baseline selects all tied nearest
training reactions using a chiral Morgan-2 side fingerprint. It ranks a panel
protein by its maximum reported MMseqs bitscore to proteins of those reactions.
No alignment hit yields a tied zero score. Parameters follow the existing CYP
audit helper and are recorded in the code/protocol binding.

ReactZyme has participant sets without side labels, so its matches are reported
as participant-set exposure, not exact transformations. Directed comparisons
retain participants and charges; differences in cofactors/protons may hide
chemical equivalence. Upstream SLEEC, foundation-model and official Horizyn
supervised exposure remain unknown. Broad training-relative homology search is
not performed; baseline alignments do not establish remote generalization.

## Run and resume

Use one idle GPU. The launcher refuses an occupied GPU or duplicate session and
never stops another job:

```bash
bash scripts/launch_activity_panels.sh 0
tail -f runs/activity_panels_nitrilase_v2/pipeline.log
```

The full run performs preparation, preflight, feature extraction, all frozen
model predictions, source audits/controls and the comparison report. No model
training is launched. Existing source data, checkpoints and CYP outputs are not
modified. Source and output receipts govern reuse; a changed protocol, model,
code or cache requires a new run directory rather than silent overwrite.

Individual stages:

```bash
../.capability-run-py/bin/python scripts/run_activity_panels.py prepare
../.capability-run-py/bin/python scripts/run_activity_panels.py preflight
CUDA_VISIBLE_DEVICES=0 ../.capability-run-py/bin/python scripts/run_activity_panels.py features --device cuda
CUDA_VISIBLE_DEVICES=0 ../.capability-run-py/bin/python scripts/run_activity_panels.py score --device cuda
../.capability-run-py/bin/python scripts/run_activity_panels.py audit
../.capability-run-py/bin/python scripts/run_activity_panels.py report
../.capability-run-py/bin/python scripts/summarize_activity_panel_run.py
```

Use `--run-root` for a separate run and `--config` for an explicitly different
protocol. Retain the original extraction/scoring `--device` when reusing those
stages. The source preparer currently implements nitrilases; additional panels
need their own audited chemistry and label preparation before inclusion.

Validation command in the existing environment (coverage plugin is absent):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../.capability-run-py/bin/python -m pytest -o addopts= -q tests/unit/test_activity_panels.py
```

## Outputs

- `data/external/activity_panels_2026/nitrilase/benchmark/manifest.json` and
  canonical query/protein/reaction/candidate tables;
- `runs/activity_panels_nitrilase_v2/protocol.json`;
- `features/`: extractor logs, cached embeddings and completion receipts;
- `scores/`: complete pair scores, variant scores and checkpoint/code hashes;
- `audit/<source>/`: pair exposure, nearest reactions, baseline scores and audit
  receipts;
- `report/summary.md`, `summary.json`, `paired_differences.csv`, per-query CSVs,
  `comparison.png` and `comparison.svg`;
- `report/readout.md`: comparison and exposure counts, plus activity-recovery
  curves in PNG/SVG, generated by `summarize_activity_panel_run.py` with a separate
  checksum receipt. This final step requires a complete comparison and also
  accepts `--run-root`.

The report lists missing methods explicitly and sets `complete=false` until all
configured comparisons exist. A partial report is never presented as a complete
model comparison.

The first complete run is `activity_panels_nitrilase_v2`. The earlier `v1` run
retains its partial outputs for traceability after an isolated-worker interpreter
path issue; those outputs are not used in the final comparison. The corrected
launcher preserves the virtual environment's Python path. All 23 focused tests
passed, including checks that training-only transfer ignores assay labels and
that catalog membership without a positive training edge is not exposure.
