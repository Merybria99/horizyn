# Paper consolidation and biological subgroup analysis

This pipeline implements evaluation/analysis only. It never trains a model,
extracts backbone features, reconstructs data, or changes the original results.

## Launch

On arginine (or another authorized healthy node), with GPUs 0,1,2 free:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
CIRCE_ANALYSIS_GPUS=0,1,2 env -u BASH_ENV -u ENV \
  bash --noprofile --norc scripts/launch_biological_residual_paper_analysis.sh
```

The default output is `runs/biological_residual_paper_analysis_v1`. Set
`CIRCE_ANALYSIS_RUN_ROOT` to another **new** `runs/biological_residual_paper_analysis_*`
directory when changing code/inputs. Existing source runs are never valid outputs.
Use `CIRCE_ANALYSIS_GPUS=0` to run sequentially on one GPU. GPU 3 is not used by
the default three-worker run; no occupied GPU processes are stopped.

The detached controller checks inputs, builds CPU biological metadata, evaluates
all three validation sets, then evaluates all three test sets and generates
reports. Within each evaluation phase, one split runs on each selected GPU.
The same launch command resumes a stopped run, reusing only completed artifacts
whose requests and output hashes still match. It does not launch duplicate
controllers. Per-split logs are `<split>/{validation,test}.log`.

```bash
tail -f runs/biological_residual_paper_analysis_v1/pipeline.log
```

## Harmonized protocol

- Reuse each split's existing final full-residual checkpoint; no checkpoint search.
- Verify every frozen F3 model tensor against that split's **epoch-29 parent**.
- Verify FP32 evaluation caches, parent identity, input hashes, reaction features,
  pair disjointness, and exact test-positive candidate membership.
- Sweep alpha `0,.025,.05,.075,.1` on validation only. Select mean bidirectional
  all-positive ReactZyme MRR, with a maximum `.005` directional validation drop;
  ties prefer smaller alpha. Test evaluates only zero and the selected alpha.
- Export both all-positive and first-positive MRR, Hit@K, P@K, mean rank and counts.
- Keep published TIGER Table 1 values visibly separate from locally reproduced
  results. Its exact evaluator equivalence remains unverified.

This is a harmonized re-evaluation after historical test inspection, not a new
untouched-test claim. Some older F3 tables used validation-selected checkpoints
and older residual runs selected alpha with first-positive MRR; their numbers
are not silently mixed into this report.

## Biological interpretation and validation

The existing Parquet audit must match actual train/validation/test pair membership.
Sequence alignment FASTAs must match actual protein sequences, and exported best
hits must match raw MMseqs output. Sequence bins require at least 80% coverage in
both directions; low-coverage and no-reported-hit queries are separate groups.

Chemistry neighbors are recomputed on CPU against ReactZyme training and the
hash-verified source-pretraining reaction inventory, then combined by maximum
similarity. Features are OR-combined radius-2, 2048-bit Morgan participant
fingerprints. They do not encode reaction direction or establish mechanistic
equivalence; fingerprint equality is not proof of identical transformations.

Other strata cover complete/partial EC annotation, native enzyme cofactor
annotation, source protein identifier exposure, original wildcard chemistry,
known-positive/training support, and UniMol2/ChIRo availability. No annotation is
passed to inference. Missing annotations remain `unknown`, never negative labels.
Reaction EC annotations are source-association aggregates, not direct mechanism labels.

Source-pretraining **sequence homology** and upstream F3/SLEEC training exposure
remain unknown. Identifier non-overlap does not establish absence of homologs.
This pipeline does not claim a complete effective-training homology audit.

Subgroup differences use matched F3 queries and 2,000 paired-query bootstrap
replicates. Groups with fewer than 20 queries are flagged. These are descriptive
intervals conditional on one training seed, not multiplicity-corrected tests or
uncertainty over training seeds. Related queries may not be independent. Known
association counts are annotation support, not complete enzyme promiscuity.

## Outputs and environment

- `manifest.json`: immutable code/input hashes, cache identities and runtime.
- `<split>/query_metadata.csv`: biological metadata with explicit unknown fields.
- `<split>/{validation,test}.{json,per_query.csv}`: consistent new evaluations.
- `reports/summary.md`: three-split F3/residual table and published TIGER references.
- `reports/test_metrics.csv`: complete bidirectional local metrics.
- `reports/validation_alpha_curves.csv`: validation-only alpha diagnostics.
- `reports/biological_strata.csv`: subgroup counts, effects and 95% intervals.
- `reports/metadata_coverage.csv`: all groups, including unknown/not-applicable.

GPU evaluation uses `../env/bin/python`. CPU metadata/reporting uses the existing
`../.capability-run-py/bin/python`, which already provides PyArrow and RDKit; no
installation or training-environment change is required.

CPU preflight, without inference:

```bash
../env/bin/python scripts/run_biological_residual_paper_analysis.py preflight
```

Rebuild reports from verified completed outputs without GPU inference:

```bash
../env/bin/python scripts/run_biological_residual_paper_analysis.py report
```
