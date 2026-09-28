# Reaction-smi generalization diagnostics

This is a validation-only diagnostic suite, not another training sweep. It compares
the existing no-classification, lambda 0.02, and **fresh** lambda 0.05 runs at the
same human epoch 9 / optimizer step 864. Immutable named snapshots are preserved
under `runs/circe_generalization_diagnostics_v1/checkpoints`. Their training/model
hyperparameters must agree except for the auxiliary weight and family weights.
The continued/resumed lambda 0.05 run is deliberately excluded.

The no-classification snapshot was previously saved inside an evaluation directory.
Only its weights are reused; no test metrics, test associations or test queries are
used to select these matched checkpoints or build the diagnostic baselines.

## Experiments

1. Compare all three models at matched training budget, in **both directions**.
2. Separate training-seen, training-unseen and mixed reaction query groups. Rank
   against the original complete validation pools first; stratify afterwards.
   Multi-positive enzymes retain all of their validation positives.
3. Stratify by molecular-composition similarity, retrieved enzyme sequence identity,
   their joint bins, positive count, actual training auxiliary-target coverage,
   missing reaction modalities and missing sequence hits.
4. Run two inexpensive training-association transfer baselines:
   - Reaction-first: nearest training chemistry, then retrieved sequence similarity
     to the enzymes annotated for that training reaction.
   - Enzyme-first: highest-bit-score training enzyme, then chemistry similarity to
     the reactions annotated for that enzyme.
5. Report paired deltas against no classification, **and 0.05 against 0.02**,
   with reaction-component cluster-bootstrap intervals. Export all available
   epoch-level loss components, weighted auxiliary terms, gradient diagnostics
   (when logged), modality weights and validation MRR from the training logs.

Metrics: ReactZyme all-positive MRR, first-positive MRR, Recall@10, and Hit@10
(the evaluator's historical `top_10`). They are not interchangeable. All-positive
MRR has a positive-count-dependent ceiling, so inspect positive-count strata.

## Launch

On a machine with the repository, feature caches, Python environment and one or
more **free** GPUs, choose their physical IDs explicitly:

```bash
env -u BASH_ENV -u ENV bash --noprofile --norc \
  /datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/launch_circe_generalization_diagnostics.sh \
  --gpus 2 --threads 8 --target-batch-size 128
```

Only use `2` if it is actually free. To evaluate all three models concurrently,
replace `--gpus 2` with three free IDs, e.g. `--gpus 0,1,2`. This is independent
checkpoint evaluation, **not DDP**. One GPU instead runs them sequentially.
The launcher refuses occupied GPUs and never stops existing jobs.

```bash
tail -f /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/circe_generalization_diagnostics_v1/pipeline.log
```

Detailed progress is in `logs/none.log`, `logs/cls002.log`, `logs/cls005.log`, and
`logs/cpu.log` inside that run directory. Preparation checks happen before these
logs appear. Closing the terminal does not stop the detached session.

After a failed run has exited, the same launch command reuses completed stages.
For GPU OOM, lower `--target-batch-size` to 64 or 32; this changes throughput, not
the candidate pools. Do not edit a running experiment's code or inputs: provenance
guards reject stale results. Changed scientific settings require a new `--output`.

CPU-only preparation or report regeneration (no GPU needed):

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
../env/bin/python scripts/run_circe_generalization_diagnostics.py prepare
../env/bin/python scripts/run_circe_generalization_diagnostics.py cpu --threads 8
# Only after all three model stages and the CPU stage finish:
../env/bin/python scripts/run_circe_generalization_diagnostics.py report --bootstrap 1000
```

## Runtime and outputs

- No retraining and no new ProT5/ReactionT5/UniMol extraction.
- Existing residue/reaction features are reused. Each checkpoint projects the
  validation candidates once; target embeddings and per-query metrics are cached.
- Molecular similarities and one MMseqs train-versus-validation search are shared
  across all methods. The CPU stage runs concurrently with GPU evaluations.
- Ranking uses bounded matrices (about 160 MB per baseline here), not a global
  training-by-training similarity matrix. Reports reuse ranks without GPU work.
- NFS can still be the bottleneck. More parallel GPUs do not guarantee a speedup;
  avoid launching duplicate jobs or copying the entire residue cache.

Results in `runs/circe_generalization_diagnostics_v1`:

- `report.md`: main seen/unseen and bidirectional table with confidence intervals.
- `summaries.csv`: every metric and diagnostic stratum, with query/cluster counts.
- `paired_differences.csv`: paired checkpoint and baseline comparisons.
- `*_per_query.json`: auditable query-level results and descriptors.
- `training_curves.json`: logged losses, gradient diagnostics and validation MRR.
- `manifest.json`, `novelty.json`: provenance, data/search definitions and coverage.

## Interpretation safeguards

- ReactZyme molecule-set strings do not reliably encode reaction sides. Morgan
  Tanimoto here describes unordered molecular composition, **not reaction-center
  novelty or catalytic mechanism**. Parse failures stay in the benchmark.
- MMseqs is an approximate search (sensitivity 7.5, E-value 0.001, at least 80%
  coverage on both sequences, at most 64 prefilter candidates by default).
  Sequence identity is the maximum among returned hits, not an exhaustive maximum
  over all training sequences. No hit is **unknown**, never zero identity.
- R-to-E sequence strata describe the ground-truth positive set: the minimum
  retrieved best-hit identity among positives with hits, with completeness reported
  separately. They do not imply a query has no close training homolog when some
  positives lack hits. E-to-R uses the query enzyme's own best retrieved identity.
- Auxiliary-target coverage describes supervision actually available during
  training, not a comprehensive independent annotation of validation proteins.
- Baselines use no validation activities as reference associations. They retain
  no-hit queries with tied scores, use sorted candidate IDs to break ties, and
  report all-tied query counts. A weak baseline with many ties is inconclusive.
- Cluster intervals account for shared reactions and multi-activity enzymes;
  they condition on this seed and these candidate pools. They are exploratory,
  not multi-seed evidence or multiplicity-adjusted significance tests.
- The next *training intervention* is intentionally conditional on the diagnostic
  findings. Do not automatically change lambda, towers, losses or negatives, and
  do not tune on the held-out test set.
