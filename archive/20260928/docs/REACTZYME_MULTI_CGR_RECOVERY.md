# Multi-equation recovery audit

`scripts/recover_reactzyme_multi_cgr.py` extends the isolated chemistry-only
recovery pass. It does not edit training, splits, or the baseline feature masks.
The default source is `runs/reactzyme_cgr_recovery_v1`; its input hashes and
mapping provenance are verified before reuse.

The new search allows 128 candidate equations, six equations per collection,
and 20,000 search nodes. A group proceeds only when the search returns one
decomposition without a truncation flag, all equations pass the balance checks,
Rhea provenance exists, the participant union equals the query after the same
water/proton/set lookup normalization, and no equation is redundant.

These tests establish chemical consistency, not biological ground truth. Search
uniqueness is relative to the finite catalogue and configured bounds. They cannot
recover physiological direction or distinguish all possible real assignments.
No target-enzyme annotations or EC labels are used to select a decomposition.

Existing constituent atom maps are reused; missing equations are mapped once on
CPU with four threads using the same model/options as the baseline. Every record
is saved atomically. Re-running the command resumes after verifying provenance:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../env/bin/python -u scripts/recover_reactzyme_multi_cgr.py \
  --output runs/reactzyme_multi_cgr_recovery_v1
```

Outputs:

- `recovery/`: expanded-search query assignments and source equations.
- `constituent_mapping/`: internal equation-grain worklist and new mapped graphs;
  these internal worklist entries are not benchmark queries.
- `query_audit.json`: one audited status for every original query.
- `{partition}_multi_cgr_index.json`: unchanged `baseline_mask`, separate
  `multi_cgr_hypothesis_mask`, per-equation graph paths, and
  `assignment_verified=false`. Missing constituent graphs disable the hypothesis
  mask, even when the remaining constituents are valid.
- `report.json`: baseline coverage and additional graph-complete hypotheses,
  separately counted. `exploratory_fraction` includes hypotheses and must never
  be reported as certified recovery coverage. Reports update before and after
  mapping; the live mapping log shows intermediate progress.

The existing detached session on slurm-node-013 is
`reactzyme_multi_cgr_recovery_v1`; its log is
`runs/reactzyme_multi_cgr_recovery_v1/pipeline.log`.

Larger-catalogue single-equation candidates from the separate coverage notebook
are not silently mixed into this pass. New outputs do not integrate a multi-CGR
encoder into the retrieval model or authorize changes to running jobs.
