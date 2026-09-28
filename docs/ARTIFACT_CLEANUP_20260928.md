# Output artifact cleanup — 2026-09-28

The output audit removed **3,529 disposable files**, totaling **1,184.82 GiB
(1.157 TiB)** of allocated storage. It also removed 545 empty scratch/cache
directories. No deletion failed. Storage is calculated from each deleted
file's allocated blocks; all deleted files had a single hard link. This is
not a before/after free-space measurement of the shared filesystem.

## What was removed

| Artifact | Files | Allocated GiB | Reason |
|---|---:|---:|---|
| MMseqs working databases | 501 | 1,113.87 | Stale preparation/search scratch under `audit_tmp`, `cluster_tmp`, or `search_tmp` |
| Historical candidate embeddings | 168 | 58.01 | Regenerable `encoded_all_checkpoints/*.targets.pt` caches from two July ablation campaigns |
| Interrupted feature writes | 2 | 12.88 | Incomplete writes with a completed sibling feature file |
| Python bytecode | 2,858 | 0.07 | Regenerated automatically when the corresponding source is imported |

The deleted candidate caches belong to
`outputs/reactzyme_latent_organization_ablation_20260714` and
`outputs/reactzyme_representation_ablation_v2_20260714`. Their encoder
checkpoints, configurations, evaluation tables, and reports remain available.
Repeating their historical all-checkpoint scoring requires re-encoding the
candidate bank.

Most scratch space came from
`runs/horizyn1_circe_v2_h200/work/preparation/audit_tmp`. Its MMseqs query,
target, prefilter, and result databases were temporary preparation products.
Completed preparation inputs and clustering outputs outside these scratch
directories were retained. Restarting that historical preparation audit may
require rebuilding its search databases.

The two interrupted writes were `esterase_audit/features/proteins.h5.partial`
and `biological_residual_controls_reaction_smi_v1/R3_unguided/cache/`
`reactzyme_tokens.h5.merge.276379.tmp`. Their completed `.h5` siblings remain.
The unfinished `full_f3_bce_seed42_gpu_cache/best.pt.tmp` was **retained**:
it is the only saved model file in that run and may still be recoverable.

## What was preserved

All trained model weights outside the two explicit incomplete feature writes,
raw and completed HDF5 features, baseline structures and alignments, run
configurations, metric tables, manuscript figures, reports, and case-study
rankings were retained. In particular, the original V4 model manifests and
their encoder/refiner weights, the F3 epoch-99 checkpoint, and the CIRCEv2
epoch-23 checkpoint remain at their original paths.

No active training or MMseqs search process was present at execution. No
tracked files were deleted. The retained pipeline code, current configs,
case-study Python files, manuscript sections, and current documentation had
no references to the removed historical cache/scratch directory names.

## Checks and limitations

The cleanup used the original output inventory, restricted filename/path
rules, and a minimum file age of 24 hours. Each file was checked again for
its type, inode, size, and modification time before deletion. Symlinks were
not followed; directories were removed only when empty.

All **14 protected model/config assets** passed SHA256 verification before
and after cleanup, including comparison with frozen V4 manifest hashes.
The maintained CLI still lists exactly V4, F3, and CIRCEv2. A separate
retention check verified **290,391 retained scientific files** against their
original sizes and modification times, with no discrepancies. All 3,529
deleted paths were confirmed absent. Results are recorded in
`retention_verification.json`.

The inventory covered `outputs`, `runs`, `case_studies/out`, and the
implementation's `outputs`, `results`, `reports`, `runs`, `lightning_logs`, and
`logs`. Nine preexisting restricted results/log folders could not be read;
these were left untouched. The audit does not claim that every historical
output is necessary: larger model and feature collections were retained
because deleting them would remove reproducibility assets.

## Audit trail

The workspace's
[`changes/artifact_cleanup_20260928/`](../../changes/artifact_cleanup_20260928)
directory contains:

- `inventory.json`: original file sizes, allocated blocks, timestamps, and scan errors.
- `plan.json`: exact deletion candidates and reasons.
- `deletions.jsonl`: one record for every actual deletion.
- `summary.json`: totals, skipped files, errors, and protected asset count.
- `protected_assets.json`: retained model/config hashes.
- `retention_verification.json`: checks of retained scientific files.
- `cleanup.py`: the one-off cleanup implementation; its default mode plans without deleting.

The code and audit metadata were subsequently published on the workspace and
implementation repositories' `ICLR2027` branches. Large local artifacts and
the full pre-cleanup inventory remain outside Git.
