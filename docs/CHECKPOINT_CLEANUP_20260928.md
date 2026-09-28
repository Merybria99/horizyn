# Checkpoint cleanup — 2026-09-28

The audit removed **239 unreferenced intermediate encoder checkpoints**.
They represented 264.79 GiB of allocated file blocks; 263.58 GiB belonged to
single-link files and was reclaimed by deletion. One deleted path was a hard
link, so its shared model contents may still occupy space through another
retained path. No deletion failed.

The original output inventory contained 1,916 `.ckpt` files. **1,677 were
retained** and verified against their original sizes and modification times.
All **14 critical V4/F3/CIRCEv2 model/config assets** also passed SHA256 checks
before and after pruning.

## Retention rules

The audit kept:

- Every checkpoint found in the inspected scientific metadata, configuration,
  reports, code, manuscript, or findings references.
- Named best, selected, last, and other non-periodic snapshots.
- The newest two distinct saved epochs in each checkpoint directory.
- Entire frozen/pretrained collections and checkpoint sets of three files or fewer.
- All checkpoints in the generalization campaign, recent `cersei_`/`cercei_`
  manuscript campaigns, original F3/F4 paper run, and CIRCEv2 runs.
- All Git-tracked checkpoints.

Only explicitly named intermediate epoch snapshots older than seven days,
with no retained reference, were eligible. Each file's inode, type, size, and
modification time were checked again before deletion. No active training was
present during the cleanup.

Most removals came from older EC-auxiliary, geometry, loss, feature-gate, and
normalization sweeps. Their selected/result-linked models, reports, numerical
results, and configuration records remain available. Replaying every historical
saved epoch would require retraining the deleted intermediate states; this
cleanup does not retain every possible retrospective diagnostic.

## Scope and evidence

The reference audit inspected 4,110 files. Large hard-negative JSON mappings
were streamed for checkpoint references. A paper note removed during manuscript
cleanup was read from its verified external archive. The reference scan had no
read/size errors for its listed sources.

This was an audit of inventoried `.ckpt` files, not a claim that every model
weight in the workspace is necessary. `.pt`/`.safetensors` models, residual
heads, pretrained weights, HDF5 features, and other data were not pruned.
Permission-restricted folders excluded by the original output inventory remain
untouched. Files and symlink aliases not covered by the original inventory
were not deletion candidates.

The workspace's
[`changes/checkpoint_cleanup_20260928/`](../../changes/checkpoint_cleanup_20260928/)
contains `plan.json`, `protected_checkpoints.json`, `deletions.jsonl`,
`summary.json`, and the one-off cleanup implementation. The compiled paper's
68 source/style/bibliography/figure files were also verified unchanged during
the separate manuscript-folder cleanup.
