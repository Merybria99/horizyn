# Parallel reaction extraction on a second node

`scripts/horizyn1_parallel_reactions.py` reuses the unchanged reaction-held-out
controller and encoder commands. It does not change the split, representations,
sampler, loss, batch size, or training schedule.

The external cache must be outside the main run. It binds to the main run's
protocol and implementation files. Three workers each use one explicitly selected
GPU (ReactionT5v2, Uni-Mol2, ChIRo); a fourth CPU worker builds chemistry features
using **train_rxns.csv** to fit the schema. All four read the existing reaction
inventory. The ChIRo molecule cache is private to the external cache; no process
opens the original cache for writing.

On tyrosine (`slurm-node-013`), after verifying GPUs 1,2,3 are free:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
tmux new-session -d -s circe_reactions_tyrosine -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=1,2,3 GPU_COUNT=3 PYTHONUNBUFFERED=1; exec ../env/bin/python scripts/horizyn1_parallel_reactions.py extract --profile h200 --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --cache-dir runs/horizyn1_reactions_tyrosine_20260910 >> runs/horizyn1_reactions_tyrosine_20260910.launch.log 2>&1'
```

On **slurm-node-014**, switch the existing controller once. The stop is graceful:
it never sends KILL in this command. If it times out, do not start a replacement;
wait and rerun the verified stop helper. Checkpointed protein work is resumed,
not discarded. Keep the extraction batching options identical to the active run.

```bash
(
set -e
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
if tmux has-session -t circe_v2_parallel_resume 2>/dev/null; then
  echo 'Resume session already exists; inspect it instead of launching twice.'
  exit 1
fi
../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --timeout 60
tmux new-session -d -s circe_v2_parallel_resume -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 PYTHONUNBUFFERED=1; exec ../env/bin/python scripts/horizyn1_parallel_reactions.py resume --profile h200 --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --cache-dir runs/horizyn1_reactions_tyrosine_20260910 --extraction-batch-size 256 --extraction-max-tokens 65536 --extraction-length-sort --extraction-padded-token-budget >> runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs/pipeline.log 2>&1'
)
```

The resume controller holds the original main-run lock. Protein extraction and
merge follow the original code path. Each reaction stage waits (up to 24 hours)
for its external validated receipt; there is no hidden fallback recomputation.
It verifies input/config signatures, complete reaction coverage where required,
valid ragged offsets, finite vectors, and content hashes. Missing Uni-Mol2/ChIRo
rows remain allowed exactly as in the original pipeline. Outputs are atomically
copied without overwriting an existing file, and import journals permit recovery
after a partial publication. The normal stage fingerprints are retained and
the journals are included in the stage inventory. Training and testing then run
normally on node-014. Do not run two controllers against the same main run.

The original pilot timing report has timestamp-only drift on the shared
filesystem. Under the main-run lock, resume independently revalidates the pilot
dependencies, feature contents, counts, configuration and derived projections
before rebinding only this report's signature. The original receipt and report
are backed up under `metadata_repairs/pilot_report_timestamps_*`. Changed inode,
size, content semantics, or any other stale artifact still fail closed; no
historical byte-equality claim is made without a historical content hash.
External feature files and receipts are fsynced before recording signatures.

Monitoring:

```bash
tail -n 30 /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_reactions_tyrosine_20260910.launch.log
tail -n 10 /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_reactions_tyrosine_20260910/logs/*worker.log
tail -n 20 /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs/pipeline.log
```

Keep the external cache immutable until the import is complete. Do not change
the existing controller/extractor code during either run. A completed original
reaction stage is still reused normally, even if it finished before the switch.
