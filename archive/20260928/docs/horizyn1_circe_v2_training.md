# CIRCE-v2 training on the reconstructed Horizyn1 corpus

This pipeline trains new CIRCE-v2 retrieval towers on the reconstructed corpus;
it does not claim to reproduce the authors' unreleased collection exactly.
The pretrained ProtT5, ReactionT5v2, UniMol2, ChIRo and SLEEC weights stay frozen.
Pooling, attention, adapters and projections are trained from initialization.

## Training specification

- Protein: cached FP16 ProtT5 residue vectors; 1,022-residue ends/centre truncation;
  frozen SLEEC scorer; the existing five-block 512-dimensional enzyme tower.
- Reaction: whole-reaction ReactionT5v2 plus independently pooled reactant/product
  UniMol2 and ChIRo, using `directional_delta`; train-fitted 617-dimensional
  chemistry features and the existing trainable modality fusion.
- One normalized vector per entity, cosine retrieval, bidirectional sampled
  multi-positive InfoNCE with equal directional weights and an active-anchor
  guard in both directions.
- Exactly 85% positive / 15% explicitly sampled-negative rows. The mixture
  *within* negatives is 50% biological / 50% random where sampling permits;
  shortages and fallback counts must be inspected in training diagnostics.
- Each 100-row batch contains **55 base-sweep positives, 30 endpoint-support
  positives and 15 negatives**. An epoch traverses the base-sweep training edges
  once, with bounded padding at the tail. At four GPUs and 100 rows/GPU its length
  is `ceil(number_of_training_pairs / 220)` steps, not division by 340; the
  endpoint-support rows are additional positives, not part of that base sweep.
- Explicit negatives have positive support at both endpoints. Their sampled
  candidate entities can be reused within the batch only after checking each
  proposed pair against the train-only negative policy. The 15% quota refers to
  sampled rows, not the fraction of negative cells in the contrastive matrix;
  other unlabelled cells are not automatically negatives.
- Auxiliary classification weight remains **zero**. Biological annotations are
  used for training-only sampling, never required from a query at inference.
- Four GPUs by default, 100 pair rows per GPU, learning rate `1e-4`, weight decay
  `0.01`, FP32 initially, at most five epochs / 100,000 optimizer steps, periodic
  fixed-panel validation every 2,000 training steps, early stopping after five
  non-improving validations. The pilot is a numerical/infrastructure acceptance
  check, not an architecture comparison or an estimate of test accuracy.

The launcher generates an immutable resolved configuration from
`configs/horizyn1_circe_v2.yaml` (`--profile base`) or the opt-in
`configs/horizyn1_circe_v2_h200.yaml` (`--profile h200`). Do not launch the
unresolved templates directly.

## Optimized four-H200 profile

Use a **new run directory** for this profile; the launcher never migrates or
modifies the old active preparation database. No local SSD is required.

| Setting | Base | H200 |
|---|---|---|
| Training precision | FP32 | BF16 mixed, with FP32 contrastive loss and sensitive SLEEC/Lorentz operations |
| Optimizer | Existing AdamW | Opt-in fused AdamW |
| Global sampled pair rows | 400 (100 × 4 GPUs) | **Unchanged: 400** |
| Training DataLoader | 2 workers/rank | 4 persistent workers/rank, prefetch 2, one Torch thread/worker |
| Attention-stat logging | Enabled | Disabled |
| ProtT5 extraction | Batch 8, token budget 4,096 | Batch 32, **padded** token budget 32,768; sequences sorted by length |
| Extraction progress/checkpoint | Every 100 proteins | Every 1,000 proteins (at most the uncommitted interval is redone) |
| Mutable preparation work | Dedicated shared run work | Dedicated shared run work; network-safe SQLite rollback journal |

The H200 profile retains the same learning rate, 85/15 row mix, negative policy,
model dimensions, frozen backbones and 1,022-residue sequence cap. It bounds
per-process Torch/BLAS threads to four. Its pilot uses the real BF16/fused-optimizer
configuration and real DataLoader settings, then verifies the checkpoint reached
the requested steps with finite parameters. OOM, nonfinite features or invalid
loss anchors stop the pipeline before full extraction. The pilot records
throughput estimates, but does not establish a guaranteed speedup.

Preparation now keeps SQLite and MMseqs temporary files in
`RUN_ROOT/work/preparation`, and publishes CSVs/manifests under `RUN_ROOT/data`.
On NFS/network storage it uses **DELETE rollback journaling, one exclusive writer,
FULL synchronous durability**, and a bounded 512 MiB SQLite cache. WAL is used
only on appropriate local filesystems. MMseqs progress remains visible remotely
in `RUN_ROOT/logs/preparation/mmseqs.log` and normal stage output in `logs/prepare.log`.
Shared-storage latency may remain a bottleneck despite these changes.

## What the staged launcher does

| Stage | Work |
|---|---|
| `preflight` | Check original audit freshness, local checkpoints/tokenizers, CPU/training environments, MMseqs and shared free space. No model inference. |
| `prepare` | Group proteins at the configured similarity/coverage thresholds, audit cross-split hits, group reactions, retain representative-owned positives, split entity groups 90/5/5, and freeze evaluation panels/full-catalog query gold. |
| `labels` | Rebuild weak mechanism/reaction-cofactor profiles using only training associations, while retaining separate native evidence and missing-label semantics. |
| `index` | Create compact memory-mapped positive/annotation indices; exclude transferred uncertain edges from candidate negatives, without using them as positives. |
| `pilot` | Select a length-stratified sample from the representatives, retain training-connected proteins, extract real frozen features, audit IDs/shapes/finiteness, benchmark shuffled residue reads, and run 20 steps of the actual four-GPU training architecture. |
| `features` | Only after the pilot passes: extract full frozen features once, merge ProtT5 through a virtual HDF5 over immutable shards, fit chemistry on training reactions, and audit feature coverage. |
| `train` | Train both CIRCE towers, automatically resuming `checkpoints/last.ckpt` after interruption. |
| `test` | Evaluate the validation-selected checkpoint against the full protein/reaction catalogs with separate heldout query filters and chunked ranking. Cache encoded embeddings for interrupted evaluations. |
| `all` | Execute all the above in order; completed, unchanged artifacts are reused. |

The main validation panel is a fixed both-cold, restricted-catalog development
panel. Full-catalog test results are a different protocol and must be labelled
accordingly. Full-catalog gold contains **all owned associations for each selected
heldout query**, including associations outside the both-cold quadrant.

No reverse chemical direction is invented as a new positive: extracted reaction
IDs are unsuffixed and follow the supplied forward SMILES. This is distinct from
training retrieval in both enzyme-to-reaction and reaction-to-enzyme directions.

## Launch detached

To replace the previous base run, stop it **on its execution node** first:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2 --timeout 30 --force-after-timeout
```

The helper verifies the controller command, user, run directory and PID start
identity. It sends TERM to that job and its observed descendants; the optional
force flag sends KILL only to verified survivors after 30 seconds. All run files
are preserved. A mismatched/stale PID is never blindly killed. Wait for the
helper to confirm shutdown before launching the replacement.

For the optimized profile, on the machine/allocation holding the four H200 GPUs:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
mkdir -p runs/horizyn1_circe_v2_h200/logs
tmux new-session -d -s horizyn1_circe_v2_h200 -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4; bash scripts/run_horizyn1_circe_v2_training.sh all --profile h200 >> runs/horizyn1_circe_v2_h200/logs/pipeline.log 2>&1'
```

No scratch path is needed. `--profile h200` defaults to the separate
`runs/horizyn1_circe_v2_h200` directory. It does not stop or reuse the old run.
Stop that run first with the verified stop helper **on the machine running it**,
if you intend to replace it; do not delete its SQLite/WAL files.

The original FP32 profile remains available:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
mkdir -p runs/horizyn1_circe_v2/logs
tmux new-session -d -s horizyn1_circe_v2 -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4; bash scripts/run_horizyn1_circe_v2_training.sh all >> runs/horizyn1_circe_v2/logs/pipeline.log 2>&1'
```

The command intentionally does **not** kill any existing tmux session or GPU job.
An existing session with the same name makes `tmux` fail safely. `tmux` is not a
Slurm allocation: obtain your site's GPU allocation first when required.

Check progress:

```bash
tail -n 60 runs/horizyn1_circe_v2_h200/logs/pipeline.log
tail -n 40 runs/horizyn1_circe_v2_h200/logs/prepare.log
tail -n 40 runs/horizyn1_circe_v2_h200/logs/preparation/mmseqs.log
cat runs/horizyn1_circe_v2_h200/current_stage.txt
tmux attach -t horizyn1_circe_v2_h200
```

Detailed logs are split by stage in `logs/`; protein extraction has one log per
GPU rank. Detach from tmux with Ctrl-B, then D. The first run spends substantial
time preparing splits and extracting features **before training epoch 1**.

Selected GPUs are checked when entering a GPU stage; a busy GPU produces an
actionable error instead of sharing silently or stopping someone else's job.
The advanced `--allow-busy-gpus` flag explicitly opts into sharing and is not
recommended for this multi-terabyte pipeline.

The reconstruction audit can be reused on another node accessing the same
shared files. Its node-local filesystem device number may differ; inode, size,
modification time and change time must still match. A device-only difference is
logged and accepted. Missing audit entries or any other changed field still
stop preflight with a detailed error. This is not a bypass for copied or changed
datasets, and the original audit is never rewritten by the launcher.

## Resume

### Permission-only metadata recovery during preparation

Changing permissions also changes file `ctime`. If this happens after a run has
bound its inputs, do not disable signature checks or delete the prepared data.
`scripts/repair_horizyn1_run_metadata.py` supports a narrowly scoped recovery
for shared-work runs that have not reached labels or extraction:

```bash
../.capability-run-py/bin/python scripts/repair_horizyn1_run_metadata.py verify \
  --run-root runs/horizyn1_circe_v2_h200 \
  --bundle-dir runs/horizyn1_circe_v2_h200/metadata_repairs/ctime_20260909
```

Verification may run alongside preparation: it does not open the live SQLite
database or change production receipts. It requires unchanged paths, inodes,
sizes and mtimes; verifies the seven historic preparation SHA-256 bindings;
reruns the entire reconstruction graph audit and compares its checks/counts;
then records fresh hashes for artifacts that lacked historical byte hashes.
That fresh audit is semantic revalidation, not proof of historical byte identity
for unhashed artifacts. A nonempty bundle is never overwritten by `verify`.

Stop the job **on its execution node**, then apply the verified bundle:

```bash
../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2_h200 --timeout 30 --force-after-timeout

../.capability-run-py/bin/python scripts/repair_horizyn1_run_metadata.py apply \
  --bundle-dir runs/horizyn1_circe_v2_h200/metadata_repairs/ctime_20260909
```

Do not continue if either command fails. Apply requires the controller, output
and work locks to be free and rehashes all verified inputs. It backs up the
original receipts, publishes the freshly verified reconstruction audit, and
changes only input ctimes in `state/prepare.json`. An interrupted apply can be
retried with the same bundle. No data, databases, file modes or pipeline code
are modified. Normal `all` preflight must run again; older native-annotation
extractor provenance is deliberately not rebased by this preparation repair.

For four H200 GPUs, this is an aggressive **unbenchmarked starting cap**, not a
measured maximum. It raises frozen-feature extraction capacity without changing
the training batch, loss, learning rate or split configuration:

```bash
tmux new-session -d -s horizyn1_circe_v2_h200_fast -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4; bash scripts/run_horizyn1_circe_v2_training.sh all --profile h200 --run-root runs/horizyn1_circe_v2_h200 --extraction-batch-size 256 --extraction-max-tokens 65536 --extraction-length-sort --extraction-padded-token-budget >> runs/horizyn1_circe_v2_h200/logs/pipeline.log 2>&1'
```

At the longest 1,022-residue inputs plus EOS, the padded budget limits each GPU
to 64 sequences rather than 256. The real pilot must establish VRAM use and
throughput; there is no automatic OOM backoff. Set extraction options before the
pilot starts: once extraction has a stage receipt, its options are immutable.
Completed preparation steps are reused, but uncommitted MMseqs work may repeat.

### Internal group-artifact mtime mismatch

The preparation database stores separate content evidence for committed groups:
path, size, mtime and SHA-256 (not ctime). If `protein_cluster_file` fails this
guard, a permission-only controller repair is insufficient. Never assume its
content is unchanged without comparing the stored SHA-256.

For an incomplete, stopped run with exactly the `entities` and `groups` stages,
`scripts/repair_horizyn1_group_evidence.py` permits only a protein-cluster mtime
update, and only if its historical path, size and SHA-256 match. Reaction-edge
evidence must match in full. It acquires all three writer locks, verifies bound
source/code hashes, backs up the entire SQLite database and any inactive rollback
journal, and updates one `groups` checkpoint value in a FULL-synchronous exclusive
transaction. It checks database integrity and rechecks both artifacts afterward.
The cluster bytes, entity/group assignments, splits, controller receipts and
pipeline code are not modified. Hot journals, WAL sidecars, downstream stages,
changed hashes or nonempty unfinished bundles are rejected.

```bash
../.capability-run-py/bin/python scripts/repair_horizyn1_group_evidence.py \
  --run-root runs/horizyn1_circe_v2_h200 \
  --bundle-dir runs/horizyn1_circe_v2_h200/metadata_repairs/group_mtime_20260909
```

Keep the bundle's full backups and `plan.json`. If interrupted before commit,
inspect the stopped state and use a fresh bundle for another attempt; unfinished
plans are not automatically replayed. When both evidence records already match,
the helper returns `already_current` without changing the database. After a
successful repair, use the same detached `all` launch command on the GPU node.

### Normal interruption recovery

After an interrupted preparation/extraction, rerun the same `all` command in a
new detached session after the old one has exited. Source/config signatures must
match. ProtT5 workers resume committed shard checkpoints. ChIRo reuses its
persistent molecule cache. Interrupted reaction-level UniMol2/ReactionT5 files
are preserved and their modality stage restarts; they are not falsely advertised
as molecule-level resumable caches.

Preparation has a durable initialization phase: interruptions while writing
ownership/state metadata or creating the empty SQLite schema can be retried
with the same command. Recovery verifies both directory locks, ownership,
bound inputs/options, and the absence of data before completing setup. Once
ingestion is enabled, missing scratch/database files still fail closed; they
are never silently recreated. Do not remove initialization markers manually.
This repair applies to new preparations made with this implementation. Runs
whose `state/prepare.json` already fingerprints older code still require a new
`RUN_ROOT`; the implementation-change guard is intentionally unchanged.

Interrupted label/index exports are also restartable: their controller-owned
partial directories are preserved under timestamped `.labels.interrupted.*` or
`.index.interrupted.*` names, then rebuilt at the original path. Foreign nonempty
directories without an owning stage manifest are never overwritten or archived.

If full features are already validated, resume training and then run the test:

```bash
tmux new-session -d -s horizyn1_circe_v2_h200_resume -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4; bash scripts/run_horizyn1_circe_v2_training.sh train --profile h200 >> runs/horizyn1_circe_v2_h200/logs/pipeline.log 2>&1 && bash scripts/run_horizyn1_circe_v2_training.sh test --profile h200 >> runs/horizyn1_circe_v2_h200/logs/pipeline.log 2>&1'
```

`train` requires the completed feature/pilot manifests. It resumes the last
checkpoint automatically; `--resume /absolute/path/to/checkpoint.ckpt` selects an
explicit checkpoint. Test checkpoint selection defaults to the validation-best
path recorded in Lightning's final checkpoint, not an arbitrary newest file.

## Paths and resource controls

The defaults target existing local installations and checkpoints. Environment
overrides include `RUN_ROOT`, `DATA_ROOT`, `PYTHON_BIN`, `SETUP_PYTHON_BIN`,
`ANNOTATION_PYTHON`, `PIPELINE_PYTHON`, `MMSEQS_BIN`, `PROTT5_MODEL`,
`REACTION_T5_MODEL_PATH`, `SLEEC_CHECKPOINT`, `COFACTOR_DICTIONARY`, and
`UNIMOL_WEIGHT_DIR`. `CIRCE_PROFILE` can select `base` or `h200` instead of a CLI
flag. `PROTT5_MODEL` must resolve to a local checkpoint directory.
No dependency installation or backbone download is hidden in the pipeline;
Hugging Face offline mode is enabled by default.

Examples of controller options:

```bash
bash scripts/run_horizyn1_circe_v2_training.sh preflight
bash scripts/run_horizyn1_circe_v2_training.sh preflight --profile h200
bash scripts/run_horizyn1_circe_v2_training.sh all --profile h200 --preparation-threads 64 --sqlite-cache-mib 512 --cpu-threads 4 --loader-workers 4 --reaction-workers 8
```

Extraction controls are `--extraction-batch-size`, `--extraction-max-tokens`,
`--extraction-progress-every`, `--extraction-checkpoint-every`, and the Boolean
`--[no-]extraction-length-sort` / `--[no-]extraction-padded-token-budget` options.
Their values are fingerprinted and recorded by the pilot. Changing them is not
silently treated as a compatible resume.

If local storage becomes available later, `--scratch-root /absolute/local/path`
(or `SCRATCH_ROOT`) is an explicit opt-in: the launcher checks local filesystem,
writability and `--min-scratch-gb` (default 100 GB), creates a unique owned run
directory, and binds it to that node. It never auto-selects local scratch and
never silently falls back to NFS for this explicit option. Shared storage remains
the default. A changed or missing bound work directory requires a new run or a
consistent stopped backup restoration, not live-database migration.

Batch size must be a positive multiple of 20 to represent the 85/15 quota exactly.
Changing sampling, splits, model weights, extraction settings or source files
invalidates the affected signed caches; use a **new** run directory when changing
the scientific configuration. Repeat any nondefault options when resuming.

There is a 6 TB initial free-space gate. The pilot computes exact truncated
residue payload from the FASTA, requires 25% headroom before full extraction, and
reports extrapolated protein extraction time. The free space is shared, not
reserved; the estimate is not an epoch-time guarantee. Virtual HDF5 files depend
on their original shard paths: **do not delete or relocate the source shards**.
The full audit checks all feature IDs/offsets and sampled protein vector blocks;
the pilot checks every protein vector. Optional molecular feature failures are
reported through per-modality coverage, not silently interpreted as zeros.

The pipeline uses a nonblocking process lock, records the controller PID and
current stage, propagates failures, and forwards termination to its own active
subprocess groups. A training launch is not proof of success: inspect the pilot
report and both-direction active-anchor diagnostics before interpreting metrics.

## Verification performed during implementation

CPU tests cover configuration invariants, immutable/resumable stage contracts,
failure propagation, exact row-ratio constraints, feature ID/shape/finiteness and
ragged-offset checks, missing virtual shards, and chemistry dimensions. Shell
syntax and CLI parsing are checked. Full feature extraction and GPU acceptance
training are deliberately left for the detached command above; they were not
launched as part of code implementation.
