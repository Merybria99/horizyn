# CIRCE-v2 on the already reconstructed Horizyn-style dataset

Use `scripts/horizyn1_circe_v2_reaction_holdout.py`, not the older
`run_horizyn1_circe_v2_training.sh`, for this protocol. Existing reconstruction,
annotation files and old run directories are never modified.

This keeps CIRCE-v2 and its training settings. It matches the published **data
preparation parameters** where reported, but does not claim the authors' exact
unreleased dataset or a full Horizyn model reproduction. The user-requested
reaction holdout is an explicit adaptation: the paper's final inference model
trained on its whole dataset.

## What is reused and what remains

| Component | Policy |
|---|---|
| Protein corpus | Reuse all 6,149,161 already clustered representatives at 80% identity. No 50% reclustering, MMseqs prefilter or cross-split alignment. |
| Retrieval positives | Reuse all 7,571,161 existing clustered pairs, including member-transferred associations. No representative-own-only filtering of retrieval gold. |
| Unpaired reaction inventory | Select the 30,926 paired reactions before grouping. The other 35 inventory entries have no raw or clustered pairs; record their original IDs/SMILES in `excluded_unpaired_reactions.csv`, not in supervised splits or zero-gold queries. Preserve the original 30,961-entry inventory. A reaction with raw pairs but no clustered pairs still causes a hard failure. |
| Reactions | Published cleanup, uncharging and metal disconnection. No additional tautomer/kekulization policy. |
| Reaction groups | 2048-bit radius-3 ECFP6 per molecule; OR per side; concatenate to 4096 bits. Group at Tanimoto **1.0**, invariant to reversal. Binary equality hashing avoids an all-pairs similarity search. |
| Chemical directions | Materialize separate forward/reverse copies and keep both in the same split. This is distinct from bidirectional retrieval. The dataloader remains `forward_only` solely to avoid augmenting those rows again. |
| Split | Seed 42, 90/5/5 train/validation/test **reaction groups**; proteins can occur in multiple splits. Ratios/seed are experiment choices, not claimed as the paper's full-corpus inference split. |
| Native annotations | Reuse completed EC labels, native cofactors and member-EC conflict flags. No archive parsing or native relabelling. |
| Weak annotations | Reaggregate the cached reaction descriptors using only representative-owned training associations. Do not copy unsplit weak targets into a train-scoped NPZ. Member-transferred retrieval positives do not become direct native or weak mechanistic evidence. |
| Unknown | Cofactor `unknown` remains missing-annotation status with zero supervision confidence. |
| CIRCE training | Model, frozen backbones, SLEEC, 85/15 row sampler, loss, disabled auxiliary classification and selected H200 profile stay unchanged. |

There are still one-time sequential CSV exports, a training-label summary
projection, compact index construction, the GPU pilot, and feature extraction.
The source dataset and annotations do **not** need reconstruction again.

The local reconstructed counts differ from the paper's 7,063,237 sequences and
8,897,870 pairs. The original reconstruction also used installed MMseqs defaults
for options not specified in the paper. Matching published settings does not
establish bit-for-bit reconstruction. See
[the published Methods](https://doi.org/10.1073/pnas.2520070123), particularly
Inference Dataset, Reaction SMILES Standardization and Augmentation, Structural
Fingerprint Clustering Protocol, and Inference Model Training.

## Cached annotation provenance

Projection binds the current cache and training-association bytes with SHA256,
checks the version/vocabulary and native-array semantics, and records its output
checksums. It rejects replaced, resized, modified-mtime or partial cache outputs.
Node-local device numbers are not identity checks. A changed ctime with the same
inode/size/mtime is explicitly recorded as a **new binding of the current cache**,
not proof of historical byte equality or a refreshed historical audit. The old
manifests/audits are not rewritten or reported as freshly passing.

## Evaluation

Validation uses a fixed, capped development panel. All positive proteins for its
reaction anchors are retained; enzyme anchors retain all their validation-reaction
positives. The validation-selected checkpoint is tested as follows:

- Reaction to enzyme: every augmented held-out test reaction against **all
  representative proteins**.
- Enzyme to reaction: proteins with test associations against **test reactions
  only**, retaining all gold within that candidate universe. Proteins may have
  appeared in training. This is not an enzyme-cold/full-reaction-catalog result.

Result JSON labels this protocol and the source-collapsed gold explicitly. The
second direction is a CIRCE evaluation extension, not an exact reproduction of
the paper's main reaction-to-enzyme benchmark. Reaction equality does not imply
all chemically similar reactions are held out together; only fingerprint-equal
groups are prohibited from crossing splits.

## Detached replacement on the H200 machine

Run this **on the machine running the old job**, inside the GPU allocation. The
subshell exits without launching if verified shutdown does not complete. The
stop helper preserves all old files; nothing here deletes/rebuilds the source
dataset or annotations.

```bash
(
set -e
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn

../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2_h200 --timeout 30 --force-after-timeout

mkdir -p runs/horizyn1_circe_v2_reaction_holdout_h200/logs
tmux new-session -d -s circe_v2_reaction_holdout -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 PYTHONUNBUFFERED=1; exec ../env/bin/python scripts/horizyn1_circe_v2_reaction_holdout.py all --profile h200 --run-root runs/horizyn1_circe_v2_reaction_holdout_h200 --extraction-batch-size 256 --extraction-max-tokens 65536 --extraction-length-sort --extraction-padded-token-budget >> runs/horizyn1_circe_v2_reaction_holdout_h200/logs/pipeline.log 2>&1'
)
```

These extraction caps carry forward the selected H200 settings; they are not a
claim of universal optimality or guaranteed memory fit. The real GPU pilot still
gates full extraction/training.

Inspect logs:

```bash
tail -n 40 /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_circe_v2_reaction_holdout_h200/logs/pipeline.log
tail -n 40 /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_circe_v2_reaction_holdout_h200/logs/prepare.log
```

The new launcher requires a new run directory and refuses an old both-cold run.
Do not edit old state files to skip stages. On restart it reuses its own unchanged
completed stages and preserves partial owned outputs before rebuilding them.

### Relaunch after the unpaired-inventory preparation failure

The fix changes an implementation included in the stage fingerprint. Use the
fresh run root below; do not remove or alter the failed run's state markers.
That failed run and its partial exports remain available for inspection.

```bash
(
set -e
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn

../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2_reaction_holdout_h200 \
  --timeout 30 --force-after-timeout

mkdir -p runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs
tmux new-session -d -s circe_v2_reaction_holdout_paired -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 PYTHONUNBUFFERED=1; exec ../env/bin/python scripts/horizyn1_circe_v2_reaction_holdout.py all --profile h200 --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --extraction-batch-size 256 --extraction-max-tokens 65536 --extraction-length-sort --extraction-padded-token-budget >> runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs/pipeline.log 2>&1'
)
```

```bash
tail -f /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs/pipeline.log
```

## Verification performed

- 17 preparation/projection/launcher tests passed in `.capability-run-py`,
  including a miniature export → training-only labels → real compact-index build.
  Unpaired-entry regression coverage verifies unchanged paired splits/queries,
  preservation of all edges and original source bytes, annotation-cache
  compatibility, malformed-ID rejection, and failure on reactions lost during
  clustering instead of silently excluding observed associations.
- 118 controller, evaluator, stop-helper, indexed-pair and CPU multimodal-training
  tests passed in `env`; the real two-step training smoke also covers materialized
  forward/reverse reaction IDs.
- A read-only chemistry check of all 30,961 reconstructed reactions found zero
  failures and 24,633 equality groups, taking 25.2 seconds on the checking node.
  This is **not** an estimate of the CSV export, indexing or feature-extraction time.
- After the unpaired-entry fix, full real-data preparation completed successfully
  in `runs/reaction_holdout_fix_check.JWRhwu/data`: 30,926 paired reactions in
  24,602 equality groups, all 7,571,161 source pairs preserved (15,142,322 after
  forward/reverse augmentation), and 35 explicitly reported unpaired entries.
  The entire raw-pair scan confirmed no excluded entry has a raw association;
  source file signatures remained unchanged. This is a standalone CPU validation
  export, not a completed stage in the replacement training run.
- The current reconstruction audit remains fresh. The resolved H200 training
  configuration passes the existing configuration validator.
- No GPU pilot, feature extraction or remote job replacement was launched during
  implementation. Those checks remain gated by the staged launcher on the
  execution node.

## Resume after the disabled-pilot candidate-set error

The pilot configuration now uses the supported `validation` candidate mode in
both sections, while keeping validation and retrieval metrics disabled. No
training objective, split, feature representation or full-run validation policy
changes. This fixes the initialization error caused by the unsupported `all`.

`scripts/repair_horizyn1_pilot_config.py` supports this exact two-line controller
change only. Without `--apply` it verifies the stopped run. With `--apply` it
backs up the configuration and state to
`metadata_repairs/pilot_candidate_set_v1`, verifies saved artifact signatures,
and updates controller/config bindings without changing completed artifacts or
marking the failed pilot passed. Config timestamp drift is accepted only with
unchanged inode/size and deterministic regeneration matching every setting
except the two corrected values; this does not prove historical byte identity.

The repair was applied to `horizyn1_circe_v2_reaction_holdout_h200_paired`.
All 11 completed stages were preserved. Normal resume fingerprint checks
confirmed preparation, labels, indexing and pilot feature extraction are skipped.
Do **not** use a fresh run directory for this repair.

Run on the H200 execution node, inside the GPU allocation:

```bash
(
set -e
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn

../.capability-run-py/bin/python scripts/stop_horizyn1_circe_v2.py \
  --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired \
  --timeout 30 --force-after-timeout

../.capability-run-py/bin/python scripts/repair_horizyn1_pilot_config.py \
  --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --apply

tmux new-session -d -s circe_v2_reaction_holdout_resume -c "$PWD" \
  'export CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 PYTHONUNBUFFERED=1; exec ../env/bin/python scripts/horizyn1_circe_v2_reaction_holdout.py all --profile h200 --run-root runs/horizyn1_circe_v2_reaction_holdout_h200_paired --extraction-batch-size 256 --extraction-max-tokens 65536 --extraction-length-sort --extraction-padded-token-budget >> runs/horizyn1_circe_v2_reaction_holdout_h200_paired/logs/pipeline.log 2>&1'
)
```

Verification of this repair:

- 30 repair/preparation/projection tests and 118 controller/evaluator/stop/index/
  CPU-training tests passed (148 total). Repair tests cover live locks, changed
  artifacts, unrelated config edits/failures, checkpoint guards, backups,
  interrupted publication and idempotence.
- The real `train_protein_pooling.py` entry point accepted the generated H200
  pilot config. Using the actual cached pilot features and unchanged model,
  a CPU smoke completed two optimizer steps and wrote a finite checkpoint.
  Smoke-only overrides were CPU/32-bit/single-device execution, batch 20,
  zero loader workers, two steps, and isolated logs/checkpoints.
- Diagnostic outputs are in `runs/pilot_config_fix_check.ZRomOq/`. No remote
  training was launched by these checks; the four-GPU pilot remains the next gate.
