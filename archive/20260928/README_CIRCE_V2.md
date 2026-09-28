# CIRCE-V2: complete training, evaluation, and Case 1 retrieval guide

This document is the operational and scientific record for the CIRCE-V2 branch.
It explains what the current experiment does, what data it uses, how its
negatives are constructed, what the loss means, how to reproduce training and
evaluation, how the Case 1 protein sequences were recovered, how the restricted
and RefSeq screens work, where every important artifact is written, and what the
current results do and do not establish.

The paths and results below describe the repository state on 2026-09-03. Unless
otherwise stated, run commands from:

```text
/datastor2/deep-proteins/EnzymeDiscovery/horizyn
```

## Contents

- [What CIRCE-V2 is](#what-circe-v2-is)
- [Repository map](#repository-map)
- [Environment and installation](#environment-and-installation)
- [Training data and split](#training-data-and-split)
- [Negative-pool construction](#negative-pool-construction)
- [Per-epoch sampling](#per-epoch-sampling)
- [Model architecture](#model-architecture)
- [Training loss](#training-loss)
- [Four-GPU training](#four-gpu-training)
- [Monitoring, stopping, and resuming](#monitoring-stopping-and-resuming)
- [Validation and checkpoint selection](#validation-and-checkpoint-selection)
- [Held-out test evaluation](#held-out-test-evaluation)
- [Metric definitions](#metric-definitions)
- [Case 1 sequence recovery](#case-1-sequence-recovery)
- [Case 1 restricted-pool retrieval](#case-1-restricted-pool-retrieval)
- [Case 1 RefSeq plus 23 retrieval](#case-1-refseq-plus-23-retrieval)
- [Structure prediction](#structure-prediction)
- [Testing and code checks](#testing-and-code-checks)
- [Known limitations](#known-limitations)
- [Reproduction checklist](#reproduction-checklist)
- [Troubleshooting](#troubleshooting)

## What CIRCE-V2 is

CIRCE-V2 is a reaction-to-enzyme dual encoder trained on the ReactZyme
reaction-SMILES split. The experiment keeps the established multimodal reaction
tower, replaces the enzyme-side EnzGFM/ESM backbone with sequence-derived ProtT5
residue embeddings, and trains with explicit annotation-derived negatives.

The key experiment is:

```text
reaction SMILES ──> multimodal reaction encoder ──> normalized 512-d query
protein sequence ─> ProtT5 residue features      ──> normalized 512-d target
                                                   cosine similarity
```

During training, half of each batch consists of observed positive
reaction-protein pairs. The other half consists of query-matched negatives,
split between annotation-derived biological negatives and random diversity
negatives whenever the available pools permit it. Unlabelled matrix cells are
not silently treated as negatives by the training loss.

“ProtT5-only” refers to the enzyme sequence backbone. It does **not** mean that
the reaction tower uses only ProtT5. The reaction tower still consumes
ReactionT5v2, UniMol2, ChIRo, and engineered reaction-chemistry features.

The current validation-selected checkpoint is epoch 23:

```text
runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt
```

It is an experimental checkpoint. It improved substantially over early
CIRCE-V2 epochs, but it did not outperform the existing F3 chemistry model on
the held-out ReactZyme test or the Case 1 restricted pool.

## Repository map

The files that define CIRCE-V2 are:

| Purpose | Path |
|---|---|
| Main experiment config | `configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml` |
| Variant C symmetric-block config | `configs/reactzyme_reaction_smi_prott5_annotation_negatives_symmetric_reaction_blocks.yaml` |
| Variant C reaction-disjoint config | `configs/reactzyme_reaction_smi_prott5_annotation_negatives_symmetric_reaction_blocks_reaction_disjoint.yaml` |
| End-to-end training launcher | `scripts/run_circe_v2.sh` |
| Reaction-disjoint launcher | `scripts/run_circe_v2_reaction_disjoint.sh` |
| Negative-pool builder | `scripts/build_annotation_negative_pools.py` |
| Reaction-cluster split builder | `scripts/build_reaction_cluster_validation.py` |
| Training entry point | `scripts/train_protein_pooling.py` |
| Typed batch sampler/data module | `horizyn/reaction_conditioned_data_module.py` |
| Sampled contrastive loss | `horizyn/losses.py` |
| Lightning training/validation logic | `horizyn/protein_pooling_lightning_module.py` |
| Model and enzyme factorization | `horizyn/model.py` |
| Configuration validation | `horizyn/config.py` |
| Held-out evaluation | `scripts/evaluate_protein_pooling.py` |
| Generic wet-lab query engine | `wet_lab/query.py` |
| Exact shard-top-K merger | `wet_lab/merge_query_shards.py` |
| Case 1 restricted config | `wet_lab/Case1/restricted_setting/query_circe_v2_epoch23.yaml` |
| Case 1 RefSeq config | `wet_lab/configs/refseq/circe_v2_epoch23_reaction_smi.yaml` |
| RefSeq+23 campaign launcher | `scripts/run_case1_refseq_circe_v2_epoch23.sh` |
| Sequence recovery table | `wet_lab/Case1/sequence_pool/final_entry_sequences.csv` |
| Unresolved entries | `wet_lab/Case1/sequence_pool/unresolved_final_entries.csv` |
| Restricted-pool comparison | `wet_lab/Case1/restricted_setting/compare_ranked_candidates.py` |
| Ground-truth augmentation/analysis | `wet_lab/Case1/refseq_setting/ground_truth_augmented.py` |
| Optional structure workflow | `wet_lab/Case1/restricted_setting/build_top25_structures.py` |

Generated training artifacts live under:

```text
runs/circe_v2_annotation_negatives/
runs/circe_v2_prott5_annotation_negatives/
```

Generated wet-lab artifacts live under:

```text
wet_lab/Case1/
wet_lab/cache/
wet_lab/logs/
wet_lab/runs/
```

These generated directories are large and many are intentionally not tracked by
Git. A code checkout alone is therefore not a complete copy of the trained
model, RefSeq database, caches, or results.

## Environment and installation

The project requires Python 3.10 or newer and a CUDA-enabled PyTorch install for
training and large-scale inference. The pinned project dependencies include
PyTorch 2.4.0 with CUDA 12.1, Lightning, Transformers 4.40.2, RDKit, HDF5,
NumPy, pandas, PyYAML, SentencePiece, Accelerate, and Weights & Biases.

Install with `uv`:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
uv sync
```

Or install the package into an existing environment:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
pip install -e .
```

The launchers accept explicit environment overrides. The environment used for
training in this workspace is:

```bash
export PROJECT_ROOT=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
export PYTHON_BIN=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python
```

The RefSeq campaign defaults to:

```text
/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python
```

Override `PYTHON_BIN` if that interpreter is unavailable on another machine.
All model, feature, CSV, HDF5, and checkpoint paths in the YAML files must also
exist on that machine.

### Weights & Biases

The CIRCE config enables online W&B logging to project `horizyn-circe-v2`, with
run name `circe-v2-prott5-annotation-negatives-seed42`. The completed canonical
run has ID `b64719oy`.

For an offline machine, either edit the config or pass a CLI override:

```bash
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/train_protein_pooling.py \
  --config configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml \
  --training.devices 4 \
  --wandb-mode offline
```

Use `--wandb-mode disabled` to turn it off. `scripts/setup_wandb_env.sh` can be
used when W&B caches and artifacts need to be redirected to a larger disk.

## Training data and split

CIRCE-V2 uses the paper-compatible ReactZyme `reaction_smi` split:

| Partition | Pair rows | Reaction queries | Candidate proteins |
|---|---:|---:|---:|
| Train | 147,393 | 6,977 | 147,299 unique train proteins |
| Validation | 16,377 | 2,448 | 16,374 |
| Test | 14,689 | 386 | 14,688 |

The principal files are:

```text
data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv
data/revised_protocols/reactzyme_paper/reaction_smi/train_rxns.csv
data/revised_protocols/reactzyme_paper/reaction_smi/validation_pairs.csv
data/revised_protocols/reactzyme_paper/reaction_smi/validation_rxns.csv
```

The split is organized around reaction SMILES. It is not a strictly
protein-disjoint split: 22 proteins occur in both training and validation, while
16,352 validation proteins are validation-only. Interpret validation as a
reaction-split retrieval measurement, not as a guaranteed unseen-family test.

ProtT5 residue embeddings are loaded from:

```text
data/standardized/retrieval_training_source_collapse/test/
  horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5
```

The directory component named `test` is historical. This HDF5 file is a shared,
unsupervised sequence-feature store. The supervised pair tables and explicit
candidate/pool IDs determine the actual train, validation, and test membership.

Proteins are represented by 1,024-dimensional ProtT5 residue vectors. The model
uses at most 1,022 residues per sequence with `ends_center` truncation: for long
proteins, residues are selected from the N-terminus, center, and C-terminus
rather than only from one end.

## Negative-pool construction

Negative pools are generated entirely from annotations already available for
the training set. CIRCE-V2 does **not** perform MMseqs search during pool
construction.

Build the pools with:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/build_annotation_negative_pools.py \
  --train-pairs data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv \
  --ec-labels data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv \
  --biofp-targets runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/biology/enzyme_biofp_soft_targets.npz \
  --output runs/circe_v2_annotation_negatives/data/train_negative_pools.json \
  --report runs/circe_v2_annotation_negatives/data/train_negative_pools_report.json \
  --max-biological 32 \
  --max-random 32 \
  --ec-prefix-depth 2 \
  --biofp-threshold 0.5 \
  --min-biofp-similarity 0.5 \
  --seed 42
```

### Identifier and reaction normalization

Common protein prefixes such as `prot_`, `uprot_`, and `nr90_` are removed when
joining annotations. For pool exclusion, a reaction signature is formed by
sorting dot-separated molecules independently on the left and right of `>>`.
This catches ordering differences but is not full RDKit graph canonicalization
and does not erase every atom-mapping or representation difference.

### Annotation eligibility

A complete EC label has exactly four non-empty components and no `-`. A protein
is rejected from typed-negative eligibility if annotations imply more than one
complete EC leaf or if incomplete and complete labels conflict. Restricting the
pools to internally consistent, complete EC annotations reduces ambiguity; it
does not prove that every chosen protein is experimentally inactive for the
query reaction.

### Biological negatives

A biological negative for reaction `r` must satisfy all of the following:

1. It is a training protein with one internally consistent complete EC label.
2. It shares the configured two-level EC prefix with at least one positive.
3. Its complete EC leaf differs from every complete positive EC leaf.
4. It shares at least one active BioFP mechanism feature with a positive.
5. Its masked BioFP similarity is at least 0.5.
6. It was observed with a different normalized training reaction signature.
7. It is not itself a known positive for `r`.

BioFP targets are binarized at 0.5. Mechanism similarity is a masked Jaccard
score. When cofactor features are comparable, the combined similarity is 75%
mechanism and 25% cofactor. Up to 32 deterministic biological negatives are
stored per reaction.

These candidates are “biologically similar” through EC-prefix and BioFP
annotation similarity. Because no sequence search is performed, they are not
guaranteed to be sequence homologs.

### Random negatives

A random negative must be an eligible training protein and must not be:

- a known positive for the query;
- already selected as a biological negative;
- assigned a complete EC leaf matching a positive leaf; or
- observed with the same normalized reaction signature.

The eligible list is seeded, shuffled, and deterministically rotated per query.
Up to 32 random negatives are stored per reaction.

### Generated-pool statistics

The current report at
`runs/circe_v2_annotation_negatives/data/train_negative_pools_report.json`
contains:

| Quantity | Value |
|---|---:|
| Training reactions | 6,977 |
| Unique training proteins | 147,299 |
| Proteins with any complete EC | 84,402 |
| Eligible biological-negative proteins | 72,251 |
| Eligible random-negative proteins | 76,924 |
| Proteins rejected for conflicting EC | 7,478 |
| Reactions with biological negatives | 5,399 |
| Reactions without biological negatives | 1,578 |
| Biological pool rows | 170,083 |
| Mean biological pool size | 24.378 |
| Maximum biological pool size | 32 |
| Random pool rows | 223,264 |
| Random pool size per reaction | 32 |

The expanded data source therefore contains 147,393 positive rows, 170,083
biological-negative rows, and 223,264 random-negative rows. Training does not
consume all 393,347 negative rows every epoch; the sampler selects a bounded
query-matched negative for each sampled positive.

## Per-epoch sampling

`TypedNegativeBatchSampler` implements the requested 50% positive / 50%
negative training batches.

With the current local batch size of 500, each GPU receives:

```text
250 observed positive rows + 250 explicit negative rows
```

On four GPUs the gathered global contrastive batch can contain up to:

```text
1,000 positives + 1,000 negatives = 2,000 rows
```

For each positive, the sampler requests a biological or random negative
according to `typed_negative_biological_fraction: 0.5`. If the requested type
is unavailable for that query, it falls back to the other available type. All
6,977 reactions have random pools, while 1,578 have no biological pool, so the
realized biological/random ratio can differ from 50/50. The guarantee is 50%
positive and 50% negative rows; it is not an exact 25%/25% split for every
reaction or every batch.

Positive rows are shuffled with `seed + epoch` and covered once per epoch before
distributed padding. DDP padding may repeat a small number of positives at the
end of an epoch. The sampler shards batches across ranks itself, so Lightning's
automatic distributed sampler is not applied on top of it.

## Model architecture

Both sides produce L2-normalized, 512-dimensional embeddings. Ranking uses their
cosine similarity.

### Reaction encoder

The reaction tower uses:

| Input | Dimension |
|---|---:|
| ReactionT5v2 | 768 |
| UniMol2 | 768 |
| ChIRo | 256 |
| Reaction chemistry vector | 617 |

The inputs are fused by the multimodal reaction-attention encoder with a
512-dimensional hidden representation. Reactant and product molecules are
handled as molecule sets. The query MLP is `[512, 4096, 4096, 512]`. Training
uses `forward_only`; validation evaluates both retrieval directions, not reverse
reaction chemistry.

### Variant C: symmetric reaction output blocks

Variant C is an opt-in extension implemented in:

```text
configs/reactzyme_reaction_smi_prott5_annotation_negatives_symmetric_reaction_blocks.yaml
```

It preserves the multimodal attention and query MLP, then projects the shared
512-dimensional reaction feature through five independent heads matching the
enzyme block order, dimensions, and fixed weights. Each head applies LayerNorm,
a learned linear projection, L2 normalization, and `sqrt(weight)` scaling. The
weighted heads are concatenated and L2-normalized again:

```text
h -> core(288), site(96), mechanism(64), cofactor(32), ec(32)
q = normalize(concat(sqrt(w_b) * normalize(head_b(h))))
```

Because the enzyme tower uses the identical ordered layout, its cosine score now
decomposes as `sum_b w_b * cosine(q_b, e_b)`. Configuration and constructor
validation reject missing, reordered, dimensionally different, or differently
weighted reaction/enzyme layouts. The implementation logs the raw normalized
block norm, weighted block norm, and weight for each reaction block.

This option is disabled by default. The epoch-23 checkpoint and its active
RefSeq campaign use the original single-vector reaction output and remain
checkpoint-compatible. Variant C adds 267,776 parameters and requires fresh
training; it cannot be enabled while loading the original epoch-23 checkpoint.

### Enzyme encoder

The enzyme representation starts exclusively from ProtT5 residue embeddings.
The final 512 dimensions are factorized into normalized blocks:

| Block | Dimensions | Fixed contribution weight | Role |
|---|---:|---:|---|
| Core | 288 | 0.55 | Raw mean-pooled sequence representation |
| Site | 96 | 0.20 | SLEEC-guided residue pooling |
| Mechanism | 64 | 0.12 | Learned residue-family attention |
| Cofactor | 32 | 0.08 | Learned residue-family attention |
| EC | 32 | 0.05 | Trainable Lorentz/hyperbolic projection |

Each block is normalized, scaled by the square root of its configured weight,
concatenated, and normalized again. The block weights are fixed in this config.
The factorized encoder already emits the final vector, so the ordinary target
MLP is frozen for this mode.

SLEEC-guided pooling uses threshold 0.34 and a frozen pretrained scorer from:

```text
checkpoints/SLEEC/
  sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt
```

Its bias scale remains trainable. The EC hyperbolic projector has no pretrained
checkpoint in this experiment and starts randomly initialized.

BioFP and EC annotations supervise construction of the negative pools; they are
not enzyme-encoder inputs at inference. The code also supports auxiliary BioFP
heads, but `biofp_aux_weight: 0.0`, so those heads add no direct auxiliary term
to the CIRCE-V2 loss.

The complete model has 108,810,973 parameters: 108,285,659 trainable and 525,314
non-trainable in the recorded run.

## Training loss

CIRCE-V2 uses `SampledMultiPositiveInfoNCELoss` with fixed inverse temperature
`beta = 10`.

For normalized reaction embedding `q_i` and protein embedding `e_j`:

```text
d(i,j) = 1 - q_i dot e_j
z(i,j) = -beta * d(i,j)
```

For reaction anchor `i`, let `P_i` be all known positive targets that are present
in the gathered global batch, and let `N_i` be the explicit biological/random
negative targets present for that anchor. Its loss is:

```text
L_i = -log( sum[j in P_i] exp(z(i,j))
            / sum[j in P_i union N_i] exp(z(i,j)) )
```

The batch loss averages anchors that have at least one in-batch positive and one
explicit negative. `positive_pair_source: all_known_in_batch` means that another
known positive for the same reaction is included in the numerator when it is
present, rather than being treated as a false negative.

Biological and random negatives have the **same per-logit loss weight**. There is
no separate negative coefficient. Their effective influence comes from how
often they are sampled, how many appear for an anchor after DDP gathering, and
how similar the model currently considers them.

Unlabelled train matrix cells are ignored. The total configured objective is the
sampled multi-positive InfoNCE term because both `lambda_residue` and
`biofp_aux_weight` are zero.

Validation pair loss is not identical to training loss: it allows unlabelled
validation cross-pairs to act as random negatives. Validation retrieval metrics,
not that pair loss alone, select the checkpoint.

## Four-GPU training

The shortest reproducible command builds the pools and then starts training:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
PROJECT_ROOT=/datastor2/deep-proteins/EnzymeDiscovery/horizyn \
PYTHON_BIN=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
CIRCE_DEVICES=4 \
bash scripts/run_circe_v2.sh
```

Train Variant C instead by selecting its isolated configuration:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
PROJECT_ROOT=/datastor2/deep-proteins/EnzymeDiscovery/horizyn \
PYTHON_BIN=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
CIRCE_DEVICES=4 \
CIRCE_CONFIG=configs/reactzyme_reaction_smi_prott5_annotation_negatives_symmetric_reaction_blocks.yaml \
bash scripts/run_circe_v2.sh
```

Variant C writes checkpoints and logs under
`runs/circe_v2_symmetric_reaction_blocks/`, leaving baseline artifacts intact.

The config uses:

| Setting | Value |
|---|---|
| GPUs | 4 |
| DDP strategy | `ddp_find_unused_parameters_true` |
| Local batch | 500 rows |
| Effective global batch | up to 2,000 rows |
| Precision | `32-true` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Maximum epochs | 30 |
| Data workers per process | 1 |
| Pinned memory | disabled |
| Seed | 42 |

To run detached in `tmux`:

```bash
tmux new-session -d -s circe_v2_train \
  "cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn && PROJECT_ROOT=/datastor2/deep-proteins/EnzymeDiscovery/horizyn PYTHON_BIN=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python CIRCE_DEVICES=4 bash scripts/run_circe_v2.sh >> runs/circe_v2_prott5_annotation_negatives/training_tmux.log 2>&1"
```

Attach with:

```bash
tmux attach -t circe_v2_train
```

Detach without stopping it by pressing `Ctrl-b`, releasing both keys, then
pressing `d`.

## Monitoring, stopping, and resuming

Check the process, recent log lines, GPUs, host memory, and disk:

```bash
tmux list-sessions
tail -n 40 runs/circe_v2_prott5_annotation_negatives/training_tmux.log
nvidia-smi
free -h
df -h /datastor2
```

Lightning metrics for the canonical run are recorded at:

```text
runs/circe_v2_prott5_annotation_negatives/logs/train/
  protein_pooling_training/version_2/metrics.csv
```

`training_balanced.log` contains multiple attempts. Treat `version_2/metrics.csv`,
W&B run `b64719oy`, and the epoch-23 checkpoint as the authoritative completed
run; `version_0` and `version_1` are superseded attempts.

To stop a detached training session gracefully:

```bash
tmux send-keys -t circe_v2_train C-c
```

Do not use `kill -9` unless graceful interruption has failed: it prevents normal
Lightning cleanup and may leave an incomplete checkpoint or cache.

Resume from a checkpoint by invoking the training entry point directly:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/train_protein_pooling.py \
  --config configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml \
  --training.devices 4 \
  --resume runs/circe_v2_prott5_annotation_negatives/checkpoints/last-v2.ckpt
```

## Validation and checkpoint selection

Validation runs once per epoch against the explicit validation candidate set and
computes both reaction-to-enzyme and enzyme-to-reaction retrieval. Checkpoints
are ranked by:

```text
val/mean_bidirectional_mrr =
    (reaction_to_enzyme_mrr + enzyme_to_reaction_mrr) / 2
```

The trainer saves the best three checkpoints plus the latest checkpoint.
Early stopping uses patience 5 and `min_delta = 0.0001` in maximization mode.
The canonical run stopped after epoch 28 because no later epoch meaningfully
improved over epoch 23.

Important validation milestones are:

| Epoch | Val loss | R→E MRR | R→E mean best rank | E→R MRR | E→R mean best rank | Mean bidirectional MRR | Balanced MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.7281 | 0.21554 | 287.32 | 0.59825 | 35.06 | 0.40689 | 0.31691 |
| 8 | 1.2334 | 0.56989 | 68.11 | 0.82981 | 16.09 | 0.69985 | 0.67572 |
| 15 | 1.0232 | 0.64197 | 84.50 | 0.85202 | 15.59 | 0.74700 | 0.73223 |
| 20 | 0.9345 | 0.66878 | 91.20 | 0.86109 | 16.64 | 0.76494 | 0.75285 |
| **23** | **0.8719** | **0.68390** | 81.97 | **0.86521** | 16.52 | **0.77456** | **0.76394** |
| 28 | 0.9357 | 0.65778 | 68.18 | 0.85405 | 15.97 | 0.75592 | 0.74318 |

The MRR trend improved strongly from epoch 0 to epoch 23 and then regressed.
Mean best rank is noisier because a small number of very poorly ranked queries
can dominate it; an epoch can improve MRR while worsening the arithmetic mean
rank of the tail.

Available canonical checkpoint files are:

```text
runs/circe_v2_prott5_annotation_negatives/checkpoints/
  protein-pooling-epoch=22.ckpt
  protein-pooling-epoch=23.ckpt
  protein-pooling-epoch=27.ckpt
  last-v2.ckpt
```

Other `-v1` or unversioned checkpoints in that directory belong to earlier
attempts and should not be substituted silently.

### Reaction-disjoint validation

The original development split is pair-random: 2,085 of its 2,448 validation
reaction IDs (85.2%) also occur in training. The official test is different:
none of its 386 reaction IDs occurs in training. The original validation score
therefore measures mostly new protein associations for already-seen reactions,
not unseen-reaction generalization.

The recommended development protocol is the primary `similarity_0p85` panel in:

```text
data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/
```

It holds out complete connected components from a hybrid Morgan-chemistry and
frozen ReactionT5 similarity graph. Its audited properties are:

| Property | Value |
|---|---:|
| Training pairs | 147,344 |
| Validation pairs | 16,426 |
| Training reactions | 6,256 |
| Validation reactions | 1,084 |
| Validation protein candidates | 16,418 |
| Train/validation reaction-ID overlap | 0 |
| Train/validation exact reaction-SMILES overlap | 0 |
| Train/validation similarity edges at cosine >= 0.85 | 0 |
| Exact protein-ID overlap | 2 |

Reaction-set chemistry normalization is fitted on the new training partition.
The negative pools are also rebuilt from that partition only. Checkpointing and
early stopping monitor `val/reaction_to_enzyme/mrr`, which matches the enzyme
discovery direction.

Launch the disjoint Variant C experiment on four free GPUs with:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
mkdir -p runs/circe_v2_symmetric_reaction_blocks_reaction_disjoint
tmux new-session -d -s circe_v2_disjoint \
  "CUDA_VISIBLE_DEVICES=0,1,2,3 CIRCE_DEVICES=4 \
   bash scripts/run_circe_v2_reaction_disjoint.sh \
   >> runs/circe_v2_symmetric_reaction_blocks_reaction_disjoint/training.log 2>&1"
```

## Held-out test evaluation

Evaluate the validation-selected epoch-23 checkpoint with the paper test
candidate protocol:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/evaluate_protein_pooling.py \
  --checkpoint runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt \
  --config runs/circe_v2_prott5_annotation_negatives/evaluation/best_epoch8/test.yaml \
  --device cuda \
  --batch-size 512 \
  --target-batch-size 1024 \
  --direction both \
  --evaluation-protocol paper_test_candidates \
  --target-embeds-cache runs/circe_v2_prott5_annotation_negatives/evaluation/best_epoch23/target_embeddings.pt \
  --output runs/circe_v2_prott5_annotation_negatives/evaluation/best_epoch23/test_both.json
```

The epoch-23 result file intentionally references the test YAML under
`best_epoch8`: that file defines the unchanged dataset/model configuration and
was reused. It does not mean that the epoch-8 checkpoint was evaluated.

Results:

| Model | Direction | Top-1 | Top-10 | Top-100 | MRR | ReactZyme MRR |
|---|---|---:|---:|---:|---:|---:|
| CIRCE-V2 epoch 8 | R→E | 0.2720 | 0.4974 | — | 0.3466 | — |
| CIRCE-V2 epoch 8 | E→R | 0.2593 | 0.4333 | — | 0.3256 | — |
| **CIRCE-V2 epoch 23** | **R→E** | **0.3109** | **0.5311** | **0.7798** | **0.3879** | **0.2630** |
| **CIRCE-V2 epoch 23** | **E→R** | **0.2884** | **0.4660** | **0.7163** | **0.3493** | **0.3493** |
| F3 chemistry baseline | R→E | 0.5337 | 0.7668 | 0.8808 | 0.6064 | 0.4009 |
| F3 chemistry baseline | E→R | 0.3902 | 0.6571 | 0.9482 | 0.4822 | — |

Bidirectional summaries:

| Model | Balanced MRR | Balanced ReactZyme MRR | Balanced Top-1 |
|---|---:|---:|---:|
| CIRCE-V2 epoch 8 | 0.3361 | — | — |
| CIRCE-V2 epoch 23 | 0.3686 | 0.3061 | 0.2996 |
| F3 chemistry baseline | 0.5443 | 0.4415 | 0.4619 |

Thus epoch 23 is better than epoch 8 within CIRCE-V2, but F3 remains materially
better on this test. The F3 reference is:

```text
runs/reactzyme_reaction_features_v1/eval/reaction_smi/
  F3_set_chemistry/test_both.json
```

## Metric definitions

For reaction-to-enzyme (R→E), each reaction ranks candidate proteins. For
enzyme-to-reaction (E→R), each enzyme ranks reaction candidates.

- **Top-K hit rate:** fraction of anchors having at least one known positive in
  the first K positions. Higher is better.
- **MRR / first-positive MRR:** mean reciprocal rank of the best-ranked known
  positive. Higher is better; 1.0 is perfect.
- **ReactZyme MRR:** for anchors with multiple positives, averages reciprocal
  ranks across all positives before averaging anchors. It is stricter than
  first-positive MRR.
- **Mean rank:** the implementation reports the arithmetic mean rank of the best
  positive for each anchor. Lower is better.
- **R-precision:** precision after retrieving R items, where R is the number of
  positives for that anchor. Higher is better.
- **Average precision:** precision integrated over the positive ranks. Higher is
  better.
- **Mean bidirectional MRR:** arithmetic mean of R→E and E→R MRR. Higher is
  better and this is the checkpoint monitor.
- **Balanced MRR:** harmonic mean of the two directional MRRs. Higher is better;
  it penalizes a model that is strong in only one direction.

Wet-lab result scores are cosine similarities between normalized embeddings.
Higher scores rank earlier, but scores are not calibrated probabilities and raw
values should not be compared across different checkpoints as if they had a
common probabilistic scale.

## Case 1 sequence recovery

Case 1 studies the D-fructose to D-tagatose epimerization query from:

```text
wet_lab/Case1/T4Ease_homologs_and_variants.xlsx
```

The atom-mapped reaction used for retrieval is:

```text
[OH:1][CH2:2][C:3](=[O:4])[C@@H:5]([OH:6])[C@@H:7]([OH:8])[C@H:9]([OH:10])[CH2:11][OH:12]>>[OH:1][CH2:2][C:3](=[O:4])[C@@H:5]([OH:6])[C@H:7]([OH:8])[C@H:9]([OH:10])[CH2:11][OH:12]
```

### Recovered entries

The final per-row sequence table is:

```text
wet_lab/Case1/sequence_pool/final_entry_sequences.csv
```

It contains 703 workbook rows:

| Workbook section | Rows |
|---|---:|
| Ranked Candidates | 29 |
| Homologs | 145 |
| Engineered Variants | 524 |
| Indirect F6P | 5 |

Sequence status counts are:

| Status | Rows |
|---|---:|
| Resolved cross-reference | 34 |
| Reference sequence | 51 |
| Exact patent sequence | 458 |
| Inferred reference | 3 |
| Reconstructed variant | 141 |
| **Resolved total** | **687** |
| **Unresolved** | **16** |

The 687 resolved rows collapse to 627 unique SHA-256 sequence hashes. The 16
unresolved rows are explicitly recorded in:

```text
wet_lab/Case1/sequence_pool/unresolved_final_entries.csv
```

Do not fill unresolved rows by guessing. Patent names, variant aliases, and
mutation labels are only sufficient when the parent sequence and the complete
mutation definition can be established.

### The different “25” counts

Several related counts appear in the Case 1 analysis:

1. The workbook has 29 `Ranked Candidates` rows; 28 resolve to sequence and 27
   of those sequences are unique.
2. Of those 27 unique sequences, 25 also occur in the restricted Homolog pool.
   This is the denominator used by the restricted top-K recovery report.
3. The first 25 workbook-ranked rows contain 24 resolved entries and 23 unique
   sequences. These 23 are the sequences added to the RefSeq augmentation.
4. Rank 8, `KoT4E M6`, is unresolved because the source does not disclose enough
   parent/mutation information. Ranks 3 and 17 are an exact sequence duplicate.

“Unique eligible sequences recovered” means: take the SHA-256 sequence
intersection between resolved Ranked Candidates and the candidate pool, inspect
the top K model hits, and count distinct eligible hashes retrieved. It is not a
claim that every workbook entry is a confirmed catalyst, and it is not ordinary
binary-classification recall.

### Sequence provenance

Use the provenance and manifest columns in:

```text
wet_lab/Case1/sequence_pool/sequence_manifest.csv
wet_lab/Case1/sequence_pool/identifier_manifest.csv
```

The sequence sources include public reference records, exact sequences disclosed
in patents, exact cross-references, inferred references, and variants
reconstructed from a known parent plus stated mutations. Preserve those labels
when reporting candidates: evidence quality is not identical across categories.

## Case 1 restricted-pool retrieval

The restricted Homolog pool contains 144 resolved workbook rows and 123 unique
sequences. Eighteen duplicate groups cover 39 rows. Its provenance composition
is 49 references, 6 resolved cross-references, 87 exact patent sequences, and 2
inferred references.

Run CIRCE-V2 against this pool:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
CUDA_VISIBLE_DEVICES=0 \
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  -m wet_lab.query \
  --config wet_lab/Case1/restricted_setting/query_circe_v2_epoch23.yaml
```

The query output is:

```text
wet_lab/Case1/restricted_setting/runs/circe_v2_epoch23/
  tagatose_4_epimerase_c52b98cac3b4/results.json
```

Compare the top 25 with the workbook evidence set:

```bash
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  wet_lab/Case1/restricted_setting/compare_ranked_candidates.py \
  --results wet_lab/Case1/restricted_setting/runs/circe_v2_epoch23/tagatose_4_epimerase_c52b98cac3b4/results.json \
  --output-dir wet_lab/Case1/restricted_setting/results/circe_v2_epoch23 \
  --top-k 25
```

The current CIRCE-V2 top 10 is:

| Rank | Entry | Description | Score |
|---:|---|---|---:|
| 1 | H044 | Ignisphaera candidate | 0.364980 |
| 2 | P21 | Patent pA07079, SEQ ID 21 | 0.342155 |
| 3 | P361 | Patent SEQ ID 361 | 0.340077 |
| 4 | P23 | Patent SEQ ID 23 | 0.324723 |
| 5 | P328 | Patent SEQ ID 328 | 0.321429 |
| 6 | H022 | CJ_RM_F4E | 0.320694 |
| 7 | P8 | Patent SEQ ID 8; same exact sequence as H022 | 0.320677 |
| 8 | P327 | Patent SEQ ID 327 | 0.320328 |
| 9 | H042 | HCX96933 | 0.316782 |
| 10 | P371 | Patent SEQ ID 371 | 0.315122 |

At K=25, CIRCE-V2 reports:

| Quantity | CIRCE-V2 | F3 restricted baseline |
|---|---:|---:|
| Rows matching Ranked Candidates | 4 | 12 |
| Direct evidence matches | 3 | 7 |
| Exact sequence matches | 4 | 12 |
| Unique selected top-25 sequences | 24 | 19 |
| Unique eligible sequences recovered | 3/25 (12%) | 7/25 (28%) |

H022/CJ_RM_F4E is an exact-sequence/evidence match to workbook rank 12. Duplicate
pool rows may receive slightly different scores only if their stored features or
metadata differ; comparisons therefore report both row-level and unique-sequence
counts.

The query engine writes `results.json`, cutoff-specific CSV/FASTA exports,
reaction-feature artifacts, and logs. It hashes the reaction and relevant model,
feature, and candidate settings into a deterministic workspace name. Candidate
length bucketing reduces padding but does not change the set or intended score
semantics.

## Case 1 RefSeq plus 23 retrieval

This is a separate, much larger workflow. It screens the CIRCE epoch-23 model
against a prepared prokaryotic RefSeq subset, then combines the exact RefSeq
top 100 with scores for the 23 unique recoverable sequences from the workbook's
first 25 ranked rows.

### What “RefSeq” means here

The local database is **not all of RefSeq**. It was built from RefSeq release 236
using the first five ordered versioned `WP_` protein FASTA files for each of the
archaea and bacteria divisions: ten input shards total.

The database manifest is:

```text
wet_lab/databases/refseq/prokaryotes/five_shards/manifest.json
```

Database statistics:

| Quantity | Value |
|---|---:|
| Downloaded source records | 3,973,220 |
| Retained proteins | 3,944,613 |
| Removed because length < 50 | 28,607 |
| Total retained residues | 1,346,228,408 |
| Retained raw length range | 50–19,170 |
| Nonstandard residues | B: 76; J: 29; U: 1,066; X: 9,817; Z: 19 |

The candidate FASTA is about 1.4 GB, metadata CSV about 509 MB, and ID order file
about 57 MB. ProtT5 residue storage is a virtual HDF5 index over three backing
shards totaling roughly 2.75 TB. Do not move or delete those backing shards: the
small virtual index depends on their paths.

ProtT5 features were produced with
`Rostlab/prot_t5_xl_half_uniref50-enc` in float16. Extraction admits up to 1,024
sequence positions; CIRCE reads at most 1,022 using `ends_center` truncation.

### Candidate count

The three RefSeq query shards contain 1,314,871 disjoint IDs each, totaling
3,944,613. The augmented evaluation scores an additional 23 workbook sequences:

```text
3,944,613 RefSeq records + 23 workbook records = 3,944,636 candidate records
```

This is a record count, not a guaranteed unique-sequence count, because the 23
sequences were not globally deduplicated by sequence hash against RefSeq.

### Launch detached

The launcher first scores the 23 workbook sequences on GPU 3, then concurrently
scores RefSeq shards on GPUs 1, 2, and 3, waits for all three, merges their exact
global top 100, and generates the augmented analysis.

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
mkdir -p wet_lab/logs/case1_refseq_circe_v2_epoch23
tmux new-session -d -s circe_refseq25 \
  "cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn && exec bash scripts/run_case1_refseq_circe_v2_epoch23.sh >> wet_lab/logs/case1_refseq_circe_v2_epoch23/controller.log 2>&1"
```

The launcher has a PID guard, preserves an interrupted query log with a timestamp,
reuses a completed query result, and resumes valid protein-embedding cache chunks.
Running the same launcher after a clean interruption is therefore the supported
resume procedure.

### Monitor

```bash
tmux has-session -t circe_refseq25
tail -n 4 wet_lab/logs/case1_refseq_circe_v2_epoch23/refseq_shard_0.log
tail -n 4 wet_lab/logs/case1_refseq_circe_v2_epoch23/refseq_shard_1.log
tail -n 4 wet_lab/logs/case1_refseq_circe_v2_epoch23/refseq_shard_2.log
nvidia-smi
free -h
df -h /datastor2
```

Attach with `tmux attach -t circe_refseq25`; detach with `Ctrl-b`, then `d`.
Stop gracefully with:

```bash
tmux send-keys -t circe_refseq25 C-c
```

At the 2026-09-03 11:57 CDT documentation snapshot, the campaign was still
running. Each shard had encoded 358,400 of 1,314,871 candidates (27.3%) at
approximately 55–57 new proteins/second. The host had about 1.4 TiB available
RAM, so memory was not the immediate bottleneck. ProtT5 HDF5 reads,
variable-length pooling, and cache writes make the job partly I/O/CPU-bound; a
momentary `nvidia-smi` utilization of 0% does not by itself indicate a stalled
process.

Treat logs and final `QUERY_EXIT=0` / `CAMPAIGN_COMPLETE` markers as the source
of truth because this progress statement will become stale.

### Output locations

| Artifact | Path |
|---|---|
| Controller and shard logs | `wet_lab/logs/case1_refseq_circe_v2_epoch23/` |
| Per-shard result workspaces | `wet_lab/runs/refseq/prokaryotes/circe_v2_epoch23_shards/` |
| Resumable target caches | `wet_lab/cache/refseq/prokaryotes/circe_v2_epoch23_reaction_smi/` |
| Merged global RefSeq result | `wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_circe_v2_epoch23_merged/results.json` |
| RefSeq+23 comparison | `wet_lab/Case1/refseq_setting/CIRCE_V2/D-fructose_to_D-tagatose/reaction_smi/refseq_plus_ground_truth_23/` |

The target cache key includes the checkpoint, model/configuration, protein store,
candidate ordering, dtype, and truncation settings. This prevents accidental
reuse across incompatible runs. Each cache is written in per-batch chunks and
consolidated on success; completed chunks allow interruption recovery.

### Why shard merging is exact for top 100

Each candidate appears in exactly one shard. Any member of the global top K must
also be within the top K of its own shard: if K candidates in that shard scored
higher, those same candidates would exclude it globally. Therefore sorting the
union of the three shard-local top-100 lists gives the exact global top 100,
provided all shards use the same model, reaction, scoring logic, and finite
scores.

### RefSeq+23 rank interpretation

The campaign stores only the RefSeq top 100. It then combines those 100 entries
with all 23 scored workbook sequences and sorts again. A workbook sequence gets
an exact augmented rank if it enters the combined top 100; otherwise its rank is
reported as `>100`. Its exact position below 100 cannot be reconstructed without
retaining more RefSeq scores.

The best of the 23 in the completed workbook-only scoring stage was
`GT_RANK_012` / CJ_RM_F4E with cosine score 0.3206815. That is not yet its final
RefSeq rank.

### Finite-value validation tradeoff

The RefSeq config sets:

```yaml
validate_finite_on_access: false
allow_uncertified_finite_skip: true
```

This avoids a prohibitively expensive scan of multi-terabyte residue HDF5 files
on every query. Encoded output batches are still checked for finite values before
being cached. For stronger offline assurance, run the certificate workflow in
`scripts/run_refseq_residue_validation.sh` or call
`scripts/validate_residue_hdf5.py` directly. That scan is itself a long,
I/O-intensive operation and should not be launched concurrently without checking
storage bandwidth.

## Structure prediction

Retrieval and structure prediction are separate. The active CIRCE-V2 RefSeq
campaign ranks sequences only; it does **not** fold them. Existing F3 structure
folders are not CIRCE-V2 structure results.

After restricted-pool retrieval, build structures for the CIRCE top 25 with:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  wet_lab/Case1/restricted_setting/build_top25_structures.py \
  --results wet_lab/Case1/restricted_setting/runs/circe_v2_epoch23/tagatose_4_epimerase_c52b98cac3b4/results.json \
  --fasta wet_lab/Case1/restricted_setting/candidate_pool/proteins.fasta \
  --sequence-manifest wet_lab/Case1/sequence_pool/sequence_manifest.csv \
  --output-dir wet_lab/Case1/restricted_setting/structures/CIRCE_V2/D-fructose_to_D-tagatose/reaction_smi \
  --gpu 0 \
  --top-k 25
```

The helper first reuses an exact sequence match from AlphaFold DB when one is
available; otherwise it predicts with `facebook/esmfold_v1`. Exact duplicate
sequences are folded only once and exposed as per-rank copies. Output names
include rank, source entry ID, and a short sequence hash, for example:

```text
rank06_H022_<hash>.pdb
by_rank/rank06_H022.pdb
```

The manifest records source accession/URL where applicable, sequence hash,
mean pLDDT, and pTM. AlphaFold DB accessions are not experimental PDB IDs, and
ESMFold outputs have no PDB accession. These structures are predictions: they do
not validate catalytic activity, substrate specificity, oligomeric assembly, or
metal/cofactor placement. RefSeq top-hit folding should be launched only after
the final merged CIRCE result exists and its FASTA export has been inspected.

## Testing and code checks

The current CIRCE implementation was checked with the focused unit suite during
development: 239 tests passed. Coverage includes:

- annotation pool construction, deterministic selection, and conflict rejection;
- typed negative batch composition and DDP sharding;
- multi-positive sampled InfoNCE masks and loss behavior;
- configuration validation;
- positive-pair handling in training and validation;
- residue-HDF5 finite checks and cache identity;
- wet-lab query caching, resumption, and shard merging.

Run the relevant suite before moving the workflow to another machine:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python -m pytest \
  tests/unit/test_annotation_negative_pools.py \
  tests/unit/test_config.py \
  tests/unit/test_losses.py \
  tests/unit/test_protein_pooling_positive_pairs.py \
  tests/unit/test_reaction_conditioned.py \
  tests/unit/test_artifacts_and_cache.py \
  tests/unit/test_wet_lab_query.py \
  tests/unit/test_wet_lab_merge_query_shards.py
```

Also run syntax and whitespace checks on modified files:

```bash
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python -m py_compile \
  scripts/build_annotation_negative_pools.py \
  scripts/train_protein_pooling.py \
  wet_lab/query.py \
  wet_lab/merge_query_shards.py
bash -n scripts/run_circe_v2.sh
bash -n scripts/run_case1_refseq_circe_v2_epoch23.sh
git diff --check
```

## Known limitations

### Training labels and sampling

- A 50/50 row balance does not balance unique reactions, EC families, or protein
  families. Reactions with more positive rows receive more anchors.
- Biological negatives are missing for 1,578 of 6,977 reactions, forcing random
  fallback for those queries.
- “Functionally incompatible” is inferred from EC, BioFP, and observed reaction
  annotations. It is not an experimental inactivity measurement.
- Incomplete/noisy annotations can put a genuine catalyst in the random-negative
  pool despite the same-EC and same-reaction exclusions.
- Without MMseqs or another sequence search, biological negatives are
  annotation-similar, not guaranteed homologous or matched in fold/domain
  architecture.
- Deterministic pools improve reproducibility but expose the model repeatedly to
  a small maximum of 32 candidates of each type per reaction.

### Model and evaluation

- The training loss uses a sampled denominator, while validation/test and
  wet-lab retrieval rank much larger candidate universes.
- The reaction split is not completely protein-disjoint and does not eliminate
  every route for sequence-family leakage.
- The frozen SLEEC scorer imposes its pretrained bias. The EC block begins from
  a random hyperbolic projection, and BioFP auxiliary supervision is disabled.
- F3 currently outperforms CIRCE-V2 on both the held-out ReactZyme test and the
  restricted Case 1 recovery measurement. CIRCE-V2 should not replace F3 based
  on the current evidence.
- Cosine scores are ranking values, not calibrated catalytic probabilities.

### RefSeq screening

- The local corpus is only the first five archaea and five bacteria FASTA shards
  from RefSeq release 236, not all NCBI Protein or all RefSeq.
- RefSeq is dominated by common organisms and protein families; rank is affected
  by database composition and domain shift.
- Long sequences are truncated and a decisive catalytic region can be omitted.
- Only the shard top 100 is retained, so exact ranks below 100 are unavailable.
- Skipping a full finite-value scan is a deliberate performance tradeoff.
- Three large residue-HDF5 backing shards are path-sensitive dependencies of the
  virtual HDF5 file.

### Case 1 evidence and structures

- Sixteen workbook rows remain unresolved, and provenance quality varies across
  public references, patents, inferred sequences, and reconstructed variants.
- The workbook `Ranked Candidates` set is a comparison/evidence set, not a clean
  collection of 25 binary-positive ground-truth labels.
- Predicted structures do not prove function, substrate specificity, catalytic
  direction, oligomer state, or experimental stability.

## Reproduction checklist

For a fresh compatible machine:

1. Check out branch `CIRCE-V2` and install the environment.
2. Verify all YAML paths, especially pretrained reaction features, the shared
   ProtT5 HDF5, SLEEC checkpoint, and reaction-chemistry schema.
3. Build typed negative pools and inspect the JSON report.
4. Run the focused test suite.
5. Launch four-GPU training through `scripts/run_circe_v2.sh`.
6. Monitor `version_*/metrics.csv` and W&B; identify the checkpoint with maximum
   `val/mean_bidirectional_mrr` rather than assuming the final epoch is best.
7. Evaluate that checkpoint with `paper_test_candidates` in both directions.
8. Run the restricted Case 1 query and compare by both row and unique sequence.
9. Prepare or mount the RefSeq database and its residue-HDF5 backing shards.
10. Launch the three-shard RefSeq+23 campaign in `tmux`; wait for
    `CAMPAIGN_COMPLETE` before interpreting the comparison directory.
11. Fold only the selected final hits and preserve provenance/confidence in the
    structure manifest.
12. Treat every computational hit as a hypothesis requiring sequence review and
    wet-lab validation.

## Troubleshooting

### Training is very slow

The model loads variable-length residue tensors and performs residue-level
pooling; the reaction tower also loads several feature modalities. Check whether
GPUs are waiting on HDF5 I/O, CPU workers, or filesystem bandwidth. Increasing
workers can help on local SSDs but can hurt a shared or already saturated HDF5
store. Confirm all four DDP ranks are alive before changing batch size.

### MRR looks unexpectedly low

Confirm the direction and candidate protocol. Reaction-to-enzyme is normally
harder because each reaction ranks thousands of proteins. First-positive MRR,
ReactZyme MRR, mean bidirectional MRR, and balanced MRR are different metrics.
Also confirm the test uses `paper_test_candidates`, the intended checkpoint, and
the matching target cache.

### A RefSeq process shows 0% GPU utilization

Inspect several minutes of progress logs, not one `nvidia-smi` snapshot. HDF5
reads, CPU work, and cache writes create low-utilization intervals. A healthy run
continues to print increasing `Encoded candidate enzymes` counts. A process in
uninterruptible `D` state can also be waiting on storage I/O.

### RefSeq stopped

Inspect `QUERY_EXIT`, the timestamped `.interrupted.*` log, free disk, kernel/OOM
messages, and whether the HDF5 backing files remain mounted. Once the prior
process is gone, rerun `scripts/run_case1_refseq_circe_v2_epoch23.sh`; valid cache
chunks are resumed automatically.

### A target cache is rejected

This is usually protective. The cache identity includes checkpoint, model
configuration, protein store, candidate order, dtype, and truncation. Do not
force reuse when one differs. Let the query create a new cache namespace.

### Duplicate proteins appear in rankings

Candidate IDs are ranked as records. Different workbook/patent IDs can encode
the same amino-acid sequence. Use sequence SHA-256 hashes and the comparison
reports for unique-sequence statistics; retain individual IDs for provenance.

### An entry has no sequence

Consult `unresolved_final_entries.csv` and the source/provenance columns. Recover
it only from a disclosed patent sequence, a reliable accession, or a fully
specified parent-plus-mutation definition. Do not infer missing residues or
mutations from the name alone.
