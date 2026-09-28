# B0 Configuration Report

This document records the exact B0 configuration used in the ReactZyme biological
latent ablation under:

```text
horizyn/runs/bio_aux_minimal_v1
```

It describes the architecture, supervision, loss, optimization recipe, checkpoint
selection, and released-test evaluation protocol represented by the frozen run
artifacts on 2026-07-20.

## Executive Summary

B0 is the control for the enzyme-side biological-label ablation. Its full name is:

```text
B0_unsupervised_control
```

The name `unsupervised_control` has a narrow meaning: B0 does **not** use BioFP or EC
labels as auxiliary supervision. B0 is still supervised end-to-end by observed
reaction-enzyme pairs through a retrieval loss.

The model is a 512-dimensional reaction-enzyme dual encoder:

```text
ReactionT5v2 + Uni-Mol2 + ChIRo
    -> molecule-set attention
    -> per-modality MLPs
    -> multimodal attention
    -> reaction MLP
    -> normalized 512-D reaction embedding

ProtT5 residue embeddings
    -> raw masked mean branch
    -> frozen-SLEEC-guided attention branch
    -> trainable Lorentz tangent branch
    -> unsupervised mechanism/cofactor residue-attention branches
    -> fixed blockwise concatenation
    -> normalized 512-D enzyme embedding

cosine distance + FullBatchMLNCELoss(beta=10, observed pairs)
```

There is no enzyme text, no reaction chemistry token, no hard-negative sampler, no
structure loss, no BioFP auxiliary loss, and no EC-supervised hyperbolic warm start.

## Experiment Scope

Three independent models are trained, one for each official ReactZyme split:

| Split | Training pairs | Train-derived validation pairs | Validation reaction queries | Validation enzyme candidates |
| --- | ---: | ---: | ---: | ---: |
| `time` | 149,554 | 16,618 | 2,156 | 16,617 |
| `enzyme_smi` | 152,748 | 16,972 | 2,464 | 16,971 |
| `reaction_smi` | 147,393 | 16,377 | 2,448 | 16,374 |

Each official training split is divided into a 90/10 training/validation partition.
The official released test subset is not used to select checkpoints. It is evaluated
only after the validation-selected checkpoints have been frozen.

The three split jobs are logically independent. The matrix launcher scheduled each
job as a four-GPU DDP run; concurrent placement on the same four GPUs was only an
execution choice and did not join their data or optimization states.

## Frozen Inputs

The large representation models are not fine-tuned. Their precomputed outputs are
loaded from HDF5 files.

### Enzyme Inputs

- ProtT5 residue embeddings: 1,024 dimensions per residue.
- Shared residue HDF5:
  `horizyn/data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5`.
- Maximum retained protein length: 1,022 residue tokens.
- Truncation policy: `ends_center`.
- SLEEC stage-1 functional-residue scorer: pretrained and frozen.
- SLEEC threshold: `0.34`.
- SLEEC scorer hidden dimension: `256`.

The SLEEC checkpoint is:

```text
horizyn/checkpoints/SLEEC/
  sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt
```

### Reaction Inputs

- ReactionT5v2 reaction embedding: 768 dimensions.
- Uni-Mol2 molecule embedding: 768 dimensions per molecule.
- ChIRo molecule embedding: 256 dimensions per molecule.
- Reaction direction: canonical forward direction only.
- Reactant and product molecule sets use separate attention poolers.
- Missing Uni-Mol2 or ChIRo modalities are allowed and masked from modality
  attention.
- Molecule sets can be normalized as self-reactions by the data loader.

Split-specific train and test features are stored under:

```text
horizyn/data/revised_protocols/reactzyme_official/features/{split}/{train|test}/
  reactiont5v2.h5
  unimol2.h5
  chiro.h5
```

## Enzyme Architecture

The model uses `ProteinPooledDualModel` with:

```yaml
pooling: sleec_guided_attention
enzyme_input_mode: raw_mean_sleec_biological_factorized
embedding_dim: 512
```

### SLEEC-Guided Residue Pooling

For residue `i`, B0 combines a learned residue-attention logit with the frozen SLEEC
scorer logit:

```text
combined_logit_i = learned_logit_i
                   + softplus(alpha)
                     * (sleec_logit_i - logit(0.34))

attention_i = softmax(combined_logit_i over valid residues)
pooled_sleec = sum_i attention_i * ProtT5_i
```

The SLEEC scorer is frozen. The learned attention layer and the positive SLEEC bias
scale are trainable. The scale starts at `1.0`.

### Five Factorized Blocks

The final enzyme vector contains five disjoint blocks:

| Block | Source | Dimensions | Norm-squared allocation |
| --- | --- | ---: | ---: |
| `core` | Masked mean of ProtT5 residues | 288 | 0.55 |
| `site` | SLEEC-guided pooled ProtT5 vector | 96 | 0.20 |
| `mechanism` | Learned family-query attention over residues | 64 | 0.12 |
| `cofactor` | Learned family-query attention over residues | 32 | 0.08 |
| `ec` | Tangent output of the Lorentz projector | 32 | 0.05 |
| **Total** |  | **512** | **1.00** |

Each branch is independently projected and L2-normalized. It is then multiplied by
the square root of its configured weight. The five weighted blocks are concatenated
and normalized again:

```text
z_enzyme = normalize(concat(
    sqrt(0.55) * normalize(core),
    sqrt(0.20) * normalize(site),
    sqrt(0.12) * normalize(mechanism),
    sqrt(0.08) * normalize(cofactor),
    sqrt(0.05) * normalize(ec)
))
```

This construction makes each configured weight its intended contribution to the
squared norm before the final normalization. Block projection dropout is `0.1`.

### Mechanism And Cofactor Branches

The residue tensor is adapted as:

```text
LayerNorm(1024) -> Linear(1024, 512) -> GELU
```

Separate learned mechanism and cofactor queries attend over the adapted residue
tokens. Their logits also receive a trainable SLEEC prior. The attended family vectors
are projected to their 64- and 32-dimensional retrieval blocks.

The architecture instantiates prediction heads for 8 mechanism labels and 10
cofactor labels. In B0, `biofp_aux_weight` is zero, so these labels and heads do not
contribute to the training objective. The mechanism and cofactor retrieval blocks are
still trained by the pair-retrieval loss. Their names therefore describe reserved
architectural slots; B0 alone does not guarantee that they learn the named biology.

BioFP target paths remain in the generated training YAML because all B0-B4 configs
share the same data schema. Loading those targets does not turn them into B0
supervision.

### Lorentz Branch

B0 does not load an EC-supervised hyperbolic checkpoint. The model nevertheless
instantiates a trainable `LorentzEnzymeProjector`:

```text
SLEEC-pooled 1024-D vector
    -> trainable Lorentz projector, hyp_dim=128
    -> 128-D tangent representation
    -> EC block projection, 32-D
```

The projector starts from random initialization and receives gradients only from
retrieval. Thus `no-ec-hyperbolic` means no EC supervision or warm start, not removal
of the Lorentz computation.

The configured target MLP dimensions are `[512, 512]`, but the biological-factorized
encoder already emits the final normalized 512-D enzyme embedding. The target MLP is
frozen and bypassed in this mode.

## Reaction Architecture

The reaction tower is `MultimodalReactionAttentionEncoder`. It has three modalities:

1. ReactionT5v2 reaction-SMILES embedding.
2. Uni-Mol2 reactant/product molecule embeddings.
3. ChIRo reactant/product molecule embeddings.

### Molecule-Set Pooling

Uni-Mol2 and ChIRo each use learned attention to pool a variable-size reactant set and
a variable-size product set. Reactant and product poolers have separate parameters.

For each molecule modality, the pooled sides are composed as:

```text
delta = product - reactant
composition = concat(reactant, product, delta, abs(delta))
```

Consequently, the Uni-Mol2 composition is 3,072-D and the ChIRo composition is
1,024-D.

### Modality MLPs And Fusion

Each modality has its own Horizyn-style MLP adapter:

```text
ReactionT5v2: 768  -> 4096 -> 4096 -> 512
Uni-Mol2:     3072 -> 4096 -> 4096 -> 512
ChIRo:        1024 -> 4096 -> 4096 -> 512
```

The three 512-D tokens are layer-normalized. A learned attention network scores each
token:

```text
Linear(512, 512) -> ReLU -> Dropout(0.0) -> Linear(512, 1)
```

Unavailable modalities are masked before softmax. Modality dropout is `0.0`, so an
available modality is not randomly removed during B0 training. The weighted token sum
is passed through the final reaction MLP:

```text
512 -> 4096 -> 4096 -> 512 -> L2 normalization
```

The resulting vector is the normalized reaction embedding `z_reaction`.

## Retrieval Objective

B0 uses cosine distance between the normalized reaction and enzyme embeddings:

```text
d_ij = 1 - z_reaction_i dot z_enzyme_j
```

The only effective optimization objective is `FullBatchMLNCELoss` with fixed inverse
temperature `beta = 10`:

```text
L_B0 = beta * mean_{(i,j) in P}(d_ij)
       + logsumexp_{all reaction-enzyme pairs (i,j)}(-beta * d_ij)
```

The positive term pulls observed pairs together. The global partition term competes
against every reaction-enzyme combination in the DDP-global batch. The formulation
supports multiple positive pairs, but it uses one global normalization rather than
separate reaction-anchored and enzyme-anchored softmax losses.

### Observed-Pair Semantics

The positive source is:

```yaml
positive_pair_source: observed_pairs
```

Only reaction-enzyme rows sampled in the current DDP-global batch are marked as
positive. If another known training match has both endpoints in the batch but its pair
row was not sampled, B0 does not recover it as a positive. This differs from
`all_known_in_batch`.

The Lightning module gathers embeddings and pair IDs from all four ranks with gradient
synchronization, concatenates them, and removes duplicate reaction and enzyme vectors
before constructing the distance matrix. With a configured batch size of 512 per DDP
process, the loss sees up to 2,048 sampled pair rows before ID deduplication.

### Disabled Losses

All additional terms are disabled:

| Term | B0 setting |
| --- | ---: |
| BioFP family BCE | `biofp_aux_weight: 0.0` |
| Residue supervision | `lambda_residue: 0.0` |
| EC hierarchy supervision | none |
| Reaction/enzyme structure alignment | none |
| Hard-negative margin loss | none |
| Capability anchor/consistency | none |
| Text alignment | none |

Therefore, for B0:

```text
total training loss = FullBatchMLNCELoss
```

## Optimization Recipe

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Scheduler | none |
| Maximum epochs | 30 |
| Precision | `32-true` |
| Float32 matmul precision | `high` |
| GPUs per split run | 4 |
| Distributed strategy | `ddp_find_unused_parameters_true` |
| Batch size | 512 per rank |
| Gradient accumulation | 1 |
| DataLoader workers | 2 per process |
| Distributed sampler | enabled |
| Validation frequency | every epoch |
| Sanity validation steps | 0 |

The run contains 99,983,581 parameters, of which 99,458,267 are marked trainable.
Precomputed ProtT5, ReactionT5v2, Uni-Mol2, and ChIRo encoders are outside this count.
The frozen SLEEC scorer is inside the model count.

## Validation And Checkpoint Selection

Validation computes both directions over the split-specific train-derived candidate
set:

- reaction-to-enzyme (`R2E`)
- enzyme-to-reaction (`E2R`)

The selection score is:

```text
val/mean_bidirectional_mrr = (R2E MRR + E2R MRR) / 2
```

Checkpoint policy:

- maximize `val/mean_bidirectional_mrr`
- save the top 3 checkpoints and `last`
- checkpoint after validation at every epoch
- early stopping patience: 5 validation epochs
- early stopping minimum delta: `1e-4`

The frozen B0 selections are:

| Split | Selected checkpoint | SHA-256 |
| --- | --- | --- |
| `time` | `protein-pooling-epoch=29.ckpt` | `54d90b9d0890bd9dde3dfb7923def6cb4ad34c095e5a25c5e352d99690edd85c` |
| `enzyme_smi` | `protein-pooling-epoch=27.ckpt` | `d27f7f86b12f8a63c5fc9e2a5ab34a09ec12db18f25fcf9b21452ed72bdec36e` |
| `reaction_smi` | `protein-pooling-epoch=28.ckpt` | `a68ce6c29e49a4f280031eaa68a10fd8ca8eddf313ffc6ea4e1522dc91fc88a5` |

## Released-Test Protocol

The frozen checkpoints are evaluated in both retrieval directions using:

```text
evaluation_protocol: paper_test_candidates
ground_truth_pairs: test_pairs_only
reaction_query_expansion: canonical_forward_only
```

Test retrieval uses the official split-specific test reactions and candidate enzymes.
The target encoding batch size is 2,048. Metrics include Top-1, Top-10, Top-100,
Top-1000, MRR, R-precision, and average precision.

The primary aggregate reported by this ablation is the mean of split-level balanced
MRR values, where balanced MRR is the arithmetic mean of R2E and E2R MRR.

## B0 Test Results

| Split | R2E MRR | R2E Top-1 | R2E Top-10 | R2E Top-100 | E2R MRR | E2R Top-1 | Balanced MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `time` | 0.7658 | 0.7001 | 0.8967 | 0.9704 | 0.7956 | 0.6971 | 0.7807 |
| `enzyme_smi` | 0.9391 | 0.9085 | 0.9860 | 0.9981 | 0.9655 | 0.9393 | 0.9523 |
| `reaction_smi` | 0.5866 | 0.5026 | 0.7461 | 0.8731 | 0.4615 | 0.3725 | 0.5240 |

Aggregate values:

| Metric | Value |
| --- | ---: |
| Mean R2E MRR | 0.7638 |
| Mean E2R MRR | 0.7409 |
| Mean balanced MRR / primary | 0.7524 |
| Worst-split R2E MRR | 0.5866 |

Test candidate and query counts:

| Split | Reaction queries | Enzyme queries/candidates |
| --- | ---: | ---: |
| `time` | 2,634 | 12,277 |
| `enzyme_smi` | 1,573 | 8,734 |
| `reaction_smi` | 386 | 14,688 |

## What B0 Measures

B0 answers this ablation question:

> How well does the fixed factorized architecture perform when every latent branch is
> trained only by reaction-enzyme retrieval pairs, without mechanism, cofactor, or EC
> label supervision?

It does **not** isolate the effect of SLEEC, Lorentz geometry, multimodal reaction
attention, or factorized block allocation. Those components are present in B0 and are
held fixed across B0-B4. B1-B4 isolate the added biological supervision or warm-start
choices relative to this architectural control.

## Logging And Artifacts

W&B:

- Entity: `omnai`
- Project: `horizyn-reactzyme-biological-latent-v1`
- Project URL: <https://wandb.ai/omnai/horizyn-reactzyme-biological-latent-v1>
- Run names:
  - `reactzyme-biological-B0-time-seed42`
  - `reactzyme-biological-B0-enzyme_smi-seed42`
  - `reactzyme-biological-B0-reaction_smi-seed42`

Authoritative local artifacts:

```text
horizyn/runs/bio_aux_minimal_v1/
  configs/B0_unsupervised_control/{split}/train.yaml
  configs/B0_unsupervised_control/{split}/test.yaml
  checkpoints/{split}/B0_unsupervised_control/
  logs/{split}/B0_unsupervised_control/
  selection/frozen_selection.json
  eval/{split}/B0_unsupervised_control/test_both.json
  eval/summary.md
```

## Implementation References

| Responsibility | Source |
| --- | --- |
| B0-B4 matrix and generated configs | `horizyn/horizyn/benchmarks/reactzyme_matrix.py` |
| Training entry point | `horizyn/scripts/train_protein_pooling.py` |
| Dual encoder and modality/factorized encoders | `horizyn/horizyn/model.py` |
| DDP gathering, positive construction, auxiliary terms | `horizyn/horizyn/protein_pooling_lightning_module.py` |
| Full-batch MLNCE implementation | `horizyn/horizyn/losses.py` |
| Reaction/enzyme datasets and loaders | `horizyn/horizyn/reaction_conditioned_data_module.py` |
| Matrix launcher | `horizyn/scripts/run_reactzyme_biological_ablation_matrix.sh` |

The generated YAML files and frozen selection manifest are the source of truth for
the executed B0 run. General-purpose configs elsewhere in the repository may describe
different historical protocols and should not be substituted when reproducing this
result.
