# Implementation Guide: Harmonic-Mean Top Five

Last updated: 2026-08-21

## Scope

This document describes the five highest-ranked Horizyn models when the six
ReactZyme test MRR cells are aggregated with the harmonic mean:

| Harmonic rank | Model | Harmonic mean | Arithmetic macro |
|---:|:--|---:|---:|
| 1 | Q5 Combined | 0.595020 | 0.6537 |
| 2 | Q3 Dense chemistry | 0.594329 | 0.6523 |
| 3 | F3 Set chemistry | 0.594292 | 0.6489 |
| 4 | Q4 Factorized modalities | 0.593975 | 0.6512 |
| 5 | Q1 E->R adapter | 0.592709 | 0.6492 |

The harmonic mean changes only leaderboard aggregation. It does not retrain a
model or alter any retrieval result.

### Q4 is not F4

The fourth-place model above is **Q4 Factorized modalities**, not the original
**F4 Factorized reaction** model.

| Model | Time E->R / R->E | Enzyme E->R / R->E | Reaction E->R / R->E | Harmonic mean |
|:--|:--:|:--:|:--:|--:|
| Q4 | 0.8202 / 0.5714 | 0.9655 / 0.6807 | 0.4687 / 0.4010 | 0.593975 |
| F4 | 0.7268 / 0.4872 | 0.9582 / 0.6520 | 0.5023 / 0.3760 | 0.563051 |

F4 jointly replaces the base reaction tower's learned modality attention with
a fixed factorized concatenation, so that change affects both retrieval
directions. It gives the strongest Horizyn Reaction-Sim E->R result (`0.5023`)
but loses substantial Time-Sim and R->E performance. Q4 retains the frozen F3
base geometry and exposes factorized raw modalities only to its E->R residual
adapter. That preserves the base R->E embedding and places Q4 in the harmonic
top five; F4 remains a specialized Reaction-Sim E->R model outside the top
five.

## Shared F3 Foundation

F3 is the common architecture. Q1, Q3, Q4, and Q5 use a freshly trained copy of
that architecture, called Q0, as their frozen parent. They do not initialize
from the historical F3 checkpoint shown separately in the leaderboard.

### Enzyme tower

The enzyme tower consumes precomputed ProT5 residue embeddings:

```text
ProT5 residues [L, 1024]
    |-- masked mean --------------------------> core branch
    |-- frozen SLEEC scorer + attention -----> site branch
    |-- trainable Lorentz tangent projection -> ec branch
    `-- SLEEC-biased family attention --------> mechanism/cofactor branches
                                                       |
                                                       v
                    normalized weighted block concatenation [512]
```

The fixed embedding allocation is:

| Branch | Dimension | Weight | Actual input |
|:--|--:|--:|:--|
| Core | 288 | 0.55 | Mean-pooled ProT5 residues |
| Site | 96 | 0.20 | SLEEC-guided attention-pooled residues |
| Mechanism | 64 | 0.12 | Learned residue query with a SLEEC attention prior |
| Cofactor | 32 | 0.08 | Learned residue query with a SLEEC attention prior |
| EC | 32 | 0.05 | Trainable Lorentz tangent projection of the pooled sequence |

Each branch is independently projected and L2-normalized. It is multiplied by
the square root of its configured weight, concatenated with the other branches,
and normalized again to produce the 512-dimensional enzyme embedding.

Important implementation detail: these F3/Q0 runs do **not** use enzyme text,
true EC labels, BioFP targets, or a BioFP auxiliary loss. The mechanism,
cofactor, and EC names denote sequence-derived latent blocks. Their prediction
heads receive no label loss because `biofp_aux_weight=0.0`, and the EC projector
is not loaded from a pretrained EC checkpoint. Retrieval supervision gives
these blocks their meaning in these particular runs.

The ProT5 embeddings are precomputed and the SLEEC scorer is frozen. The
SLEEC-guided pooler, block projections, family attention, and F3 retrieval
modules remain trainable during joint F3/Q0 training.

### Reaction tower

The base reaction tower receives four precomputed modalities:

| Modality | Input dimension | Representation |
|:--|--:|:--|
| ReactionT5v2 | 768 | Forward reaction/string embedding |
| UniMol2 | 768 per molecule | Learned attention pooling over the molecule set |
| ChIRo | 256 per molecule | Learned attention pooling over chiral molecule features |
| Set chemistry | 617 | Train-fitted descriptors, fingerprints, set statistics, and cofactors |

The 617-dimensional set-chemistry vector is:

```text
512  mean Morgan fingerprint, radius 2
 64  sixteen molecular descriptors x {sum, mean, max, std}
  4  component-count and molecule-set validity/diversity statistics
 37  fixed-capacity train-vocabulary core-cofactor indicator block
---
617
```

ReactionT5v2, UniMol2, ChIRo, and chemistry inputs are projected into trainable
modality tokens. F3/Q0 uses unconstrained learned attention over those tokens,
then an MLP output projection and final normalization produce a 512-dimensional
reaction embedding. Molecule features use `side_composition=molecule_set`, so
the representation emphasizes the unordered participant set rather than an
explicit product-minus-reactant transformation.

### F3/Q0 base training

F3 and Q0 are trained independently, once per ReactZyme split, with the same
recipe:

| Setting | Value |
|:--|:--|
| Trainable scope | Enzyme and reaction projection/fusion modules jointly |
| Loss | `FullBatchMLNCELoss` |
| Positive definition | All observed enzyme-reaction pairs in the global batch |
| Inverse temperature | Fixed `beta=10` |
| Epochs | Up to 30 |
| Learning rate | `1e-4` |
| Weight decay | `1e-2` |
| Batch size | 512 on four-GPU DDP |
| Selection | Validation mean bidirectional MRR |

The Q-series adapter models initialize from the validation-selected Q0
checkpoint for the corresponding split.

## Shared E2R Adapter Stage

Q1, Q3, Q4, and Q5 all add one `E2RReactionAdapter` to Q0. This is a
direction-specific residual head, not a second retrieval model or an ensemble:

```text
z_e2r = normalize(z_base + sigmoid(g) * Delta(adapter_inputs))
```

`g` is one learned scalar initialized so `sigmoid(g)=0.1`. `Delta` is:

```text
LayerNorm -> Linear(hidden=512) -> GELU -> Dropout(0.1) -> Linear(512)
```

During this stage the complete Q0 model is frozen and kept in evaluation mode;
only the adapter and its optional input projections are trainable. The adapter
is called only for enzyme-to-reaction retrieval. Reaction-to-enzyme retrieval
uses the unchanged Q0 reaction embedding. This is why Q1/Q3/Q4/Q5 have
identical R->E scores in the current test table.

Common adapter training settings are:

| Setting | Value |
|:--|:--|
| Stage | `training_stage=e2r_adapter` |
| Loss | `MultiAlignmentRetrievalLoss` |
| Direction weights | `lambda_e2r=1`, `lambda_r2e=0` |
| Positives | Observed pairs, multi-positive anchor loss |
| Identity regularizer | `0.05 * mean(1 - cosine(z_e2r, z_base))` |
| Epochs | Up to 15 |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Early stopping | Validation E->R ReactZyme MRR, patience 4 |

## 1. Q5 Combined

Q5 enables every Q-series adapter input and the hard-negative objective in one
training run:

```text
frozen Q0 reaction embedding [512]
    + factorized raw modality block [512]
    + dense transformation projection [128]
                 |
                 v
       one gated residual E2R adapter
```

Q5 is not a sequential composition of trained Q1, Q2, Q3, and Q4 checkpoints.
It initializes directly from Q0, then jointly trains one adapter containing all
the enabled components.

### Factorized raw modalities

The adapter separately projects and normalizes the raw inputs:

| Input | Adapter dimension | Fixed weight |
|:--|--:|--:|
| ReactionT5v2 | 128 | 0.25 |
| UniMol2 molecule-set mean | 192 | 0.375 |
| ChIRo molecule-set mean | 96 | 0.1875 |
| Set chemistry | 96 | 0.1875 |

Each block is multiplied by `sqrt(weight)` and concatenated. Missing-modality
masks remove unavailable blocks and renormalize the remaining weights. The
result is a fixed-allocation 512-dimensional side channel that prevents the
base attention mixture from being the adapter's only view of reaction inputs.

### Dense transformation input

Q5 also projects an 1114-dimensional train-fitted transformation vector to 128
dimensions:

| Dense block | Dimension |
|:--|--:|
| Signed product-minus-reactant Morgan fingerprint | 512 |
| DRFP reaction fingerprint | 512 |
| Standardized descriptor delta | 16 |
| Molecular side-statistic delta | 8 |
| Hashed mapped reaction-center labels | 64 |
| Parse and mapping availability masks | 2 |

Descriptor normalization is fitted on training reactions only. Directional
Rhea reconstructions and mapped center labels are used when available; masks
make missing chemistry explicit.

### Hard negatives

Hard-negative pools are mined from the trained Q1 model in the E->R direction.
For each enzyme anchor, mining retains up to 256 high-ranked non-positive
reaction candidates. Q5 batches use 48 enzyme anchors, up to 2 positives per
anchor, and 8 mined negatives per anchor before filling the batch.

The loss adds an E->R hard-negative term with weight `0.25`, top-k `64`, and
margin `0.05`. It compares the positive log-sum-exp with the hardest negative
log-sum-exp through a softplus margin loss.

Why it performs well: Q5 preserves Q0's R->E geometry while giving E->R a
controlled raw-modality view, explicit reaction transformations, and difficult
candidate discrimination. It is first under arithmetic, geometric, and
harmonic aggregation.

## 2. Q3 Dense Chemistry

Q3 uses the same frozen Q0 and residual E2R adapter, but enables only the dense
1114-dimensional transformation vector:

```text
[Q0 reaction embedding 512 ; dense projection 128]
    -> gated residual adapter -> E2R reaction embedding 512
```

It does not use factorized raw modality blocks and does not use mined
hard-negative batches or the hard-negative loss. Its adapter input dimension is
640.

Q3 tests whether explicit product-minus-reactant chemistry can correct E->R
ranking without replacing the base embedding. Its near-tie with Q5 indicates
that the dense transformation branch carries most of the useful balancing
signal, while Q5's additional branches provide a small aggregate improvement.

## 3. F3 Set Chemistry

F3 is the jointly trained base model described above. It has no E2R-only
adapter, no warm start, no dense directional vector, and no mined hard
negatives.

```text
reaction modalities -> learned attention -> z_reaction
ProT5/SLEEC blocks   -> fixed-layout fusion -> z_enzyme
                         |
                         v
        bidirectional full-batch multi-positive retrieval loss
```

Both retrieval directions therefore share exactly the same reaction and enzyme
embeddings. This coupling is F3's main advantage and limitation:

- It preserves strong Reaction-Sim E->R behavior and ranks third under the
  harmonic mean.
- It cannot improve E->R independently; changing reaction geometry for E->R
  also changes R->E.
- Its learned modality attention can overuse the 617-dimensional chemistry
  token because no explicit block-capacity constraint is applied.

## 4. Q4 Factorized Modalities

Q4 uses Q1's residual adapter plus the fixed-allocation raw modality side
channel described under Q5:

```text
[Q0 reaction embedding 512 ; factorized raw modalities 512]
    -> gated residual adapter -> E2R reaction embedding 512
```

It does not use the 1114-dimensional dense transformation vector and does not
use hard-negative sampling or loss. Its adapter input dimension is 1024.

Q4 isolates whether direct access to ReactionT5v2, UniMol2, ChIRo, and set
chemistry helps after the base attention encoder has already compressed them.
The fixed dimensions and weights prevent one raw modality from consuming the
entire side channel, but they do not encode an explicit reaction direction.

## 5. Q1 E2R Adapter

Q1 is the minimal direction-specific intervention:

```text
Q0 reaction embedding 512
    -> gated residual adapter
    -> E2R reaction embedding 512
```

The adapter receives no raw modality blocks and no dense transformation input.
It learns only a nonlinear residual transformation of the frozen Q0 embedding.
Its input dimension is 512.

Q1 answers the narrowest audit question: can E->R improve by changing only the
reaction candidate geometry while preserving R->E exactly? The identity loss
and gate initialization constrain that correction to stay close to Q0. Q3, Q4,
and Q5 retain this same mechanism and differ only in additional adapter inputs
and, for Q5, hard-negative training.

## Direct Comparison

| Component | Q5 | Q3 | F3 | Q4 | Q1 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Jointly trains base towers | No | No | Yes | No | No |
| Frozen Q0 parent | Yes | Yes | No | Yes | Yes |
| E2R-only residual adapter | Yes | Yes | No | Yes | Yes |
| Factorized raw modalities | Yes | No | No | Yes | No |
| Dense 1114-d transformation | Yes | Yes | No | No | No |
| Mined E2R hard negatives | Yes | No | No | No | No |
| Changes R->E embedding | No | No | Yes | No | No |
| Primary adapter selection metric | E->R MRR | E->R MRR | N/A | E->R MRR | E->R MRR |

## Source Map

- Leaderboard: [`README_reactzyme_tiger_top10.md`](README_reactzyme_tiger_top10.md)
- [F3 seed-42 configs](../../../runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry)
- [Q-series seed-42 configs](../../../runs/reactzyme_e2r_pareto_v1/configs/seed42)
- [Q-series campaign definition](../horizyn/horizyn/benchmarks/reactzyme_e2r_pareto.py)
- [Q-series launcher and mining stage](../horizyn/scripts/run_reactzyme_e2r_pareto_campaign.sh)
- [Model implementations](../../../horizyn/model.py): `E2RReactionAdapter`, `EnzymeBiologicalFactorizedEncoder`, and `MultimodalReactionAttentionEncoder`
- [Adapter freezing and identity loss](../../../horizyn/protein_pooling_lightning_module.py)
- [Retrieval and hard-negative losses](../../../horizyn/losses.py): `MultiAlignmentRetrievalLoss`
- [Dense chemistry builder](../horizyn/horizyn/capability/reaction_dense_features.py)
- [Set-chemistry builder](../../../horizyn/capability/reaction_set_features.py)
