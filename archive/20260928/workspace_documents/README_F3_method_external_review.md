# F3 Set-Chemistry Retrieval Model: Implementation and Training Method

Last updated: 2026-08-23

This note describes the exact F3 implementation in a self-contained form so
that it can be reviewed independently. It separates what the model actually
does from the biological names used for some latent enzyme blocks.

## 1. One-Paragraph Summary

F3 is a single-stage, dual-tower enzyme-reaction retrieval model. The enzyme
tower converts frozen ProT5 residue embeddings into a 512-dimensional vector
using mean pooling, SLEEC-guided residue attention, two learned residue-query
branches, and a small trainable Lorentz tangent branch. The reaction tower
combines frozen ReactionT5v2, UniMol2, and ChIRo features with a 617-dimensional
handcrafted molecule-set chemistry vector through learned modality attention.
Both towers are trained jointly, but only the pooling, projection, and fusion
layers are updated. Training uses a global full-batch multi-positive NCE loss
over the enzyme-reaction pairs present in the distributed batch. F3 has no
enzyme text, no biological-label loss, no hard-negative stage, no directional
adapter, and no warm start from another retrieval checkpoint.

## 2. Task and Retrieval Geometry

Each observed pair is a reaction ID `r` and an enzyme/protein ID `e`. The model
learns two unit-normalized vectors:

```text
z_r = reaction_tower(reaction features) in R^512
z_e = enzyme_tower(ProT5 residue features) in R^512
score(r, e) = cosine(z_r, z_e)
```

The same two embeddings and the same cosine score are used in both directions:

- `E->R`: an enzyme is the query and all reactions in the candidate pool are ranked.
- `R->E`: a reaction is the query and all enzymes in the candidate pool are ranked.

There are no direction-specific heads in F3. Any change to either tower affects
both retrieval directions.

## 3. What Makes F3 Different

F3 came from a controlled reaction-feature ablation:

```text
F1: ReactionT5v2 + UniMol2 + ChIRo over an unordered molecule set
F3: F1 + a train-fitted 617-dimensional molecule-set chemistry vector
```

Therefore, the defining F3 intervention is the fourth reaction modality. F3
does not add a new loss, a new training stage, or biological supervision.

## 4. Data Protocol

Three independent F3 models are trained, one for each released ReactZyme split:

| Split | Train pairs | Validation pairs | Held-out test pairs |
|:--|--:|--:|--:|
| Time | 149,554 | 16,618 | 12,287 |
| Enzyme-Sim | 152,748 | 16,972 | 8,739 |
| Reaction-Sim | 147,393 | 16,377 | 14,689 |

For each split, the released training pairs are shuffled with seed 42 and split
90/10 at the pair-row level. The released test files are not included in model
selection and are byte-identical to the benchmark release. Exact train and
validation pairs do not overlap.

The pair-row validation design is important: the same enzyme or reaction can
still occur in both train and validation. It is therefore an internal
validation set, not a simulation of the held-out Time, Enzyme-Sim, or
Reaction-Sim distribution shift.

ReactZyme stores these reaction records as unordered dot-separated participant
sets rather than explicit reactant/product equations. The data loader converts
a set `S` to the pseudo-reaction `S>>S` when a feature extractor expects
reaction syntax. Consequently, F3 deliberately does not encode reaction
direction or a product-minus-reactant transformation.

## 5. Enzyme Tower

### 5.1 Input

For protein `e`, ProT5 is run before retrieval training and supplies residue
embeddings:

```text
H_e in R^(L x 1024)
```

Sequences are capped at 1,022 residue tokens. Long sequences use the
`ends_center` truncation policy. ProT5 is not updated during F3 training.

### 5.2 SLEEC-guided site pooling

A pretrained SLEEC residue scorer produces one functional score/logit per
residue. The scorer is frozen. A separate learned linear attention logit is
combined with the centered SLEEC logit:

```text
a_i = learned_logit(H_i)
      + softplus(alpha) * (SLEEC_logit_i - logit(0.34))

w_i = softmax(a_i over non-padding residues)
h_site_raw = sum_i w_i H_i
```

The scale `alpha` and the learned attention projection are trainable. The SLEEC
checkpoint itself remains fixed.

### 5.3 Five fixed-layout blocks

The final enzyme embedding is divided into five blocks:

| Block name | Dimension | Fixed weight | Actual source |
|:--|--:|--:|:--|
| Core | 288 | 0.55 | Mean-pooled ProT5 residues |
| Site | 96 | 0.20 | SLEEC-guided pooled ProT5 residues |
| Mechanism | 64 | 0.12 | Learned residue query with SLEEC-logit bias |
| Cofactor | 32 | 0.08 | Learned residue query with SLEEC-logit bias |
| EC | 32 | 0.05 | Trainable Lorentz tangent projection of the SLEEC-pooled vector |

The core and site branches use `LayerNorm -> Linear -> Dropout(0.1)`.

For the mechanism and cofactor branches, residues first pass through:

```text
LayerNorm(1024) -> Linear(1024, 512) -> GELU
```

Two learned query vectors attend to these adapted residues. Each query's
attention logits also receive a trainable positive multiple of the frozen
SLEEC residue logits. The resulting 512-dimensional pooled vectors are
projected to 64 and 32 dimensions.

For the EC-named branch, no EC-pretrained checkpoint is loaded. A randomly
initialized `LorentzEnzymeProjector` maps the SLEEC-pooled 1024-dimensional
vector to a 128-dimensional tangent representation, using curvature 0.25 and a
hard tangent clip of 5.0. This projector and the final `128 -> 32` projection
are trained only by retrieval supervision.

Each block `b_k` is normalized independently, scaled by the square root of its
fixed weight `lambda_k`, concatenated, and normalized again:

```text
z_e = normalize(concat_k(sqrt(lambda_k) * normalize(b_k)))
```

The dimensions sum to 512 and the weights sum to 1. The block weights are not
learned.

### 5.4 Critical interpretation

The names `mechanism`, `cofactor`, and `EC` describe intended latent roles, not
supervised variables in F3. Although classifier heads exist in the module, F3
sets `biofp_aux_weight=0.0`. It supplies no mechanism labels, cofactor labels,
true EC labels, capability vectors, or enzyme text. Only the retrieval loss
shapes these branches.

## 6. Reaction Tower

### 6.1 Frozen input modalities

For each unordered participant set, F3 loads four precomputed modalities:

| Modality | Raw size | Processing during F3 training |
|:--|--:|:--|
| ReactionT5v2 | 768 per reaction | Trainable modality MLP |
| UniMol2 | 768 per molecule | Trainable molecule attention, then modality MLP |
| ChIRo | 256 per molecule | Trainable molecule attention, then modality MLP |
| Set chemistry | 617 per reaction | Trainable modality MLP |

The large ReactionT5v2, UniMol2, and ChIRo encoders are frozen because their
outputs are precomputed. Missing UniMol2, ChIRo, or chemistry entries are
allowed and are masked out of modality attention.

UniMol2 and ChIRo molecule embeddings are pooled independently with masked
learned attention. Because `side_composition=molecule_set`, only the single
stored participant set is pooled. No reactant/product delta is computed.

### 6.2 The 617-dimensional chemistry vector

The set-chemistry input is constructed with RDKit as follows:

```text
512  mean Morgan fingerprint across valid molecules, radius 2
 64  16 molecular descriptors aggregated by sum, mean, max, and std
  4  molecule-set statistics
 37  fixed-capacity core-cofactor indicator block
---
617
```

The 16 descriptors are exact molecular weight, heavy atoms, hetero atoms,
formal charge, H-bond donors, H-bond acceptors, TPSA, logP, rotatable bonds,
ring count, aromatic fraction, stereocenter count, wildcard atoms, nitrogen
atoms, oxygen atoms, and combined phosphorus/sulfur atoms.

The four set statistics are component count, unique-component fraction,
valid-component fraction, and wildcard-component fraction. Atom-map numbers
are removed before canonicalization.

Descriptor mean and standard deviation are fitted only on valid training
reactions for that split. The cofactor vocabulary is also fitted only on the
training reactions. In the three F3 runs the active vocabulary contains eight
labels: `ATP`, `CoA`, `FMN`, `FeS_cluster`, `NAD`, `NADP`, `SAM`, and
`quinone`. The remaining slots in the fixed 37-dimensional capacity are zero.
Validation and test reactions are transformed with the training schema.

Chemistry feature coverage was complete for all reaction records. Roughly 51
to 52 percent of reactions had at least one recognized core cofactor.

### 6.3 Modality projection and fusion

Each raw modality is projected to a 512-dimensional token using an independent
MLP with two 4,096-wide ReLU hidden layers and no modality-level output
normalization or dropout. The four tokens receive LayerNorm, then a learned
attention network computes one scalar per token:

```text
ell_m = Linear(512, 512) -> ReLU -> Linear(512, 1)
alpha_m = softmax(ell_m over available modalities)
h_r = sum_m alpha_m * token_m
```

There is no modality dropout, chemistry dropout, entropy penalty, fixed prior,
or capacity constraint in F3. The fused vector passes through another MLP with
two 4,096-wide ReLU hidden layers and a 512-dimensional output, followed by L2
normalization:

```text
z_r = normalize(MLP(h_r))
```

This is an unconstrained single-head modality attention mechanism. It can
learn a different mixture for every reaction, but it can also over-rely on a
high-capacity or easy modality such as the handcrafted chemistry token.

That concern is visible in the training logs. Averaged over the seven logged
training batches from each selected checkpoint epoch, the modality weights
were:

| Split | ReactionT5v2 | UniMol2 | ChIRo | Set chemistry |
|:--|--:|--:|--:|--:|
| Time, epoch 26 | 0.1291 | 0.0137 | 0.0070 | 0.8503 |
| Enzyme-Sim, epoch 28 | 0.0850 | 0.0299 | 0.0071 | 0.8780 |
| Reaction-Sim, epoch 28 | 0.1281 | 0.0236 | 0.0085 | 0.8398 |

These are averages of per-sample dynamic attention values, not four fixed
learned coefficients. Nevertheless, they show that the trained F3 reaction
tower is operationally dominated by the set-chemistry branch.

## 7. Frozen and Trainable Parameters

Frozen or precomputed during F3 retrieval training:

- ProT5 sequence encoder.
- ReactionT5v2 reaction encoder.
- UniMol2 molecular encoder.
- ChIRo molecular encoder.
- Pretrained SLEEC residue scorer.
- Handcrafted set-chemistry features and their train-fitted schema.

Trainable jointly from random initialization:

- SLEEC-guided pooling attention and its bias scale.
- Enzyme core, site, mechanism, cofactor, and EC projections.
- Mechanism/cofactor learned queries and residue key/value adapters.
- The Lorentz projector used by the EC-named latent block.
- UniMol2 and ChIRo molecule-set attention poolers.
- All reaction modality MLPs, modality attention, and output MLP.

There is no staged enzyme pretraining, no retrieval warm start, and no
post-training adapter stage in F3.

## 8. Training Objective

Within each GPU batch, duplicate reaction and enzyme IDs are encoded once.
Across four DDP processes, embeddings and pair IDs are gathered with gradient
synchronization and deduplicated again. With a configured batch size of 512 per
process, the global loss sees up to 2,048 sampled pair rows before ID
deduplication.

Let `Q` be the unique reactions in the global distributed batch, `T` the unique
enzymes, and `P_batch` the observed pair rows sampled into that batch. For
unit-normalized embeddings:

```text
d_ij = 1 - cosine(z_r_i, z_e_j)
beta = 10

L_F3 = beta * mean_{(i,j) in P_batch}(d_ij)
       + logsumexp_{i in Q, j in T}(-beta * d_ij)
```

Equivalently, it increases the logits of sampled positive pairs while using
every other reaction-enzyme combination in the global batch in one shared
partition function.

Important details:

- `beta=10` is fixed, not learned.
- `positive_pair_source=observed_pairs` means only positive rows actually
  sampled into the current global batch are marked positive.
- If another known positive combination is present through its two endpoints
  but its pair row was not sampled, it can be treated as a negative.
- The loss weights positive pairs equally, not reaction anchors or enzyme
  anchors equally. High-degree entities can therefore contribute more often.
- The objective is globally symmetric, but it is not the average of an `E->R`
  row-softmax loss and an `R->E` column-softmax loss.
- No hard-negative mining, auxiliary biological loss, structure loss,
  direction-balancing term, or attention regularizer is active.

## 9. Optimization Procedure

Each of the three split-specific models uses the same settings:

| Setting | Value |
|:--|:--|
| Random seed | 42 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-2` |
| Epoch limit | 30 |
| Precision | FP32 (`32-true`) |
| GPUs | 4 |
| Distributed strategy | DDP with unused-parameter detection |
| Batch size | 512 per process |
| Gradient accumulation | 1 |
| Validation frequency | Every epoch |
| Early-stop patience | 5 epochs |
| Early-stop minimum delta | `1e-4` |
| Checkpoint monitor | Arithmetic mean of validation `E->R` and `R->E` MRR |

There is no learning-rate scheduler in this configuration. All three runs
reached the 30-epoch limit. The selected checkpoints were:

| Split | Selected epoch | Validation E->R MRR | Validation R->E MRR | Mean |
|:--|--:|--:|--:|--:|
| Time | 26 | 0.968735 | 0.832372 | 0.900553 |
| Enzyme-Sim | 28 | 0.967148 | 0.850928 | 0.909038 |
| Reaction-Sim | 28 | 0.963265 | 0.848917 | 0.906091 |

The validation `mrr` used for checkpoint selection is conventional reciprocal
rank of the first positive. The code also computes all-positive ReactZyme MRR,
but F3's checkpoint monitor does not use it. This matters most for `R->E`,
where a reaction can have many valid enzymes.

## 10. Held-Out Evaluation

After validation checkpoint selection, each model is evaluated once on its
released test split. Ranking uses cosine similarity over the official paper
candidate pools and canonical forward-only IDs. Hit@k succeeds when any known
positive is in the top k. The reported MRR below is ReactZyme's all-positive
definition:

```text
MRR(query) = mean_{positive candidate p}(1 / rank(p))
```

Current recomputed test results are:

| Split | E->R H@1 | E->R H@5 | E->R H@10 | E->R MRR | R->E H@1 | R->E H@5 | R->E H@10 | R->E MRR |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Time | 0.6974 | 0.9208 | 0.9478 | 0.7997 | 0.7304 | 0.8667 | 0.9077 | 0.5644 |
| Enzyme-Sim | 0.9385 | 0.9958 | 0.9971 | 0.9662 | 0.9180 | 0.9809 | 0.9879 | 0.6800 |
| Reaction-Sim | 0.3902 | 0.5651 | 0.6571 | 0.4822 | 0.5337 | 0.6969 | 0.7668 | 0.4009 |

The arithmetic mean of the six MRR cells is 0.6489. For context, TIGER
ESM2Text reports 0.5735 over the same six summary cells. F3 exceeds TIGER in
five cells, but TIGER remains better on Reaction-Sim `E->R` (0.5180 versus
0.4822).

## 11. What F3 Is Not

To avoid an incorrect review, F3 should not be described as any of the
following:

- It is not a TIGER-style text-fusion model.
- It is not an enzyme biological-fingerprint supervision model.
- It does not train ProT5, ReactionT5v2, UniMol2, or ChIRo end to end.
- It does not use true EC labels, mechanism labels, or enzyme cofactor labels.
- It does not represent explicit reactant-to-product direction.
- It does not use all known in-batch positives.
- It does not use hard negatives or a reranker.
- It does not use separate `E->R` and `R->E` embeddings.
- It is a single-seed result, not a multi-seed estimate.

## 12. Main Strengths

1. The test protocol is split-specific and uses untouched released test files.
2. Chemistry preprocessing statistics and the cofactor vocabulary are fitted on
   training reactions only.
3. Both retrieval directions share one compact 512-dimensional space.
4. F3 adds chemically meaningful local and global molecule-set features while
   retaining pretrained learned representations.
5. The result is produced by one joint training stage and one retrieval loss.

## 13. Main Weaknesses and Possible Failure Modes

1. Pair-row validation is much easier than the held-out similarity splits and
   can contain train/validation entity overlap. Validation MRR around 0.90 does
   not predict Reaction-Sim test performance reliably.
2. Checkpoint selection uses first-positive MRR, whereas final ReactZyme `R->E`
   reporting uses all-positive MRR.
3. `observed_pairs` can create false negatives for known associations not
   sampled as rows in the current batch.
4. Global MLNCE weights pair occurrences rather than anchors, so high-degree
   reactions or enzymes can dominate the learned geometry.
5. The unordered `S>>S` representation discards reaction direction. It can
   model participant compatibility but not the chemical transformation itself.
6. The reaction attention is unconstrained and its 4,096-wide modality MLPs are
   large. The selected-epoch logs assign 84 to 88 percent of average attention
   to set chemistry, so chemistry shortcutting is an observed risk rather than
   only a theoretical one.
7. The mechanism, cofactor, and EC enzyme block names are not guaranteed to be
   semantically identifiable because they receive no corresponding labels.
8. The EC-named Lorentz projector is randomly initialized and retrieval-trained;
   calling it an EC hierarchy representation would overstate the method.
9. The fixed enzyme block weights were manually chosen and were not selected by
   a dedicated ablation in this F-series experiment.
10. Only seed 42 was run, so small differences from nearby models may be random
    variation.

## 14. Questions for an External Reviewer

The most useful review would address these questions:

1. Is global pair-normalized MLNCE appropriate for a many-to-many retrieval
   graph, or should it be replaced by anchor-balanced bidirectional objectives?
2. Should known associations among the in-batch endpoints always be expanded
   into positives to remove false negatives?
3. Should checkpoint selection use all-positive ReactZyme MRR, a harmonic mean
   across directions, or a held-out validation split designed to mimic each
   benchmark shift?
4. Is an unordered participant set sufficient for this benchmark, or is an
   explicit transformation representation required to improve Reaction-Sim
   `E->R` without sacrificing other cells?
5. How should modality capacity and attention be constrained so the 617-vector
   helps without becoming a chemistry shortcut?
6. Are the five enzyme blocks defensible without direct supervision, and is the
   random Lorentz branch useful or merely extra capacity?
7. Which single change is most likely to improve Reaction-Sim `E->R` while
   preserving F3's five stronger test cells?

## 15. Ready-to-Paste Review Prompt

```text
Please review the F3 enzyme-reaction retrieval method described below as a
machine-learning methods reviewer. Focus on whether the architecture and loss
match the many-to-many retrieval problem, whether the validation and checkpoint
protocol can explain poor held-out Reaction-Sim E->R generalization, and whether
the handcrafted chemistry modality is likely to help or create shortcuts.

Please distinguish implementation bugs from modeling limitations. Rank your
recommended changes by expected impact and experimental cost. In particular,
assess: (1) observed-pair versus all-known positives, (2) global MLNCE versus
anchor-balanced bidirectional losses, (3) first-positive versus all-positive
checkpoint selection, (4) unordered participant sets versus directional
reaction features, (5) unconstrained modality attention, and (6) unsupervised
mechanism/cofactor/EC-named enzyme blocks.

Do not assume that the biological names of the enzyme blocks imply label
supervision. The model uses no enzyme text or biological-label loss.

[Paste Sections 1-13 of this document here.]
```

## 16. Source Map

- Exact F3 configs: `horizyn/runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/`
- F-series definition: `horizyn/configs/benchmarks/reactzyme_paper/reaction_features.yaml`
- Enzyme and reaction modules: `horizyn/horizyn/model.py`
- Distributed training and checkpoint metrics: `horizyn/horizyn/protein_pooling_lightning_module.py`
- MLNCE implementation: `horizyn/horizyn/losses.py`
- Set-chemistry builder: `horizyn/horizyn/capability/reaction_set_features.py`
- Split manifest: `horizyn/data/revised_protocols/reactzyme_paper/manifest.json`
- Test results: `horizyn/runs/reactzyme_reaction_features_v1/eval/`
- TIGER comparison: `documents/README_reactzyme_tiger_top10.md`
