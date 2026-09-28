# Horizyn Losses

This document explains the loss functions used in the Horizyn codebase, what each one
optimizes, where it is implemented, how it is configured, and how to interpret the
logged loss components.

The codebase has three main loss families:

1. Retrieval losses for reaction-enzyme dual-encoder training.
2. Hyperbolic EC hierarchy losses for enzyme or reaction hyperbolic pretraining.
3. Residue-level SLEEC / auxiliary BCE losses.

The retrieval losses are the ones that directly affect reaction-to-enzyme and
enzyme-to-reaction benchmark performance. The hyperbolic and residue losses shape
representations before or alongside retrieval, but they are not the main retrieval
objective unless explicitly enabled by the active training path.

## Important Terminology

### Query And Target

In the retrieval modules:

- Query usually means reaction.
- Target usually means enzyme.

The core distance matrix is:

```text
dists[i, j] = distance(reaction_i, enzyme_j)
```

Lower distance means a better match. Most losses internally convert distances to
retrieval logits:

```text
logit[i, j] = -beta * dists[i, j]
```

So a smaller distance gives a larger logit.

### Beta

`beta` is the inverse temperature. Larger `beta` makes the softmax sharper and makes
the loss focus more strongly on small distance differences.

The base retrieval class stores beta as `logbeta` for numerical stability:

```text
beta = exp(logbeta)
```

If `learn_beta: true`, beta is trainable and can be clipped by `beta_min` and
`beta_max`.

### Positive Pair Source

`positive_pair_source` is not itself a loss, but it strongly changes the supervision
given to retrieval losses.

Supported modes:

```yaml
training:
  loss:
    positive_pair_source: observed_pairs
```

or:

```yaml
training:
  loss:
    positive_pair_source: all_known_in_batch
```

`observed_pairs` uses only the paired reaction-enzyme examples that appear in the
current batch.

`all_known_in_batch` uses all known training positives between reactions and enzymes
that are present in the DDP-global batch. If a reaction and a known matching enzyme
both appear anywhere in the global batch, that pair is marked positive.

This is important for multi-positive retrieval. With `observed_pairs`, a valid known
enzyme may be treated as a negative if it is not the observed paired item. With
`all_known_in_batch`, those known matches are recovered as positives.

## Retrieval Losses

Retrieval losses live in:

```text
horizyn/losses.py
```

They are built through:

```python
build_horizyn_loss(...)
```

and wired from config in:

```text
scripts/train_protein_pooling.py
horizyn/protein_pooling_lightning_module.py
horizyn/lightning_module.py
horizyn/reaction_conditioned_lightning_module.py
```

### FullBatchNCELoss

`FullBatchNCELoss` is the abstract base class for full-batch NCE-style retrieval
losses.

It does not compute a training loss by itself. It provides:

- beta initialization
- optional learnable beta
- beta clipping
- input validation

The expected inputs are:

```text
dists:      [num_queries, num_targets]
query_idx:  [num_positive_pairs]
target_idx: [num_positive_pairs]
```

`query_idx[k]` and `target_idx[k]` identify one positive pair:

```text
reaction query_idx[k] matches enzyme target_idx[k]
```

Input validation checks:

- `dists` is rank 2.
- indices are `torch.long`.
- positive pair indices are non-empty.
- query and target index tensors have equal length.
- indices are in range.
- distances are finite.

### FullBatchMLNCELoss

Config names:

```yaml
training:
  loss:
    name: FullBatchMLNCELoss
```

Aliases:

```yaml
name: mlnce
```

This is the original full-batch multi-label NCE retrieval loss.

Conceptual formula:

```text
L = beta * mean_{(i,j) in positives}(dists[i,j])
    + logsumexp_{all i,j}(-beta * dists[i,j])
```

The first term pulls positive pairs closer. The second term is a global partition
function over every reaction-enzyme pair in the batch. Since all in-batch pairs enter
the partition, the rest of the batch acts as negatives.

Main behavior:

- Uses the whole DDP-global batch as the contrastive candidate pool.
- Handles multi-positive data by accepting many positive `(query_idx, target_idx)`
  pairs.
- Weights each positive pair equally.
- Does not explicitly average by reaction anchor or enzyme anchor.

The last point matters. If some reactions have many known positives and others have
few, `FullBatchMLNCELoss` gives more total weight to the reactions with many positive
pairs. That can be acceptable, but it does not perfectly match balanced retrieval
metrics such as balanced MRR.

Practical interpretation:

- Good baseline when the dataset mostly behaves like one-to-one or uniformly
  multi-positive retrieval.
- Benefits from large batches because more in-batch negatives improve the partition.
- Can be less aligned with reaction-to-enzyme performance when the number of known
  positives per reaction is highly uneven.

Logged component:

```text
train/loss_mlnce
val/loss_mlnce
```

depending on the module and logging prefix.

### BidirectionalAnchorBalancedSupConLoss

Config names:

```yaml
training:
  loss:
    name: BidirectionalAnchorBalancedSupConLoss
```

Aliases:

```yaml
name: bidirectional_anchor_balanced_supcon
name: anchor_balanced_supcon
```

This is a bidirectional supervised contrastive loss. It is anchor-balanced instead of
pair-balanced.

It builds a boolean positive matrix:

```text
positive_mask[i, j] = true if reaction_i matches enzyme_j
```

Then it computes two losses.

Reaction-to-enzyme:

```text
R2E_i = logsumexp_j(logit[i,j])
        - logsumexp_{j positive for i}(logit[i,j])

R2E = mean_i(R2E_i)
```

Enzyme-to-reaction:

```text
E2R_j = logsumexp_i(logit[i,j])
        - logsumexp_{i positive for j}(logit[i,j])

E2R = mean_j(E2R_j)
```

Total:

```text
loss = 0.5 * (R2E + E2R)
       + direction_balance_weight * abs(R2E - E2R)
```

Main behavior:

- Each reaction anchor contributes once to R2E, regardless of how many positives it
  has.
- Each enzyme anchor contributes once to E2R, regardless of how many positives it has.
- The optional direction gap term discourages improving one retrieval direction while
  damaging the other.

Practical interpretation:

- Better aligned with balanced retrieval metrics than pair-weighted MLNCE.
- Useful when both R2E and E2R matter.
- With single positive per anchor, it behaves like bidirectional cross entropy.
- With multiple positives per anchor, it uses the positive logsumexp, so any known
  positive can satisfy the anchor.

Logged components:

```text
loss_r2e
loss_e2r
loss_direction_gap
loss_weighted_direction_gap
```

### MultiAlignmentRetrievalLoss

Config names:

```yaml
training:
  loss:
    name: MultiAlignmentRetrievalLoss
```

Aliases:

```yaml
name: multi_alignment_retrieval
name: multi_alignment
```

This is the most flexible retrieval loss. It starts from the same anchor-balanced
R2E and E2R supervised contrastive terms, then optionally adds hard-negative and
structure-aware terms.

Overall formula:

```text
loss =
  lambda_r2e_normalized * R2E
+ lambda_e2r_normalized * E2R
+ lambda_direction_gap * abs(R2E - E2R)
+ lambda_r2e_hard_neg * R2E_hard_neg
+ lambda_rr * RR
+ lambda_ee * EE
+ lambda_gw * GW
```

`lambda_r2e` and `lambda_e2r` are normalized internally:

```text
lambda_r2e_normalized = lambda_r2e / (lambda_r2e + lambda_e2r)
lambda_e2r_normalized = lambda_e2r / (lambda_r2e + lambda_e2r)
```

So these two weights define the relative direction priority.

#### Cross-Domain R2E And E2R Terms

The R2E and E2R terms are the same anchor-balanced supervised contrastive losses used
by `BidirectionalAnchorBalancedSupConLoss`.

Use these knobs to steer direction:

```yaml
training:
  loss:
    lambda_r2e: 0.85
    lambda_e2r: 0.15
```

Higher `lambda_r2e` prioritizes reaction-to-enzyme retrieval. Higher `lambda_e2r`
prioritizes enzyme-to-reaction retrieval.

#### Direction Gap Term

```yaml
training:
  loss:
    lambda_direction_gap: 0.05
```

This adds:

```text
lambda_direction_gap * abs(R2E - E2R)
```

It discourages large imbalance between directions. This can be useful if an R2E-heavy
run starts destroying E2R performance.

#### R2E Hard Negative Term

```yaml
training:
  loss:
    lambda_r2e_hard_neg: 0.2
    r2e_hard_neg_top_k: 256
    r2e_hard_neg_margin: 0.0
```

For each reaction anchor, the loss looks at the hardest negative enzymes in the batch.
It compares their combined score to the combined positive score:

```text
hard_loss_i =
  softplus(
    logsumexp(top_k_negative_logits)
    - logsumexp(positive_logits)
    + margin
  )
```

Then it averages over valid reaction anchors.

Main behavior:

- Directly targets R2E mistakes.
- Focuses on plausible wrong enzymes that the model currently ranks high.
- Can improve R2E metrics if the candidate pool contains meaningful hard negatives.
- Can over-focus on batch artifacts if `top_k` is too high or the positives are
  incomplete.

#### RR Structure Term

```yaml
training:
  loss:
    lambda_rr: 0.05
    tau_rr: 0.1
```

RR means reaction-reaction structure loss.

The module can build EC-derived reaction weights. Reactions sharing deeper EC prefixes
receive larger positive weights. The current depth weights are:

```text
shared EC depth 2 -> 0.25
shared EC depth 3 -> 0.50
shared EC depth 4 -> 1.00
```

The RR loss is a weighted supervised contrastive loss over reaction embeddings:

```text
RR_i = - weighted average log probability of EC-related reactions
RR = mean_i(RR_i)
```

Main behavior:

- Pulls EC-related reactions together.
- Can improve smoothness of the reaction representation.
- May hurt split generalization if EC similarity is too coarse or leaks the wrong
  inductive bias.

#### EE Structure Term

```yaml
training:
  loss:
    lambda_ee: 0.05
    tau_ee: 0.1
```

EE means enzyme-enzyme structure loss.

It is the enzyme-side analog of RR. Enzymes sharing EC prefixes are treated as
weighted positives.

Main behavior:

- Pulls EC-related enzymes together.
- Can stabilize enzyme geometry.
- May reduce fine-grained discrimination if enzymes with the same high-level EC are
  not interchangeable for the retrieval task.

#### GW Geometry Term

```yaml
training:
  loss:
    lambda_gw: 0.01
    tau_gw: 0.1
    gw_max_anchors: 512
    symmetric_gw: true
```

GW means a detached Gromov-Wasserstein-like geometry alignment regularizer.

It compares:

- reaction-reaction geometry
- enzyme-enzyme geometry
- a transported version of one geometry into the other space

In this implementation, embeddings are normalized and within-domain geometries are
detached before the squared error is computed. The transport matrix comes from the
current cross-domain similarities.

Main behavior:

- Encourages the reaction space and enzyme space to have compatible neighborhood
  structure.
- Can regularize alignment beyond direct positives.
- Can slow training and over-regularize if the learned cross-transport is poor early
  in training.

#### Structure Term Gating

`MultiAlignmentRetrievalLoss` receives:

```text
structure_terms_enabled
```

This allows training to use RR, EE, and GW while validation can disable them. The
protein pooling module supports:

```yaml
training:
  loss:
    apply_structure_terms_on_val: false
```

That is usually desirable because validation loss should reflect retrieval alignment
more directly, not auxiliary structure penalties.

Logged components:

```text
loss_r2e
loss_e2r
loss_cross
loss_r2e_weight
loss_e2r_weight
loss_direction_gap
loss_weighted_direction_gap
loss_r2e_hard_neg
loss_weighted_r2e_hard_neg
loss_rr
loss_ee
loss_gw
loss_weighted_rr
loss_weighted_ee
loss_weighted_gw
```

### HorizynFGWLoss

Config names:

```yaml
training:
  loss:
    name: HorizynFGWLoss
```

Aliases:

```yaml
name: fgw
name: horizyn_fgw
```

This is an older composite retrieval loss.

It starts with `FullBatchMLNCELoss` and optionally adds structure-aware regularizers:

```text
loss = MLNCE
       + lambda_r * RR_external_similarity
       + lambda_e * EE_external_similarity
       + lambda_g * GW
```

#### MLNCE Base

The base retrieval term is the pair-balanced MLNCE objective:

```text
MLNCE = beta * mean(positive distances)
        + logsumexp(-beta * all distances)
```

This is different from `MultiAlignmentRetrievalLoss`, whose base retrieval term is
anchor-balanced R2E / E2R supervised contrastive loss.

#### External Similarity RR / EE Terms

FGW uses externally supplied similarity matrices:

```text
reaction_similarity
enzyme_similarity
```

Pairs with similarity above thresholds become positives:

```yaml
training:
  loss:
    delta_r: 0.5
    delta_e: 0.5
```

Temperatures:

```yaml
training:
  loss:
    tau_r: 0.1
    tau_e: 0.1
```

Regularizer weights:

```yaml
training:
  loss:
    lambda_r: 0.05
    lambda_e: 0.05
```

Main behavior:

- Pulls externally similar reactions together.
- Pulls externally similar enzymes together.
- The quality of the external similarity features matters a lot.

#### FGW Geometry Term

```yaml
training:
  loss:
    lambda_g: 0.01
    tau_t: 0.1
    symmetric_gw: true
```

This aligns reaction and enzyme within-domain geometries using a soft transport from
reaction-enzyme similarities.

Unlike `MultiAlignmentRetrievalLoss`, this implementation does not detach the
within-domain geometry terms in the same way.

Logged components:

```text
loss_mlnce
loss_rr
loss_ee
loss_gw
loss_weighted_rr
loss_weighted_ee
loss_weighted_gw
```

## Positive Pair Construction In Retrieval Training

The protein pooling module deduplicates repeated reactions and enzymes inside the
batch before computing the global loss.

In DDP training, each rank first computes local embeddings. Then the module gathers
variable-sized tensors and IDs across all ranks. The loss is computed on the
DDP-global unique reaction and enzyme sets.

The positive indices are then built in one of two ways.

### observed_pairs

For each observed pair in the batch:

```text
positive reaction index = index of observed reaction ID
positive enzyme index   = index of observed enzyme ID
```

This mirrors the dataloader pairs exactly.

### all_known_in_batch

The datamodule exposes:

```text
_train_query_to_targets
```

For each unique reaction in the global batch, the module checks all known train
target enzymes. If the target enzyme is also present in the global batch, that pair is
added as a positive.

This makes the loss more correct for many-to-many reaction-enzyme supervision.

## Hyperbolic EC Hierarchy Losses

These losses live in:

```text
horizyn/hyperbolic_enzyme.py
```

They are used by:

```text
scripts/pretrain_hyperbolic_enzyme.py
scripts/pretrain_hyperbolic_reaction.py
```

The enzyme and reaction hyperbolic pretraining scripts use the same broad objective:

```text
loss_total =
  loss_rank
+ alpha_radial * loss_radial
+ alpha_radius_target * loss_radius_target
+ alpha_centroid_cone * loss_centroid_cone
+ alpha_ec_entailment_effective * loss_ec_entailment
```

`alpha_ec_entailment_effective` is zero until the EC entailment warmup has passed.

### HierarchyRankingLoss

This is a triplet margin ranking loss in Lorentz space.

For an anchor, positive, and negative:

```text
loss = relu(margin + distance(anchor, positive) - distance(anchor, negative))
```

The margin is based on EC hierarchy depth:

```text
margin = base_margin * max(depth_positive - depth_negative, 0)
```

Main behavior:

- If the positive shares a deeper EC prefix than the negative, it should be closer.
- If the model already ranks the positive closer by enough margin, the triplet gives
  zero loss.

Logged values include:

```text
loss_rank
num_triplets
mean_positive_distance
mean_negative_distance
mean_depth_positive
mean_depth_negative
```

### same_ec4_radius_loss

This is logged as `loss_radial` in the pretraining scripts.

It finds pairs with complete shared EC4 labels and penalizes radius differences:

```text
loss = mean(abs(radius_i - radius_j))
```

Main behavior:

- Enzymes or reactions with the exact same EC4 should have similar distance from the
  Lorentz origin.
- This encourages same-function points to occupy a similar hierarchy depth.

If there are no same-EC4 pairs in the batch, the loss returns zero.

### radius_target_loss

This is logged as `loss_radius_target`.

It penalizes points that move farther from the Lorentz origin than a configured
radius target:

```text
loss = mean(relu(radius - radius_target)^2)
```

Main behavior:

- Prevents hyperbolic embeddings from drifting too far outward.
- Helps control numerical stability and hierarchy depth.

If `radius_target <= 0`, the loss returns zero.

### LorentzEntailmentConeLoss

This is the primitive cone violation loss used by the centroid and EC-prefix
entailment losses.

For a child point and a parent point, it computes:

- `phi`: the angle of the child relative to the parent.
- `omega`: the parent cone aperture.
- `violation = relu(phi - eta * omega)`.

The loss is:

```text
loss = violation ^ cone_loss_power
```

Main behavior:

- A child should lie inside the parent entailment cone.
- Violations mean the child is geometrically inconsistent with the hierarchy.

### NonParametricCentroidConeLoss

This is logged as `loss_centroid_cone`.

It builds EC-prefix centroids from the current batch:

```text
EC 1
EC 1.1
EC 1.1.1
EC 1.1.1.1
```

The centroids are computed by:

1. Mapping Lorentz points to tangent space.
2. Averaging points with the same EC prefix.
3. Mapping the centroid back to Lorentz space.

It then applies entailment cone loss to:

- child-prefix centroid -> parent-prefix centroid
- item point -> deepest available prefix centroid

Main behavior:

- Adds hierarchy constraints without learning explicit EC class embeddings.
- Depends on having enough items per EC prefix in the batch.

Logged values include:

```text
loss_centroid_cone
num_centroid_cone_terms
```

### ECPrefixEntailmentLoss

This is logged as `loss_ec_entailment`.

It also builds batch EC-prefix centroids, then applies Lorentz cone entailment between:

- deeper EC prefix centroids and their parent prefix centroids
- each item and its deepest available EC prefix centroid

Compared with `NonParametricCentroidConeLoss`, this implementation also logs richer
diagnostics:

```text
loss_ec_entailment
num_ec_entailment_terms
ec_entailment_violation_rate
mean_ec_entailment_angle
mean_ec_entailment_aperture
```

Main behavior:

- Encourages the hyperbolic space to encode EC parent-child hierarchy.
- Warmup is supported so the model can first learn basic ranking structure before
  cone entailment becomes active.

## SLEEC And Residue-Level Losses

These losses are separate from the reaction-enzyme retrieval objective.

### confidence_aware_stage1_loss

Implemented in:

```text
horizyn/sleec_stage1.py
```

Used by:

```text
scripts/train_sleec_stage1.py
```

This trains the SLEEC stage-1 residue classifier.

Total:

```text
loss = supervised_BCE + lambda_pseudo * pseudo_BCE
```

The supervised term is:

```text
binary_cross_entropy_with_logits(supervised_logits, supervised_labels)
```

Optionally, `supervised_pos_weight` can rebalance positive residues.

The pseudo-label term is also BCE-with-logits, but it is confidence-gated:

```text
confidence = max(sigmoid(logit), 1 - sigmoid(logit))
weight = 1[confidence > confidence_threshold]
pseudo_loss = mean(weight * BCE(pseudo_logits, pseudo_labels))
```

Main behavior:

- Learns residue-level active-site or functional residue scores.
- Uses pseudo labels only when the model is confident enough.
- Does not directly optimize retrieval MRR.

Logged components:

```text
supervised_loss
pseudo_loss
pseudo_selected
pseudo_coverage
```

### Protein Pooling Residue Supervision Loss

Implemented in:

```text
horizyn/protein_pooling_lightning_module.py
```

This is an optional auxiliary loss during retrieval training.

It is active only if:

```yaml
training:
  lambda_residue: <positive value>
```

and residue labels plus valid masks are present.

It computes:

```text
residue_loss = BCEWithLogits(sleec_logits[valid_residues], labels[valid_residues])
```

Then retrieval loss is updated:

```text
loss_total = retrieval_loss + lambda_residue * residue_loss
```

Logged components:

```text
loss_residue
loss_weighted_residue
```

Main behavior:

- Keeps SLEEC residue scores aligned with available residue-level labels.
- Can regularize the enzyme pooling module.
- May compete with retrieval if `lambda_residue` is too large.

## Attention Statistics Are Not Losses

Some functions compute attention diagnostics, for example:

```text
mean_attention_entropy
attention_mass_above_threshold
```

These are logging metrics, not training losses, unless manually added to the training
objective elsewhere.

They help diagnose pooling behavior:

- high entropy means diffuse attention
- low entropy means concentrated attention
- mass above threshold measures how much attention falls on high-scoring residues

## How To Interpret Logged Loss Names

Lightning logs the total loss and individual components. The prefix depends on the
stage:

```text
train/...
val/...
```

Common names:

```text
loss
loss_step
loss_epoch
```

These are the total objective used by the optimizer or the validation step.

For retrieval training, total loss can include:

```text
retrieval loss
+ optional weighted residue loss
```

For `MultiAlignmentRetrievalLoss`, inspect these first:

```text
train/loss_r2e
train/loss_e2r
train/loss_cross
train/loss_r2e_hard_neg
train/loss_weighted_r2e_hard_neg
```

For R2E-focused runs, useful signs are:

- `loss_r2e` decreases.
- `loss_r2e_hard_neg` decreases or stabilizes at a lower value.
- validation `reaction_to_enzyme/mrr` improves.
- `loss_e2r` does not explode.
- balanced MRR does not collapse.

Do not compare absolute loss values across different loss families too directly.
`FullBatchMLNCELoss`, anchor-balanced supervised contrastive loss, and hard-negative
augmented loss have different scales.

## Which Loss To Use For Which Goal

### General Baseline

Use:

```yaml
training:
  loss:
    name: FullBatchMLNCELoss
    positive_pair_source: observed_pairs
```

This is the simplest and most stable baseline.

### Correct Multi-Positive Supervision

Use:

```yaml
training:
  loss:
    name: FullBatchMLNCELoss
    positive_pair_source: all_known_in_batch
```

or:

```yaml
training:
  loss:
    name: MultiAlignmentRetrievalLoss
    positive_pair_source: all_known_in_batch
```

This avoids treating known positives as negatives when they occur in the same global
batch.

### Improve Balanced R2E And E2R

Use:

```yaml
training:
  loss:
    name: BidirectionalAnchorBalancedSupConLoss
    positive_pair_source: all_known_in_batch
    direction_balance_weight: 0.0
```

Optionally add a small direction balance weight if one side dominates.

### Prioritize Reaction-To-Enzyme Retrieval

Use:

```yaml
training:
  loss:
    name: MultiAlignmentRetrievalLoss
    positive_pair_source: all_known_in_batch
    lambda_r2e: 0.85
    lambda_e2r: 0.15
    lambda_r2e_hard_neg: 0.2
    r2e_hard_neg_top_k: 256
    r2e_hard_neg_margin: 0.0
```

Monitor:

```text
val/reaction_to_enzyme/mrr
val/enzyme_to_reaction/mrr
val/balanced_mrr
```

If R2E improves but E2R collapses, increase `lambda_e2r` or add
`lambda_direction_gap`.

### Add EC Structure Carefully

Use:

```yaml
training:
  loss:
    name: MultiAlignmentRetrievalLoss
    lambda_rr: 0.01
    lambda_ee: 0.01
```

Start with small values. RR and EE can help if EC hierarchy matches the desired
retrieval behavior, but they can also blur distinctions between enzymes or reactions
that share coarse EC prefixes.

### Use FGW-Style Regularization

Use:

```yaml
training:
  loss:
    name: HorizynFGWLoss
    lambda_r: 0.05
    lambda_e: 0.05
    lambda_g: 0.01
```

This is more experimental for retrieval because the base loss is still MLNCE and the
structure terms depend heavily on external similarity quality.

## Practical Takeaways

For the current objective of improving reaction-to-enzyme retrieval:

1. `MultiAlignmentRetrievalLoss` is the most directly relevant loss.
2. `positive_pair_source: all_known_in_batch` is important for avoiding false
   negatives in multi-positive supervision.
3. `lambda_r2e > lambda_e2r` directly prioritizes R2E.
4. `lambda_r2e_hard_neg` focuses training on the difficult enzymes that R2E retrieval
   currently confuses.
5. `lambda_direction_gap` is a guardrail when R2E gains come at too much E2R cost.
6. RR, EE, and GW are regularizers, not guaranteed improvements. They should be
   tested one at a time.
7. Absolute loss values are less important than validation retrieval metrics,
   especially `val/reaction_to_enzyme/mrr` for the current goal.

## Source Map

Main retrieval losses:

```text
horizyn/losses.py
```

Retrieval loss wiring:

```text
horizyn/protein_pooling_lightning_module.py
horizyn/lightning_module.py
horizyn/reaction_conditioned_lightning_module.py
scripts/train_protein_pooling.py
```

Hyperbolic EC losses:

```text
horizyn/hyperbolic_enzyme.py
scripts/pretrain_hyperbolic_enzyme.py
scripts/pretrain_hyperbolic_reaction.py
```

SLEEC and residue losses:

```text
horizyn/sleec_stage1.py
scripts/train_sleec_stage1.py
horizyn/protein_pooling_lightning_module.py
```
