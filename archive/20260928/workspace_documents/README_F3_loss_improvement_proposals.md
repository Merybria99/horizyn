# F3 Loss Improvement Proposals

Date: 2026-08-14

## Scope

This report analyzes loss-level improvements for the F3 `set_chemistry`
configuration. The objective is to improve retrieval performance, especially
reaction-similarity E->R MRR, without changing the F3 reaction or enzyme
representation stacks and without sacrificing the five ReactZyme result cells
where F3 is already competitive with TIGER.

The analysis is based on:

- the current F3 configurations and loss implementation;
- the corrected ReactZyme all-positive MRR implementation;
- the biological and graph-degree analysis of the three ReactZyme splits;
- the previous R-series loss and sampling experiments;
- the ReactZyme, FGW-CLIP, and TIGER training objectives.

## Current F3 Baseline

F3 uses:

- enzyme: ProtT5 residues, frozen SLEEC scoring, SLEEC-guided attention, and
  the factorized enzyme projection;
- reaction: ReactionT5v2, UniMol2, ChIRo, and 617-dimensional train-fitted
  molecule-set chemistry features;
- multimodal reaction attention followed by an MLP projection;
- 512-dimensional normalized retrieval embeddings;
- four-GPU DDP with batch size 512 per GPU;
- observed-pair `FullBatchMLNCELoss`;
- fixed inverse temperature `beta=10`;
- no BioFP auxiliary loss, hard-negative loss, EC loss, or geometry loss.

The corrected paper-test MRR results are:

| Split | E->R | R->E |
|---|---:|---:|
| Time | 0.8013 | 0.5644 |
| Enzyme similarity | 0.9650 | 0.6800 |
| Reaction similarity | 0.4822 | 0.4009 |
| Six-cell macro | 0.6490 | |

The main remaining comparison gap is reaction-similarity E->R: TIGER
ESM2Text reports 0.518, while F3 reports 0.4822. F3 already exceeds TIGER on
reaction-similarity R->E under the corrected all-positive metric.

## Diagnosis of the Current Loss

F3 currently minimizes:

```text
L_MLNCE = -mean_{(i,j) in P}(s_ij)
          + logsumexp_{i,j}(s_ij)
```

where `P` is the sampled set of positive enzyme-reaction pairs and `s_ij` is
the temperature-scaled similarity.

This is a global pair-distribution objective. It has useful optimization
properties and has performed well in this repository, but it is not the same
as a bidirectional query-ranking objective.

### 1. Pair weighting does not match query weighting

Every positive pair has equal weight. A reaction associated with 1,000
enzymes therefore contributes roughly 1,000 times as much positive gradient as
a reaction associated with one enzyme.

ReactZyme metrics first compute a result for each query and then average over
queries. Consequently, the training objective and evaluation measure weight
the association graph differently.

### 2. Reaction-degree imbalance is extreme

The reaction-similarity test split contains 386 reactions and 14,689 pairs.
The ten largest reaction groups contain 56.7% of the pairs, and reaction degree
has a Gini coefficient of 0.901.

The global MLNCE objective can therefore learn reaction popularity and hub
geometry instead of requiring every reaction anchor to produce a useful local
ranking. This is especially problematic in the reaction-similarity split,
where every test reaction is absent from training and train-reaction frequency
cannot transfer directly.

### 3. `observed_pairs` creates avoidable false negatives

F3 uses:

```yaml
positive_pair_source: observed_pairs
```

If an enzyme and reaction are both present in a DDP-global batch but their
known association was not one of the sampled rows, that association remains in
the denominator without being marked positive. The repository already
supports `all_known_in_batch`, which constructs the positive mask from every
known training association represented in the batch.

This does not solve unknown or missing biological annotations, but it removes
false negatives that are known to the training graph.

### 4. Positive cardinality differs by direction

Most enzyme queries have one associated reaction. Reaction queries can have
many associated enzymes. A single positive aggregation rule is consequently
not ideal for both directions:

- E->R is approximately a single-positive classification/ranking problem;
- R->E is a highly variable multi-positive ranking problem.

The corrected ReactZyme R->E MRR averages the reciprocal rank of every valid
enzyme for each reaction. A loss that only requires one easy positive to rank
well is not sufficiently aligned with this metric.

### 5. Checkpoint selection currently uses the wrong MRR variant

F3 checkpoints and early-stops on:

```yaml
checkpoint_monitor: val/mean_bidirectional_mrr
```

In the current metric implementation, `mrr` uses the rank of the best positive.
The corrected `reactzyme_mrr` instead averages reciprocal rank over every
positive associated with the query.

Every loss experiment should therefore use:

```yaml
logging:
  checkpoint_monitor: val/mean_bidirectional_reactzyme_mrr

training:
  early_stopping:
    monitor: val/mean_bidirectional_reactzyme_mrr
```

Without this correction, a better multi-positive loss can produce a better
checkpoint that is not selected by early stopping.

## Proposed Loss Variants

### L0: Corrected FullBatchMLNCE control

Keep the F3 loss and architecture unchanged, but use:

```yaml
positive_pair_source: all_known_in_batch
checkpoint_monitor: val/mean_bidirectional_reactzyme_mrr
```

Purpose:

- establish a correctness-controlled F3 baseline;
- measure the effect of removing known false negatives;
- separate loss changes from checkpoint-selection changes.

Expected impact: small but low risk. Previous R-series results found little
aggregate difference between observed and all-known MLNCE, but the corrected
checkpoint metric was not used in that comparison.

### L1: Degree-tempered FullBatchMLNCE

Keep the global partition but weight positive pairs by reaction degree:

```text
w_ij proportional to degree(reaction_i)^(-alpha)

L_degree = -sum_{(i,j) in P}(normalized(w_ij) * s_ij)
           + logsumexp_{i,j}(s_ij)
```

Test:

- `alpha=0.25`;
- `alpha=0.50` as the primary setting;
- `alpha=0.75`.

Do not begin with `alpha=1.0`, because complete inverse-degree weighting can
give rare, potentially noisy reactions excessive influence.

Purpose:

- retain the simple and stable F3 objective;
- reduce reaction-hub domination;
- discourage train-reaction popularity shortcuts;
- isolate degree correction without changing the batch sampler.

### L2: Bidirectional anchor-positive-mass InfoNCE

Use one normalized softmax per reaction anchor and one per enzyme anchor:

```text
L_anchor(i) = logsumexp(all logits for anchor i)
              - logsumexp(positive logits for anchor i)

L = lambda_e2r * mean_i(L_e2r(i))
    + lambda_r2e * mean_j(L_r2e(j))
```

This is already approximated by `BidirectionalAnchorBalancedSupConLoss`.

Purpose:

- test the effect of replacing global pair normalization with query
  normalization;
- reproduce the central geometric difference between F3 and TIGER-style
  symmetric InfoNCE.

Limitation: `logsumexp` over positives optimizes the total positive probability
mass. One easy positive can dominate that mass, so the loss is better aligned
with first-positive retrieval than with corrected all-positive R->E MRR. This
variant is an important diagnostic but is not the preferred final objective.

### L3: Decoupled all-positive bidirectional InfoNCE

For each anchor `i` and positive `j`, compare that positive against only the
known negatives:

```text
L_all_positive(i) =
    mean_{j in P_i} log(
        1 + sum_{k not in P_i} exp(s_ik - s_ij)
    )
```

Compute this in both directions and average over anchors:

```text
L_cardinality = gamma * mean_enzyme_anchors(L_e2r)
                + (1 - gamma) * mean_reaction_anchors(L_r2e)
```

Properties:

- every known positive must outrank the negatives;
- positives do not compete against other valid positives;
- every anchor contributes equal total weight;
- every positive within an anchor contributes equal weight;
- the R->E term directly reflects corrected ReactZyme all-positive MRR.

Recommended initial weights:

- `gamma=0.55` for reaction-similarity training;
- `gamma=0.50` for time and enzyme-similarity training;
- fixed `beta=10` for the first comparison.

This formulation is inspired by FGW-CLIP's explicit multi-positive association
matrix, but adds per-anchor and per-positive normalization to prevent reaction
hubs from dominating the loss.

### L4: Hybrid MLNCE plus cardinality-aware ranking

This is the primary recommendation:

```text
L_hybrid = (1 - lambda) * L_MLNCE
           + lambda * L_cardinality
```

Use:

- `positive_pair_source: all_known_in_batch`;
- `lambda=0` at initialization;
- linear warmup to `lambda=0.30` over epochs 1-5;
- `gamma=0.55` for reaction similarity and `0.50` otherwise;
- fixed `beta=10` initially.

Rationale:

- FullBatchMLNCE preserves the stable global association geometry responsible
  for F3's strong five-cell performance;
- the cardinality-aware component adds query-normalized ranking gradients;
- warmup lets the towers establish coarse cross-domain alignment before
  emphasizing local rankings;
- the relatively small `lambda` limits regression risk.

This variant has the best expected performance/simplicity tradeoff.

### L5: Hybrid loss with separate learned temperatures

Extend L4 with independent inverse temperatures:

```text
beta_e2r != beta_r2e
```

Recommended settings:

- initialize both at `10`;
- constrain both to `[3, 30]`;
- optimize the logarithm of each temperature;
- add a small penalty toward the initial value;
- log both values separately to W&B.

E->R and R->E use different candidate spaces and have very different positive
cardinality. Separate temperatures let each directional softmax learn an
appropriate ranking sharpness. TIGER provides evidence that a learned
temperature works with symmetric retrieval, while this variant adapts it to
the asymmetry of ReactZyme.

### L6: Soft reciprocal-rank auxiliary

Approximate each positive rank using:

```text
soft_rank(i,j) = 1
    + sum_{k not in P_i} sigmoid((s_ik - s_ij) / tau_rank)
```

Add a small top-heavy ranking term:

```text
L = L_hybrid + 0.05 * mean(log(soft_rank))
```

Use only the top 64-128 in-batch negatives for each anchor to control memory
and computation.

Purpose:

- make the final optimization more directly sensitive to reciprocal rank;
- focus gradients on candidates that can affect the top of the list.

Risk: differentiable ranking losses are less stable and more sensitive to
temperature. This should only be tested after selecting a stable hybrid loss.

### L7: Class-balanced sigmoid EBM-NCE

Treat every known association independently:

```text
L_sigmoid = -mean_positive(log(sigmoid(s_ij - b)))
            -eta * mean_negative(log(sigmoid(-(s_ij - b))))
```

Use a learned bias `b` and class-balanced negative weight `eta`.

Advantages:

- multiple valid enzymes do not compete with each other;
- there is no single global softmax partition;
- the objective naturally supports an arbitrary multi-positive matrix.

Risks:

- negative calibration is sensitive to the positive/negative ratio;
- independent binary discrimination is less directly rank-normalized than
  InfoNCE;
- missing annotations can be treated as false negatives.

This is a useful secondary comparison, not the first candidate to replace F3.

## Variants Not Recommended Initially

### Naive hard-negative margins

ReactZyme explicitly identifies negative construction as unresolved because an
unannotated pair is not necessarily biologically negative. Previous local
hard-negative experiments were also inconsistent. Hard-negative mining should
therefore not be combined with the first loss comparison.

If introduced later, it should:

- exclude all known positives;
- use semi-hard rather than only hardest negatives;
- use a small weight, approximately `0.05-0.10`;
- avoid forcing chemically or functionally plausible unannotated pairs far
  apart.

### Strong EC, GW, or BioFP auxiliary losses

These objectives alter latent organization and no longer provide a clean test
of the retrieval loss. FGW-CLIP also reports that a large EC loss can dominate
the retrieval objective. They should only be added after selecting the best
cross-domain loss.

### Unnormalized all-positive summation

Summing a loss over all positives without dividing by anchor degree would
continue to let reaction hubs dominate. Every proposed multi-positive loss
must average first within each anchor and then across anchors.

## Recommended Experiment Order

| Run | Objective | Purpose | Priority |
|---|---|---|---:|
| L0 | Corrected F3 MLNCE, all-known positives | Correctness baseline | 1 |
| L1 | L0 plus degree weighting, `alpha=0.50` | Simple hub correction | 2 |
| L2 | Anchor-positive-mass InfoNCE | Query-normalization diagnostic | 3 |
| L3 | Pure decoupled all-positive InfoNCE | Metric-aligned replacement | 4 |
| L4 | 0.70 MLNCE + 0.30 decoupled loss | Recommended hybrid | 1 |
| L5 | L4 plus separate learned temperatures | Directional calibration | 2 |
| L6 | L4 plus 0.05 soft-rank loss | Top-rank refinement | 5 |
| L7 | Class-balanced sigmoid EBM-NCE | Alternative multi-label objective | 6 |

For initial screening, keep the architecture, data, optimizer, epochs, and
batch construction identical to F3. This is necessary to attribute differences
to the loss.

After screening:

1. run L0, L1, L3, and L4 on all three splits with seed 42;
2. repeat the two strongest variants with three seeds;
3. only then add learned temperatures or a rank auxiliary;
4. evaluate the selected checkpoint once on the released paper test set.

The observed reaction-similarity E->R variation between equivalent F3/Q0
evaluations is approximately 0.0156. A single-seed improvement smaller than
roughly 0.02 should therefore not be treated as decisive.

## Selection Criteria

Primary metric:

```text
mean of the six corrected ReactZyme MRR cells
```

Targeted metric:

```text
reaction-similarity E->R MRR
```

Guardrails:

- reaction-similarity E->R should improve beyond F3's 0.4822;
- reaction-similarity R->E should remain at least approximately 0.395;
- no previously strong F3 cell should fall by more than 0.01;
- improvements should be consistent across at least three seeds;
- checkpoints must be selected using
  `val/mean_bidirectional_reactzyme_mrr`, not first-positive MRR.

The practical target is to exceed 0.50 reaction-similarity E->R while retaining
a six-cell macro MRR near or above 0.649.

## Final Recommendation

The first implementation should be L4:

```text
0.70 * all-known FullBatchMLNCE
+ 0.30 * cardinality-aware bidirectional decoupled InfoNCE
```

with a five-epoch auxiliary-weight warmup, fixed `beta=10`, and corrected
ReactZyme-MRR checkpoint selection.

This is safer than replacing MLNCE outright. It preserves the globally stable
F3 objective while explicitly correcting its two largest weaknesses:

1. pair-weighted rather than query-weighted training;
2. identical treatment of nearly single-positive E->R and highly
   multi-positive R->E retrieval.

If L4 improves validation consistently, the next extension should be L5 with
separate bounded directional temperatures.

## References

- Hua et al., [ReactZyme: A Benchmark for Enzyme-Reaction
  Prediction](https://arxiv.org/pdf/2408.13659).
- [Multi-Alignment Contrastive Learning for Enzyme-Reaction
  Retrieval](https://arxiv.org/pdf/2512.08508).
- [TIGER: Text-Informed Generalized Enzyme-Reaction
  Retrieval](https://arxiv.org/pdf/2605.24489).
- `horizyn/horizyn/losses.py`.
- `horizyn/horizyn/protein_pooling_lightning_module.py`.
- `horizyn/docs/reactzyme_split_biology_report.md`.
- `documents/README_reactzyme_all_ablations.md`.

## Implemented Campaign

The loss campaign is implemented under:

```text
horizyn/runs/reactzyme_f3_loss_ablation_v1
```

It generates 72 reusable train/test config pairs: eight losses, three splits,
and seeds `42`, `17`, and `73`. The controller executes only the intended 36
training runs: all eight variants at seed 42, followed by seeds 17 and 73 for
the two validation-selected variants.

The shared controller supports `setup`, `screen`, `replicate`, `freeze`,
`test`, and `all` phases:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn

./scripts/run_reactzyme_f3_loss_campaign.sh \
  runs/reactzyme_f3_loss_ablation_v1 \
  --phase all \
  --detach
```

Inspect the complete schedule without starting work:

```bash
./scripts/run_reactzyme_f3_loss_campaign.sh \
  runs/reactzyme_f3_loss_ablation_v1 \
  --phase all \
  --dry-run
```

Each loss also has a dedicated launcher, for example:

```bash
./scripts/run_reactzyme_f3_loss_L4.sh \
  runs/reactzyme_f3_loss_ablation_v1 \
  --phase screen \
  --detach
```

All training jobs use GPUs `0,1,2,3`. Training runs are serialized behind a
host-specific lock; test evaluation can use GPUs 0-2 concurrently. Runtime
caches, temporary files, W&B state, checkpoints, and logs remain under the run
root rather than the home directory.

Setup materializes canonical isomeric reaction CSVs and regenerates
ReactionT5v2, UniMol2, ChIRo, and 617-dimensional set-chemistry artifacts. HDF5
artifacts without matching SMILES mode, source CSV hash, normalizer version,
extractor version, and coverage are rejected rather than reused.
