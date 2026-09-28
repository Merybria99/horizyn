# Post-Ablation Improvement Options

Last updated: 2026-08-24

## Scope

This note reviews the strongest retained ReactZyme ablations and defines the
smallest next campaign that could improve F3 without relying on the rejected
reaction-cluster-held-out checkpoint proxy.

All paper-test values below use the official candidate pools and ReactZyme's
all-positive MRR. Validation-only observations are identified explicitly.

## Rejected Approach

The reaction-cluster-held-out split at similarity 0.85 is not suitable as the
primary checkpoint selector for the released Reaction-Sim benchmark.

- It selected epoch 10 over epoch 28.
- Epoch 10 test MRR was `0.4275` E->R and `0.3481` R->E.
- A diagnostic epoch-28 test was better in both directions at `0.4624` and
  `0.3647` even though cluster validation strongly preferred epoch 10.
- The repartition also withheld 1,031 official-training reaction IDs and
  replaced 14,747 official-training pairs with 14,698 former validation pairs.

The run remains available for provenance, but it must not be used in model
selection, the leaderboard, or final checkpoint promotion.

## Best Evidence From Existing Ablations

| Result | Strongest evidence | Limitation |
|:--|:--|:--|
| F3 set chemistry | Best balanced one-stage model, macro MRR `0.6490`; Reaction-Sim `0.4822` E->R and `0.4009` R->E | Reaction fusion assigns about 84% of its weight to set chemistry on Reaction-Sim |
| F4 factorized reaction | Best internal Reaction-Sim E->R at `0.5023` | Replacing adaptive fusion with a fixed identity path lowers Time-Sim and R->E; macro `0.6171` |
| Q6 bidirectional adapters | Best retained macro `0.6593`; Reaction-Sim R->E `0.4089` | Single seed, two-stage, direction-specific, and Reaction-Sim E->R remains `0.4682` |
| S1 modality normalization | Versus S0, Reaction-Sim test improved from `0.4368/0.3609` to `0.4788/0.3824` E->R/R->E | Only seed 42 was tested; attention still assigns about 79% to chemistry |
| S4 bounded attention plus consistency | Three-seed validation winner; mean validation macro `0.6439`; official test improves the Time/Enzyme cells | Three-seed Reaction-Sim test is only `0.4596` E->R and `0.3691` R->E |
| L5 hybrid loss plus directional temperatures | Improved seed-42 test macro from L0 `0.6011` to `0.6061` | Gain is almost entirely Time-Sim; Reaction-Sim changes by less than `0.002`; replication stopped early |
| Historical R07 | Strong simple ProtT5-mean + ReactionT5v2/UniMol2 result | Protocol is not paper-comparable and requires a clean rerun |

## Causal Interpretation

### What should be preserved

1. ReactionT5v2 is essential. Removing it in F2 collapses macro MRR from the
   F1 range to `0.3076`.
2. Train-fitted 617-D set chemistry is useful. F3 improves over F0 by about
   `0.0127` macro MRR.
3. The shared one-stage F3 geometry remains the strongest simplicity/performance
   reference.
4. Fixed modality capacity is useful specifically for unseen-reaction E->R.
   F4 is the clearest evidence.

### What should not be added again

1. Biological auxiliary supervision: B1-B4 all lose to the unsupervised B0
   control.
2. Hard-negative mining: Q2 loses Reaction-Sim E->R and does not improve the
   macro; stronger historical negatives also regress.
3. Dense directional/RXNMapper/Rhea features: Q3 and F5/F6 do not provide a
   reliable gain over simpler controls.
4. Chemistry dropout by itself: S3 regresses Time-Sim and does not beat the
   bounded control on Reaction-Sim validation.
5. The cluster-held-out checkpoint selector described above.

## Main Architectural Hypothesis

F3 and F4 expose complementary behavior:

```text
F3: adaptive attention + unrestricted output MLP
    -> strong five-cell performance
    -> chemistry-dominated reaction representation

F4: fixed normalized modality blocks + identity output
    -> best Reaction-Sim E->R
    -> insufficient adaptation for Time-Sim and R->E
```

The repository already implements the missing interpolation as
`output_projection: residual_mlp`:

```text
z_factorized = concat(
    sqrt(0.25)   * normalize(ReactionT5v2_128),
    sqrt(0.375)  * normalize(UniMol2_192),
    sqrt(0.1875) * normalize(ChIRo_96),
    sqrt(0.1875) * normalize(SetChemistry_96)
)

z_reaction = normalize(
    z_factorized + sigmoid(g) * Delta(LayerNorm(z_factorized))
)
```

Initialize `sigmoid(g)=0.10`. This retains F4's explicit block geometry while
allowing one shared reaction embedding to learn a gated correction. It is a
single retrieval head, requires no adapter stage, and uses no new data source.
The sigmoid bounds the scalar gate but not the MLP output norm, so the run must
log the residual/base norm ratio. Add a small cosine identity penalty (`0.02`)
or cap the gate if that ratio grows beyond `0.30`.

## Recommended Campaign

### Stage 0: close the existing S4 result (complete)

The already frozen S4 checkpoints for seeds 42, 17, and 73 were evaluated on
the official test protocol on 2026-08-22. Their mean E->R/R->E MRR values were
`0.8113/0.5661` on Time-Sim, `0.9705/0.6810` on Enzyme-Sim, and
`0.4596/0.3691` on Reaction-Sim. S4 therefore does not solve the unseen-reaction
failure and should not be promoted over F3.

### Stage 1: four seed-42 architecture screens

| ID | Configuration | Question |
|:--|:--|:--|
| M0 | Exact audited F3/S0 control | Establish same-code rerun variance |
| M1 | M0 plus scale-preserving modality L2 normalization | Confirm the simplest S1 gain |
| M2 | F4 factorized concatenation plus gated residual MLP, gate `0.10`, identity weight `0.02` | Preserve F4 generalization while recovering adaptive capacity |
| M3 | M2 plus detached chemistry-free consistency, weight `0.03` | Test the only shortcut regularizer with three-seed validation support |

Keep all other components fixed: official split-specific training data,
ReactionT5v2, UniMol2, ChIRo, train-fitted set chemistry, the current enzyme
tower, observed-pair FullBatchMLNCE, and corrected all-positive validation
metrics.

### Stage 2: replication

Promote at most two variants using standard validation only, then repeat them
at seeds 17 and 73. Do not use the released test to choose an epoch, gate,
consistency weight, or variant.

### Stage 3: official test

Freeze the three validation-selected checkpoints per promoted variant and run
the official test once. Report mean and standard deviation across seeds.

Suggested acceptance conditions relative to the matched M0 control:

- no six-cell test MRR regression larger than `0.005`;
- Reaction-Sim E->R improvement of at least `0.010`;
- Reaction-Sim R->E at least preserved;
- six-cell macro improvement larger than the control's seed standard deviation.

## Lower-Priority Follow-Ups

1. Port L5's separate directional temperatures to the winning architecture
   only after the architecture screen. L5 does not solve Reaction-Sim by itself.
2. Rerun the simple R07 enzyme/reaction stack under the exact current paper
   protocol as a complexity control.
3. If M2 underfits, test one residual gate initialization (`0.20`) rather than
   opening a broad gate sweep.
4. If M2 preserves Reaction-Sim but loses Time-Sim, add the S4 consistency term
   before adding any direction-specific adapter.

## Recommendation

S4 testing is complete. For new training, prioritize M1 and M2. M1 is the
lowest-risk improvement supported by test evidence; M2 is the cleanest
untested architecture that directly combines the observed strengths of F3 and
F4. Do not restart biological supervision, hard-negative mining, or
cluster-held-out checkpoint selection.

## Execution Status

The M0-M3 seed-42 screen was launched on 2026-08-24 under
`horizyn/runs/reactzyme_f3_geometry_v1`. It uses one H200 per variant and a
per-device batch size of 2,048, preserving the audited four-GPU effective
global batch size. All persistent caches, logs, checkpoints, and W&B state are
inside the project tree; only short-lived multiprocessing sockets use `/tmp`.

W&B project: `horizyn-reactzyme-f3-geometry-v1`.
