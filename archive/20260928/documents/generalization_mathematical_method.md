# Independent retrieval with training support and reaction anchors

This note defines the frozen methods and their limitations. Phase two was developed after phase one's external failure; phase four followed the phase-two evaluations. Their reused-panel checks are exploratory. The registered measured panels and their exposure histories remain separate. Outcome claims belong in [the findings](../../../findings.md).

The implementation references are [the residual towers and loss](../../../horizyn/generalization_residual.py), [original anchors](../horizyn/semantic_anchors.py), [density gating](../horizyn/generalization_density.py), [smooth reaction responses](../horizyn/semantic_smooth.py), and [the composed encoder](../horizyn/generalization_phase2.py). Both immutable recipes and their relationship are recorded in [phase one](../../../runs/generalization_20260919_2251/frozen_recipe.json) and [phase two](../../../runs/generalization_20260919_2251/phase2/frozen_recipe.json).

## Fixed data and independent endpoints

Let `x_e` and `x_r` be native frozen F3 protein and reaction vectors. Let `N(x)=x/||x||₂`, with the implementation's treatment of zero norms. The training graph `T` contains recorded positive enzyme–reaction edges. All dictionaries, centers, density references and association lookups are fitted from `T` and its features. Inference never inserts candidate/query labels into this graph.

The original F3 score is the dot product of independently normalized endpoints. Residual towers produce `delta_e(x)=0.2 MLP_e(x)` and `delta_r(x)=0.2 MLP_r(x)`. Each tower is512→1024→512 with LayerNorm, GELU and an initially zero final layer. F3 itself remains frozen.

Full-graph training minimizes bidirectional cross entropy against uniform distributions over each node's **recorded** positive neighbors. Every training reaction/protein is in the corresponding denominator. A cosine identity penalty discourages moving training endpoints away from F3. Unrecorded true activities remain possible false negatives; the objective does not establish experimental inactivity.

## Phase one: residuals plus a sparse training dictionary

The first learned endpoints were `h_e=N(x_e+delta_e(x_e))` and `h_r=N(x_r+delta_r(x_r))`. Their identity loss averages `1−cos(h,N(x))` over training examples. An average penalty does not bound the movement of each training point or any unseen input.

The semantic coordinates are indexed by training reactions `j=1,…,M`. Protein means are individually normalized, centered using training proteins, and normalized again. For enzyme `e`, retrieve 32 training proteins by cosine. Its reaction-coordinate weight is

```text
u_e[j] = maximum exp((cos(e,t) − nearest_cosine(e))/0.03)
         over retrieved training proteins t recorded as positive for reaction j,
         or 0 when no retrieved protein has that association.
a_e = N(u_e)
```

Reaction features concatenate four independently normalized and training-centered modalities: ReactionT5v2, UniMol2, ChIRo and set chemistry. Phase one retained only the 16 most similar training reactions:

```text
u_r[j] = exp((cos(r,j) − nearest_cosine(r))/0.03) for a retained reaction j,
         or 0 otherwise.
a_r = N(u_r)
```

The deployed endpoints concatenate `sqrt(0.75) h` and `sqrt(0.25) a`. Their dot product equals a 75% residual score plus 25% semantic score in exact arithmetic. Each representation uses only that endpoint and the fixed training dictionary.

Two structural limitations matter. First, disjoint semantic supports give exactly zero score. If every candidate has zero semantic score for a query, candidate-index tie order contains all ranking information. Second, large residual changes can occur outside the training-average identity constraint. These mechanisms were visible in training/validation diagnostics: 24.20% of unseen-reaction validation positive edges had zero phase-one anchor score, and unseen reaction endpoints moved more on average than seen ones. The [support](../../../runs/generalization_20260919_2251/phase2/smooth_anchors/phase1_hard_anchor_validation_support.json) and [geometry](../../../runs/generalization_20260919_2251/phase2/smooth_anchors/graph_geometry_diagnostic.json) artifacts quantify these observations. They motivate changes; they do not alone prove what caused any individual biological ranking.

## Phase two: support-dependent residuals and dense reaction responses

Phase two chooses a gate separately for each endpoint using its nearest training neighbor in **normalized F3 space**. For endpoint type `s`, let `c_s(x)` be that nearest cosine. Training leave-one-out nearest-neighbor similarities define thresholds `q25_s` and `q95_s`; protein threshold estimation uses the registered deterministic 4096-protein sample, and reaction estimation uses all training reactions.

```text
g_s(x) = clip((c_s(x) − q25_s)/(q95_s − q25_s), 0, 1)
h_s(x) = N(x + g_s(x) delta_s(x))
```

Equal thresholds use the implementation's explicit threshold indicator. The gate depends on the endpoint and fixed training references, so it preserves independent encoding. At low support it suppresses the learned correction. Nearest similarity is a support heuristic, not a calibrated probability of correctness; high-support inputs can still be wrong, and the residual can still be large where the gate is high.

The protein semantic map remains the 32-neighbor association transfer above. Reaction responses now include **every** training reaction:

```text
v_r[j] = exp((cos(r,j) − maximum_k cos(r,k))/0.03)
a_r = N(v_r)
```

The maximum subtraction is a rowwise numerical stabilization that disappears after normalization. A positive numerical floor protects representability. This removes the hard reaction-support cutoff: in exact arithmetic, any nonempty nonnegative enzyme anchor vector has a positive dot product with the reaction vector. Protein top-k membership and maximum association pooling remain piecewise operations; the entire method is not globally smooth. Small positive scores can also be uninformative, so elimination of zero rows is not evidence of biological accuracy.

The final composition again uses 75% density-gated learned endpoints and 25% raw semantic endpoints. It does not mix an already F3-blended semantic score a second time. In code, the actual concatenated endpoint vectors are scored rather than adding independently rounded score matrices.

## Numerical contract, storage and selection

The second-phase residual forward pass and final normalization use FP64, then return FP32 vectors. Neighbor cosine calculations use FP64 accumulation with FP32 stored responses. Retrieval scores are accumulated in FP64 and rounded once to FP32. Exact score ties use stable candidate order.

The unchanged parent-F3 control normalizes native vectors once in FP32. A separate `baseline_fp64` normalizes those same vectors once in FP64 and casts to FP32. A zero density gate matches the latter mathematical/numerical convention; it need not be bitwise equal to the original FP32-normalized baseline. Both controls must be reported so normalization-induced rank changes are visible.

The enzyme semantic map remains sparse, even though reaction responses are dense. A reusable enzyme index therefore stores a dense 512-dimensional learned block plus CSR semantic coordinates. Scoring uses the same FP64 block dot products and final FP32 output. [Verification records](../../../runs/generalization_20260919_2251/phase2/production_transfer_checks/reaction_smi.json) cover exact full/subset/singleton and sparse/dense equivalence; corresponding files cover the other two splits.

The second-phase selection objective equally weights seen- and unseen-reaction all-positive MRR in both directions, retaining four aggregate/unseen safeguards against the original F3 baseline. All-positive MRR averages reciprocal ranks over known positives within a query, then averages queries. It differs from first-positive MRR and from recall@k. Shared hyperparameters transfer across splits, while density thresholds fit each split's training data. Seeds 42/17/73 vary the residual head, not the pretrained/F3 backbone or the data.

Finally, unseen reaction IDs do not imply remote chemistry: 4 of 363 ID-unseen validation reactions have byte-identical raw features to training reactions, and most remaining queries still have close training neighbors. Training-label memory, homolog exposure, incomplete activity labels, lost reaction direction in participant-set inputs, and assay conditions all limit biological interpretation. The method preserves independent encoders and a reproducible retrieval score; prospective activity and utility require separate evidence.

## Phase four: hybrid reaction-neighbor geometry

This extension leaves the density-gated residual, its checkpoints, anchor
temperatures, protein-neighbor count and final composition weight unchanged.
Let `z_raw` be the phase-two normalized/centered raw endpoint representation and
`z_native=N(x)` the native F3 representation, without an additional fitted center.
For each endpoint type, construct

```text
z_eta = concat(sqrt(1−eta) z_raw, sqrt(eta) z_native)
cos_eta(i,j) = (1−eta) cos_raw(i,j) + eta cos_native(i,j)
```

The equality is mathematical; the implementation builds the concatenated vectors
and applies the frozen accumulation/rounding convention. The native view uses
FP64 normalization rounded to FP32. At `eta=0`, code delegates to the original
phase-two path so that the raw-only control exactly reproduces it.

The fixed nine-pair validation screen used `eta_e, eta_r ∈ {0,0.5,1}`. It selected
`eta_e=0, eta_r=0.5`. Therefore the enzyme anchor index is identical to phase two,
while the reaction's dictionary responses become

```text
w_r[j] = exp((cos_0.5(r,j) − maximum_k cos_0.5(r,k))/0.03)
a_r = N(w_r)
```

The existing exponential numerical floor is retained. The final score is still
the dot product of independently constructed weighted endpoints, with 75% learned
and 25% semantic weight. Each official split uses only its own training reaction
dictionary; shared eta values transfer unchanged. Protein pooling has not been
replaced, and no query-specific candidate information is introduced.

The implementation is [HybridPhase4Encoder](../horizyn/generalization_phase4.py),
with the [frozen recipe](../../../runs/generalization_20260919_2251/phase4/frozen_recipe.json).
Exact full/subset/singleton, chunked and sparse/dense checks passed for all nine
split/seed models. Despite its validation gain, this variant regressed on
Reaction-Sim E→R and did not improve the new measured aminotransferase panel over
F3. It remains a failed extension, not a replacement selected from test outcomes.

## Post-evaluation training-mean calibration

This follow-up was developed after the preceding official and external outcomes
were available. Its finite screen uses training/validation only, but repeated
external panels cannot provide new independent confirmation for it.

Let `e` and `r` denote the complete frozen phase-two endpoint vectors. Training
means use only endpoints occurring in actual training edges. The reaction mean
is uniform over training reactions; the enzyme mean is either uniform over
training proteins or weighted by the training reaction-balanced edge marginal.
The candidate score is

```text
s(e,r) = dot(e,r) − gamma_e dot(e,mean_r) − gamma_r dot(r,mean_e)
e_aug  = concat(e, 1, −gamma_e dot(e,mean_r))
r_aug  = concat(r, −gamma_r dot(r,mean_e), 1)
```

These independently computed augmented vectors preserve dot-product retrieval.
They must **not** be normalized after augmentation. Each mean is accumulated in
FP64 in fixed training-ID/chunk order. Bias coordinates use FP64 dot products
cast to FP32. The actual FP32 augmented endpoints are scored with the existing
FP64 accumulation and final FP32 rounding; subtracting biases from an already
rounded phase-two score is a different numerical operation. When both coefficients
are zero, the implementation delegates exactly to phase two.

The fixed 50-tuple screen uses `gamma_e,gamma_r ∈ {0,0.25,0.5,1,2}` and the two
enzyme-mean schemes. It maximizes the same four-cell balanced validation MRR,
with aggregate/unseen safeguards in both directions against **both** phase two
and precision-matched F3, allowing at most 0.005 decline in each guarded cell.
It selects the uniform-protein mean, `gamma_e=0`, `gamma_r=1`: balanced MRR
0.635186→0.639560. This subtracts a training-derived reaction-candidate affinity
in E→R. In R→E it adds only a query constant and cannot improve mathematical
rankings. Final FP32 rounding can perturb exact ties, so those numerical effects
are separately checked and cannot be interpreted as learned retrieval gains.

The validation-selected augmented vectors reproduce exact subset/singleton
endpoints and scores. No query-constant rank changes occurred on validation.
See [the finite screen and independent algebra review](../../../runs/generalization_20260919_2251/post_evaluation_calibration/README.md).
Transfer results belong to a separately recorded exploratory evaluation; they
do not change the identities or protocols of the earlier frozen models.
