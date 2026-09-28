# Final validation-selected hybrid reaction geometry

This is an exploratory extension after phases1–3, including inspected official
benchmark, Case1, P450 and nitrilase outcomes. Those panels are development-exposed.
The aminotransferase panel was acquired by a separate curator; its assay table
was visible during source verification, but no model-selection agent accessed
its activity labels or model outcomes. All aminotransferase outcomes remain
sealed until this new model and its inference implementation are frozen.

## Question and fixed screen

The existing semantic branch uses raw ProtT5 means and raw multimodal chemistry.
Does the functional information learned by native F3 improve its within-endpoint
neighbor geometry? This differs from mixing an F3 score with an anchor score:
the added view changes training-neighbor evidence before association transfer.

The preregistered grid is exactly nine pairs:
`eta_enzyme × eta_reaction ∈ {0,0.5,1}²`. Native F3 vectors are normalized in
FP64 and stored in FP32, without a fitted center. Existing raw-view training
centers and dictionaries remain unchanged. Each endpoint's similarity uses an
actual concatenation of `sqrt(1−eta)*raw_unit` and `sqrt(eta)*native_unit`, followed
by the existing canonical dot-product convention. The zero-native-view path
delegates to the original implementation and must exactly reproduce phase2.

Protein neighbors remain32 with temperature0.03 and maximum association pooling.
Reaction responses include every training reaction at temperature0.03. The
final score composition remains75% frozen density-gated residual and25% semantic
evidence. No new temperature, neighbor count, composition weight, residual
checkpoint or external-panel threshold is selected.

The selector is the equal mean of seen/unseen-reaction all-positive MRR in both
directions, retaining the four0.005 safeguards against precision-matched F3.
All training/validation data, source hashes and numerical requirements were
pinned in `phase4/hybrid_anchors/protocol.json` before the screen. Its SHA256 is
`9cb8744f86dbb97c0aa34c53e0001d4bcd805f1cd9ec593d33da104c9c0e6c8b`.

## Selection and required production checks

The selected pair is **eta_enzyme=0, eta_reaction=0.5**, with balanced validation
MRR0.6395120844 and aggregate MRR0.8011356592. Phase2's corresponding values are
0.6351863295 and0.8000403643. All nine configurations pass the baseline guards.
The raw-only condition exactly reproduces phase2 endpoints and positive ranks.
This result supports the added learned reaction view; it does not support
replacing raw protein means with native F3 protein geometry in this screen.

Before external inference, require exact checkpoint reload, full/subset/singleton
endpoint equivalence and sparse/dense score equivalence. Fit each split's native
training dictionary from that split's training cache only; transfer the selected
eta values and other hyperparameters unchanged. Seeds42/17/73 vary only the
already trained residual heads. They share the frozen F3 backbone and selected
geometry rule. Pin all nine split/seed tuples, feature identities and source
hashes in a new immutable freeze; do not edit the phase2 freeze or source files.

## Evaluation commitment

Primary: `phase4_seed42`. Replications: seeds17/73. The new semantic-only
`hybrid_anchor_only` control uses the selected geometry with composition weight1.
Retain the previous frozen phase2 primary, density-only, F3_native and F3_fp64
as comparators. Compare official ReactZyme all-positive MRR, Hit@1 and Hit@10
on the unchanged candidate pools. Report paired query intervals and E→R
reaction-group sensitivity rather than treating dependent enzyme queries as
independent evidence.

Case1 retains paper-only, paper-plus-patent and conditional workbook evidence
tiers. P450 retains its reaction-permutation conditioning check and the original
overlap strata. Nitrilase retains its complete measured-panel denominators and
conditional non-detect policy. No favorable control or replication replaces
the new primary after evaluation.

The new aminotransferase assay is the independent outcome check for the frozen
models, with curation exposure and training overlap explicitly disclosed. Its
full25×18 panel is primary; the sequence-complete subset and preregistered
exposure rectangles are diagnostic. All candidate-pool changes in these
rectangles must be named and applied identically to all compared methods.
No threshold, variant or new model will be selected from its outcomes.
