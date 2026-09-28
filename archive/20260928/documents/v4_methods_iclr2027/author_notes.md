# Notes for editing and submission

The methods files specify what was implemented. They deliberately do not turn the biological loss into a claim of demonstrated biological interpretability or generalization.

V4 without added biological supervision is the sole proposed method. Its phase-2 objective is full-graph uniform-positive cross-entropy plus ten times the identity penalty. EC, cofactor-category, and bond-change losses appear only in the V4+Bio ablation. Their effects are mixed and do not establish a robust benefit. Do not relabel V4+Bio metrics as V4 results: primary results use the saved no-biology checkpoints, including Case1 recovery of 10/12 unique literature catalysts. Later matched-data Horizyn retraining is stronger on ReactZyme; describe empirical advantages only where supported.

## Author's writing style

Use the supplied problem-formulation passage as the prose model: define the objects and notation first, introduce the objective in connected explanatory paragraphs, and link architecture and training with explicit transitions and section references. Describe what each component computes and why it enters the formulation. Keep the main section within two to two-and-a-half pages; place detailed specifications and derivations in the appendix. The revised main text uses `\method{}` and separate problem-formulation, enzyme-encoder, reaction-encoder, alignment, refinement, and scoring subsections.

The final dot-product notation must refer to the complete neural-plus-dictionary embeddings. Distinguish the full association graph, training graph, and query-dependent candidate pools. Use `f_R,f_E` for encoder functions to avoid overloading the fused protein vector `f_e`. The two residual heads are modality-specific, not retrieval-direction-specific heads.

The expanded enzyme subsection follows the author's functional-residue motivation and paragraph structure. The supplied additive-logit expression `w^T h_i + rho u_i` does not describe the evaluated multiview model: SLEEC defines a separate frozen, centered/clipped and uniformly smoothed view. There is no learned SLEEC mixing coefficient in that attention. Its normalized attention weights should not be described as calibrated probabilities of catalytic function. The four other views learn from retrieval associations in phase 1; the separate biological ablation refines the heads in phase 2 while those views remain fixed.

## Important implementation distinctions

1. **Phase 1 includes two attention penalties.** Both entropy and diversity coefficients are 0.01 in the frozen V4 configurations. Earlier prose that described phase 1 only as retrieval alignment was incomplete. They act on raw learned-slot attention, before its uniform mixture.
2. **The protein gate is feature-wise.** It has K+2 weights for each of 256 hidden coordinates, not one scalar per view. Its uniform mixture gives a minimum weight of 0.05/(K+2), or 0.05/6 at the default K=4.
3. **In the biological ablation, the added loss cannot train the residue queries.** F3 is frozen in phase 2. Claims about biologically specializing residue slots would need evidence from a different variant with F3 updates.
4. **The two retrieval losses differ.** Original MLNCE is not the phase-1 objective. Phase 2 is uniform-positive cross-entropy with all positives in its denominator, not the phase-1 decoupled loss.
5. **The ablation-only relative biological loss has two kinds of confidence normalization.** Weighted distance means divide by confidence mass; the outside factor divides that same within-category mass by pair count, preserving reduced evidence strength. Do not omit that factor.
6. **The dictionary's raw molecular features use the forward reactant bank.** On ReactZyme this bank is the complete self-reaction participant set; on EnzymeMap it is the physical reactant side. It is not the trainable directional-delta feature. The dictionary protein mean can use more cached residues than the truncated F3 input.
7. **Inference differs from the refinement fit.** Phase 2 was fitted before tripling the fused protein branch; at inference, it receives that adjusted protein embedding and uses half its training residual coefficient. The EnzymeMap checkpoint already stores the factor-three change.
8. **“From scratch” has a limited meaning.** Target-specific retrieval layers are freshly initialized; ProtT5, reaction/molecular feature models, and SLEEC are pretrained resources.
9. **V4 is a recipe with separate target fits.** It is not one checkpoint trained jointly on ReactZyme and EnzymeMap. The primary Case1 deployment uses the Reaction-Sim-trained model.
10. **The chemistry descriptor is fixed, but its projection is learned.** It concatenates 512 averaged Morgan-fingerprint coordinates, 64 aggregated molecular properties, four participant-set statistics, and 37 cofactor-indicator positions. Only the 64 property coordinates are standardized, using training reactions; the cofactor vocabulary is also fitted on training reactions. Descriptor construction does not use enzyme--reaction association labels. The learned chemistry MLP and modality attention determine its contribution to retrieval. This participant summary does not itself encode which bonds change or which side a molecule occupies.

11. **Molecule attention and modality attention perform different operations.** Only Uni-Mol2 and ChIRo pool molecule embeddings separately by side; their two compositions, ReactionT5v2, and the chemistry descriptor form the four projected sources. ReactZyme copies its participant set to both branches. Independent pooling parameters can give unequal outputs, so their contrast does not establish a physical reaction direction.

12. **K=4 is retained as a compromise, not a proven optimum.** The completed K=1/2/4/8 ablation reports all 16 fresh seed-42 fits of original V4. K=2 leads EnzymeMap BEDROC85 in both settings; K=4 leads EF5 and EF10, with mixed ReactZyme trade-offs. The K=4 control is freshly retrained and must remain separate from historical primary results. Changing K changes projection and gate parameter counts as well as the number of summaries. There is one seed, no parameter-matched control, and no K=0 arm; do not infer statistical significance, a causal benefit of adding any learned views, or the same sensitivity under V4+Bio. Tables use all-positive ReactZyme MRR, BEDROC percentages, and fold enrichment.

## Claims supported by the completed experiments

The original V4 and phase-2 biological weight-1 variant both exceed the 14 recorded literature benchmark point comparisons, subject to the ReactZyme metric-definition distinction and input-resource differences in the experimental section. This does not include the stronger locally retrained Horizyn comparator. This does not establish that added biological supervision caused those wins. In the phase-2-only experiment, EnzymeMap BEDROC85 slightly decreases versus V4 and is nearly identical under shuffled labels. The Case1 unique-sequence improvement is one literature catalyst moving from rank 26 to 25; paper-supported recovery in the original 144-entry panel remains 10/15. Cofactor removal retains that crossing and slightly improves conditional AUROC.

Fresh-F3 relative biology gives a stronger but mixed EnzymeMap signal across three seeds, with a positive average BEDROC85 change and a negative average EF5 change. Those are separate experiments, not the phase-2-only `all_1` result. Do not merge the favorable parts of different configurations into one method result.

The losses, gates, or attention maps alone do not establish that an individual dimension or residue query represents a named biochemical property. Cohort recovery is not calibrated activity prediction. Case1 is a retrospective literature panel with related constructs and heterogeneous assay evidence, not a new prospective wet-lab study.

## Experimental selection and resource disclosure

The declared final recipe uses epoch 10 and phase-2 step 100, but it was reached through repeated exploratory benchmark inspection. Do not describe the overall search as a pristine, single-use test protocol. Keeping a fixed final recipe does not erase that development history.

The association graph is target-specific. The biological variant also uses extra annotations on training endpoints, including reaction-only Rhea resources for ReactZyme; native protein EC annotations may describe activities beyond the retained graph. Matching downstream associations does not imply identical total information across competitors.

The generic phase-2 registry field saying `external_screening` describes delegation of validation; for this recipe the retained checkpoint is the declared fixed step 100. Do not infer validation-based checkpoint selection from that generic string. EnzymeMap screening uses BEDROC85, BEDROC20, EF5, and EF10, not MRR. ReactZyme all-positive MRR and its near-tie sensitivity belong in the experimental protocol.

## Supporting reports

- [Original V4 explanation](../v4_extensive_explanation.md)
- [Biological architecture and interpretation](../v4_biological_signal_architecture.md)
- [Weight-1 matched controls](../v4_relative_biology_phase2_weight1_controls_20260921.md)
- [Case1 biological controls](../v4_relative_biology_phase2_case1_controls_20260921.md)
- [Additional seed and shuffled-label controls](../v4_biology_additional_controls_20260921.md)
- [Final experiment audit](../v4_biology_final_audit_20260921.md)

The seed-sensitivity report has a historical footer stating that all shuffled/removal studies were seed-42-only. Later controls in the additional-controls report supersede that statement: relative-weight-1 shuffled labels were evaluated at seeds 42/43/44, and mechanism removal additionally at seed 43. The methods draft does not repeat the outdated footer.

## Notation convention

Use bold symbols for vectors and matrices, and plain indexed symbols for scalar coordinates and attention weights. Reserve `h_i` for the 1024-dimensional residue feature; raw protein views are `x_e^P` (global ProtT5 mean), `x_e^S` (SLEEC-weighted mean), and `x_e^k` (query k). All projected views `y_e^v` have 256 dimensions. Keep the enzyme index on fused vectors and gate weights; its suppression on residue vectors and residue attention is stated explicitly. Use the same numerical normalization for queries and keys.

Keep phase-1 base vectors `b`, phase-2 training vectors `u^tr`, inference neural vectors `u`, dictionary vectors `a`, and final retrieval vectors `z` distinct. The biological ablation uses renormalized `u^tr`, not final `z`; it is not part of the proposed training objective. Phase-specific scalar logits are `s^(1)` and `s^(2)`. The training adjacency is `Y^tr`; modality sets are `M_r`, participant multisets are `P_r`, and molecular property vectors are `d(m)`.

Use the author's feature-source labels consistently: the fusion label set is `I={P,S,1,...,K}`, and gate rows follow that order. Introduce fusion with “We combine the extracted enzyme features ... by projecting”; retain formal mathematical prose rather than explanatory instructions to the reader.
