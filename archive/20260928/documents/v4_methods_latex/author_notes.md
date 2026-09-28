# Notes for editing and submission

The methods files specify what was implemented. They deliberately do not turn the biological loss into a claim of demonstrated biological interpretability or generalization.

## Important implementation distinctions

1. **Phase 1 includes two attention penalties.** Both entropy and diversity coefficients are 0.01 in the frozen V4 configurations. Earlier prose that described phase 1 only as retrieval alignment was incomplete. They act on raw learned-slot attention, before its uniform mixture.
2. **The protein gate is feature-wise.** It has six weights for each of 256 hidden coordinates, not one scalar per view. Its uniform mixture gives a minimum weight of 0.05/6.
3. **The current added biological loss cannot train the residue queries.** F3 is frozen in phase 2. Claims about biologically specializing residue slots would need evidence from a different variant with F3 updates.
4. **The two retrieval losses differ.** Original MLNCE is not the phase-1 objective. Phase 2 is uniform-positive cross-entropy with all positives in its denominator, not the phase-1 decoupled loss.
5. **The relative biological loss has two kinds of confidence normalization.** Weighted distance means divide by confidence mass; the outside factor divides that same within-category mass by pair count, preserving reduced evidence strength. Do not omit that factor.
6. **The dictionary's raw molecular features use the forward reactant bank.** On ReactZyme this bank is the complete self-reaction participant set; on EnzymeMap it is the physical reactant side. It is not the trainable directional-delta feature. The dictionary protein mean can use more cached residues than the truncated F3 input.
7. **Inference differs from the refinement fit.** Phase 2 was fitted before tripling the fused protein branch; at inference, it receives that adjusted protein embedding and uses half its training residual coefficient. The EnzymeMap checkpoint already stores the factor-three change.
8. **“From scratch” has a limited meaning.** Target-specific retrieval layers are freshly initialized; ProtT5, reaction/molecular feature models, and SLEEC are pretrained resources.
9. **V4 is a recipe with separate target fits.** It is not one checkpoint trained jointly on ReactZyme and EnzymeMap. The primary Case1 deployment uses the Reaction-Sim-trained model.

## Claims supported by the completed experiments

The original V4 and phase-2 biological weight-1 variant both exceed the 14 recorded primary benchmark point comparisons. This does not establish that added biological supervision caused those wins. In the phase-2-only experiment, EnzymeMap BEDROC85 slightly decreases versus V4 and is nearly identical under shuffled labels. The Case1 unique-sequence improvement is one literature catalyst moving from rank 26 to 25; paper-supported recovery in the original 144-entry panel remains 10/15. Cofactor removal retains that crossing and slightly improves conditional AUROC.

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
