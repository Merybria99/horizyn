# Exploratory second phase

Started after the first frozen evaluation on 2026-09-20. The first recipe and
all of its results remain immutable. It did **not** meet the joint objective:
Case 1 primary recovery fell to zero, and Reaction-Sim E→R remained below the
published TIGER value. Any subsequent evaluation on these same panels is an
exploratory follow-up, not a new untouched-test claim.

Three mechanisms are being investigated using training and standard validation
features/labels only:

1. Smooth semantic transfer: remove the exact-zero support intersection created
   by independently truncated protein and reaction neighbors.
2. Globally constrained linear adaptation: contrastively fit raw-feature CCA
   representations while bounding the operator norm of their displacement.
   A bound on every input is stronger than a mean penalty on training inputs.
3. Density-aware adaptation: quantify and limit residual displacement outside
   regions supported by training examples, with independently computed endpoint
   gates so retrieval remains a dual encoder.

Before any second-phase results, the shared selection objective is fixed as the
equal-weight mean of **seen-reaction and unseen-reaction** all-positive MRR in
both directions (four cells). This prevents recurrent reaction labels from
dominating selection. The existing four aggregate/unseen safeguards remain:
no direction/stratum may fall more than 0.005 below F3. Aggregate-optimized
choices are separately named diagnostics. Distance-stratified
results must be reported beside aggregate results. All configurations and failed
trials are retained. No Case 1 or P450 score is used in these fitting scripts.
An additional literature panel, if suitable data exist, will be chosen by source
quality and feature availability before model scores are inspected.

These experiments cannot erase exposure to the first evaluation. They may
provide a better model and evidence about failure mechanisms, but any claim of
generalization must identify which evidence remained untouched.
