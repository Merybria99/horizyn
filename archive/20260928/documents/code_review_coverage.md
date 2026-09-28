# Workspace code review coverage

The review covered the readable active source in the workspace and the nested
Horizyn, EnzymeCAGE, VenusRXN and vendored ReactZyme implementations. Architecture,
data flow, training objectives, evaluation, feature extraction and wet-lab query
paths were examined across four parallel reviews. Critical F3/CIRCE paths were
read in depth and exercised with focused checks; remaining vendor source was
opened and structurally inventoried. This is not a claim that every vendored
line received an equally deep semantic audit.

The durable file-level inventories and detailed findings are under
[`runs/generalization_20260919_2251/code_review`](../../../runs/generalization_20260919_2251/code_review).
The initial source snapshot and its SHA manifest preserve the working-tree
baseline, including substantial changes that existed before this campaign.

OS permissions blocked `case_studies/`, `eda/`, top-level `ReactZyme/`, and some
data/vendor paths. No permissions were changed. Installed environments, package
caches, binary checkpoints, generated outputs and historical source archives
were excluded from active-source review. The readable vendored ReactZyme
retrieval implementation supplied the benchmark protocol audit.

Changes made in this campaign address the retrieval investigation and scoped
correctness issues. Other findings, including SABIO retry-cache handling,
accession whitespace parsing and heterogeneous activity-threshold semantics,
remain documented rather than being changed incidentally.

Both the workspace index repository and the nested `horizyn` repository are on
`research/f3-circev2-generalization-20260919`. Pre-existing user changes remain
in place; a branch switch alone does not turn this dirty workspace into a clean
baseline. The saved initial snapshot is the reference for isolating task edits.
