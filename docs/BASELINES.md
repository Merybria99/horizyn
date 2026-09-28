# Baselines and external dependencies

Only **V4, F3, and CIRCEv2** are maintained pipeline implementations. External
comparators remain available for reproducing scientific comparisons. They do
not extend the selectable pipeline families.

## Comparator evidence retained

| Comparator | Source / integration | Recorded evidence |
|---|---|---|
| Released ReactZyme families | `.deps/ReactZyme`; historical training adapters under `archive/20260928/scripts/reactzyme_public_*.py` | [Results](../runs/reactzyme_public_baselines_20260921/results.md), [comparison](../runs/reactzyme_public_baselines_20260921/comparison.md) |
| CREEP, released with CARE | `.deps/CARE`; archived `reactzyme_public_creep.py` and `enzymemap_public_creep.py` | ReactZyme campaign above; [completed EnzymeMap test](../runs/enzymemap_public_creep_20260924_seed42/complete.json), [selection](../runs/enzymemap_public_creep_20260924_seed42/selection.json) |
| CLIPZyme | Original campaign feature/model assets and preserved screening/embedding scripts | [Embedding comparison](../runs/clipzyme_embedding_comparison_20260924/), screening provenance in the current manuscript and original run receipts |
| EnzymeCAGE | Workspace `EnzymeCAGE/` and retained preparation assets | Historical preparation/comparison records; public source availability alone does not establish a completed matched-data result |
| Horizyn refits | Historical `reactzyme_public_horizyn.py`, `enzymemap_public_horizyn.py`, and source snapshots | Retained campaign results; the current manuscript omits this comparator from quantitative tables because its adaptation/optimization protocol differs |
| TIGER and FGW-CLIP | Published reference rows and original literature audit | [Release audit](../archive/20260928/documents/reactzyme_released_competitors_20260921.md); no local reproduction is implied |

The manuscript's included comparisons and protocol disclosures take priority
over the broader campaign's planned comparator list. Early campaign documents
may say "running" or "pending"; consult completed receipts and result files
for actual outcomes. In particular, the CREEP EnzymeMap test completed on
2026-09-24, with the validation-selected epoch-39 checkpoint.

## Retired integrations

The unused **HoroPCA**, **ChiENN**, and historical **EnzGFM** source trees now
live in `archive/20260928/third_party/`. Their contents, licenses, and repository
metadata were preserved. They have no imports or configured paths in the
maintained pipelines and case-study runners. EnzGFM was used for an earlier
backbone-control campaign; its weights, extracted features, and results remain
at their original paths. It is not an active V4 component. ChiENN is distinct
from ChIRo.

Older baseline launchers are already under `archive/20260928/scripts/`.
Historical commands may require restoring their original directory layout
and environment. Archived programs are not supported pipeline commands.

Root third-party repositories and pretrained weight collections were retained,
including VenusRXN and EnzymeCAGE. They contain scientific reference code or
data and were not deleted merely because they are outside the main pipeline.

## Dependencies still required

V4 feature extraction still uses **ProtT5, SLEEC, ReactionT5v2, Uni-Mol2, and
ChIRo**. `.deps/ChIRo`, its pretrained checkpoint, Uni-Mol utilities, model
caches, and the extraction environments remain in place. These components
must not be confused with retired alternative architectures or baselines.

The historical Ligns repository remains available for the SLEEC/MSA provenance
described in the manuscript appendix. None of its assets were removed.

See [the documentation/baseline cleanup report](DOCUMENTATION_CLEANUP_20260928.md)
for exact moved paths, content hashes, link repairs, and validation.
