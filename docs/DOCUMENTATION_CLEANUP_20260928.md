# Documentation and retired-baseline cleanup — 2026-09-28

The maintained documentation now directs readers to dictionary-free V4, F3,
and CIRCEv2. Earlier proposals, dictionary/biology studies, historical comparison
notes, and superseded paper drafts were moved to the dated archive. Scientific
results were preserved rather than deleted.

## Changes

- Archived 13 old top-level method/proposal notes under
  `archive/20260928/workspace_documents/`; the data-preparation guide and
  literature PDFs remain in the workspace's `documents/`.
- Archived the old implementation `documents/` contents together under
  `archive/20260928/documents/`, preserving campaign reports, ablation plots,
  evidence tables, and early LaTeX/PDF draft bundles.
- Moved the unused HoroPCA, ChiENN, and historical EnzGFM code repositories from
  `.deps/` to `archive/20260928/third_party/`. Their licenses and repository
  metadata remain intact. Snapshot tooling now follows all three relocated
  repositories.
- Added concise documentation indexes and [a baseline source/evidence
  registry](BASELINES.md). The findings entry page now points to the current
  guide; the full dated findings log explicitly identifies itself as historical.
- Repaired 123 existing local Markdown links after relocation. Scientific
  prose, recorded scores, tables, and archived code were not rewritten.
- Corrected the completed CREEP EnzymeMap run's stale launch-time status and
  linked its selection and test receipts. Its original README is in the
  pre-edit snapshot.

The migration contains **754 files, including 70 Markdown files**, across
documentation bundles and the three vendor repositories. Every file matched
its original hash immediately after moving. Only Markdown link relocation
subsequently changed archived documentation bytes; both original and final
hashes are recorded. Vendor code and README contents were preserved exactly.

## Retained material

CARE/CREEP and ReactZyme comparator code remains available, along with all
recorded baseline weights, feature banks, evaluation tables, and scientific
reports. EnzymeCAGE, VenusRXN, Ligns, root third-party repositories, and
pretrained model collections were retained. ChIRo remains at its original
path because the supported pipelines use it for reaction features; it is
distinct from the retired ChiENN integration.

The current manuscript repository was not changed. Markdown reports inside
original run/data/case-study directories remain beside their evidence. Only
the explicitly identified stale CREEP README status was updated. Existing
presentation material, data specifications, source snapshots, and licenses
were retained.

No new training was launched. Archived programs and draft recipes are
research records, not maintained entry points. Repeating old commands may
require restoring their original layout and runtime. The three pipeline
commands are documented in [PIPELINES.md](PIPELINES.md).

## Verification and audit trail

The maintained import check passes, and the CLI lists V4, F3, and CIRCEv2.
Content verification covers all 754 migrated files, including their repository
metadata. Markdown link checks cover the new entry pages and indexes;
historically broken links are not treated as current documentation contracts.
The final maintained suite passed **873 tests** in 38.58 seconds, with 39
existing dependency/Lightning warnings. Its result is recorded in
`test_result.json` in the evidence directory.

[`changes/documentation_baseline_cleanup_20260928/`](../../changes/documentation_baseline_cleanup_20260928/)
contains the original-content `before.tar.gz`, exact `migration.json`,
`summary.json`, content/link `verification.json`, and the test result.

The cleaned code, documentation, readable historical source, and selected audit
metadata were subsequently published on `ICLR2027`. Retired vendor repositories
remain local; publication records their upstream URLs/revisions in
`archive/20260928/third_party/repositories.json`. The 13 preexisting restricted
archive files remain local and were not added to Git. All maintained source
files are included in the published branch.
