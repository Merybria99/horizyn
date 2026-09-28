# Historical source archive

These programs, configurations, tests, and documents were removed from the
active implementation when the project was organized around V4, F3, and
CIRCEv2 on 2026-09-28. They retain their original contents and directory layout
for research traceability. They are not supported pipeline entry points.

The archive includes abandoned architecture variants, dictionary/semantic
scoring, experimental calibration methods, one-off campaign launchers, and
their obsolete tests and documentation. Some files have preexisting filesystem
read restrictions; moving them preserves those files and their permissions.

`migration.json` records original paths, archive paths, and readable-file
SHA256 hashes. The pre-cleanup readable-source snapshot and Git diffs are at
`../../../changes/v4_cleanup_20260928/` relative to this directory.

Datasets, checkpoint weights, saved run configurations, feature caches,
recorded evaluation outputs, and manuscripts retain their original paths.
Source snapshots inside old run directories are scientific artifacts and were
not reorganized.

Use [the maintained pipeline API](../../horizyn/pipelines/) and
[its guide](../../docs/PIPELINES.md) for current training and inference.

## Additional documentation and retired dependencies

`workspace_documents/` contains the old top-level method/proposal notes.
`documents/` contains September campaign reports, biology/dictionary studies,
ablation figures, and superseded LaTeX draft bundles. Their scientific content
is preserved; existing local Markdown links were relocated where needed.
The current manuscript remains in its separate original repository.

`third_party/` contains the retired HoroPCA, ChiENN, and EnzGFM repositories.
Weights and original benchmark outputs were retained outside the archive.
ChIRo, CARE/CREEP, and ReactZyme dependencies remain in their original locations.

The supplemental migration manifest, original-content snapshot, and link
repair records are in
[`changes/documentation_baseline_cleanup_20260928`](../../../changes/documentation_baseline_cleanup_20260928/).
See [the current baseline index](../../docs/BASELINES.md) for source and result
locations. Archived commands may require their historical layout and runtime.
