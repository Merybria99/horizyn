#!/usr/bin/env python3
"""Consolidate completed embedding diagnostics into the original README.

This combines reviewed Markdown and existing figures. It does not compute
metrics, select models, launch training or overwrite the companion reports.
The base-comparison source uses paths relative to the output root, matching
the original report generator's conventions.
"""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/public_embedding_comparison_20260924'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(text):
    matches = list(re.finditer(r'^## (.+)$', text, flags=re.M))
    return {m.group(1): text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def rebase(text, origin):
    def replace(match):
        target = match.group(1)
        if target.startswith(('http:', 'https:', '#')):
            return match.group(0)
        path, sep, fragment = target.partition('#')
        rebased = Path(os.path.relpath((origin / path).resolve(), OUT.resolve())).as_posix()
        return '](' + rebased + (sep + fragment if sep else '') + ')'
    return re.sub(r'\]\(([^)]+)\)', replace, text)


def lower_headings(text):
    return re.sub(r'^(#{2,5}) ', r'\1# ', text, flags=re.M)


def build():
    paths = dict(base=OUT / 'report_sources/base_comparison.md',
                 protocol=OUT / 'neighborhood_curves/protocol_explanation.md',
                 curves=OUT / 'neighborhood_curves/README.md',
                 levels=OUT / 'ec_levels/README.md')
    docs = {k: p.read_text() for k, p in paths.items()}
    parts = {k: sections(text) for k, text in docs.items()}
    base, protocol, curves, levels = (parts[k] for k in ['base', 'protocol', 'curves', 'levels'])

    def include(kind, titles):
        origin = OUT if kind == 'base' else paths[kind].parent
        return '\n\n'.join('### ' + title + '\n\n' + lower_headings(rebase(parts[kind][title], origin)) for title in titles)

    text = [
        '# Public checkpoint embedding analysis — consolidated report',
        'Updated 24 September 2026. This is the original report, now containing all completed embedding analyses from this campaign, their nine figures, full result tables and protocol explanations. The companion documents remain reproducible sources; the completed results can be read here without switching between reports. This report does not claim that every planned training or screening experiment is finished.',
        '**Main finding:** the current dictionary-free CERSEI has stronger enzyme functional neighborhoods in several comparisons, most consistently at broad and intermediate EC resolution. The advantage weakens substantially at EC4, reaction-space comparisons are mixed, and favorable local geometry does not establish better exact-pair retrieval or wet-lab activity. The near-ties and counterexamples are retained throughout.',
        '## Experiment inventory and completion status',
        '| Experiment | Scope and models | Status and location in this report |\n'
        '| --- | --- | --- |\n'
        '| Enzyme neighborhoods at EC1–EC4 | Three ReactZyme splits: CERSEI, Horizyn, CREEP, ProtT5; EnzymeMap: CERSEI, Horizyn, CLIPZyme, ProtT5 | Complete: k=1–50, coverage and per-class analysis; Figures 7–8 |\n'
        '| Common complete-annotation cohort | Same EC4-annotated proteins, candidate banks and neighbor rankings across EC1–EC4 | Complete: macro and query-mean metrics; Figure 9 |\n'
        '| Original enzyme EC3 comparison | Identical bank/query protocol with both historical uncertainty conventions retained | Complete: Figure 1 and paired differences |\n'
        '| Enzyme and reaction neighborhood curves | EC3 protein labels; ReactZyme association-derived EC3 and EnzymeMap native-rule reaction labels | Complete: Figures 5–6 |\n'
        '| Cross-modal retrieval diagnostics | Both directions, full candidate pools, positive coverage and complete ReactZyme MRR context | Complete for available fixed exports: Figure 2 and tables |\n'
        '| Chemistry-matched reaction analysis | Same-rule versus different-rule comparisons with matched participant similarity and native mapping-quality filtering | Complete: 427 EnzymeMap queries and 55,704 matched comparisons |\n'
        '| CLEAN transfer control | Exact CLEAN training-sequence overlap removed; EC1–EC4 cosine plus EC3 native-L2 sensitivity | Complete on its separate matched transfer banks: Figure 3 and tables |\n'
        '| EnzymeCAGE coverage and native P450 panel | Common-pool coverage audit; independent pocket branch and native pair-score diagnostics | Scoped analysis complete: Figure 4. Full common-pool competitor comparison is unavailable |\n'
        '| CREEP EnzymeMap refit and final test geometry | Official EnzymeMap training split and full-library screening | Pending at this update: epoch 30 validation; no final selection, test summary or completed embedding export |',
        'The status above is a report snapshot, not a live monitor. CREEP validation values are not substituted for final test results. [CREEP run protocol and progress files](../enzymemap_public_creep_20260924_seed42/README.md) remain the source for that separate ongoing experiment. Case 1 and the broader training campaign are documented in [project findings](../../findings.md); they are related application/benchmark experiments, not additional measurements performed by this embedding analysis.',
        '## Scientific questions and full neighborhood protocol',
        include('protocol', ['Purpose: what are we looking for?', 'Protocol: from a fixed checkpoint to a plotted point',
                             'How this relates to the other experiments', 'What evidence would strengthen the interpretation?']),
        '## Scope and checkpoint provenance',
        base['Scope and checkpoint provenance'],
        '## Enzyme functional organization at EC1–EC4',
        'These are the complete hierarchy results. The primary curves change annotation eligibility with the EC level; the separate complete-annotation control fixes the proteins and neighbor lists. The latter explicitly reports a different averaging scheme when showing query-mean agreement.',
        include('levels', ['What the four levels ask', 'Main result at k=10', 'Complete neighborhood-size curves',
                          'Protocol and annotation coverage', 'Control: the same completely annotated proteins at all four levels',
                          'Interpretation by level']),
        '## Original EC3 comparison and uncertainty sensitivity',
        base['Matched-data functional neighborhoods'],
        'These EC3 results are retained to connect the hierarchy extension to the initial analysis. They do not imply an equal advantage at EC4. The complete-annotation and per-level coverage controls above make that distinction explicit.',
        '## Enzyme and reaction neighborhood-size curves',
        include('curves', ['EnzymeMap', 'ReactZyme', 'Exact values and persistence across k']),
        '## Retrieval and alignment',
        base['Alignment and retrieval'],
        '## CLEAN transfer analysis',
        base['CLEAN: protein-only transfer control'],
        include('levels', ['CLEAN transfer control at all levels']),
        '## EnzymeCAGE coverage and native panel',
        base['EnzymeCAGE: available coverage and native panel'],
        '## Uncertainty, fairness and limits of interpretation',
        base['Uncertainty, fairness, and checks'],
        '**Additional limitation identified in the curve extension:** the fixed-class, family-weighted bootstrap degenerates for EnzymeMap native-rule agreement because members of each rule share a resampling component. Its zero-width intervals must not be read as zero uncertainty. Figures 5–9 therefore show descriptive points/curves without confidence bands. The original numerical summaries and both uncertainty conventions remain preserved for audit. None of these comparisons provides training-seed uncertainty.',
        'The conclusion is conditional on each annotation definition, eligible bank, query sample and averaging scheme. Shared EC labels need not imply the same substrate specificity. Test-evaluation membership does not establish that a sequence or reaction was unseen in training. Candidate-to-query homology/chemistry filters do not independently decontaminate training. The full-EC near-ties, the Time EC4 loss to Horizyn, the reaction-space CREEP counterexamples and the complete ReactZyme MRR results prevent a universal superiority claim.',
        '## Reproducibility, artifacts and checks',
        'The initial public-checkpoint analysis was developed on `research/embedding-public-checkpoints-20260924`; later curve and EC-level extensions were added during the `research/creep-enzymemap-20260924` campaign. Existing fitted checkpoints and manuscript files were left unchanged by these analyses. Computation for the extensions ran on CPU while the separate CREEP training used the GPUs.',
        '| Analysis | Checks and source artifacts |\n'
        '| --- | --- |\n'
        '| Initial public-checkpoint audit | [Protocol](protocol.json), [validation](validation.json), [executed notebook](analysis.ipynb), per-split receipts and result files |\n'
        '| Complete neighborhood curves | [64 k=10/50 checks](neighborhood_curves/validation.json), [curve values](neighborhood_curves/curves.csv), [per-class values](neighborhood_curves/per_class.csv), [source hashes](neighborhood_curves/sources.json), [export QA](neighborhood_curves/export_qa.json) |\n'
        '| EC1–EC4 extension | [128 k=10/50 checks](ec_levels/validation.json), [6,400 curve rows](ec_levels/curves.csv), [coverage](ec_levels/scope.json), [per-class results](ec_levels/per_class.csv), [source hashes](ec_levels/sources.json), [export QA](ec_levels/export_qa.json) |\n'
        '| Report consolidation | [Source and link validation](consolidated_report_receipt.json); this step changes documentation only |',
        'The EC3 per-query curves reproduce the earlier artifacts exactly. The complete-annotation control verifies identical neighbor lists across EC levels and pointwise hierarchy nesting. The maximum discrepancy from saved point summaries is 3.33e−16 for the 64 curve checks and 5.55e−16 for the 128 EC-level checks. These are reproducibility checks, not evidence of biological superiority.',
        '| Figure | Vector export |\n'
        '| --- | --- |\n'
        '| 1: EC3 functional neighborhoods | [PDF](01_matched_functional_neighborhoods.pdf) |\n'
        '| 2: Cross-modal positive coverage | [PDF](02_matched_alignment.pdf) |\n'
        '| 3: CLEAN transfer control | [PDF](03_clean_transfer_control.pdf) |\n'
        '| 4: EnzymeCAGE native diagnostic | [PDF](04_enzymecage_native_panel.pdf) |\n'
        '| 5: EnzymeMap neighborhood curves | [PDF](neighborhood_curves/05_enzymemap_neighborhood_curves.pdf) |\n'
        '| 6: ReactZyme neighborhood curves | [PDF](neighborhood_curves/06_reactzyme_neighborhood_curves.pdf) |\n'
        '| 7: EC1–EC4 overview | [PDF](ec_levels/07_enzyme_ec1_ec4_agreement.pdf) |\n'
        '| 8: Complete EC-level curves | [PDF](ec_levels/08_enzyme_ec_neighborhood_curves.pdf) |\n'
        '| 9: Common-annotation cohort | [PDF](ec_levels/09_enzyme_ec_complete_cohort.pdf) |',
        'All nine figures also have SVG and PNG exports beside their PDFs. The source reports are [neighborhood curves](neighborhood_curves/README.md) and [all EC levels](ec_levels/README.md). Their relative links are rebased when incorporated here. Rebuild this original report with `python3 horizyn/scripts/consolidate_public_embedding_report.py` from the workspace root. Consolidation reads saved Markdown and does not run experiments.',
    ]
    functional = ROOT / 'runs/cersei_crossmodal_functional_recovery_20260925/README.md'
    figure_count = 9
    if functional.exists():
        paths['functional'] = functional
        extra = functional.read_text().split('\n', 1)[1].strip()
        extra = extra.replace('**Figure 1.**', '**Figure 10.**').replace('**Figure 2.**', '**Figure 11.**')
        text.extend(['## Cross-modal functional recovery beyond recorded partners',
                     lower_headings(rebase(extra, functional.parent))])
        figure_count += 2
    result = '\n\n'.join(text) + '\n'
    if functional.exists():
        result = result.replace('Updated 24 September 2026.', 'Updated 25 September 2026.')
        result = result.replace('their nine figures', 'their eleven figures')
        result = result.replace(
            '| CREEP EnzymeMap refit and final test geometry | Official EnzymeMap training split and full-library screening | Pending at this update: epoch 30 validation; no final selection, test summary or completed embedding export |',
            '| CREEP EnzymeMap refit and final test | Official EnzymeMap training split and full-library screening | Complete: frozen selection and final test scores are used in the new cross-modal diagnostic; older same-modality figures were not rerun |\n'
            '| Functional recovery beyond recorded partners | All eligible test reactions; CERSEI, Horizyn, CREEP and EnzymeMap CLIPZyme; two exclusion policies and five CERSEI stages | Complete: Figures 10–11, all k=1–50, random/ceiling controls and paired intervals |')
        result = result.replace(
            'The status above is a report snapshot, not a live monitor. CREEP validation values are not substituted for final test results. [CREEP run protocol and progress files](../enzymemap_public_creep_20260924_seed42/README.md) remain the source for that separate ongoing experiment.',
            'The status above is a report snapshot, not a live monitor. CREEP EnzymeMap has a [completion receipt](../enzymemap_public_creep_20260924_seed42/complete.json), [frozen selection](../enzymemap_public_creep_20260924_seed42/selection.json), and [final test results](../enzymemap_public_creep_20260924_seed42/test/evaluation/summary.json). Its scores now enter the cross-modal functional-recovery analysis; the earlier same-modality figures retain their original comparator scope.')
        result = result.replace(
            'CREEP EnzymeMap has no final export in this frozen comparison; the training run continues separately.',
            'CREEP EnzymeMap was absent from these older frozen panels. Its completed test scores are now included in the cross-modal analysis below; these panels were not rerun.')
        result = result.replace(
            '| 9: Common-annotation cohort | [PDF](ec_levels/09_enzyme_ec_complete_cohort.pdf) |',
            '| 9: Common-annotation cohort | [PDF](ec_levels/09_enzyme_ec_complete_cohort.pdf) |\n'
            '| 10: Functional recovery beyond recorded partners | [PDF](../cersei_crossmodal_functional_recovery_20260925/functional_recovery_curves.pdf) |\n'
            '| 11: Refinement effects on functional recovery | [PDF](../cersei_crossmodal_functional_recovery_20260925/refinement_effects_at10.pdf) |')
        result = result.replace('All nine figures also have SVG and PNG exports beside their PDFs.',
                                'All eleven figures also have SVG and PNG exports beside their PDFs.')
        result = result.replace(
            '**Main finding:** the current dictionary-free CERSEI has stronger enzyme functional neighborhoods in several comparisons, most consistently at broad and intermediate EC resolution.',
            '**Main finding:** the current dictionary-free CERSEI has stronger enzyme functional neighborhoods in several comparisons, most consistently at broad and intermediate EC resolution. The new cross-modal diagnostic also supports EC3-compatible retrieval beyond recorded partner identities, most clearly on Enzyme-Sim and EnzymeMap, including a partner-sequence-component exclusion.')
    joint = ROOT / 'runs/cersei_joint_space_figure_20260925/README.md'
    if joint.exists():
        paths['joint_space'] = joint
        extra = joint.read_text().split('\n', 1)[1].strip()
        extra = extra.replace('**Figure 1.**', '**Figure 12.**').replace('**Figure 2.**', '**Figure 13.**')
        result += '\n## Enzyme-space visualization with cross-modal recovery\n\n' + lower_headings(rebase(extra, joint.parent)) + '\n'
        figure_count += 2
        result = result.replace('their eleven figures', 'their thirteen figures')
        result = result.replace('All eleven figures also have SVG and PNG exports beside their PDFs.',
                                'All thirteen figures also have SVG and PNG exports beside their PDFs.')
        result = result.replace(
            '| Functional recovery beyond recorded partners |',
            '| Joint-space illustration | Identical 720 annotated EnzymeMap enzymes; Horizyn, CREEP, CLIPZyme, CERSEI; fixed t-SNE and quantitative recovery | Complete: composite Figure 12 and all prespecified seeds in Figure 13 |\n| Functional recovery beyond recorded partners |')
        row = '| 11: Refinement effects on functional recovery | [PDF](../cersei_crossmodal_functional_recovery_20260925/refinement_effects_at10.pdf) |'
        result = result.replace(row, row + '\n| 12: Enzyme-space maps and recovery | [PDF](../cersei_joint_space_figure_20260925/joint_space_composite.pdf) |\n'
                                '| 13: All prespecified projection seeds | [PDF](../cersei_joint_space_figure_20260925/projection_seed_sensitivity.pdf) |')
    cluster = ROOT / 'runs/cersei_enzyme_cluster_metrics_20260925/README.md'
    if cluster.exists():
        paths['enzyme_cluster_metrics'] = cluster
        extra = cluster.read_text().split('\n', 1)[1].strip()
        result += '\n## Quantitative enzyme organization on the map cohort\n\n' + lower_headings(rebase(extra, cluster.parent)) + '\n'
    # Relative image/data links from every source must resolve in this document.
    for target in re.findall(r'\]\(([^)]+)\)', result):
        if target.startswith(('http:', 'https:', '#')): continue
        target_path = OUT / target.split('#')[0]
        if target_path.name == 'consolidated_report_receipt.json': continue
        assert target_path.exists(), target
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', result)
    assert len(images) == len(set(images)) == figure_count, images
    (OUT / 'README.md').write_text(result)
    receipt = dict(documentation_only=True, source_sha256={str(p): sha(p) for p in paths.values()},
                   script_sha256=sha(Path(__file__)), output_sha256=sha(OUT / 'README.md'),
                   embedded_figures=images, figure_count=figure_count, local_links_valid=True,
                   experiments_rerun=False, pending='No incomplete experiment is represented as a completed comparison', words=len(result.split()))
    (OUT / 'consolidated_report_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['figure_count', 'local_links_valid', 'experiments_rerun', 'words']}))


if __name__ == '__main__':
    build()
