#!/usr/bin/env python3
"""Phase2 exploratory Case1/P450 readout with both recipe lineages verified.

Retains the existing phase1 evaluator/metric implementation unchanged. The
manifest's `freeze` is the phase1 feature lineage and `phase2_freeze` binds all
new predictions. Both panels were already opened in phase1: these are repeat,
exploratory evaluations, not independent confirmation panels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import generalization_external_evaluate as external
from generalization_nitrilase_evaluate import load_phase2_scores


def run(args):
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    record = manifest['phase2_freeze']
    phase2_path = external.resolve(base, record['path'])
    phase2 = external.checked_identity(phase2_path, record['sha256'])
    frozen = json.loads(phase2_path.read_text())
    if frozen.get('schema') != 'phase2_generalization_frozen_recipe_v1' or not frozen.get('frozen_before_new_external_evaluation'):
        raise ValueError('Require the completed phase2 freeze')
    original = manifest['freeze']
    if original['sha256'] != frozen['original_frozen_recipe']['sha256']:
        raise ValueError('Original lineage and phase2 recipe disagree')
    if frozen.get('canonical_reference_method') != manifest['baseline_method']:
        raise ValueError('Paired reference differs from phase2 freeze')
    provenance = {}
    loaded = {}
    for method in manifest['methods']:
        loaded[method['label']] = {}
        for panel, shape in [('case1',(1,123)), ('p450',(191,490))]:
            catalog = args.audit_root / (panel+'_audit/features/catalog.json')
            loaded[method['label']][panel] = load_phase2_scores(method['panels'][panel], base, catalog, provenance, phase2, shape)
    external.run(args)
    case_metadata = external.case1_metadata(args.audit_root/'case1_audit')
    ties = {}
    for label, panels in loaded.items():
        scores = panels['case1'][0]
        values, counts = np.unique(scores,return_counts=True)
        expected = {}
        for label_set, indices in case_metadata['indices'].items():
            indices = list(indices)
            expected[label_set] = {}
            for k in external.CASE_CUTS:
                recovered = sum(max(0,min(int(np.count_nonzero(scores==scores[i])), k-int(np.count_nonzero(scores>scores[i]))))/int(np.count_nonzero(scores==scores[i])) for i in indices)
                expected[label_set][str(k)] = dict(expected_recovered=recovered, expected_recall=recovered/len(indices))
        p450_unique = [len(np.unique(row)) for row in panels['p450']]
        ties[label] = dict(case1_unique_scores=len(values),case1_all_scores_tied=bool(len(values)==1),
            case1_candidates_in_exact_ties=int(counts[counts>1].sum()),case1_uniform_tie_expected_recovery=expected,
            p450_all_scores_tied_reaction_queries=sum(n==1 for n in p450_unique),
            p450_reaction_queries_with_any_exact_ties=sum(n<490 for n in p450_unique))
    external.write_json(args.output/'exact_score_tie_audit.json',ties)
    summary_path = args.output/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['phase2_freeze'] = phase2
    summary['evaluation_status'] = 'Exploratory repeated Case1/P450 assessment after phase1 failure; neither panel is independent phase2 confirmation.'
    summary['protocol']['score_contract'] = 'Frozen phase2 predictions use canonical_dot FP64 accumulation rounded to FP32. F3_native normalizes native endpoints once in FP32; F3_fp64 normalizes once in FP64 and rounds endpoints to FP32. Primary paired reference is F3_fp64.'
    summary['protocol']['phase1_preservation'] = 'All phase1 results remain unchanged in their separate output directory.'
    summary['exact_score_ties'] = ties
    summary['protocol']['tie_audit'] = 'The inherited stable-order metrics are accompanied by explicit tie counts and uniform-tie expected Case1 recovery; apparent catalog-order success under constant scores is not evidence.'
    external.write_json(summary_path,summary)
    readout = args.output/'readout.md'
    readout.write_text('These are **exploratory phase2 repeats** on the already opened Case1 and P450 panels. The independent, phase2-sequestered assay check is nitrilase, which was nevertheless previously evaluated in this workspace. The primary paired reference is F3_fp64; F3_native isolates the earlier numerical convention.\n\n'+readout.read_text())
    complete_path = args.output/'complete.json'
    receipt = json.loads(complete_path.read_text())
    receipt['schema'] = 'phase2_exploratory_external_evaluation_receipt_v1'
    receipt['phase2_freeze'] = phase2
    receipt['phase2_evaluator'] = external.identity(__file__)
    receipt['dual_freeze_guard'] = external.identity(Path(__file__).with_name('generalization_nitrilase_evaluate.py'))
    receipt['phase2_source_checks'] = provenance
    receipt['outputs'] = {p.name:external.identity(p) for p in sorted(args.output.iterdir()) if p.is_file() and p!=complete_path}
    external.write_json(complete_path,receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--audit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-split',choices=['reaction_smi','enzyme_smi','time'],default='reaction_smi')
    parser.add_argument('--bootstrap-replicates',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20260919)
    run(parser.parse_args())


if __name__=='__main__':
    main()
