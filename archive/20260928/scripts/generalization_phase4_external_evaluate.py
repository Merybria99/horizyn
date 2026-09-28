#!/usr/bin/env python3
"""Authenticate phase4 extensions before reusing immutable external metrics.

The aminotransferase mode defaults to authentication only. All source and model
freezes must authenticate before any new-panel assay labels enter an evaluator.
"""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import generalization_external_evaluate as external
import generalization_phase2_external_evaluate as phase2_external
import generalization_nitrilase_evaluate as nitrilase
from horizyn.generalization_phase4 import validate_phase4_bundle
from horizyn.generalization_phase2 import validate_phase2_bundle

METHOD_ROLES={'phase4_seed42':'primary','phase4_seed17':'diagnostic_seed',
    'phase4_seed73':'diagnostic_seed','hybrid_anchor_only':'diagnostic_component',
    'phase2_seed42':'earlier_primary_comparison','phase2_seed17':'earlier_diagnostic_seed',
    'phase2_seed73':'earlier_diagnostic_seed','density_only':'diagnostic_component',
    'smooth_only':'diagnostic_component','F3_native':'numerical_control','F3_fp64':'paired_reference'}


def bind_methods(methods):
    if [m['label'] for m in methods]!=list(METHOD_ROLES) or [m['label'] for m in methods if m.get('primary')]!=['phase4_seed42']:
        raise ValueError('Retain all eleven methods with the fixed primary and roles')
    bindings=[('primary',42,'selected'),('seed17',17,'selected'),('seed73',73,'selected'),('hybrid_anchor_only',42,'selected'),
        ('primary',42,'selected'),('seed17',17,'selected'),('seed73',73,'selected'),
        ('density_only',42,'selected'),('smooth_only',42,'selected'),('primary',42,'baseline'),('primary',42,'baseline_fp64')]
    for method,(variant,seed,key) in zip(methods,bindings):
        entries=list(method['panels'].values()) if 'panels' in method else [method['scores']]
        if not entries:raise ValueError('Every fourth-phase method requires scores')
        for entry in entries:
            if entry.get('score_key')!=key:
                raise ValueError('Method label and score key disagree')
            if entry.get('expected_variant',variant)!=variant or entry.get('expected_seed',seed)!=seed:
                raise ValueError('Fourth-phase method label and declared variant/seed differ')
            entry['expected_variant']=variant;entry['expected_seed']=seed
    return methods


def check_method_binding(entry,receipt,bundle,phase):
    if receipt.get('phase')!=phase or receipt.get('labels_used') is not False:
        raise ValueError('Prediction must come from the declared label-free phase')
    if bundle.get('split')!='reaction_smi':
        raise ValueError('External panels require the reaction_smi training split')
    if 'expected_variant' not in entry or 'expected_seed' not in entry:
        raise ValueError('Method bindings are required before score authentication')
    if bundle.get('variant')!=entry['expected_variant'] or bundle.get('seed')!=entry['expected_seed']:
        raise ValueError('Actual bundle variant/seed differs from its method label')


def load_bound_phase2_scores(entry,base,catalog,provenance,phase2,shape):
    path=external.resolve(base,entry['path']);receipt_path=path.parent/'complete.json'
    receipt=json.loads(receipt_path.read_text())
    bundle_path=external.resolve(receipt_path.parent,receipt['bundle']['path'])
    external.checked_identity(bundle_path,receipt['bundle']['sha256'])
    bundle=json.loads(bundle_path.read_text())
    check_method_binding(entry,receipt,bundle,'exploratory_phase2')
    validate_phase2_bundle(bundle,bundle_path.parent)
    return nitrilase.load_phase2_scores(entry,base,catalog,provenance,phase2,shape)


def label_defined_rank_association(scores,rates,censored,spearman):
    """Keep every nonconstant assay-order query; constant predictions earn zero."""
    result={}
    for direction,s,a,c in [('reaction_to_enzyme',scores,rates,censored),
                          ('enzyme_to_reaction',scores.T,rates.T,censored.T)]:
        records=[]
        for i,(x,y,z) in enumerate(zip(s,a,c)):
            # The number of observed assay-order classes depends only on labels.
            classes=len(np.unique(y[~z]))+int(z.any())
            eligible=classes>1
            constant=bool(np.all(x==x[0]))
            value=(0. if constant else spearman(x,y,z)) if eligible else None
            records.append(dict(query_index=i,spearman=value,label_eligible=eligible,
                constant_prediction=constant,quantified_count=int((~z).sum()),censored_count=int(z.sum())))
        values=[r['spearman'] for r in records if r['label_eligible']]
        result[direction]=dict(per_query=records,mean_spearman=float(np.mean(values)) if values else None,
            eligible_query_count=len(values),omitted_constant_activity_queries=len(records)-len(values),
            constant_prediction_on_eligible_queries=sum(r['label_eligible'] and r['constant_prediction'] for r in records),
            constant_prediction_policy='Zero information (0 correlation) on label-eligible queries; eligibility is identical across methods.')
    return result


def load_phase4_scores(entry,base,catalog,provenance,phase4,phase2,shape):
    path=external.resolve(base,entry['path'])
    receipt_path=path.parent/'complete.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('phase')!='exploratory_phase4' or receipt.get('labels_used') is not False:
        raise ValueError('Require label-free fourth-phase inference')
    frozen_record=receipt.get('phase4_frozen_recipe')
    if not frozen_record or frozen_record['sha256']!=phase4['sha256'] or str(Path(frozen_record['path']).resolve())!=phase4['path']:
        raise ValueError('Fourth-phase prediction freeze differs from the declared freeze')
    bundle_path=external.resolve(receipt_path.parent,receipt['bundle']['path'])
    external.checked_identity(bundle_path,receipt['bundle']['sha256'])
    bundle=json.loads(bundle_path.read_text())
    check_method_binding(entry,receipt,bundle,'exploratory_phase4')
    if bundle['phase4_frozen_recipe']!=frozen_record:
        raise ValueError('Prediction and fourth-phase bundle lineage differ')
    validate_phase4_bundle(bundle,bundle_path.parent)
    if 'expected_bundle' in entry and receipt['bundle']!=entry['expected_bundle']:
        raise ValueError('Score entry uses a different preregistered bundle')
    if entry.get('score_key')!='selected' or 'expected_variant' not in entry or 'expected_seed' not in entry:
        raise ValueError('Fourth-phase score loading requires mandatory method bindings')
    if bundle['variant']!=entry['expected_variant']:
        raise ValueError('Method label and frozen variant disagree')
    if bundle['seed']!=entry['expected_seed']:
        raise ValueError('Method label and frozen residual seed disagree')
    provenance[phase4['path']]=phase4
    return nitrilase.load_phase2_scores(entry,base,catalog,provenance,phase2,shape)


def validate_manifest(path):
    path=path.resolve();manifest=json.loads(path.read_text());base=path.parent
    phase4=external.checked_identity(external.resolve(base,manifest['phase4_freeze']['path']),manifest['phase4_freeze']['sha256'])
    freeze=json.loads(Path(phase4['path']).read_text())
    phase2=external.checked_identity(external.resolve(base,manifest['phase2_freeze']['path']),manifest['phase2_freeze']['sha256'])
    if freeze['schema']!='phase4_hybrid_frozen_recipe_v1' or not freeze['frozen_before_new_external_evaluation']:
        raise ValueError('Require frozen fourth-phase selection before external evaluation')
    if freeze['phase2_frozen_recipe']['sha256']!=phase2['sha256']:
        raise ValueError('Fourth-phase and second-phase lineage disagree')
    for record in freeze['implementation_sources']:
        external.checked_identity(record['path'],record['sha256'])
    pinned={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['implementation_sources']}
    required=[Path(__file__),ROOT/'scripts/generalization_external_evaluate.py',
              ROOT/'scripts/generalization_phase2_external_evaluate.py',ROOT/'scripts/generalization_nitrilase_evaluate.py']
    for source in required:
        if pinned.get(str(source.resolve()))!=external.sha256(source):
            raise ValueError('The fourth-phase freeze must pin every reused evaluator source: '+str(source))
    bind_methods(manifest['methods'])
    if manifest['baseline_method']!='F3_fp64':raise ValueError('Retain the fixed paired reference')
    return manifest,phase4,phase2


def run(args):
    manifest,phase4,phase2=validate_manifest(args.manifest)
    base=args.manifest.resolve().parent;provenance={}
    if args.panel=='aminotransferase':
        path=args.audit_root/'aminotransferase_audit/evaluate_panel.py'
        freeze=json.loads(Path(phase4['path']).read_text())
        pinned={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['implementation_sources']}
        if pinned.get(str(path.resolve()))!=external.sha256(path):
            raise ValueError('Fourth-phase freeze does not pin the immutable aminotransferase evaluator')
        spec=importlib.util.spec_from_file_location('aminotransferase_frozen_evaluator',path)
        panel=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)
        original=manifest['phase2_panel_manifest']
        original_path=external.resolve(base,original['path'])
        external.checked_identity(original_path,original['sha256'])
        old_manifest,prereg,catalog,scores,keep,provenance=panel.authenticate(original_path)
        # Freeze must pin both source and endpoint preregistration, not amend either.
        catalog_path=args.audit_root/'aminotransferase_audit/features/catalog.json'
        for method in manifest['methods'][:4]:
            scores[method['label']]=load_phase4_scores(method['scores'],base,catalog_path,provenance,phase4,phase2,(18,25))
        for method in manifest['methods'][4:]:
            load_bound_phase2_scores(method['scores'],base,catalog_path,provenance,phase2,(18,25))
        scores={m['label']:scores[m['label']] for m in manifest['methods']}
        provenance[str(args.manifest.resolve())]=external.identity(args.manifest)
        provenance[str(Path(__file__).resolve())]=external.identity(__file__)
        # Earlier seven entries must retain their exact immutable score sources.
        if any({k:new['scores'].get(k) for k in old['scores']}!=old['scores'] for new,old in zip(manifest['methods'][4:],old_manifest['methods'])):
            raise ValueError('Original seven-method score provenance changed')
        amendment=manifest.get('ordinal_association_amendment',{})
        if amendment.get('eligibility')!='label_defined' or amendment.get('constant_prediction_value')!=0:
            raise ValueError('Require the disclosed label-defined quantitative endpoint amendment')
        # Source remains immutable: this separately frozen extension supplies the
        # prespecified amended endpoint for every method, before outcome access.
        panel.rank_association=lambda s,a,c:label_defined_rank_association(s,a,c,panel.censored_spearman)
        panel.evaluate_authenticated(args,manifest,prereg,catalog,scores,keep,provenance)
        if args.evaluate:
            summary_path=args.output/'summary.json';summary=json.loads(summary_path.read_text())
            summary['phase4_freeze']=phase4
            summary['method_roles']=METHOD_ROLES
            summary['ordinal_association_amendment']=amendment
            summary['phase2_preservation']='Original phase2 evaluator/preregistration unchanged; all seven methods retained. The separately frozen extension uses label-defined quantitative query eligibility for every method.'
            external.write_json(summary_path,summary)
            complete=args.output/'complete.json';receipt=json.loads(complete.read_text())
            receipt.update(phase4_freeze=phase4,phase4_wrapper=external.identity(__file__),
                outputs={p.name:external.identity(p) for p in sorted(args.output.iterdir()) if p.is_file() and p!=complete})
            external.write_json(complete,receipt)
        return
    shapes={'case1':(1,123),'p450':(191,490)} if args.panel=='case1_p450' else {'nitrilase':(38,18)}
    for method in manifest['methods'][:4]:
        for name,shape in shapes.items():
            entry=method['panels'][name] if args.panel=='case1_p450' else method['scores']
            catalog=args.audit_root/(name+'_audit/features/catalog.json')
            load_phase4_scores(entry,base,catalog,provenance,phase4,phase2,shape)
    for method in manifest['methods'][4:]:
        for name,shape in shapes.items():
            entry=method['panels'][name] if args.panel=='case1_p450' else method['scores']
            catalog=args.audit_root/(name+'_audit/features/catalog.json')
            load_bound_phase2_scores(entry,base,catalog,provenance,phase2,shape)
    if not args.evaluate:
        if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh authentication directory')
        args.output.mkdir(parents=True,exist_ok=True)
        external.write_json(args.output/'scores_authenticated.json',dict(phase4_freeze=phase4,provenance=provenance,outcomes_computed=False))
        return
    if args.panel=='case1_p450':phase2_external.run(args)
    else:nitrilase.run(args)
    summary_path=args.output/'summary.json';summary=json.loads(summary_path.read_text())
    summary['phase4_freeze']=phase4
    summary['method_roles']=METHOD_ROLES
    summary['evaluation_status']='Exploratory fourth-phase repeat of a previously opened panel; all earlier results remain immutable. All eleven prespecified methods retained.'
    external.write_json(summary_path,summary)
    readout=args.output/'readout.md'
    if readout.exists():readout.write_text('Fourth-phase exploratory extension. The fixed new primary is phase4_seed42; phase2_seed42 remains the earlier primary comparison. All earlier outputs are preserved.\n\n'+readout.read_text())
    complete=args.output/'complete.json';receipt=json.loads(complete.read_text())
    receipt.update(schema='phase4_external_evaluation_receipt_v1',phase4_freeze=phase4,
        phase4_source_checks=provenance,phase4_wrapper=external.identity(__file__),
        outputs={p.name:external.identity(p) for p in sorted(args.output.iterdir()) if p.is_file() and p!=complete})
    external.write_json(complete,receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',choices=['case1_p450','nitrilase','aminotransferase'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--audit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--release-note',default='')
    parser.add_argument('--training-split',choices=['reaction_smi','enzyme_smi','time'],default='reaction_smi')
    parser.add_argument('--bootstrap-replicates',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20260919)
    args=parser.parse_args()
    if args.evaluate and not args.release_note.strip():parser.error('Record the parent release note before outcome evaluation')
    run(args)


if __name__=='__main__':main()
