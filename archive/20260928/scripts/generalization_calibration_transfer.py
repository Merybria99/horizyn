#!/usr/bin/env python3
"""Train-only transfer, provenance guards and frozen post-evaluation calibration."""
from __future__ import annotations
import argparse,datetime,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_retrieval import checked_artifact,sha256,canonical_dot
from horizyn.generalization_calibration import MeanAffinityCalibration
from generalization_full_graph import atomic_json
from generalization_composed_calibration import fit_training_means

RUN=ROOT/'runs/generalization_20260919_2251';OUT=RUN/'post_evaluation_calibration/transfer'
RULE=dict(gamma_enzyme=0.,gamma_reaction=1.,enzyme_mean_scheme='uniform_proteins')
SPLITS=['reaction_smi','enzyme_smi','time'];SEEDS=[42,17,73]

def record(p):return dict(path=str(Path(p).resolve()),sha256=sha256(p))

def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']

def authenticate_closure(freeze):
    if freeze.get('schema')!='post_evaluation_calibration_transfer_freeze_v1' or not freeze.get('frozen_before_new_predictions') or freeze.get('development_exposed') is not True or freeze.get('rule')!=RULE:
        raise ValueError('Require frozen development-exposed fixed calibration rule')
    for row in freeze['implementation_sources']:checked_artifact(row)
    for key in ['phase2_frozen_recipe','original_frozen_recipe','validation_selection']:checked_artifact(freeze[key])
    return freeze

def validate_freeze(record_):return authenticate_closure(json.loads(checked_artifact(record_).read_text()))

def checked_prediction(freeze_record,score_path,catalog_path,split,seed,score_key):
    """Authenticate a complete prediction without reading any assay labels."""
    freeze=validate_freeze(freeze_record);score_path=Path(score_path);receipt_path=score_path.parent/'complete.json';receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='generalization_predictions_v1' or receipt.get('phase')!='post_evaluation_calibration' or receipt.get('labels_used') is not False:raise ValueError('Wrong prediction phase or label use')
    if not same(receipt['calibration_freeze'],freeze_record) or receipt['output_sha256']!=sha256(score_path):raise ValueError('Prediction freeze/output mismatch')
    if receipt['inputs']['catalog']['sha256']!=sha256(catalog_path):raise ValueError('Prediction/catalog mismatch')
    if score_key not in ['selected','parent','baseline','baseline_fp64']:raise ValueError('Unregistered score key')
    rows=[row for row in freeze['approved_models'] if row['split']==split and row['seed']==seed and same(row['parent_bundle'],receipt['bundle']) and same(row['calibration_state'],receipt['calibration_state'])]
    if len(rows)!=1:raise ValueError('Unapproved model/mean tuple')
    inputs=[row for row in freeze['approved_predictions'] if row['split']==split and row['seed']==seed and same(row['input_receipt'],receipt['input_receipt']) and same(row['parent_scores'],receipt['parent_reference']['scores'])]
    if len(inputs)!=1:raise ValueError('Unapproved prediction input/control tuple')
    for key in ['bundle','calibration_state','input_receipt']:checked_artifact(receipt[key])
    from horizyn.generalization_phase2 import validate_phase2_bundle
    parent=json.loads(checked_artifact(receipt['bundle']).read_text());validate_phase2_bundle(parent,Path(receipt['bundle']['path']).parent)
    if parent['split']!=split or parent['seed']!=seed:raise ValueError('Parent split/seed mismatch')
    for key in ['phase2_frozen_recipe','frozen_recipe']:
        reference='original_frozen_recipe' if key=='frozen_recipe' else key
        if not same(receipt[key],freeze[reference]):raise ValueError('Prediction parent freeze mismatch')
    inp=json.loads(checked_artifact(receipt['input_receipt']).read_text())
    if inp.get('schema')!='generalization_feature_bundle_receipt_v1' or inp.get('freeze_sha256')!=freeze['original_frozen_recipe']['sha256']:raise ValueError('Invalid input receipt lineage')
    for key in ['catalog','base','protein_means','reaction_features']:
        if not same(inp['inputs'][key],receipt['inputs'][key]):raise ValueError('Receipt input mismatch: '+key)
    if not receipt.get('parent_scores_exact') or not receipt.get('zero_scores_exact') or not all(receipt['subset_parity'].values()):raise ValueError('Numerical replay failed')
    for item in receipt['parent_reference'].values():checked_artifact(item)
    with np.load(score_path) as f:values=f[score_key]
    catalog=json.loads(Path(catalog_path).read_text());shape=(len(catalog.get('reactions',catalog.get('query_ids'))),len(catalog['proteins']))
    if values.dtype!=np.float32 or values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid calibrated scores')
    return values,receipt

@torch.inference_mode()
def fit(args):
    from horizyn.generalization_phase2 import ComposedPhase2Encoder
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(args.device)
    for phase in ['phase2','phase4']:
        for item in json.loads((RUN/phase/'frozen_recipe.json').read_text())['implementation_sources']:checked_artifact(item)
    if (OUT/'fit_complete.json').exists():raise ValueError('Train-only transfer already complete')
    OUT.mkdir(parents=True,exist_ok=True);approved=[]
    for split in SPLITS:
        features=RUN/('features' if split=='reaction_smi' else 'features_'+split)
        paths={k:features/k for k in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','protein_mean.h5','reaction_features.npz']};records={k:record(p) for k,p in paths.items()}
        catalog=json.loads(paths['catalog.json'].read_text())
        with np.load(paths['pairs.npz']) as f:train=f['train']
        tr,te=np.unique(train[:,0]),np.unique(train[:,1])
        if len(np.unique(train,axis=0))!=len(train) or catalog['train_reactions']!=[catalog['reactions'][i] for i in tr] or catalog['train_proteins']!=[catalog['proteins'][i] for i in te]:raise ValueError('Actual training selector mismatch')
        with np.load(paths['f3_features.npz']) as f:be=torch.tensor(f['proteins'],device=args.device);br=torch.tensor(f['train_reactions'],device=args.device)
        with h5py.File(paths['protein_mean.h5']) as f:
            if f['ids'].asstr()[:].tolist()!=catalog['proteins'] or not f['complete'][:].all():raise ValueError('Mean source mismatch')
            raw=torch.tensor(f['vectors'][:],device=args.device)
        with np.load(paths['reaction_features.npz']) as f:
            blocks={k:torch.tensor(f[k],device=args.device) for k in ['t5v2','unimol2','chiro','chemistry']};masks={k:torch.tensor(f[k+'_mask'],device=args.device) for k in blocks}
        for seed in SEEDS:
            directory=OUT/'models'/split/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            bundle=RUN/'phase2/models'/split/f'seed{seed}'/'bundle.json';parent=record(bundle)
            model,spec=ComposedPhase2Encoder.from_bundle(bundle,args.device)
            if spec['feature_manifest_sha256']!=records['manifest.json']['sha256']:raise ValueError('Parent feature mismatch')
            atomic_json(directory/'protocol.json',dict(schema='calibration_train_only_transfer_fit_v1',rule=RULE,training_sources=records,parent_bundle=parent,
                source=record(Path(__file__)),mean_fitter=record(ROOT/'scripts/generalization_composed_calibration.py'),module=record(ROOT/'horizyn/generalization_calibration.py'),
                split=split,seed=seed,training_only=True,validation_used=False,test_or_external_used=False))
            mr,me=fit_training_means(model,be,br,blocks,masks,raw,train,tr,te,directory)
            calibration=MeanAffinityCalibration(mr,me['uniform_proteins'],0.,1.)
            ir=np.unique(np.linspace(0,len(tr)-1,31,dtype=int));ie=np.unique(np.linspace(0,len(te)-1,193,dtype=int))
            r=model.encode_reactions(br[ir],{k:v[tr[ir]] for k,v in blocks.items()},{k:v[tr[ir]] for k,v in masks.items()})
            e=model.encode_enzymes(be[te[ie]],raw[te[ie]],batch_size=512);ra,ea=calibration.encode_reactions(r),calibration.encode_enzymes(e)
            zero=MeanAffinityCalibration(mr,me['uniform_proteins'],0.,0.);full=canonical_dot(ra,ea)
            checks=dict(zero_endpoints=torch.equal(zero.encode_reactions(r),r) and torch.equal(zero.encode_enzymes(e),e),zero_scores=torch.equal(canonical_dot(zero.encode_reactions(r),zero.encode_enzymes(e)),canonical_dot(r,e)),
                subset_endpoints=torch.equal(ra[::3],calibration.encode_reactions(r[::3])) and torch.equal(ea[::7],calibration.encode_enzymes(e[::7])),
                subset_scores=torch.equal(full[::3,::7],canonical_dot(calibration.encode_reactions(r[::3]),calibration.encode_enzymes(e[::7]))),
                singleton_scores=torch.equal(full[:1,:1],canonical_dot(calibration.encode_reactions(r[:1]),calibration.encode_enzymes(e[:1]))))
            if not all(checks.values()):raise ValueError('Training-only calibration parity check failed')
            state=dict(schema='p2_calibration_state_v1',rule=RULE,split=split,seed=seed,parent_bundle=parent,training_sources=records,reaction_mean=mr.cpu(),enzyme_mean=me['uniform_proteins'].cpu(),fit_protocol=record(directory/'protocol.json'))
            torch.save(state,directory/'calibration.pt');atomic_json(directory/'preflight_check.json',dict(training_fixture_shape=list(full.shape),checks=checks,labels_used=False))
            if split=='reaction_smi' and seed==42:
                original=torch.load(RUN/'post_evaluation_calibration/train_means.pt',map_location='cpu',weights_only=False)
                if not torch.equal(mr.cpu(),original['reaction_mean']) or not torch.equal(me['uniform_proteins'].cpu(),original['enzyme_means']['uniform_proteins']):raise ValueError('Validation-selected means were not replayed exactly')
            row=dict(split=split,seed=seed,parent_bundle=parent,calibration_state=record(directory/'calibration.pt'),training_sources=records,preflight_check=record(directory/'preflight_check.json'))
            approved.append(row);atomic_json(OUT/'fit_progress.json',approved);print(json.dumps(dict(split=split,seed=seed,complete=True,all_checks=True)),flush=True)
            del model
    atomic_json(OUT/'fit_complete.json',dict(schema='calibration_train_only_transfer_complete_v1',rule=RULE,approved_models=approved,training_only=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fit']);p.add_argument('--device',default='cuda:0');args=p.parse_args();fit(args)

if __name__=='__main__':main()
