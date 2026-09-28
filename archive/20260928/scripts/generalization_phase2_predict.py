#!/usr/bin/env python3
"""Label-free prediction with a frozen exploratory phase-two bundle."""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import canonical_dot,sha256
from horizyn.semantic_anchors import row_unit
from generalization_predict import validate_input_receipt
from generalization_full_graph import atomic_json


def run(args):
    if args.batch_size<=0:raise ValueError('Positive prediction batch size required')
    if (args.output/'complete.json').exists():raise ValueError('Choose a fresh immutable prediction directory')
    args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');device=torch.device(args.device)
    model,spec=ComposedPhase2Encoder.from_bundle(args.bundle,device)
    receipt=validate_input_receipt(args,spec)
    catalog=json.loads(args.catalog.read_text());reactions=catalog.get('reactions',catalog.get('query_ids'))
    if reactions is None or len(set(reactions))!=len(reactions) or len(set(catalog['proteins']))!=len(catalog['proteins']):raise ValueError('Missing or duplicate catalog IDs')
    with np.load(args.base) as source:
        be=torch.tensor(source['proteins'],device=device);br=torch.tensor(source[args.reaction_key],device=device)
    with h5py.File(args.protein_means) as source:
        if list(source['ids'].asstr()[:])!=catalog['proteins']:raise ValueError('Protein means/catalog ID order mismatch')
        if 'complete' in source and not source['complete'][:].all():raise ValueError('Incomplete protein means')
        means=torch.tensor(source['vectors'][:],device=device)
    if len(be)!=len(catalog['proteins']) or len(br)!=len(reactions):raise ValueError('Native base arrays/catalog rows disagree')
    with np.load(args.reaction_features) as source:
        blocks={k:torch.tensor(source[k],device=device) for k in model.modalities}
        masks={k:torch.tensor(source[k+'_mask'],device=device) for k in model.modalities}
    if any(len(x)!=len(br) for x in blocks.values()) or any(x.shape!=(len(br),) for x in masks.values()):raise ValueError('Reaction block/mask dimensions disagree')
    with torch.inference_mode():
        e,de=model.encode_enzymes(be,means,args.batch_size,return_diagnostics=True)
        r,dr=model.encode_reactions(br,blocks,masks,args.batch_size,return_diagnostics=True)
        outputs=dict(selected=canonical_dot(r,e),baseline=canonical_dot(F.normalize(br,dim=1),F.normalize(be,dim=1)),
                     baseline_fp64=canonical_dot(row_unit(br),row_unit(be)))
    if not all(torch.isfinite(value).all() for value in outputs.values()):raise ValueError('Nonfinite retrieval scores')
    np.savez(args.output/'scores.npz',**{k:v.cpu().numpy() for k,v in outputs.items()})
    np.savez(args.output/'support_diagnostics.npz',**{endpoint+'_'+k:v.cpu().numpy() for endpoint,diag in [('enzyme',de),('reaction',dr)] for k,v in diag.items() if v is not None})
    if args.save_embeddings:torch.save(dict(enzymes=e.cpu(),reactions=r.cpu()),args.output/'encoded_endpoints.pt')
    shutil.copyfile(args.catalog,args.output/'catalog.json')
    manifest=dict(schema='generalization_predictions_v1',phase='exploratory_phase2',labels_used=False,
        bundle=dict(path=str(args.bundle.resolve()),sha256=sha256(args.bundle)),
        phase2_frozen_recipe=spec['phase2_frozen_recipe'],frozen_recipe=spec['frozen_recipe'],input_receipt=receipt,
        inputs={key:dict(path=str(getattr(args,key).resolve()),sha256=sha256(getattr(args,key))) for key in ('base','catalog','protein_means','reaction_features')},
        reaction_key=args.reaction_key,shape=list(outputs['selected'].shape),embedding_dimension=e.shape[1],
        score_keys=list(outputs),baseline_contracts=spec['baselines'],score_contract=spec['score'],
        batch_size=args.batch_size,device=str(device),torch_version=torch.__version__,source_sha256=sha256(__file__),
        elapsed_seconds=time.monotonic()-started,output_sha256=sha256(args.output/'scores.npz'),
        support_diagnostics_sha256=sha256(args.output/'support_diagnostics.npz'))
    atomic_json(args.output/'complete.json',manifest)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=manifest['shape'],phase=manifest['phase'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','base','catalog','protein-means','reaction-features','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--input-receipt',type=Path);p.add_argument('--reaction-key',default='reactions')
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--save-embeddings',action='store_true');run(p.parse_args())


if __name__=='__main__':main()
