#!/usr/bin/env python3
"""Validate train-bank endpoint refinement inside the existing SLEEC phase-2 model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import canonical_dot, sha256
from horizyn.generalization_transport import transport_embeddings, refine_phase2
from horizyn.semantic_anchors import centered_unit, reaction_features
from generalization_full_graph import validation_data, atomic_json, eligible
from generalization_metrics import evaluate_scores
from generalization_smooth_anchors import robust_value
from generalization_phase2_official_evaluate import checked_official_truth

RUN = ROOT / 'runs/generalization_20260919_2251'
GRID = [dict(enzyme_strength=e, reaction_strength=r)
        for e, r in itertools.product((0., .1, .25, .5), repeat=2)]


def load_features(path, device):
    catalog = json.loads((path / 'catalog.json').read_text())
    with np.load(path / 'f3_features.npz') as data:
        enzymes = torch.as_tensor(data['proteins'], device=device)
        reactions = torch.as_tensor(data['reactions'], device=device)
    with h5py.File(path / 'protein_mean.h5') as data:
        if data['ids'].asstr()[:].tolist() != catalog['proteins']:
            raise ValueError('Protein means and catalog disagree')
        if not data['complete'][:].all():
            raise ValueError('Incomplete protein means')
        means = torch.as_tensor(data['vectors'][:], device=device)
    with np.load(path / 'reaction_features.npz') as data:
        modalities = ('t5v2', 'unimol2', 'chiro', 'chemistry')
        blocks = {k: torch.as_tensor(data[k], device=device) for k in modalities}
        masks = {k: torch.as_tensor(data[k + '_mask'], device=device) for k in modalities}
    if len(enzymes) != len(catalog['proteins']) or len(reactions) != len(catalog['reactions']):
        raise ValueError('Native embedding axes disagree')
    return catalog, enzymes, reactions, means, blocks, masks


def reaction_keys(model, blocks, masks):
    return reaction_features(blocks, model.density.reaction_centers, masks, model.modalities)


@torch.inference_mode()
def train_values(model, endpoint, base, raw, blocks=None, masks=None):
    """Use the exact self-anchor support bound when it proves gate saturation.

    Every submitted row is a training endpoint. For a raw-space gate, its own
    raw anchor is present. Its self dot product is therefore a lower bound on
    maximum support, avoiding a quadratic training-against-training search.
    Fall back to the original encoder whenever that proof does not apply.
    """
    density = model.density
    gate = density.selected_gate
    if gate['space'] == 'raw' and gate['endpoints'] in ('both', endpoint):
        keys = (centered_unit(raw, density.protein_center) if endpoint == 'enzyme'
                else reaction_keys(model, blocks, masks))
        high = density.thresholds['raw'][endpoint][str(gate['upper_quantile'])]
        if bool((keys.double().square().sum(1) >= high + 1e-6).all()):
            delta = density._residual_delta(base, endpoint)
            scales = base.new_full((len(base),), gate['cap'])
            result = density._normalize(base, delta, scales)
            sample = torch.linspace(0, len(base)-1, min(17, len(base)), device=base.device).long().unique()
            if endpoint == 'enzyme':
                reference = density.encode_enzymes(base[sample], raw[sample], batch_size=32)
            else:
                reference = density.encode_reactions(base[sample],
                    {k:v[sample] for k,v in blocks.items()}, {k:v[sample] for k,v in masks.items()}, batch_size=32)
            if not torch.equal(reference, result[sample]):
                raise ValueError('Training self-anchor optimization failed exact parity')
            return result
    if endpoint == 'enzyme':
        return density.encode_enzymes(base, raw, batch_size=256)
    return density.encode_reactions(base, blocks, masks, batch_size=256)


def summarize(evaluation):
    summary = evaluation['summary']
    return {d: summary[d]['all']['reactzyme_mrr'] for d in summary}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('reaction_smi','enzyme_smi','time'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'registry.json').exists():
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    features = RUN / {'reaction_smi':'features','enzyme_smi':'features_enzyme_smi','time':'features_time'}[args.split]
    bundle = RUN / 'phase2/models' / args.split / 'seed42/bundle.json'
    model, spec = ComposedPhase2Encoder.from_bundle(bundle, device)
    if sha256(features / 'manifest.json') != spec['feature_manifest_sha256']:
        raise ValueError('Phase-2 model belongs to different training features')
    registry = dict(created_utc=datetime.now(timezone.utc).isoformat(), split=args.split,
        schema='phase2_training_neighbor_refinement_v1', grid=GRID, neighbors=32, temperature=.03,
        source_sha256=sha256(Path(__file__)), module_sha256=sha256(ROOT/'horizyn/generalization_transport.py'),
        parent_bundle=dict(path=str(bundle),sha256=sha256(bundle)),
        train_manifest_sha256=sha256(features/'manifest.json'),
        selection='Maximum balanced seen/unseen all-positive validation MRR; each direction aggregate/unseen within 0.005 of phase2',
        test_used_for_selection=False, retains_sleec=True, ensembles=False, hubness_correction=False)
    atomic_json(out/'registry.json',registry)
    started=time.monotonic()
    catalog, be, br, means, blocks, masks = load_features(features, device)
    with np.load(features/'pairs.npz') as data:
        pairs={k:data[k] for k in ('train','validation')}
    tr,te=np.unique(pairs['train'][:,0]),np.unique(pairs['train'][:,1])
    if [catalog['proteins'][i] for i in te] != catalog['train_proteins']:
        raise ValueError('Training enzyme selector mismatch')
    if [catalog['reactions'][i] for i in tr] != catalog['train_reactions']:
        raise ValueError('Training reaction selector mismatch')
    vr,ve,truth=validation_data(catalog,pairs)
    ek=centered_unit(means,model.density.protein_center)
    rk=reaction_keys(model,blocks,masks)
    ev=train_values(model,'enzyme',be[te],means[te])
    rv=train_values(model,'reaction',br[tr],None,
                    {k:v[tr] for k,v in blocks.items()},{k:v[tr] for k,v in masks.items()})
    bank=dict(enzyme_keys=ek[te].cpu(),enzyme_values=ev.cpu(),reaction_keys=rk[tr].cpu(),reaction_values=rv.cpu(),
              training_protein_ids=catalog['train_proteins'],training_reaction_ids=catalog['train_reactions'],registry=registry)
    torch.save(bank,out/'training_bank.pt')
    transported_e=transport_embeddings(ek[ve],ek[te],ev)
    transported_r=transport_embeddings(rk[vr],rk[tr],rv)
    e=model.encode_enzymes(be[ve],means[ve],batch_size=256)
    r=model.encode_reactions(br[vr],{k:v[vr] for k,v in blocks.items()},
                            {k:v[vr] for k,v in masks.items()},batch_size=256)
    records=[];baseline=None;best=None
    for recipe in GRID:
        er=refine_phase2(e,transported_e,recipe['enzyme_strength'],alpha=model.alpha)
        rr=refine_phase2(r,transported_r,recipe['reaction_strength'],alpha=model.alpha)
        evaluation=evaluate_scores(canonical_dot(rr,er),truth)
        summary=evaluation['summary']
        if baseline is None:baseline=summary
        record=dict(recipe=recipe,validation=summary,value=robust_value(summary),
                    eligible=eligible(summary,baseline,.005),elapsed_seconds=time.monotonic()-started)
        records.append(record)
        if record['eligible'] and (best is None or record['value']>best['value']):best=record
        atomic_json(out/'validation.json',dict(records=records,best=best))
        print(json.dumps(dict(recipe=recipe,value=record['value'],eligible=record['eligible'],
                              validation=summarize(evaluation),elapsed_seconds=record['elapsed_seconds'])),flush=True)
        del evaluation,er,rr
    selection=dict(selected=best,baseline=baseline,registry_sha256=sha256(out/'registry.json'),
                   training_bank_sha256=sha256(out/'training_bank.pt'),test_used_for_selection=False,
                   selected_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'selection.json',selection)
    if not args.evaluate_test:return
    # Freeze selection before opening official test labels or producing test predictions.
    del e,r,transported_e,transported_r,be,br,means,blocks,masks,ek,rk
    test=RUN/f'features_test_{args.split}'
    tc,be,br,means,blocks,masks=load_features(test,device)
    te_keys=bank['enzyme_keys'].to(device);tr_keys=bank['reaction_keys'].to(device)
    transported_e=transport_embeddings(centered_unit(means,model.density.protein_center),te_keys,ev)
    transported_r=transport_embeddings(reaction_keys(model,blocks,masks),tr_keys,rv)
    e=model.encode_enzymes(be,means,batch_size=256)
    r=model.encode_reactions(br,blocks,masks,batch_size=256)
    recipe=best['recipe']
    er=refine_phase2(e,transported_e,recipe['enzyme_strength'],alpha=model.alpha)
    rr=refine_phase2(r,transported_r,recipe['reaction_strength'],alpha=model.alpha)
    score=canonical_dot(rr,er);base_score=canonical_dot(r,e)
    np.savez(out/'test_scores.npz',selected=score.cpu().numpy(),baseline_phase2=base_score.cpu().numpy())
    provenance={}
    checked_catalog,edges=checked_official_truth(test,args.split,spec['frozen_recipe']['sha256'],provenance)
    if checked_catalog!=tc:raise ValueError('Test catalog changed during prediction')
    truth=dict(reaction_index=edges[:,0],enzyme_index=edges[:,1])
    selected=evaluate_scores(score,truth);original=evaluate_scores(base_score,truth)
    atomic_json(out/'test_summary.json',dict(split=args.split,selection_sha256=sha256(out/'selection.json'),
        test_used_for_selection=False,test_results_exploratory=True,recipe=recipe,
        selected=selected['summary'],baseline_phase2=original['summary'],
        candidate_proteins=len(tc['proteins']),candidate_reactions=len(tc['reactions']),
        scores_sha256=sha256(out/'test_scores.npz'),truth_provenance=provenance))
    print(json.dumps(dict(test=summarize(selected),baseline=summarize(original))),flush=True)


if __name__=='__main__':main()
