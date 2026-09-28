#!/usr/bin/env python3
"""Focused contracts plus a real full-batch forward/backward for both pilot arms."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


class PairDataset:
    def __init__(self, rows):
        self.keys=list(range(len(rows)))
        self.tuple_dataset={i:dict(query_id=q,target_id=p) for i,(q,p) in enumerate(rows)}
    def __len__(self):return len(self.keys)


class Contracts(unittest.TestCase):
    def test_all_known_positive_mask(self):
        from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
        fake=SimpleNamespace(positive_pair_source='all_known_in_batch',
            _VALID_POSITIVE_PAIR_SOURCES={'all_known_in_batch','observed_pairs'},device=torch.device('cpu'),
            trainer=SimpleNamespace(datamodule=SimpleNamespace(_train_query_to_targets={'r1':{'p1','p2'},'r2':{'p2'}})))
        q,t=ProteinPooledLitModule._positive_pair_indices(fake,['r1','r2'],['p1','p2'],['r1','r2'],['p1','p2'])
        self.assertEqual(set(zip(q.tolist(),t.tolist())),{(0,0),(0,1),(1,1)})

    def test_loss_protects_all_positives_and_balances_reactions(self):
        from horizyn.losses import DecoupledAllPositiveInfoNCELoss
        loss=DecoupledAllPositiveInfoNCELoss(beta=2.,lambda_r2e=1.,lambda_e2r=0.,unknown_negative_weight=.5)
        distances=torch.tensor([[.1,.3,1.1],[.8,.2,1.3]],requires_grad=True)
        q=torch.tensor([0,0,1]);p=torch.tensor([0,1,1])
        value=loss(distances,q,p)
        logits=-2*distances
        row0=(torch.nn.functional.softplus(torch.log(.5*logits[0,2].exp())-logits[0,:2])).mean()
        row1=torch.nn.functional.softplus(torch.log(.5*(logits[1,0].exp()+logits[1,2].exp()))-logits[1,1])
        torch.testing.assert_close(value,(row0+row1)/2)
        value.backward()
        self.assertTrue((distances.grad[q,p]>0).all())
        duplicate=loss(distances.detach(),torch.tensor([0,0,0,1]),torch.tensor([0,1,1,1]))
        torch.testing.assert_close(value.detach(),duplicate)

    def test_sampler_keeps_structured_anchors_after_first_cycle(self):
        from horizyn.reaction_conditioned_data_module import DirectionalHardNegativeBatchSampler
        rows=[(f'r{i%3}_f',f'p{i}') for i in range(90)]
        dataset=PairDataset(rows)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pools.json'
            path.write_text(json.dumps({'r0':['p1','p2'],'r1':['p0','p2'],'r2':['p0','p1']}))
            sampler=DirectionalHardNegativeBatchSampler(dataset,batch_size=8,hard_negative_pools_path=path,
                anchor_queries_per_batch=1,positives_per_query=1,negatives_per_query=2,seed=42)
            anchors=[];batches=[]
            for batch in sampler:
                self.assertEqual(len(set(batch)),8)
                self.assertGreaterEqual(sampler.last_statistics['structured_rows'],3)
                anchors.extend(sampler.last_statistics['anchors']);batches.append(batch)
            counts=Counter(anchors)
            self.assertLessEqual(max(counts.values())-min(counts.values()),1)
            sampler.set_epoch(0)
            self.assertEqual(batches,list(sampler))
            for q in sampler.anchor_queries:
                for row in sampler._build_anchor_rows(q,__import__('random').Random(1))[1:]:
                    target=dataset.tuple_dataset[row]['target_id']
                    self.assertNotIn(target,sampler.query_positive_targets[q])

    def test_directional_features_change_on_reversal(self):
        from horizyn.model import MultimodalReactionAttentionEncoder
        r=torch.tensor([[1.,3.]]);p=torch.tensor([[4.,2.]])
        forward=MultimodalReactionAttentionEncoder._compose_side_pair(r,p)
        reverse=MultimodalReactionAttentionEncoder._compose_side_pair(p,r)
        self.assertFalse(torch.equal(forward,reverse))
        torch.testing.assert_close(forward[:,4:6],-reverse[:,4:6])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/enzymecage_f3_combined_20260918'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output.resolve()
    sys.path.insert(0,str(out/'code'))
    torch.set_num_threads(4);torch.manual_seed(42)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Contract tests failed')
    from horizyn.config import load_config
    from horizyn.training_options import reaction_data_module_kwargs,protein_pooling_model_kwargs
    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    from horizyn.training_warm_start import _load_partial_model_warm_start
    import lightning.pytorch as pl
    evidence={}
    for name in ['control','combined']:
        config=load_config(out/name/'configs/train.yaml')
        dm=ReactionConditionedDataModule(**reaction_data_module_kwargs(config))
        dm.setup('fit')
        module=ProteinPooledLitModule(**protein_pooling_model_kwargs(config))
        _load_partial_model_warm_start(module,config.training.init_from_checkpoint)
        trainer=pl.Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
        trainer.datamodule=dm;module.trainer=trainer
        module.to(args.device);module.train()
        loader=dm.train_dataloader();batch=next(iter(loader))
        batch=module.transfer_batch_to_device(batch,torch.device(args.device),0)
        queries=list(dict.fromkeys(batch['query_id']));targets=list(dict.fromkeys(batch['target_id']))
        q,t=module._positive_pair_indices(queries,targets,list(batch['query_id']),list(batch['target_id']))
        observed=len(set(zip(batch['query_id'],batch['target_id'])))
        loss,size,attention,components=module._compute_full_batch_loss(batch,return_attention_stats=True)
        if not torch.isfinite(loss):raise ValueError('Nonfinite real batch loss')
        loss.backward()
        gradient_names=[]
        for key,value in module.named_parameters():
            if value.grad is not None:
                if not torch.isfinite(value.grad).all():raise ValueError(f'Nonfinite gradient: {key}')
                gradient_names.append(key)
        directional={}
        if name=='combined':
            for key in ['model.query_encoder.unimol_projection.main_nn.0.weight','model.query_encoder.chienn_projection.main_nn.0.weight']:
                parameter=dict(module.named_parameters())[key];width=parameter.shape[1]//4
                norm=float(parameter.grad[:,2*width:].norm().item());directional[key]=norm
                if norm<=0:raise ValueError(f'Directional difference weights receive no gradient: {key}')
            known={(queries[i],targets[j]) for i,j in zip(q.tolist(),t.tolist())}
            expected={(qid,pid) for qid in queries for pid in dm._train_query_to_targets.get(qid,[]) if pid in set(targets)}
            if known!=expected:raise ValueError('Real batch mask misses known training associations')
        evidence[name]=dict(loss=float(loss.detach()),batch_rows=size,unique_reactions=len(queries),unique_enzymes=len(targets),
            observed_positive_pairs=observed,numerator_positive_pairs=len(q),finite_gradient_tensors=len(gradient_names),
            directional_difference_gradient_norms=directional)
        print('REAL_BATCH',name,json.dumps(evidence[name]),flush=True)
        del module,dm,trainer,loader,batch,loss;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    (out/'preflight.json').write_text(json.dumps(dict(passed=True,contract_tests=result.testsRun,real_batches=evidence),indent=2)+'\n')


if __name__=='__main__':main()
