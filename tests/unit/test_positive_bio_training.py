"""Loss neutrality, sparse data, distributed reduction and actual training plumbing."""
import copy

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from horizyn.capability.enzyme_capability_dataset import BioFPTargetDataset, TargetWithBioFPTargetDataset
from horizyn.config_validation import _validate_enzyme_inputs
from horizyn.positive_bio_loss import positive_anchor_loss
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.training_options import protein_pooling_model_kwargs
from tests.unit.test_enzyme_multiview_integration import TinyData, batch, config, make_module, scorer_checkpoint


LABELS = {'ec':['1.1.1.1','2.1.1.1'], 'cofactor':['NAD_NADP','PLP'], 'mechanism':['redox','transfer']}


def annotations(rows=4):
    result = {}
    for family in LABELS:
        result[f'biofp_{family}_positive_indices'] = torch.tensor([[0,1],[-1,-1],[1,-1],[0,-1]])[:rows]
        result[f'biofp_{family}_confidence'] = torch.tensor([[1.,.5],[0.,0.],[1.,0.],[1.,0.]])[:rows]
    return result


def test_positive_loss_missing_entries_are_neutral():
    distance = torch.tensor([[.2,.8],[.4,.6],[.5,.3]], requires_grad=True)
    targets = {'biofp_ec_positive_indices':torch.tensor([[0,1],[-1,-1],[1,-1]]),
               'biofp_ec_confidence':torch.tensor([[1.,3.],[0.,0.],[1.,0.]])}
    loss, metrics = positive_anchor_loss({'biofp_alignment_ec':distance}, targets, {'ec':1.})
    assert loss.item() == pytest.approx((.65+.3)/2)
    loss.backward()
    torch.testing.assert_close(distance.grad, torch.tensor([[.125,.375],[0.,0.],[0.,.5]]))
    assert metrics['biofp_ec_active'].item() == 2
    assert metrics['biofp_ec_active_labels'].item() == 3


def test_inactive_families_do_not_dilute_active_ones_and_all_missing_backpropagates():
    a, b = torch.ones(2,2,requires_grad=True), torch.ones(2,2,requires_grad=True)
    targets = {'biofp_ec_positive_indices':torch.tensor([[0],[-1]]),
               'biofp_ec_confidence':torch.tensor([[1.],[0.]]),
               'biofp_cofactor_positive_indices':torch.full((2,1),-1),
               'biofp_cofactor_confidence':torch.zeros(2,1)}
    details = {'biofp_alignment_ec':a, 'biofp_alignment_cofactor':b}
    loss,_ = positive_anchor_loss(details,targets,{'ec':1.,'cofactor':1.})
    assert loss.item() == 1
    loss.backward(); assert torch.count_nonzero(b.grad) == 0
    targets['biofp_ec_positive_indices'].fill_(-1)
    targets['biofp_ec_confidence'].zero_()
    a.grad=None; b.grad=None
    loss,_ = positive_anchor_loss(details,targets,{'ec':1.,'cofactor':1.})
    assert loss.item() == 0
    loss.backward(); assert torch.count_nonzero(a.grad) == torch.count_nonzero(b.grad) == 0


@pytest.mark.parametrize('bad', [-2,2])
def test_invalid_sparse_indices_rejected(bad):
    with pytest.raises(ValueError,match='Out-of-range'):
        positive_anchor_loss({'biofp_alignment_ec':torch.ones(1,2)},
            {'biofp_ec_positive_indices':torch.tensor([[bad]]),'biofp_ec_confidence':torch.ones(1,1)}, {'ec':1.})


def test_sparse_dataset_and_unannotated_candidate(tmp_path):
    path=tmp_path/'positive.npz'
    np.savez_compressed(path,ids=np.array(['p0']),ec_positive_indices=np.array([[0,-1]]),
                        ec_confidence=np.array([[1.,0.]]),ec_vocab_size=np.array(2))
    dataset=BioFPTargetDataset(path)
    class Candidates:
        keys=['p0','unknown']
        def __getitem__(self,key): return {'target_vec':torch.ones(3)}
    attached=TargetWithBioFPTargetDataset(Candidates(),dataset)
    assert attached.missing_count == 1
    assert attached['p0']['biofp_ec_positive_indices'].tolist() == [0,-1]
    assert attached['unknown']['biofp_ec_positive_indices'].tolist() == [-1,-1]
    assert attached['unknown']['biofp_ec_confidence'].sum() == 0


def test_configuration_forward_and_checkpoint(scorer_checkpoint,tmp_path):
    value=config(scorer_checkpoint)
    value.model['biofp']={'positive_labels':LABELS}
    value.data.update(protein_biofp_targets_path='positive.npz',protein_biofp_vocab_path='positive.json')
    value.training.loss.update(biofp_aux_mode='positive_anchor',biofp_aux_weight=.02,
        biofp_normalize_active_families=True,biofp_family_weights={x:1. for x in LABELS})
    _validate_enzyme_inputs(value)
    options=protein_pooling_model_kwargs(value)
    assert options['biofp_aux_mode']=='positive_anchor' and options['biofp_positive_labels']==LABELS
    model=ProteinPooledLitModule(**options)
    model._biofp_auxiliary_loss(pooling_details={'biofp_alignment_'+f:torch.ones(4,2) for f in LABELS},
                              biofp_targets=annotations())
    payload={'state_dict':model.state_dict(),'hyper_parameters':dict(model.hparams),'pytorch-lightning_version':pl.__version__}
    path=tmp_path/'model.ckpt'; torch.save(payload,path)
    reloaded=ProteinPooledLitModule.load_from_checkpoint(path, map_location='cpu',sleec_checkpoint_path=None)
    assert reloaded.hparams.biofp_aux_mode=='positive_anchor'
    for name,tensor in model.state_dict().items(): torch.testing.assert_close(tensor,reloaded.state_dict()[name])


def test_cpu_training_logs_positive_loss_with_attention_logging_off(scorer_checkpoint,tmp_path):
    module=make_module(scorer_checkpoint,biofp_aux_mode='positive_anchor',biofp_aux_weight=.02,
        biofp_positive_labels=LABELS,biofp_family_weights={x:1. for x in LABELS},
        biofp_normalize_active_families=True,biofp_aux_warmup_epochs=0,
        positive_pair_source='all_known_in_batch')
    sample=batch(); sample.update(annotations())
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_steps=2,max_epochs=1,
        logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
        num_sanity_val_steps=0,default_root_dir=tmp_path)
    before=copy.deepcopy(module.model.multiview_encoder.state_dict())
    trainer.fit(module,datamodule=TinyData(sample))
    assert any('weighted_biofp' in key for key in trainer.callback_metrics)
    assert any(not torch.equal(before[k],v) for k,v in module.model.multiview_encoder.state_dict().items() if 'readout' in k and 'anchors' not in k)


def _ddp_worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1,1,bias=False))
        with torch.no_grad(): model.module.weight.fill_(1)
        distances=model(torch.tensor([[2.]]))
        loss,counts=positive_anchor_loss({'biofp_alignment_ec':distances},
            {'biofp_ec_positive_indices':torch.tensor([[0 if rank==0 else -1]]),
             'biofp_ec_confidence':torch.tensor([[1. if rank==0 else 0.]])}, {'ec':1.})
        assert loss.item()==2 and counts['biofp_ec_active'].item()==1
        loss.backward()
        torch.testing.assert_close(model.module.weight.grad,torch.tensor([[2.]]))
    finally: dist.destroy_process_group()


def test_gradient_diagnostics_separate_families_sharing_slots(scorer_checkpoint):
    module=make_module(scorer_checkpoint)
    module.biofp_family_weights={'ec':1.,'cofactor':1.}
    shared=torch.tensor([1.,2.],requires_grad=True)
    retrieval=shared[0]
    ec,cofactor=shared[0]*2,shared[1]*3
    result=module._biofp_gradient_diagnostics(retrieval,ec+cofactor,
        {'biofp_shared_ec':shared,'biofp_shared_cofactor':shared},.02,
        family_losses={'weighted_biofp_ec':ec,'weighted_biofp_cofactor':cofactor})
    assert result['biofp_ec_gradient_cosine']==1
    assert result['biofp_cofactor_gradient_cosine']==0
    assert result['biofp_ec_gradient_ratio'].item()==pytest.approx(.04)
    assert result['biofp_cofactor_gradient_ratio'].item()==pytest.approx(.06)


def test_distributed_empty_rank_matches_global_mean(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)
