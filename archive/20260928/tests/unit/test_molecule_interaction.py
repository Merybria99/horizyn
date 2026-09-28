import copy

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
import yaml

from horizyn.model import MoleculeSetInteractionPooling, MultimodalReactionAttentionEncoder
from scripts.run_circe_molecule_interaction import make_config


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_permutation_invariance_and_padding():
    torch.manual_seed(42)
    model=MoleculeSetInteractionPooling(8).eval()
    x=torch.randn(3,5,8)
    mask=torch.tensor([[1,1,1,0,0],[1,0,1,1,1],[1,1,1,1,1]],dtype=torch.bool)
    order=torch.tensor([4,1,3,0,2])
    expected,weights=model(x,mask,True)
    actual,shuffled_weights=model(x[:,order],mask[:,order],True)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(shuffled_weights,weights[:,order],atol=1e-6,rtol=1e-5)
    padded=x.masked_fill(~mask[...,None],float('nan'))
    torch.testing.assert_close(model(padded,mask),expected)
    assert not weights[~mask].any()


def test_empty_sets_are_finite_and_gradients_flow():
    model=MoleculeSetInteractionPooling(8)
    x=torch.randn(2,3,8,requires_grad=True)
    output,weights=model(x,torch.tensor([[1,1,0],[0,0,0]],dtype=torch.bool),True)
    assert torch.isfinite(output).all() and not output[1].any() and not weights[1].any()
    output.square().sum().backward()
    assert model.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert x.grad[0,:2].abs().sum()>0 and not x.grad[1].any()
    assert torch.isfinite(model(torch.empty(2,0,8))).all()


def small_encoder(**overrides):
    options=dict(input_dim=16,output_dim=16,unimol_dim=8, widths=[16],
                 use_reaction_model=False,use_chienn=False,use_reaction_chemistry=False,
                 side_composition='molecule_set',reaction_pooling='interaction',
                 separate_side_poolers=False,modality_fusion='mean')
    options.update(overrides)
    return MultimodalReactionAttentionEncoder(**options)


def test_encoder_has_one_modality_and_ignores_disabled_inputs():
    encoder=small_encoder().eval()
    x=torch.randn(3,4,8)
    options=dict(reactant_embeddings=x,product_embeddings=x,has_unimol2=torch.tensor([1,1,0]))
    output=encoder(**options)
    changed=encoder(**options,reaction_embedding=torch.full((3,100),float('nan')),
                    reaction_chemistry_vector=torch.full((3,12),float('nan')))
    torch.testing.assert_close(output,changed)
    assert torch.isfinite(output).all()
    assert tuple(encoder.modality_names)==('unimol2',)
    assert encoder.reaction_projection is None and encoder.chienn_projection is None
    assert encoder.reaction_chemistry_projection is None
    assert encoder.unimol_reactant_pooling is encoder.unimol_product_pooling


@pytest.mark.parametrize('option',[
    {'use_reaction_model':True},{'use_chienn':True},{'use_reaction_chemistry':True,'reaction_chemistry_dim':10},
    {'side_composition':'directional_delta'},
])
def test_interaction_rejects_accidental_hybrid_inputs(option):
    with pytest.raises(ValueError,match='UniMol2-only'):small_encoder(**option)


@pytest.mark.parametrize('devices',[1,2,3,4])
def test_generated_configuration_preserves_control_recipe(tmp_path,devices):
    from horizyn.config import DotDict,validate_config
    base=dict(seed=42,data=dict(train_pairs_path='train.csv',test_pairs_path='validation.csv',
        train_reactions_path='train_rxns.csv',test_reactions_path='validation_rxns.csv',
        protein_residue_embeds_path='proteins.h5',train_batch_size=512,
        reaction_representation='multimodal_reaction_attention',reaction_unimol2_embeds_path='unimol.h5',
        train_reaction_t5v2_embeds_path='t5.h5'),
        model=dict(name='ProteinPooledDualModel',query_encoder_type='multimodal_reaction_attention',
            query_encoder_dims=[512,4096,512],target_encoder_dims=[512,512],embedding_dim=512,
            reaction_attention_pooling={'separate_side_poolers':True},
            enzyme_block_fusion={'dims':{'core':512},'weights':{'core':1.0}},
            sleec_pooling={'freeze_scorer':True}),
        training=dict(max_epochs=30,precision='32-true',loss=dict(name='DecoupledAllPositiveInfoNCELoss',
            positive_pair_source='all_known_in_batch',unknown_negative_weight=.5,biofp_aux_weight=0)),
        logging={'wandb':{'enabled':False}})
    before=copy.deepcopy(base)
    cfg=make_config(base,tmp_path,devices,864)
    assert base==before
    validate_config(DotDict(cfg))
    assert cfg['data']['train_batch_size']*devices==1536
    assert cfg['training']['loss']==base['training']['loss']
    assert cfg['model']['enzyme_block_fusion']==base['model']['enzyme_block_fusion']
    assert cfg['model']['sleec_pooling']==base['model']['sleec_pooling']
    assert cfg['training']['precision']==base['training']['precision']
    for name in ('model','chiro','chemistry','directional'):
        assert cfg['data'][f'reaction_use_{name}'] is False
    assert cfg['data']['train_reaction_t5v2_embeds_path'] is None
    assert cfg['training']['init_from_checkpoint'] is None
    assert cfg['training']['max_steps']==864


def test_two_training_steps_checkpoint_reload_and_real_evaluator(tmp_path):
    """Exercise ordinary CSV data, all-known positives, both towers and evaluation."""
    import h5py
    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    from scripts.evaluate_protein_pooling import evaluate_checkpoint,CONFIGURED_FORWARD_CANDIDATES
    rng=np.random.default_rng(42)
    reactions=tmp_path/'reactions.csv';pairs=tmp_path/'pairs.csv'
    reactions.write_text('reaction_id,reaction_smiles\nr0,CCO.CCN\nr1,CCC.CCO\n')
    pairs.write_text('pr_id,reaction_id,protein_id\n0,r0,p0\n1,r1,p1\n2,r0,p2\n3,r1,p3\n')
    with h5py.File(tmp_path/'proteins.h5','w') as f:
        f['ids']=np.array(['p0','p1','p2','p3'],dtype='S');f['offsets']=np.arange(0,17,4)
        f['vectors']=rng.normal(size=(16,8)).astype('float32')
    with h5py.File(tmp_path/'unimol.h5','w') as f:
        # r1 deliberately missing: preserve its positives and candidate entry.
        f['ids']=np.array(['r0_f'],dtype='S')
        for side in ('reactant','product'):
            f[f'{side}_offsets']=np.array([0,2])
            f[f'{side}_vectors']=rng.normal(size=(2,8)).astype('float32')
    data=dict(train_pairs_path=str(pairs),test_pairs_path=str(pairs),
              train_reactions_path=str(reactions),test_reactions_path=str(reactions),
              protein_residue_embeds_path=str(tmp_path/'proteins.h5'),residue_dim=8,
              reaction_representation='multimodal_reaction_attention',
              reaction_unimol2_embeds_path=str(tmp_path/'unimol.h5'),reaction_unimol_dim=8,
              reaction_use_model=False,reaction_use_chiro=False,reaction_use_chemistry=False,
              reaction_allow_missing_unimol2=True,normalize_molecule_sets_as_self_reactions=True,
              reaction_direction_mode='forward_only',train_batch_size=4,num_workers=0,
              standardize_reactions=False,validation_enabled=False)
    dm=ReactionConditionedDataModule(**data)
    model=ProteinPooledLitModule(query_encoder_dims=[16,16],target_encoder_dims=[8,16],
          embedding_dim=16,residue_dim=8,pooling='mean',
          query_encoder_type='multimodal_reaction_attention',reaction_unimol_dim=8,
          reaction_use_model=False,reaction_use_chienn=False,reaction_use_chemistry=False,
          reaction_side_composition='molecule_set',reaction_pooling='interaction',
          reaction_separate_side_poolers=False,reaction_modality_fusion='mean',
          reaction_modality_encoder_widths=[16],loss_name='DecoupledAllPositiveInfoNCELoss',
          positive_pair_source='all_known_in_batch',unknown_negative_weight=.5,
          biofp_aux_weight=0,log_attention_stats=False)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_steps=2,max_epochs=2,
                       logger=False,enable_checkpointing=False,enable_progress_bar=False,
                       enable_model_summary=False,limit_val_batches=0)
    trainer.fit(model,datamodule=dm)
    assert trainer.global_step==2 and torch.isfinite(trainer.callback_metrics['train/loss'])
    ckpt=tmp_path/'model.ckpt';trainer.save_checkpoint(ckpt)
    restored=ProteinPooledLitModule.load_from_checkpoint(ckpt,map_location='cpu',weights_only=False)
    assert restored.hparams.reaction_pooling=='interaction'
    config={'data':data,'model':{'name':'ProteinPooledDualModel','query_encoder_dims':[16,16],
            'query_encoder_type':'multimodal_reaction_attention',
            'target_encoder_dims':[8,16],'embedding_dim':16,'reaction_use_chiro':False},
            'training':{'max_epochs':2}}
    ids=tmp_path/'candidates.txt';ids.write_text('p0\np1\np2\np3\n')
    config['data']['validation_retrieval_candidate_ids_path']=str(ids)
    path=tmp_path/'validation.yaml';path.write_text(yaml.safe_dump(config))
    result=evaluate_checkpoint(str(ckpt),str(path),'cpu',2,2,False,direction='both',
                              evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
                              per_query_output=str(tmp_path/'queries.json'))
    assert result['num_reaction_candidates']==2 and result['num_enzyme_candidates']==4
    assert (tmp_path/'queries.json').exists()
