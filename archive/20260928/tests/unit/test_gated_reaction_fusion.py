import copy
from pathlib import Path

import pytest
import torch
import yaml

from horizyn.gated_reaction_fusion import GatedReactionFusion
from horizyn.model import MultimodalReactionAttentionEncoder
from scripts.run_circe_gated_fusion import make_config


@pytest.fixture(autouse=True)
def limit_threads():
    original = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def encoder(variant='gated_attention_concat'):
    return MultimodalReactionAttentionEncoder(input_dim=16, output_dim=12, widths=[24],
        num_layers=1, reaction_model_dim=10, unimol_dim=8, chienn_dim=6,
        reaction_pooling='interaction', side_composition='molecule_set',
        separate_side_poolers=False, modality_token_layer_norm=True,
        modality_fusion=variant)


def inputs():
    return dict(reactant_embeddings=torch.randn(3,4,8), product_embeddings=torch.zeros(3,1,8),
        reactant_padding_mask=torch.tensor([[False,False,True,True]]*3),
        reaction_embedding=torch.randn(3,10), has_unimol2=torch.tensor([True,True,False]),
        reactant_chirality_embeddings=torch.randn(3,4,6), product_chirality_embeddings=torch.zeros(3,1,6),
        reactant_chirality_padding_mask=torch.tensor([[False,False,True,True]]*3),
        has_chiro=torch.tensor([True,False,True]))


def test_masking_all_missing_and_gradients():
    model = GatedReactionFusion(16, 3)
    tokens = torch.randn(3,3,16, requires_grad=True)
    mask = torch.tensor([[True,True,True],[True,False,True],[False,False,False]])
    poisoned = tokens.masked_fill(~mask[...,None], float('nan'))
    output, info = model(poisoned, mask, True)
    assert torch.isfinite(output).all()
    assert output[~mask].eq(0).all()
    assert info['cross_modal_gates'][~mask].eq(0).all()
    torch.testing.assert_close(info['cross_modal_gates'][mask], torch.full_like(info['cross_modal_gates'][mask], .1))
    assert info['cross_modal_attention'][1,:,:,1].eq(0).all()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('variant',['concat','gated_attention_concat'])
def test_encoder_permutation_masks_and_checkpoint(variant):
    torch.manual_seed(5)
    model = encoder(variant).eval(); batch = inputs()
    expected, info = model(**batch, return_attention=True)
    perm = torch.tensor([2,0,3,1])
    altered = {k:v.clone() for k,v in batch.items()}
    for prefix in ('reactant', 'reactant_chirality'):
        altered[f'{prefix}_embeddings'] = altered[f'{prefix}_embeddings'][:,perm]
        altered[f'{prefix}_padding_mask'] = altered[f'{prefix}_padding_mask'][:,perm]
        altered[f'{prefix}_embeddings'].masked_fill_(altered[f'{prefix}_padding_mask'][...,None], float('nan'))
    altered['reactant_embeddings'][2] = float('nan')
    altered['reactant_chirality_embeddings'][1] = float('nan')
    actual = model(**altered)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert info['modality_weights_are_availability']
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3))
    actual.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored = encoder(variant).eval(); restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(**batch), expected)


def test_shared_initialization_and_rng_are_matched():
    torch.manual_seed(42); control = encoder('concat'); after_control = torch.randn(10)
    torch.manual_seed(42); gated = encoder(); after_gated = torch.randn(10)
    torch.testing.assert_close(after_control, after_gated)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[name], value)


def test_config_preserves_enzyme_and_loss(tmp_path):
    from horizyn.config import DotDict
    from horizyn.config_validation import _validate_reaction_attention
    base = dict(seed=42, data={}, model=dict(target_encoder_dims=[512,512],
        query_encoder_type='multimodal_reaction_attention',
        reaction_attention_pooling={'separate_side_poolers':True},
        enzyme_block_fusion={'dims':{'core':512},'weights':{'core':1.0}},
        sleec_pooling={'freeze_scorer':True}),
        training=dict(precision='32-true',loss=dict(name='DecoupledAllPositiveInfoNCELoss',
            positive_pair_source='all_known_in_batch',unknown_negative_weight=.5,biofp_aux_weight=0)),
        logging={'wandb':{'enabled':False}})
    before = copy.deepcopy(base)
    for devices in (1,2,3,4):
        for variant in ('concat','gated'):
            cfg = make_config(base,tmp_path,tmp_path/'t5.h5',devices,864,variant)
            _validate_reaction_attention(DotDict(cfg))
            assert cfg['training']['loss'] == base['training']['loss']
            assert cfg['model']['enzyme_block_fusion'] == base['model']['enzyme_block_fusion']
            assert cfg['model']['sleec_pooling'] == base['model']['sleec_pooling']
            assert cfg['model']['target_encoder_dims'] == base['model']['target_encoder_dims']
            assert cfg['training']['precision'] == base['training']['precision']
            assert cfg['data']['train_batch_size']*devices == 1536
            assert cfg['data']['reaction_use_chiro'] and cfg['data']['reaction_use_model']
            assert not cfg['data']['reaction_use_chemistry'] and not cfg['data']['reaction_use_directional']
    assert base == before


@pytest.mark.parametrize('variant', ['concat','gated_attention_concat'])
def test_training_checkpoint_reload_and_evaluation(tmp_path, variant):
    import h5py
    import numpy as np
    import lightning.pytorch as pl
    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    from scripts.evaluate_protein_pooling import evaluate_checkpoint, CONFIGURED_FORWARD_CANDIDATES
    rng = np.random.default_rng(4)
    (tmp_path/'reactions.csv').write_text('reaction_id,reaction_smiles\nr0,CCO.CCN\nr1,CCC.CCO\n')
    (tmp_path/'pairs.csv').write_text('pr_id,reaction_id,protein_id\n0,r0,p0\n1,r1,p1\n2,r0,p2\n3,r1,p3\n')
    with h5py.File(tmp_path/'proteins.h5','w') as f:
        f['ids']=np.array(['p0','p1','p2','p3'],dtype='S');f['offsets']=np.arange(0,17,4)
        f['vectors']=rng.normal(size=(16,8)).astype('float32')
    for name,dim in [('unimol',8),('chiro',6)]:
        with h5py.File(tmp_path/f'{name}.h5','w') as f:
            f['ids']=np.array(['r0_f'],dtype='S')
            for side in ('reactant','product'):
                f[f'{side}_offsets']=np.array([0,2]); f[f'{side}_vectors']=rng.normal(size=(2,dim)).astype('float32')
    with h5py.File(tmp_path/'t5.h5','w') as f:
        f['ids']=np.array(['r0_f','r1_f'],dtype='S'); f['vectors']=rng.normal(size=(2,10)).astype('float32')
    data=dict(train_pairs_path=str(tmp_path/'pairs.csv'),test_pairs_path=str(tmp_path/'pairs.csv'),
        train_reactions_path=str(tmp_path/'reactions.csv'),test_reactions_path=str(tmp_path/'reactions.csv'),
        protein_residue_embeds_path=str(tmp_path/'proteins.h5'),residue_dim=8,
        reaction_representation='multimodal_reaction_attention',
        reaction_unimol2_embeds_path=str(tmp_path/'unimol.h5'),reaction_unimol_dim=8,
        reaction_chiro_embeds_path=str(tmp_path/'chiro.h5'),reaction_chiro_dim=6,
        reaction_t5v2_embeds_path=str(tmp_path/'t5.h5'),reaction_model_dim=10,
        reaction_use_model=True,reaction_use_chiro=True,reaction_use_chemistry=False,
        reaction_allow_missing_unimol2=True,reaction_allow_missing_chiro=True,
        normalize_molecule_sets_as_self_reactions=True,reaction_direction_mode='forward_only',
        train_batch_size=4,num_workers=0,standardize_reactions=False,validation_enabled=False)
    dm=ReactionConditionedDataModule(**data)
    model=ProteinPooledLitModule(query_encoder_dims=[16,16],target_encoder_dims=[8,16],
        embedding_dim=16,residue_dim=8,pooling='mean',query_encoder_type='multimodal_reaction_attention',
        reaction_unimol_dim=8,reaction_model_dim=10,reaction_chienn_dim=6,
        reaction_use_model=True,reaction_use_chienn=True,reaction_use_chemistry=False,
        reaction_side_composition='molecule_set',reaction_pooling='interaction',
        reaction_separate_side_poolers=False,reaction_modality_fusion=variant,
        reaction_modality_encoder_widths=[16],loss_name='DecoupledAllPositiveInfoNCELoss',
        positive_pair_source='all_known_in_batch',unknown_negative_weight=.5,biofp_aux_weight=0,
        log_attention_stats=True,attention_logging_interval=1)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_steps=2,max_epochs=2,logger=False,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0)
    trainer.fit(model,datamodule=dm)
    assert torch.isfinite(trainer.callback_metrics['train/loss'])
    if variant=='gated_attention_concat':
        assert any('reaction_fusion/gate_' in key for key in trainer.callback_metrics)
    checkpoint=tmp_path/'last.ckpt';trainer.save_checkpoint(checkpoint)
    restored=ProteinPooledLitModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False)
    assert restored.hparams.reaction_modality_fusion==variant
    candidates=tmp_path/'candidates.txt'; candidates.write_text('p0\np1\np2\np3\n')
    data['validation_retrieval_candidate_ids_path']=str(candidates)
    config={'data':data,'model':{'name':'ProteinPooledDualModel','query_encoder_dims':[16,16],
        'target_encoder_dims':[8,16],'embedding_dim':16,'query_encoder_type':'multimodal_reaction_attention',
        'reaction_use_chiro':True},'training':{'max_epochs':2}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    result=evaluate_checkpoint(str(checkpoint),str(tmp_path/'config.yaml'),'cpu',2,2,False,
        direction='both',evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
        per_query_output=str(tmp_path/'queries.json'))
    assert result['num_reaction_candidates']==2 and result['num_enzyme_candidates']==4
