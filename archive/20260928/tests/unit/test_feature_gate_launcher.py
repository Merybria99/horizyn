import copy

import pytest

from scripts.run_circe_feature_gate import make_config, verify


@pytest.mark.parametrize('devices', [1, 2, 3, 4])
@pytest.mark.parametrize('enzyme_tower', ['baseline', 'multiview'])
def test_matched_configs_preserve_reference_recipe(tmp_path, devices, enzyme_tower):
    from horizyn.config import DotDict, validate_config
    base = dict(seed=42, data=dict(train_pairs_path='train.csv', train_reactions_path='train_rxns.csv',
        validation_pairs_path='val.csv', validation_reactions_path='val_rxns.csv',
        protein_residue_embeds_path='proteins.h5', reaction_representation='multimodal_reaction_attention',
        reaction_unimol2_embeds_path='unimol.h5', reaction_chiro_embeds_path='chiro.h5',
        train_reaction_t5v2_embeds_path='t5.h5'),
        model=dict(name='ProteinPooledDualModel', query_encoder_type='multimodal_reaction_attention',
            query_encoder_dims=[512,4096,4096,512], target_encoder_dims=[512,512], embedding_dim=512,
            reaction_multimodal_attention=dict(modality_encoder_widths=[4096,4096]),
            enzyme_input_mode='raw_mean_sleec_biological_factorized',
            enzyme_block_fusion={'dims':{'core':256,'site':256},'weights':{'core':.5,'site':.5}},
            sleec_pooling={'freeze_scorer':True,'checkpoint_path':'sleec.ckpt'}, biofp={'hidden_dim':512}),
        training=dict(precision='32-true',learning_rate=1e-4,weight_decay=.01,
            loss=dict(name='DecoupledAllPositiveInfoNCELoss', positive_pair_source='all_known_in_batch',
                      unknown_negative_weight=.5,biofp_aux_weight=0)), logging={'wandb':{'enabled':False}})
    original = copy.deepcopy(base)
    configs = [make_config(base, tmp_path, tmp_path/'t5.h5', devices, 30, variant, enzyme_tower)
               for variant in ('feature', 'scalar')]
    assert base == original
    for config in configs:
        validate_config(DotDict(config))
        assert config['model']['query_encoder_dims'] == base['model']['query_encoder_dims']
        for key in ('target_encoder_dims', 'sleec_pooling'):
            assert config['model'][key] == base['model'][key]
        if enzyme_tower == 'baseline':
            for key in ('enzyme_block_fusion', 'biofp'):
                assert config['model'][key] == base['model'][key]
            assert 'enzyme_attention_regularization' not in config['training']
        else:
            assert config['model']['enzyme_input_mode'] == 'raw_mean_sleec_multiview'
            assert config['model']['enzyme_multiview']['num_slots'] == 4
            assert config['model']['enzyme_multiview']['uniform_mix'] == .05
            assert config['training']['enzyme_attention_regularization'] == dict(entropy_weight=.01, diversity_weight=.001)
            assert not {'enzyme_block_fusion', 'biofp', 'hyperbolic_encoder'} & config['model'].keys()
        for key in ('loss', 'learning_rate', 'weight_decay', 'precision'):
            assert config['training'][key] == base['training'][key]
        assert config['training']['init_from_checkpoint'] is None
        assert config['ablation']['enzyme_tower'] == enzyme_tower
        assert config['training']['max_steps'] == -1
        assert config['training']['max_epochs'] == 30
        assert config['data']['train_batch_size'] * devices == 1536
        assert config['logging']['save_top_k'] == -1
        assert config['model']['reaction_pooling'] == 'attention'
        for section in ('data', 'model'):
            assert config[section]['reaction_use_model'] and config[section]['reaction_use_chiro']
            assert not config[section]['reaction_use_chemistry']
            assert not config[section]['reaction_use_directional']
    feature, scalar = copy.deepcopy(configs)
    feature['ablation'] = scalar['ablation']
    feature['model']['reaction_multimodal_attention']['fusion'] = 'scalar_gate'
    assert feature == scalar


def test_verify_detects_mutated_config(tmp_path):
    from scripts.run_circe_generalization_diagnostics import sha
    path = tmp_path/'train.yaml'; path.write_text('seed: 42\n')
    manifest = dict(sources=[], code={}, configs={'train.yaml': sha(path)})
    verify(tmp_path, manifest)
    path.write_text('seed: 43\n')
    with pytest.raises(ValueError, match='Config changed'):
        verify(tmp_path, manifest)


@pytest.mark.parametrize('devices,epochs,variant', [(0,30,'feature'), (5,30,'scalar'), (4,0,'feature'), (4,30,'gated')])
def test_bad_arguments_rejected(tmp_path, devices, epochs, variant):
    with pytest.raises(ValueError):
        make_config({}, tmp_path, tmp_path/'t5.h5', devices, epochs, variant)


def test_detached_launch_forwards_enzyme_tower(tmp_path, monkeypatch):
    import sys
    from scripts import run_circe_feature_gate as launcher
    launches = []
    monkeypatch.setattr(launcher, 'check_gpus', lambda gpus: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, **kwargs: launches.append(command))
    monkeypatch.setattr(sys, 'argv', ['launcher', 'launch', '--gpus', '0,1',
        '--enzyme-tower', 'multiview', '--positive-biology', '--biofp-aux-weight', '0.02',
        '--output', str(tmp_path/'fresh')])
    launcher.main()
    assert len(launches) == 1
    assert launches[0][:3] == ['tmux', 'new-session', '-d']
    assert '--enzyme-tower multiview' in launches[0][-1]
    assert '--gpus 0,1' in launches[0][-1]
    assert '--positive-biology' in launches[0][-1]
    assert '--biofp-aux-weight 0.02' in launches[0][-1]
    assert '--ec-labels ' in launches[0][-1]


def test_positive_biology_config_retains_nonzero_weight(tmp_path):
    from horizyn.config import DotDict, validate_config
    from tests.unit.test_enzyme_multiview_integration import config
    base = config().to_dict() if hasattr(config(), 'to_dict') else dict(config())
    base['model'].update(enzyme_input_mode='raw_mean_sleec_biological_factorized',
        reaction_multimodal_attention={'modality_encoder_widths':[4096,4096]})
    base['data'].update(train_pairs_path='train.csv',train_reactions_path='rxns.csv',
        protein_residue_embeds_path='proteins.h5',residue_dim=8)
    base['seed']=42
    base['logging']={'wandb':{}}
    labels={'ec':['1.1.1.1'],'cofactor':['NAD_NADP'],'mechanism':['redox']}
    annotations={'families':labels,'targets':str(tmp_path/'targets.npz'),'vocab':str(tmp_path/'vocab.json')}
    result=make_config(base,tmp_path,tmp_path/'t5.h5',4,30,'feature','multiview',annotations,.02)
    assert result['training']['loss']['biofp_aux_weight']==.02
    assert result['training']['loss']['biofp_aux_mode']=='positive_anchor'
    assert result['training']['loss']['biofp_aux_warmup_epochs']==3
    assert result['model']['biofp']['positive_labels']==labels
    assert result['data']['train_batch_size']==384
    from horizyn.config_validation import _validate_enzyme_inputs
    _validate_enzyme_inputs(DotDict(result))
