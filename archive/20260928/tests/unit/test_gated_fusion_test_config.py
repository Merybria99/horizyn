import copy
from scripts.test_circe_gated_fusion import test_config as make_test_config


def test_heldout_config_preserves_model_and_replaces_validation(tmp_path):
    base = {'model': {'fusion': 'gated_attention_concat'}, 'training': {
        'loss': {'name': 'unchanged'}, 'validation_retrieval_candidate_ids_path': 'old'},
        'data': {'test_pairs_path': 'validation.csv',
                 'validation_pairs_path': 'validation.csv',
                 'reaction_t5v2_embeds_path': 'old.h5'}}
    original = copy.deepcopy(base)
    cfg = make_test_config(base, tmp_path/'output', tmp_path/'test_pairs.csv',
                           tmp_path/'test_rxns.csv', tmp_path/'features')
    assert base == original
    assert cfg['model'] == base['model']
    assert cfg['training']['loss'] == base['training']['loss']
    for prefix in ('test', 'validation'):
        assert cfg['data'][f'{prefix}_pairs_path'] == str(tmp_path/'test_pairs.csv')
        assert cfg['data'][f'{prefix}_reactions_path'] == str(tmp_path/'test_rxns.csv')
    for section in ('data', 'training'):
        assert cfg[section]['validation_retrieval_candidate_ids_path'] == str(tmp_path/'test_candidate_ids.txt')
    assert cfg['data']['reaction_t5v2_embeds_path'] == str(tmp_path/'output/reactiont5_participants.h5')
    for modality in ('unimol2', 'chiro'):
        assert cfg['data'][f'reaction_{modality}_embeds_path'] == str(tmp_path/f'features/{modality}.h5')
