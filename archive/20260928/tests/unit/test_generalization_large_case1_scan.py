"""The input-only sampling and forced-reference aliases are part of the protocol."""
import numpy as np
import pytest
from scripts.generalization_large_case1_scan import alias_catalog, tile_selection


def test_seeded_tiles_cover_only_fixed_distinct_blocks_in_source_order():
    tiles, indices = tile_selection(3944613)
    again, again_indices = tile_selection(3944613)
    assert tiles == again
    assert np.array_equal(indices, again_indices)
    assert len(tiles) == 256 and len({t['tile_index'] for t in tiles}) == 256
    assert np.all(np.diff(indices) > 0)
    assert indices.min() >= 0 and indices.max() < 3944613
    assert np.array_equal(indices, np.concatenate([np.arange(t['start'], t['stop']) for t in tiles]))


def test_global_match_appends_if_not_selected_and_reuses_if_selected():
    lit = {'groups': [
        {'representative_id': 'A', 'sequence_sha256': 's1', 'all_entry_ids': ['A', 'A_alias']},
        {'representative_id': 'B', 'sequence_sha256': 's2', 'all_entry_ids': ['B']},
        {'representative_id': 'C', 'sequence_sha256': 's3', 'all_entry_ids': ['C']},
    ]}
    matches = {'A': {'refseq_ids': ['WP_present']}, 'B': {'refseq_ids': ['WP_absent']}}
    ids, aliases = alias_catalog(['WP_first', 'WP_present'], lit, matches)
    assert ids == ['WP_first', 'WP_present', 'LIT_B', 'LIT_C']
    assert [r['candidate_index'] for r in aliases] == [1, 2, 3]
    assert [r['already_in_selected_background'] for r in aliases] == [True, False, False]
    assert aliases[0]['all_entry_ids'] == ['A', 'A_alias']


def test_ambiguous_duplicates_and_oversampling_rejected():
    with pytest.raises(ValueError): alias_catalog(['WP_A', 'WP_A'], {'groups': []}, {})
    with pytest.raises(ValueError): tile_selection(1, tile_count=2)
