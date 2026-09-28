"""Pure synthetic tests; no background feature reads or real activity labels."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import generalization_large_case1_evaluate as evaluator

class RankingTests(unittest.TestCase):
    def test_unique_scores(self):
        r,order=evaluator.reference_ranks(np.array([.2,.8,.5],np.float32),[0,1])
        np.testing.assert_array_equal(r['stable_rank'],[3,1])
        np.testing.assert_array_equal(r['best_rank'],r['worst_rank'])
        np.testing.assert_allclose(r['expected_reciprocal_rank'],[1/3,1])

    def test_uniform_ties_remove_order_advantage(self):
        r,_=evaluator.reference_ranks(np.zeros(100,np.float32),[0,99])
        summary=evaluator.tier_summary(r,{0,1},100,cuts=(1,10))
        self.assertAlmostEqual(summary['cutoffs']['1']['uniform_tie_expected_recovered'],.02)
        self.assertEqual(summary['cutoffs']['1']['stable_recovered'],1)
        self.assertEqual(summary['cutoffs']['1']['guaranteed_recovered'],0)
        self.assertEqual(summary['cutoffs']['1']['possible_recovered'],2)
        self.assertAlmostEqual(summary['uniform_tie_expected_all_positive_mrr'],sum(1/x for x in range(1,101))/100)

    def test_tie_at_cut_boundary(self):
        r,_=evaluator.reference_ranks(np.array([3.,2.,2.,2.,1.],np.float32),[1,3,4])
        np.testing.assert_array_equal(r['best_rank'],[2,2,5]);np.testing.assert_array_equal(r['worst_rank'],[4,4,5])
        s=evaluator.tier_summary(r,{0,1,2},5,cuts=(3,100))
        self.assertAlmostEqual(s['cutoffs']['3']['uniform_tie_expected_recovered'],4/3)
        self.assertEqual(s['cutoffs']['100']['uniform_tie_expected_recovered'],3)

    def test_order_permutation_preserves_expected_metrics(self):
        s=np.array([1,2,2,0,2],np.float32);indices=[1,4]
        a,_=evaluator.reference_ranks(s,indices)
        permutation=np.array([4,0,3,2,1]);mapped=[int(np.where(permutation==i)[0][0]) for i in indices]
        b,_=evaluator.reference_ranks(s[permutation],mapped)
        np.testing.assert_allclose(a['expected_reciprocal_rank'],b['expected_reciprocal_rank'])
        self.assertEqual(evaluator.tier_summary(a,{0,1},5)['cutoffs']['1']['uniform_tie_expected_recall'],evaluator.tier_summary(b,{0,1},5)['cutoffs']['1']['uniform_tie_expected_recall'])

    def test_invalid_scores_or_indices(self):
        for scores,indices in [([1,np.nan],[0]),([1,2],[0,0]),([1,2],[2]),([],[])]:
            with self.assertRaises(ValueError):evaluator.reference_ranks(np.array(scores),indices)

class AliasTests(unittest.TestCase):
    def example(self):
        pids=[f'P{i:03d}' for i in range(123)]
        literature={'proteins':pids,'groups':[dict(representative_id=p,sequence_sha256=f'{i:064x}',all_entry_ids=[p]) for i,p in enumerate(pids)]}
        ids=['WP_0','WP_1','WP_2']+['LIT_'+p for p in pids[2:]]
        aliases=[]
        for i,r in enumerate(literature['groups']):
            index=i+1
            aliases.append(dict(r,candidate_id=ids[index],candidate_index=index,already_in_selected_background=i<2))
        return {'groups':aliases},{'proteins':ids,'query_ids':['q']},literature

    def matches(self,literature):
        return {p:dict(sequence_sha256=literature['groups'][i]['sequence_sha256'],refseq_ids=['WP_'+str(i+1)]) for i,p in enumerate(literature['proteins'][:2])}

    def test_all_123_sequences_once(self):
        aliases,catalog,literature=self.example()
        self.assertEqual(len(evaluator.validate_aliases(aliases,catalog,literature,3,self.matches(literature))),123)

    def test_duplicate_candidate_mapping_rejected(self):
        aliases,catalog,literature=self.example();aliases['groups'][1]['candidate_index']=1
        with self.assertRaises(ValueError):evaluator.validate_aliases(aliases,catalog,literature,3,self.matches(literature))

    def test_reordered_append_rejected(self):
        aliases,catalog,literature=self.example()
        for i,j in [(2,3),(3,2)]:
            aliases['groups'][i]['candidate_index']=j+1;aliases['groups'][i]['candidate_id']=catalog['proteins'][j+1]
        with self.assertRaises(ValueError):evaluator.validate_aliases(aliases,catalog,literature,3,self.matches(literature))

    def test_changed_sequence_rejected(self):
        aliases,catalog,literature=self.example();aliases['groups'][5]['sequence_sha256']='f'*64
        with self.assertRaises(ValueError):evaluator.validate_aliases(aliases,catalog,literature,3,self.matches(literature))

    def test_false_selected_match_rejected(self):
        aliases,catalog,literature=self.example()
        with self.assertRaises(ValueError):evaluator.validate_aliases(aliases,catalog,literature,3,{})

    def test_globally_matching_unsampled_is_appended(self):
        aliases,catalog,literature=self.example();matches=self.matches(literature)
        matches['P002']=dict(sequence_sha256=literature['groups'][2]['sequence_sha256'],refseq_ids=['WP_not_sampled'])
        self.assertEqual(len(evaluator.validate_aliases(aliases,catalog,literature,3,matches)),123)

class ChunkTests(unittest.TestCase):
    def test_complete_ordered_partition(self):
        protocol=dict(tiles=[dict(tile_index=2,start=8192,stop=12288),dict(tile_index=963,start=3944448,stop=3944613)],background_count=4261,candidate_count=4384)
        receipts=[dict(tile_index=2,rows=4096),dict(tile_index=963,rows=165),dict(tile_index='literature',rows=123)]
        evaluator.validate_chunk_partition(receipts,protocol)
        for broken in (receipts[:-1],receipts[::-1],receipts+[receipts[-1]],receipts[:2]+[dict(tile_index='literature',rows=122)]):
            with self.assertRaises(ValueError):evaluator.validate_chunk_partition(broken,protocol)

class SelectionTests(unittest.TestCase):
    def test_exact_seed_and_order(self):
        tiles=np.sort(np.random.default_rng(20260920).choice(964,256,replace=False))
        count=sum(min((int(t)+1)*4096,3944613)-int(t)*4096 for t in tiles)
        actual,indices=evaluator.selected_indices({'tiles':tiles.tolist(),'background_count':count})
        np.testing.assert_array_equal(actual,tiles);self.assertEqual(len(indices),count)
        wrong=tiles.tolist();wrong[0]=964
        with self.assertRaises(ValueError):evaluator.selected_indices({'tiles':wrong,'background_count':count})
        with self.assertRaises(ValueError):evaluator.selected_indices({'tiles':tiles[::-1].tolist(),'background_count':count})

if __name__=='__main__':unittest.main()
