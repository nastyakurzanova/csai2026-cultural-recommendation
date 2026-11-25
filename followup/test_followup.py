"""Check the score composition, temporal boundaries, coverage and saved inference."""
import unittest
import numpy as np
import pandas as pd
import json
from run import HERE, ROOT, P, readjl
from recommender import gate, predict_scores, markov_kernel, MarkovHeritageRecommender

OUT = HERE/'results'


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(OUT/'predictions.csv')
        cls.ctx = readjl(ROOT/'data/contexts_main.jsonl') + readjl(OUT/'transfer_contexts.jsonl')
        cls.lookup = {c['context_id']:c for c in cls.ctx}

    def test_zero_weight_reduces_exactly_to_markov(self):
        m=MarkovHeritageRecommender(OUT/'models'/'Buda','Markov')
        x=next(c for c in self.ctx if c['city']=='Buda' and c['warm'])
        b=m.model.batch([x])
        score,_=predict_scores(b,m.markov,None,m.model.pop,dict(eta=0,alpha=.3,beta=.5))
        np.testing.assert_array_equal(score,m.markov[b['last']])

    def test_gate_fallback_and_nonzero_change(self):
        b=dict(length=np.array([3,3,3,3]),metadata_available=np.array([1,1,0,1]),
               drift=np.array([.8,.8,.8,.0]),long_available=np.array([1,0,1,1]))
        np.testing.assert_allclose(gate(b,.25,.25),[.35,1,.25,.25])
        np.testing.assert_allclose(gate(b,.25,0),[.25,1,.25,.25])
        self.assertTrue(np.all((gate(b,.95,.5)>=0) & (gate(b,.95,.5)<=1)))

    def test_transfer_has_no_development_users_or_future_history(self):
        old=pd.read_csv(ROOT/'source/processed/sequence_audit.csv',dtype={'userID':str})
        old_ids=set(old[old.city.isin(P['reused_test'])].userID)
        cut=json.loads((OUT/'transfer_cutoffs.json').read_text())
        for c in self.ctx:
            if c['city'] not in P['new_cities']:
                continue
            self.assertLess(max(c['prefix_times']),c['target_time'])
            self.assertTrue(all(t<c['sequence_start'] for t in c['old_end_times']))
            self.assertEqual(c['transfer_user'],c['user'] not in old_ids)
            if c['split']=='test':
                self.assertGreater(c['sequence_start'],cut[c['city']]['validation_end'])

    def test_markov_counts_and_row_distributions(self):
        cut=json.loads((OUT/'transfer_cutoffs.json').read_text())
        seq=readjl(OUT/'transfer_sequences.jsonl')
        mu=json.loads((OUT/'parameters.json').read_text())['markov_prior']
        for c in P['new_cities']:
            r=MarkovHeritageRecommender(OUT/'models'/c,'Markov')
            T=np.zeros_like(r.model.transitions)
            for s in seq:
                if s['city']==c and s['end']<=cut[c]['train_end']:
                    ids=[r.model.index[i] for i in s['items']]
                    for i,j in zip(ids,ids[1:]):
                        T[i,j]+=1
            np.testing.assert_array_equal(T,r.model.transitions)
            np.testing.assert_allclose(r.markov,markov_kernel(r.model,mu))
            np.testing.assert_allclose(r.markov.sum(1),1)
            np.testing.assert_array_equal(np.diag(r.markov),0)

    def test_all_models_have_identical_contexts_per_subset(self):
        for _,g in self.pred.groupby(['evaluation','city','subset']):
            self.assertFalse(g.duplicated(['model','context_id']).any())
            ref=set(g[g.model.eq('Markov')].context_id)
            for _,m in g.groupby('model'):
                self.assertEqual(ref,set(m.context_id))

    def test_saved_recommender_matches_evaluated_top5(self):
        for (city,name),part in self.pred[self.pred.subset.eq('all')].groupby(['city','model']):
            row=part.iloc[0];c=self.lookup[row.context_id]
            r=MarkovHeritageRecommender(OUT/'models'/city,name)
            got=r.recommend(c['prefix'],c['old_items'])
            self.assertEqual([p['poiID'] for p in got['recommendations']],json.loads(row.top5))
            self.assertNotIn(c['prefix'][-1],[p['poiID'] for p in got['recommendations']])
            self.assertAlmostEqual(got['alpha'],row.alpha)


if __name__=='__main__':
    unittest.main(verbosity=2)
