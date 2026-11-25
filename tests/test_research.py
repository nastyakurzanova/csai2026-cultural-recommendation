"""Invariant checks for leakage, ranking, diffusion and deployment round trip."""
import unittest,sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from model import normalize,gate_alpha,ppr_kernel,make_features,ranks,metric_arrays,HeritageRecommender
from experiment import readjl

class ResearchTests(unittest.TestCase):
    def test_metric_known_ranks(self):
        m=metric_arrays(np.array([1,2,5,6,10,11]));np.testing.assert_allclose(m['NDCG5'],[1,1/np.log2(3),1/np.log2(6),0,0,0]);np.testing.assert_allclose(m['HR5'],[1,1,1,0,0,0]);self.assertAlmostEqual(m['MRR'][-1],1/11)
    def test_candidate_mask_and_static_ties(self):
        rank,order=ranks(np.array([[1.,1.,1.]]),np.array([0]),np.array([1]),np.array([.2,.9,.8]),np.array([11,22,33]))
        self.assertEqual(order.tolist(),[[2,0,1]]);self.assertEqual(rank.tolist(),[2])
    def test_gate_boundaries(self):
        n=np.array([0,1,2,100]);a=gate_alpha(np.ones(4),n,.25,1,3);self.assertTrue(np.all((a>=.25)&(a<=1)));self.assertTrue(np.all(np.diff(a)>=0));np.testing.assert_allclose(gate_alpha(np.zeros(4),n,.25,1,3),.25)
    def test_diffusion_against_independent_power_iteration(self):
        F=np.array([[1,1,0],[1,0,1],[0,1,1]],float)
        K,R,P,W=ppr_kernel(F,.3,np.ones(3)/3,return_full=True)
        seed=np.zeros(6);seed[0]=1;x=seed.copy()
        for _ in range(1000):x=.3*seed+.7*(x@P)
        np.testing.assert_allclose(x,R[0],atol=1e-10);np.testing.assert_allclose(K.sum(1),1);np.testing.assert_allclose(np.diag(K),0)
    def test_shuffle_preserves_degrees_and_relations(self):
        items=pd.DataFrame({'poiID':[1,2,3,4],'category':['a','a','b','b']})
        edges=pd.DataFrame({'poiID':[1,1,2,3,4,4],'property':['P84']*6,'target':['Q1','Q2','Q1','Q3','Q2','Q3']})
        F,k,r,_=make_features(items,edges);S,ks,rs,sw=make_features(items,edges,shuffle_seed=23)
        self.assertEqual(k,ks);np.testing.assert_allclose((F>0).sum(0),(S>0).sum(0));np.testing.assert_allclose((F>0).sum(1),(S>0).sum(1));self.assertTrue(all(x[0]=='class' for x in r if x[0]=='class'));self.assertGreater(sum(sw.values()),0)
    def test_no_future_in_context(self):
        for c in readjl(ROOT/'data/contexts_main.jsonl'):
            self.assertLess(max(c['prefix_times']),c['target_time'])
            self.assertTrue(not c['old_end_times'] or max(c['old_end_times'])<c['sequence_start'])
            self.assertNotEqual(c['prefix'][-1],c['target'])
    def test_sequence_disjoint_splits(self):
        seen={}
        cut=json.loads((ROOT/'data/temporal_cutoffs.json').read_text())
        for c in readjl(ROOT/'data/contexts_main.jsonl'):
            key=(c['city'],c['user'],c['seqID']);self.assertEqual(seen.setdefault(key,c['split']),c['split'])
            if c['split']=='train':self.assertLessEqual(c['sequence_end'],cut[c['city']]['train_end'])
            if c['split']=='validation':self.assertGreater(c['sequence_start'],cut[c['city']]['train_end'])
            if c['split']=='test':
                self.assertGreater(c['sequence_start'],cut[c['city']]['validation_end'])
                self.assertGreater(c['target_time'],cut[c['city']]['validation_end'])
    def test_deployment_matches_saved_experiment(self):
        contexts={c['context_id']:c for c in readjl(ROOT/'data/contexts_main.jsonl')}
        top=pd.read_csv(ROOT/'results/top5_predictions.csv');pred=pd.read_csv(ROOT/'results/predictions.csv')
        for city in ['Buda','Vien','Edin']:
            z=top[(top.city==city)&(top.model=='CulturalAdaptive')].iloc[0];c=contexts[z.context_id]
            m=HeritageRecommender(ROOT/'models'/city);out=m.recommend(c['prefix'],c['old_items'])
            self.assertEqual([r['poiID'] for r in out['recommendations']],json.loads(z.top5))
            self.assertTrue(all(r['poiID']!=c['prefix'][-1] for r in out['recommendations']))
            row=pred[(pred.context_id==c['context_id'])&(pred.model=='CulturalAdaptive')&(pred.subset=='warm')].iloc[0]
            self.assertAlmostEqual(out['short_weight'],row.alpha)
    def test_global_flickr_cluster_accounting(self):
        p=pd.read_csv(ROOT/'results/predictions.csv');x=p[(p.split=='test')&(p.subset=='warm')&p.city.isin(['Buda','Vien'])&(p.model=='CulturalFixed')]
        self.assertEqual(x.user.nunique(),47);self.assertEqual(len(x),272);self.assertEqual(x[['city','user']].drop_duplicates().shape[0],52)
        contexts=readjl(ROOT/'data/contexts_main.jsonl')
        for city in ['Buda','Vien','Edin']:
            for split in ['validation','test']:
                for subset in ['all','warm']:
                    expected=sum(c['city']==city and c['split']==split and (subset=='all' or c['warm']) for c in contexts)
                    group=p[(p.city==city)&(p.split==split)&(p.subset==subset)]
                    self.assertEqual(group.model.nunique(),12)
                    self.assertTrue((group.groupby('model').size()==expected).all())
    def test_source_photo_accounting(self):
        p=ROOT/'source/processed';photos=pd.read_csv(p/'photos_clean.csv');v=pd.read_csv(p/'visits_prepared.csv')
        self.assertEqual(len(photos),int(v.photo_count.sum()));self.assertEqual(len(photos),153155)
    def test_feature_provenance(self):
        e=pd.read_csv(ROOT/'data/cultural_edges.csv');self.assertTrue(e.property.isin(['P84','P149','P138']).all());self.assertFalse(e.statement_id.isna().any());self.assertFalse(e.duplicated(['city','poiID','property','target']).any());self.assertEqual(len(e),120)
    def test_saved_probabilistic_kernels(self):
        for c in ['Buda','Vien','Edin']:
            z=np.load(ROOT/f'models/{c}/weights.npz')
            for key in ['collaborative','cultural_kernel','category_kernel','flat_kernel','cultural_hybrid']:
                np.testing.assert_allclose(z[key].sum(1),1,atol=1e-9);self.assertTrue(np.all(z[key]>=0));np.testing.assert_allclose(np.diag(z[key]),0)
    def test_ease_closed_form_optimality(self):
        for c in ['Buda','Vien','Edin']:
            z=np.load(ROOT/f'models/{c}/weights.npz');params=json.loads((ROOT/'results/selected_parameters.json').read_text());B=z['ease_weights'];G=z['gram']
            grad=(G+params['EASE']['l2']*np.eye(len(G)))@B-G;np.fill_diagonal(grad,0)
            np.testing.assert_allclose(grad,0,atol=1e-8);np.testing.assert_allclose(np.diag(B),0)
    def test_markov_counts_use_training_sequences_only(self):
        ss=readjl(ROOT/'data/sequences_main.jsonl');cut=json.loads((ROOT/'data/temporal_cutoffs.json').read_text())
        for c in ['Buda','Vien','Edin']:
            model=HeritageRecommender(ROOT/'models'/c).model;expected=np.zeros_like(model.transitions)
            for s in ss:
                if s['city']!=c or s['end']>cut[c]['train_end']:continue
                for a,b in zip(s['items'],s['items'][1:]):expected[model.index[a],model.index[b]]+=1
            np.testing.assert_array_equal(model.transitions,expected)
    def test_positive_graph_variant_loads_and_has_positive_component(self):
        for c in ['Buda','Vien','Edin']:
            model=HeritageRecommender(ROOT/'models'/c,'graph_positive')
            self.assertGreater(model.parameters['CulturalFixedForced']['eta'],0)
            self.assertGreater(np.linalg.norm(model.kernel-model.model.collaborative),0)
            ids=model.model.ids.tolist();out=model.recommend(ids[:2],ids[2:5]);self.assertEqual(out['variant'],'graph_positive')
if __name__=='__main__':unittest.main(verbosity=2)
