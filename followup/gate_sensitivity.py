"""Exploratory diagnostic added after validation selected the constant gate."""
from pathlib import Path
import json
import pandas as pd
from run import HERE, ROOT, P, METRICS, subset, readjl, paired_inference, atomic_csv
from recommender import MarkovHeritageRecommender, predict_scores, ranks
from model import metric_arrays


def main(out=HERE/'results'):
    v = pd.read_csv(out/'validation_grid.csv')
    q = v[(v.model=='AdaptivePositive') & v.beta.ne(0)].copy()
    selected = q.sort_values('validation_user_macro_NDCG5', ascending=False, kind='stable').iloc[0]
    beta = float(selected.beta)
    assert beta
    contexts = readjl(ROOT/'data/contexts_main.jsonl') + readjl(out/'transfer_contexts.jsonl')
    fixed = pd.read_csv(out/'predictions.csv', float_precision='round_trip')
    rows = []
    for cohort, cities in [('reused',P['reused_test']),('transfer',P['new_cities'])]:
        for city in cities:
            r = MarkovHeritageRecommender(out/'models'/city, 'GraphPositive')
            cs = [c for c in contexts if c['city']==city and c['split']=='test'
                  and (cohort=='reused' or c['transfer_user'])]
            b = r.model.batch(cs)
            sc,a = predict_scores(b,r.markov,r.graph,r.model.pop,dict(r.params,beta=beta,c=P['gate_c']))
            rk,_ = ranks(sc,b['target'],b['last'],r.model.pop,r.model.ids)
            vals = metric_arrays(rk)
            for i,c in enumerate(cs):
                row = dict(evaluation=cohort,city=city,user=c['user'],context_id=c['context_id'],
                           model='AdaptiveNonzero',alpha=a[i],rank=int(rk[i]),**{m:vals[m][i] for m in METRICS})
                rows.extend(dict(row,subset=s) for s in ('all','two_profiles','warm') if subset(c,s))
    pred=pd.DataFrame(rows)
    atomic_csv(pred,out/'nonzero_gate_predictions.csv')
    combined=pd.concat([fixed[fixed.model.eq('GraphPositive')],pred],ignore_index=True)
    comparisons=[]
    for cohort,cities in [('reused',P['reused_test']),('transfer',P['new_cities'])]:
        for s in ('all','two_profiles','warm'):
            d=combined[combined.evaluation.eq(cohort)]
            stat=paired_inference(d,'AdaptiveNonzero','GraphPositive',subset=s,cities=cities)
            f=d[(d.subset==s)&(d.model=='GraphPositive')][['context_id','alpha']]
            a=d[(d.subset==s)&(d.model=='AdaptiveNonzero')][['context_id','alpha']]
            changed=f.merge(a,on='context_id',validate='one_to_one',suffixes=('_fixed','_adaptive'))
            comparisons.append(dict(stat,evaluation=cohort,beta=beta,
                                    contexts_with_changed_weight=int((abs(changed.alpha_fixed-changed.alpha_adaptive)>1e-12).sum())))
    atomic_csv(pd.DataFrame(comparisons),out/'nonzero_gate_contrasts.csv')
    (out/'nonzero_gate_notes.json').write_text(json.dumps(dict(
        status='Exploratory follow-up, added after observing the selected beta=0; no claims of independent confirmation',
        selection='Best nonzero beta from the original frozen validation grid; no test-based selection',
        beta=beta, validation_NDCG5=float(selected.validation_user_macro_NDCG5)),indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':
    main()
