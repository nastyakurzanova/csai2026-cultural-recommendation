"""Fit, tune on development validation only, and evaluate both study cohorts."""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import os
import sys

# Small dense kernels are faster and easier to reproduce with one BLAS thread.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from prepare_transfer import prepare, readjl
from recommender import CityModel, graph_kernel, markov_kernel, predict_scores, ranks
from experiment import best_search, paired_inference
from model import metric_arrays
from research_io import atomic_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = json.loads((HERE / 'protocol.json').read_text())
METRICS = ['NDCG5', 'HR5', 'MRR', 'NDCG10', 'HR10']


def subset(c, name):
    return (name == 'all' or
            (name == 'two_profiles' and bool(c['old_items']) and len(c['prefix']) >= 2) or
            (name == 'warm' and c['warm']))


def summaries(pred, cohorts):
    rows = []
    for (cohort, group, name), part in pred.groupby(['evaluation', 'subset', 'model'], sort=False):
        stats = part.groupby(['city', 'user'])[METRICS].mean()
        for city, q in stats.groupby(level='city'):
            rows.append(dict(evaluation=cohort, subset=group, model=name, city=city,
                             contexts=int((part.city == city).sum()), users=len(q), **q.mean().to_dict()))
        rows.append(dict(evaluation=cohort, subset=group, model=name, city='MACRO',
                         contexts=len(part), users=part.user.nunique(), **stats.groupby(level='city').mean().mean().to_dict()))
    return pd.DataFrame(rows)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    paths = ['protocol.json', 'scope.json', 'prepare_transfer.py', 'recommender.py', 'run.py']
    lock = {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in paths}
    (out / 'protocol_lock.json').write_text(json.dumps(lock, indent=2))
    prepare(out)
    old_items, new_items = pd.read_csv(ROOT / 'data/items.csv'), pd.read_csv(out / 'transfer_items.csv')
    items = pd.concat([old_items, new_items], ignore_index=True)
    edges = pd.concat([pd.read_csv(ROOT / 'data/cultural_edges.csv'), pd.read_csv(out / 'transfer_edges.csv')])
    sequences = readjl(ROOT / 'data/sequences_main.jsonl') + readjl(out / 'transfer_sequences.jsonl')
    contexts = readjl(ROOT / 'data/contexts_main.jsonl') + readjl(out / 'transfer_contexts.jsonl')
    cutoffs = json.loads((ROOT / 'data/temporal_cutoffs.json').read_text()) | json.loads((out / 'transfer_cutoffs.json').read_text())
    dev, old, new = P['development'], P['reused_test'], P['new_cities']
    models, traininfo, city_edges = {}, [], {}
    for city in old + new:
        train = [s for s in sequences if s['city'] == city and s['end'] <= cutoffs[city]['train_end']]
        city_edges[city] = edges[edges.city.eq(city)]
        m = CityModel.fit(city, items[items.city.eq(city)], city_edges[city], train)
        models[city] = m
        traininfo.append(dict(city=city, places=len(m.ids), train_sequences=len(train),
                              observed_train_transitions=int(m.transitions.sum()),
                              cultural_edges=len(city_edges[city]),
                              places_with_edges=city_edges[city].poiID.nunique()))
    atomic_csv(pd.DataFrame(traininfo), out / 'city_metadata.csv')
    v = {c: models[c].batch([x for x in contexts if x['city'] == c and x['split'] == 'validation' and x['warm']]) for c in dev}
    assert all(len(x['target']) for x in v.values())
    val_log = []
    mu = best_search('Markov', [dict(mu=x) for x in P['prior']],
                     lambda c,b,h: markov_kernel(models[c],h['mu'])[b['last']], models, v, val_log)['mu']
    base = {c: markov_kernel(m, mu) for c, m in models.items()}
    cache = {}

    def kernel(c, h):
        kind, r = h['kind'], h.get('restart', .3)
        key = c, kind, r
        if key not in cache:
            cache[key] = graph_kernel(models[c], city_edges[c], kind, r)[0]
        return cache[key]

    def score(c, b, h):
        g = kernel(c, h) if h['eta'] else None
        return predict_scores(b, base[c], g, models[c].pop, h)

    variants = {'Markov': dict(eta=0., alpha=1., beta=0., kind='none')}
    for name, kind in [('Category','category'), ('Flat','flat'), ('Graph','graph')]:
        rs = P['restart'] if kind != 'flat' else [.3]
        zero = [dict(kind=kind, eta=0., alpha=1., restart=rs[0], beta=0.)]
        positive = [dict(kind=kind, eta=e, alpha=a, restart=r, beta=0.)
                    for e, a, r in product(P['eta'][1:], P['alpha'], rs)]
        variants[name+'Free'] = best_search(name+'Free', zero + positive,
                                            lambda c,b,h: score(c,b,h)[0], models,v,val_log)
        # Reuse evaluated configs instead of re-running identical predictions.
        candidates = [x for x in val_log if x['model']==name+'Free' and x['eta']>0]
        top = max(candidates, key=lambda x: round(x['validation_user_macro_NDCG5'], 12))
        variants[name+'Positive'] = {k:top[k] for k in ('kind','eta','alpha','restart','beta')}
    for mode in ('Free','Positive'):
        fixed = variants['Graph'+mode]
        grid = [dict(fixed, beta=b, c=P['gate_c']) for b in P['beta']] if fixed['eta'] else [dict(fixed,c=P['gate_c'])]
        variants['Adaptive'+mode] = best_search('Adaptive'+mode,grid,
                                                lambda c,b,h:score(c,b,h)[0],models,v,val_log)
    variants['CategoryMatched'] = dict(variants['GraphPositive'], kind='category')
    variants['FlatMatched'] = dict(variants['GraphPositive'], kind='flat', restart=.3)
    calibration = dict(markov_prior=mu, variants=variants, development_cities=dev,
                       validation_contexts=sum(len(b['target']) for b in v.values()))
    (out / 'parameters.json').write_text(json.dumps(calibration, indent=2))
    atomic_csv(pd.DataFrame(val_log), out / 'validation_grid.csv')

    rows, predictions, cohorts = [], {}, {'reused': old, 'transfer': new}
    for cohort, cities in cohorts.items():
        for city in cities:
            m = models[city]
            raw = [x for x in contexts if x['city']==city and x['split']=='test' and
                   (cohort!='transfer' or x['transfer_user'])]
            assert all(x['sequence_start'] > cutoffs[city]['validation_end'] for x in raw)
            if not raw:
                raise ValueError(f'Empty full test cohort in {city}')
            b = m.batch(raw)
            predictions[city] = b
            saved = {}
            for name, h in variants.items():
                scores, alpha = score(city,b,h)
                rank, order = ranks(scores, b['target'], b['last'], m.pop, m.ids)
                stats = metric_arrays(rank)
                for i,c in enumerate(raw):
                    top = json.dumps(m.ids[order[i,:min(5,len(m.ids)-1)]].tolist(), separators=(',',':'))
                    common = dict(evaluation=cohort, city=city, user=c['user'], context_id=c['context_id'],
                                  model=name, target=c['target'], rank=int(rank[i]), alpha=alpha[i],
                                  graph_weight=h['eta'], top5=top, **{k: stats[k][i] for k in METRICS})
                    rows.extend(dict(common, subset=s) for s in ('all','two_profiles','warm') if subset(c,s))
                if h['eta']:
                    saved['kernel_'+name] = kernel(city,h)
            m.save(out/'models'/city, {}, dict(saved,markov=base[city]))
            meta = out/'models'/city/'metadata.json'
            meta.write_text(json.dumps(json.loads(meta.read_text()) | calibration, indent=2, ensure_ascii=False))
    pred = pd.DataFrame(rows)
    assert not pred.duplicated(['evaluation','subset','model','context_id']).any()
    atomic_csv(pred,out/'predictions.csv')
    result = summaries(pred, cohorts)
    atomic_csv(result,out/'metrics.csv')

    contrasts = P['primary_contrasts'] + [
        ['GraphPositive','FlatMatched','all','matched_graph_vs_flat'],
        ['GraphPositive','Markov','all','positive_graph_vs_markov'],
        ['AdaptivePositive','GraphPositive','warm','adaptation_warm'],
    ]
    inference = []
    for cohort,cities in cohorts.items():
        p = pred[pred.evaluation.eq(cohort)]
        for a,b,s,name in contrasts:
            x,y = p[(p.model==a)&(p.subset==s)],p[(p.model==b)&(p.subset==s)]
            assert set(x.context_id)==set(y.context_id)
            r = paired_inference(p,a,b,subset=s,cities=cities)
            inference.append(dict(r,evaluation=cohort,contrast=name,
                                  primary=(cohort=='transfer' and [a,b,s,name] in P['primary_contrasts'])))
    designated = [r for r in inference if r['primary'] and 'p_two_sided' in r]
    largest = 0.
    for i,r in enumerate(sorted(designated,key=lambda x:x['p_two_sided'])):
        largest = max(largest,min(1.,(len(designated)-i)*r['p_two_sided']))
        r['p_holm'] = largest
    atomic_csv(pd.DataFrame(inference),out/'contrasts.csv')

    placebo = []
    h = variants['GraphPositive']
    for seed in [None] + [P['seed'] + i for i in range(20)]:
        values, swaps = [], []
        for city in new:
            m,b = models[city],predictions[city]
            g,s = graph_kernel(m,city_edges[city],h['kind'],h['restart'],seed)
            score0,_ = predict_scores(b,base[city],g,m.pop,h)
            r,_ = ranks(score0,b['target'],b['last'],m.pop,m.ids)
            z = pd.DataFrame(dict(u=b['users'],v=metric_arrays(r)['NDCG5']))
            values.append(z.groupby('u').v.mean().mean());swaps.append(s)
            placebo.append(dict(seed='original' if seed is None else seed,city=city,
                                NDCG5=values[-1],accepted_swaps=s))
        placebo.append(dict(seed='original' if seed is None else seed,city='MACRO',
                            NDCG5=np.mean(values),accepted_swaps=sum(swaps)))
    atomic_csv(pd.DataFrame(placebo),out/'graph_placebo.csv')
    print(result[(result.city=='MACRO') & ((result.evaluation=='transfer')&(result.subset=='all') |
                                           (result.evaluation=='reused')&(result.subset=='warm'))][
        ['evaluation','subset','model','contexts','users','NDCG5','HR5','MRR']].to_string(index=False))
    print(pd.DataFrame(inference).to_string(index=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE/'results')
    main(p.parse_args().out)
