"""Independent checks of raw inputs, data lineage, full test scores and metrics."""
import argparse
from collections import defaultdict
from itertools import groupby
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FOLLOW=ROOT/'followup/results'
OUT=ROOT/'audit/results'
PROCESSED=ROOT/'source/processed'
RAW=ROOT/'source/raw'
CITIES=['Buda','Vien','Edin','Delh','Glas','Osak','Pert','Toro']
DEV=CITIES[:3]
NEW=CITIES[3:]
KEYS=['NDCG5','HR5','MRR','NDCG10','HR10']
sys.path.insert(0,str(ROOT/'src'))
from experiment import paired_inference


def lines(p):
    return [json.loads(r) for r in p.read_text().splitlines() if r]


def read(p):
    return pd.read_csv(p,float_precision='round_trip')


def norm(a):
    s=a.sum(axis=-1,keepdims=True)
    return np.divide(a,s,out=np.zeros_like(a,dtype=float),where=s!=0)


def recalc(r):
    r=np.asarray(r)
    return dict(NDCG5=np.where(r<=5,1/np.log2(r+1),0),HR5=(r<=5).astype(float),
                MRR=1/r,NDCG10=np.where(r<=10,1/np.log2(r+1),0),HR10=(r<=10).astype(float))


def validate_metrics(frame):
    assert (frame['rank']>0).all()
    for k,v in recalc(frame['rank']).items():
        np.testing.assert_allclose(frame[k],v,rtol=1e-12,atol=1e-13)


def by_city_macro(g,key='NDCG5'):
    return float(g.groupby(['city','user'])[key].mean().groupby('city').mean().mean())

def rebuild_kernel(F,kind,restart,pop):
    if kind=='flat':
        lengths=np.sqrt((F**2).sum(1,keepdims=True))
        V=np.divide(F,lengths,out=np.zeros_like(F),where=lengths!=0)
        K=V@V.T
    else:
        n=len(F);W=np.block([[np.zeros((n,n)),F],[F.T,np.zeros((F.shape[1],F.shape[1]))]])
        A=norm(W);empty=np.flatnonzero(A.sum(1)==0);A[empty,empty]=1
        K=(restart*np.linalg.inv(np.eye(len(A))-(1-restart)*A))[:n,:n]
    K=np.maximum(K,0)
    K[np.diag_indices_from(K)]=0
    K[K.sum(1)<=1e-15]=pop
    K[np.diag_indices_from(K)]=0
    return norm(K)


def main(previous):
    OUT.mkdir(exist_ok=True,parents=True)
    checks=[]
    def passed(area,observation,count=None):
        checks.append(dict(area=area,result='пройдено',observation=observation,count=count))
    with zipfile.ZipFile(RAW/'data-ijcai15.zip') as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        for i in members:
            assert z.read(i)==(RAW/i.filename).read_bytes(),i.filename
    passed('Архив Flickr','Все извлечённые файлы побайтно совпадают с архивом авторов',len(members))
    allraw=[]
    for f in sorted((RAW/'data-ijcai15/userVisits-ijcai15').glob('userVisits-*.csv')):
        df=pd.read_csv(f,sep=';',dtype={'photoID':str,'userID':str});df['city']=f.stem.split('-')[1];allraw.append(df)
    allraw=pd.concat(allraw,ignore_index=True)
    clean=read(PROCESSED/'photos_clean.csv')
    visits=read(PROCESSED/'visits_prepared.csv')
    raw_keys=['city','photoID','userID','poiID','dateTaken','seqID']
    invalid=~allraw.dateTaken.between(1,1451606399)
    assert len(allraw)==153208 and invalid.sum()==53
    rk=allraw.loc[~invalid,raw_keys].astype({'photoID':str}).sort_values(raw_keys).reset_index(drop=True)
    ck=clean[raw_keys].astype({'photoID':str}).sort_values(raw_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(rk,ck,check_dtype=False)
    assert clean.photoID.notna().all() and not ck.duplicated().any()
    assert visits.photo_count.sum()==len(clean)==153155 and len(visits)==33778
    passed('Фотографии и посещения','153 208 строк; 53 даты вне диапазона; остальные строки совпали по шести исходным полям. Сумма снимков по посещениям совпадает.',len(clean))
    assert not visits[['city','userID','seqID','visit_id']].duplicated().any()
    # Reconstruct maximal constant-place runs independently, including both alias merges.
    alias={(r.city,r.original_poiID):r.poiID for r in read(PROCESSED/'poi_aliases.csv').itertuples()}
    photos=sorted(rk.itertuples(index=False),key=lambda r:(r.city,r.userID,r.seqID,r.dateTaken,r.photoID))
    rows=[]
    for (city,user,sid,poi),g in groupby(photos,key=lambda r:(r.city,r.userID,r.seqID,alias[r.city,r.poiID])):
        ts=[r.dateTaken for r in g]
        rows.append((city,user,sid,poi,ts[0],ts[-1],len(ts)))
    columns=['city','userID','seqID','poiID','start_time','end_time','photo_count']
    assert rows==list(visits[columns].itertuples(index=False,name=None))
    passed('Восстановление посещений','Независимо построены последовательные группы снимков одного места после объединения дублей; места, время и размеры групп совпали.',len(visits))
    cat=pd.concat([read(ROOT/'data/items.csv'),read(FOLLOW/'transfer_items.csv')],ignore_index=True)
    edges=pd.concat([read(ROOT/'data/cultural_edges.csv'),read(FOLLOW/'transfer_edges.csv')],ignore_index=True)
    assert not cat.duplicated(['city','poiID']).any()
    assert cat.lat.between(-90,90).all() and cat.lon.between(-180,180).all()
    assert len(cat)==135 and len(edges)==164 and not edges.duplicated(['city','poiID','property','target']).any()
    snap=defaultdict(list)
    for f in (RAW/'wikidata').glob('*.json'):
        for q,entity in json.loads(f.read_text()).get('entities',{}).items():
            for prop,statements in entity.get('claims',{}).items():
                for s in statements:
                    snap[s['id']].append((q,prop,entity.get('lastrevid'),s))
    refs=0;other_entity=[]
    cats=cat.set_index(['city','poiID'])
    for r in edges.itertuples(index=False):
        match=[(q,s) for q,p,rev,s in snap[r.statement_id] if p==r.property and rev==r.snapshot_revision and
              s.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id')==r.target]
        assert match, (r.city,r.poiID,r.statement_id)
        q,s=match[0]
        assert s['rank']==r.rank!='deprecated'
        assert len(s.get('references',[]))==r.reference_count
        assert json.loads(r.qualifiers_json)==s.get('qualifiers',{})
        assert r.source_url.endswith('/'+q)
        if q!=cats.loc[(r.city,r.poiID)].qid:
            other_entity.append(dict(city=r.city,poiID=r.poiID,canonical_qid=cats.loc[(r.city,r.poiID)].qid,
                                     source_qid=q,statement_id=r.statement_id))
        refs+=bool(r.reference_count)
    pd.DataFrame(other_entity,columns=['city','poiID','canonical_qid','source_qid','statement_id']).to_csv(OUT/'merged_source_edges.csv',index=False)
    passed('Культурные связи','Все рёбра найдены в исходных ответах Wikidata. Сверены сущность, редакция, тип, цель, ранг, квалификаторы и число ссылок.',len(edges))
    context=lines(ROOT/'data/contexts_main.jsonl')+lines(FOLLOW/'transfer_contexts.jsonl')
    seq=lines(ROOT/'data/sequences_main.jsonl')+lines(FOLLOW/'transfer_sequences.jsonl')
    lookup={c['context_id']:c for c in context}
    assert len(lookup)==len(context)
    cuts=json.loads((ROOT/'data/temporal_cutoffs.json').read_text())|json.loads((FOLLOW/'transfer_cutoffs.json').read_text())
    scope={city:set(cat[cat.city.eq(city)].poiID) for city in CITIES}
    hist=defaultdict(list);found=set()
    oldids=set(read(PROCESSED/'sequence_audit.csv').query('city in @DEV').userID)
    for x in sorted(seq,key=lambda r:(r['city'],r['user'],r['start'],r['seqID'])):
        city,user=x['city'],x['user'];old=hist[city,user];times=x['times'];items=x['items']
        assert set(items)<=scope[city] and len(items)==len(times) and all(a<b for a,b in zip(times,times[1:]))
        assert not old or old[-1][1]<x['start']
        for i in range(1,len(items)):
            cid=f'{city}:{user}:{x["seqID"]}:{i}';y=lookup[cid];found.add(cid)
            assert y['prefix']==items[:i] and y['prefix_times']==times[:i] and y['target']==items[i]
            assert y['target_time']==times[i] and y['target']!=y['prefix'][-1]
            assert y['old_items']==[v for v,_ in old] and y['old_end_times']==[t for _,t in old]
            assert y['warm']==(len(old)>=3 and i>=2)
            if y['split']=='test':assert x['start']>cuts[city]['validation_end']
            if y['split']=='validation':assert x['start']>cuts[city]['train_end'] and x['end']<=cuts[city]['validation_end']
            if y['split']=='train':assert x['end']<=cuts[city]['train_end']
            if city in NEW:assert y['transfer_user']==(user not in oldids)
        old.extend((v,x['end']) for v in items)
    assert found==set(lookup)
    passed('История и разделение','Все контексты заново составлены из последовательностей. Целевое событие следует за префиксом; повторные визиты разрешены. Границы и исключение исходных пользователей проверены.',len(context))
    p1=read(ROOT/'results/predictions.csv');p2=read(FOLLOW/'predictions.csv')
    pred=p2[p2.subset.eq('all')].copy()
    for frame in (p1,read(ROOT/'results/exploratory_predictions.csv'),p2,read(FOLLOW/'nonzero_gate_predictions.csv')):
        validate_metrics(frame)
        assert not frame.duplicated([k for k in ['evaluation','split','subset','model','context_id'] if k in frame]).any()
        assert frame.context_id.isin(lookup).all()
    passed('Метрики по рангу','NDCG@5, HR@5, MRR, NDCG@10 и HR@10 независимо пересчитаны для каждой сохранённой строки основных серий.',len(p1)+len(p2))
    orig=read(ROOT/'results/metrics.csv')
    for (split,subset,model,city),group in p1.groupby(['split','subset','model','city']):
        given=orig[(orig.split==split)&(orig.subset==subset)&(orig.model==model)&(orig.city==city)].iloc[0]
        expected=group.groupby('user')[KEYS].mean().mean().to_numpy()
        np.testing.assert_allclose(expected,given[KEYS].astype(float),atol=1e-12,rtol=1e-12)
    agg=read(FOLLOW/'metrics.csv')
    for (ev,su,model),g in p2.groupby(['evaluation','subset','model']):
        a=agg[(agg.evaluation==ev)&(agg.subset==su)&(agg.model==model)].set_index('city')
        for city,sub in g.groupby('city'):
            np.testing.assert_allclose(sub.groupby('user')[KEYS].mean().mean(),a.loc[city,KEYS].astype(float),atol=1e-12)
            assert len(sub)==a.loc[city,'contexts'] and sub.user.nunique()==a.loc[city,'users']
        np.testing.assert_allclose([by_city_macro(g,k) for k in KEYS],a.loc['MACRO',KEYS].astype(float),atol=1e-12)
    passed('Агрегирование','Средние пересчитаны сначала по прогнозам пользователя, затем по пользователям города и по городам с одинаковым весом.',len(orig)+len(agg))
    params=json.loads((FOLLOW/'parameters.json').read_text())
    h3rows=[];cityinfo=[];contextinfo=[];rounding=[]
    for city in CITIES:
        d=FOLLOW/'models'/city
        model=json.loads((d/'metadata.json').read_text())
        with np.load(d/'weights.npz',allow_pickle=False) as z:
            weights={k:z[k] for k in z.files}
        c=read(d/'items.csv');ids=c.poiID.to_numpy();i={q:j for j,q in enumerate(ids)}
        pop=np.full(len(ids),1e-9);tr=np.zeros((len(ids),len(ids)))
        for r in seq:
            if r['city']!=city or r['end']>cuts[city]['train_end']:continue
            ix=[i[x] for x in r['items']]
            for j in ix:pop[j]+=1
            for j,k in zip(ix,ix[1:]):tr[j,k]+=1
        pop=norm(pop);np.testing.assert_allclose(pop,weights['pop'],atol=1e-14)
        np.testing.assert_array_equal(tr,weights['transitions'])
        base=tr+params['markov_prior']*pop[None,:]
        base[np.diag_indices_from(base)]=0
        base[base.sum(1)==0]=pop
        base[np.diag_indices_from(base)]=0;base=norm(base)
        np.testing.assert_allclose(base,weights['markov'],atol=1e-13)
        el=edges[edges.city.eq(city)]
        keys=[tuple(q) for q in model['feature_keys']]
        F=np.zeros((len(ids),len(keys)))
        for j,poi in enumerate(ids):
            F[j,keys.index(('class',c.category.iloc[j]))]=1
            for prop,gr in el[el.poiID.eq(poi)].groupby('property'):
                for value in gr.target:
                    F[j,keys.index((prop,value))]=1/len(gr)
        np.testing.assert_array_equal(F,weights['features'])
        cur=pred[pred.city.eq(city)]
        for name,g in cur.groupby('model'):
            contexts=[lookup[v] for v in g.context_id]
            n=len(contexts);L=np.zeros((n,len(ids)));S=L.copy()
            last=np.array([i[y['prefix'][-1]] for y in contexts])
            target=np.array([i[y['target']] for y in contexts])
            for j,y in enumerate(contexts):
                for q in y['prefix']:S[j,i[q]]+=1
                for q in y['old_items']:L[j,i[q]]+=1
            Ll=norm(L);Ss=norm(S)
            idx=np.array([r!='class' for r,_ in model['feature_keys']])
            Lf=Ll@weights['features'][:,idx];Sf=Ss@weights['features'][:,idx]
            available=(Lf.sum(1)>0)&(Sf.sum(1)>0)
            drift=.5*np.abs(norm(Lf)-norm(Sf)).sum(1);drift[~available]=0
            h=params['variants'][name]
            alpha=np.clip(h['alpha']+h.get('beta',0)*drift*S.sum(1)/(S.sum(1)+h.get('c',3)),0,1)
            alpha[L.sum(1)==0]=1
            np.testing.assert_allclose(alpha,g.alpha,atol=1e-12)
            scores=base[last].copy()
            if h['eta']:
                K=weights['kernel_'+name]
                assert (K>=0).all()
                np.testing.assert_allclose(K.sum(1),1,atol=1e-12)
                fk=F[:,~idx] if h['kind']=='category' else F
                np.testing.assert_allclose(K,rebuild_kernel(fk,h['kind'],h.get('restart',.3),pop),atol=1e-12)
                q=(1-alpha[:,None])*(Ll@K)+alpha[:,None]*(Ss@K)
                q[np.arange(n),last]=0
                q[q.sum(1)==0]=pop;q[np.arange(n),last]=0
                scores=(1-h['eta'])*scores+h['eta']*norm(q)
            for j,(row,y) in enumerate(zip(g.itertuples(index=False),contexts)):
                candidates=[k for k in range(len(ids)) if k!=last[j]]
                order=sorted(candidates,key=lambda k:(-scores[j,k],-pop[k],ids[k]))
                assert order.index(target[j])+1==row.rank and [int(ids[k]) for k in order[:5]]==json.loads(row.top5),(city,name,y['context_id'])
                stable=sorted(candidates,key=lambda k:(-round(float(scores[j,k]),14),-pop[k],ids[k]))
                rank14=stable.index(target[j])+1
                rounding.append(dict(evaluation=row.evaluation,city=city,user=y['user'],model=name,
                    rank=row.rank,rank_rounded=rank14,top5_changed=stable[:5]!=order[:5],NDCG5=row.NDCG5,
                    rounded_NDCG5=float(recalc(rank14)['NDCG5'])))
                if name=='Markov':
                    poporder=sorted(candidates,key=lambda k:(-pop[k],ids[k]))
                    rank_pop=poporder.index(target[j])+1
                    common=dict(evaluation=row.evaluation,subset='all',city=city,user=y['user'],context_id=y['context_id'])
                    h3rows.extend([dict(common,model='Markov',rank=row.rank,**{k:getattr(row,k) for k in KEYS}),
                                   dict(common,model='Popularity',rank=rank_pop,**{k:float(v) for k,v in recalc(rank_pop).items()})])
                    distance=(float(c.lat.iloc[last[j]]),float(c.lon.iloc[last[j]]),float(c.lat.iloc[target[j]]),float(c.lon.iloc[target[j]]))
                    la,lo,lb,lblo=map(math.radians,distance)
                    km=12742*math.asin(math.sqrt(min(1,math.sin((la-lb)/2)**2+math.cos(la)*math.cos(lb)*math.sin((lo-lblo)/2)**2)))
                    contextinfo.append(dict(city=city,cohort=row.evaluation,context_id=y['context_id'],user=y['user'],
                        target=y['target'],last_poi=y['prefix'][-1],target_visited=y['target'] in y['old_items']+y['prefix'],
                        km_from_last=km,old_visits=len(y['old_items']),prefix_visits=len(y['prefix']),
                        gap_days=(y['sequence_start']-max(y['old_end_times']))/86400 if y['old_end_times'] else None,
                        both_have_cultural_features=bool(available[j]),cultural_distance=drift[j],
                        observed_transition_count=int(tr[last[j],target[j]]),markov_rank=row.rank,popularity_rank=rank_pop))
        freq=el.groupby(['property','target']).poiID.nunique()
        cityinfo.append(dict(city=city,places=len(c),edges=len(el),places_with_edges=el.poiID.nunique(),
                             feature_values=len(freq),shared_feature_values=int((freq>=2).sum()),
                             referenced_edges=int((el.reference_count>0).sum())))
    passed('Обучающие матрицы','Независимо посчитаны частоты мест и переходов только до отсечки каждого из восьми городов.',len(CITIES))
    passed('Графовые матрицы','Все наборы признаков собраны заново из рёбер; ядра диффузии проверены обращением матрицы и сопоставлены с сохранёнными.',len(CITIES))
    passed('Полное ранжирование','Независимый расчёт совпал с каждым тестовым рангом и топ-5 для всех вариантов второй серии.',len(pred))
    stability=[]
    for (ev,name),gg in pd.DataFrame(rounding).groupby(['evaluation','model']):
        stability.append(dict(evaluation=ev,model=name,contexts=len(gg),changed_target_ranks=int((gg['rank']!=gg.rank_rounded).sum()),
                              changed_top5=int(gg.top5_changed.sum()),delta_NDCG5=by_city_macro(gg,'rounded_NDCG5')-by_city_macro(gg)))
    pd.DataFrame(stability).to_csv(OUT/'rounding_sensitivity.csv',index=False)
    hh=pd.DataFrame(h3rows);hh.to_csv(OUT/'h3_predictions.csv',index=False)
    diag=pd.DataFrame(contextinfo);diag.to_csv(OUT/'context_diagnostics.csv',index=False)
    cd=pd.DataFrame(cityinfo);cd.to_csv(OUT/'coverage.csv',index=False)
    data=hh[hh.evaluation.eq('transfer')]
    h3=paired_inference(data,'Markov','Popularity',subset='all',cities=NEW)
    h3['Markov']=by_city_macro(data[data.model.eq('Markov')]);h3['Popularity']=by_city_macro(data[data.model.eq('Popularity')])
    h3['user_macro']={m:float(g.groupby('user').NDCG5.mean().mean()) for m,g in data.groupby('model')}
    h3['leave_one_city_out']={city:by_city_macro(data[(data.city.ne(city))&(data.model=='Markov')])-
                                 by_city_macro(data[(data.city.ne(city))&(data.model=='Popularity')]) for city in NEW}
    h3['city_results']={city:{m:float(g.groupby('user').NDCG5.mean().mean()) for m,g in part.groupby('model')}
                        for city,part in data.groupby('city')}
    h3['supported']=bool(h3['delta']>0 and h3['ci_low']>0 and h3['p_two_sided']<.05)
    (OUT/'h3_summary.json').write_text(json.dumps(h3,indent=2))
    newdiag=diag[diag.cohort.eq('transfer')]
    paired=pred[pred.evaluation.eq('transfer')].pivot(index='context_id',columns='model',values='rank')
    global_cut=max(cuts[c]['validation_end'] for c in ['Buda','Vien'])
    after={x for x in newdiag.context_id if lookup[x]['target_time']>global_cut}
    timing=[
        dict(contrast='H3',**paired_inference(data[data.context_id.isin(after)],'Markov','Popularity',subset='all',cities=NEW)),
        dict(contrast='H1',**paired_inference(p2[p2.evaluation.eq('transfer')&p2.context_id.isin(after)],
                                              'GraphPositive','CategoryMatched',subset='all',cities=NEW)),
        dict(contrast='end_to_end',**paired_inference(p2[p2.evaluation.eq('transfer')&p2.context_id.isin(after)],
                                                    'AdaptiveFree','Markov',subset='all',cities=NEW)),
    ]
    pd.DataFrame(timing).to_csv(OUT/'common_date_sensitivity.csv',index=False)
    (OUT/'interpretation.json').write_text(json.dumps(dict(
        transfer_graph_better=int((paired.GraphPositive<paired.Markov).sum()),
        transfer_graph_same=int((paired.GraphPositive==paired.Markov).sum()),
        transfer_graph_worse=int((paired.GraphPositive>paired.Markov).sum()),
        transfer_repeats=int(newdiag.target_visited.sum()),
        transfer_within_km=int(newdiag.km_from_last.le(1).sum()),
        transfer_before_final_dev_cutoff=sum(lookup[x]['target_time']<=global_cut for x in newdiag.context_id),
        transfer_with_both_profiles=int((newdiag.old_visits.gt(0)&newdiag.prefix_visits.ge(2)).sum()),
        transfer_with_both_profiles_and_metadata=int((newdiag.old_visits.gt(0)&newdiag.prefix_visits.ge(2)&newdiag.both_have_cultural_features).sum()),
        transfer_linked_values=int(cd[cd.city.isin(NEW)].shared_feature_values.sum()),
        transfer_feature_values=int(cd[cd.city.isin(NEW)].feature_values.sum()),
    ),indent=2))
    cityout=[]
    for (cohort,city),g in diag.groupby(['cohort','city']):
        two=g[g.old_visits.gt(0)&g.prefix_visits.ge(2)]
        cityout.append(dict(cohort=cohort,city=city,contexts=len(g),users=g.user.nunique(),
                            repeat_targets=int(g.target_visited.sum()),median_distance_km=float(g.km_from_last.median()),
                            within_1km=int(g.km_from_last.le(1).sum()),two_profiles=len(two),
                            both_metadata=int(two.both_have_cultural_features.sum()),
                            median_gap_days=g.gap_days.median(),unknown_transition=int(g.observed_transition_count.eq(0).sum())))
    pd.DataFrame(cityout).to_csv(OUT/'diagnostic_summary.csv',index=False)
    files=['source/processed/photos_clean.csv','source/processed/visits_prepared.csv',
           'data/contexts_main.jsonl','data/cultural_edges.csv','results/predictions.csv',
           'results/metrics.csv','results/paired_inference.csv','results/selected_parameters.json',
           'followup/results/transfer_contexts.jsonl','followup/results/transfer_edges.csv',
           'followup/results/predictions.csv','followup/results/metrics.csv','followup/results/parameters.json',
           'followup/results/contrasts.csv','followup/results/graph_placebo.csv','followup/results/nonzero_gate_contrasts.csv']
    hashes=[]
    manifest=OUT/'rebuild_hashes.csv'
    frozen=read(manifest).set_index('file').sha256.to_dict() if manifest.exists() else {}
    if previous or frozen:
        for rel in files:
            x=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
            y=hashlib.sha256((previous/rel).read_bytes()).hexdigest() if previous else frozen[rel]
            assert x==y,rel
            hashes.append(dict(file=rel,sha256=x,identical=True))
        pd.DataFrame(hashes).to_csv(manifest,index=False)
        passed('Повторный запуск','Ключевые файлы обеих серий и подготовки совпали с предыдущей версией побайтно.',len(hashes))
    (OUT/'audit_summary.json').write_text(json.dumps(dict(checks=checks,objects=len(cat),edges=len(edges),
        referenced_edges=refs,claims_from_merged_entities=len(other_entity),rebuild_matches=len(hashes)),ensure_ascii=False,indent=2))
    print(json.dumps(h3,indent=2));print(pd.DataFrame(checks).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path)
    main(p.parse_args().previous)
