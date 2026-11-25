"""Locked temporal evaluation, validation tuning, robustness and paired inference."""
from pathlib import Path
import json,os,time,hashlib
import numpy as np,pandas as pd
from scipy.sparse.csgraph import connected_components
from model import CityModel,make_features,ppr_kernel,flat_kernel,row_kernel,gate_alpha,ranks,metric_arrays,normalize
from research_io import atomic_csv

ROOT=Path(__file__).resolve().parents[1]
CONF=json.loads((ROOT/'protocol.json').read_text());G=CONF['grids'];SEED=CONF['seed']
MAIN=CONF['main_cities'];CITIES=MAIN+[CONF['external_city']]
def readjl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def macro(values,batches):
    vals=[]
    for city,b in batches.items():
        v=values[city];df=pd.DataFrame({'u':b['users'],'v':v})
        if len(df):vals.append(df.groupby('u').v.mean().mean())
    return float(np.mean(vals)) if vals else float('nan')
def metric_score(scores,batches,models):
    values={}
    for city,b in batches.items():
        rank,_=ranks(scores[city],b['target'],b['last'],models[city].pop,models[city].ids)
        values[city]=metric_arrays(rank)['NDCG5']
    return macro(values,batches)
def combine(C,K,eta):return (1-eta)*C+eta*K
def best_search(name,configs,scorefun,models,batches,records):
    best=None;bestval=-np.inf
    for config in configs:
        scores={city:scorefun(city,b,config) for city,b in batches.items()}
        val=metric_score(scores,batches,models);records.append({'model':name,**config,'validation_user_macro_NDCG5':val})
        if val>bestval+1e-12:bestval=val;best=config.copy()
    print(name,best,'validation',round(bestval,6),flush=True)
    return best

def paired_inference(pred,model_a,model_b,subset='warm',metric='NDCG5',cities=MAIN,draws=10000):
    x=pred[(pred.model==model_a)&(pred.subset==subset)&pred.city.isin(cities)][['context_id','city','user',metric]]
    y=pred[(pred.model==model_b)&(pred.subset==subset)&pred.city.isin(cities)][['context_id',metric]]
    z=x.merge(y,on='context_id',validate='one_to_one',suffixes=('_a','_b'));z['delta']=z[metric+'_a']-z[metric+'_b']
    u=z.groupby(['city','user']).delta.mean().reset_index()
    available=[c for c in cities if (u.city==c).any()]
    if not len(u):return {'a':model_a,'b':model_b,'subset':subset,'metric':metric,'status':'no paired eligible users'}
    users=sorted(u.user.unique());ui={v:i for i,v in enumerate(users)}
    D=np.zeros((len(users),len(available)));M=np.zeros_like(D)
    for row in u.itertuples():i=ui[row.user];j=available.index(row.city);D[i,j]=row.delta;M[i,j]=1
    point=float((D.sum(0)/M.sum(0)).mean());rng=np.random.default_rng(SEED)
    boot=[];perm=[]
    for start in range(0,draws,500):
        count=min(500,draws-start);weights=rng.multinomial(len(users),np.ones(len(users))/len(users),size=count)
        den=weights@M;num=weights@D;means=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        vals=np.mean(means,axis=1);boot.extend(vals[np.isfinite(vals)])
        signs=rng.choice([-1,1],size=(count,len(users)));perm.extend(((signs@D)/M.sum(0)).mean(1))
    boot=np.asarray(boot);perm=np.asarray(perm)
    return {'a':model_a,'b':model_b,'subset':subset,'metric':metric,'delta':point,'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'p_two_sided':float((1+np.sum(np.abs(perm)>=abs(point)-1e-14))/(1+len(perm))),'users_global':len(users),'contexts':len(z),'cities':'|'.join(available),'bootstrap_draws_used':len(boot)}

def aggregate(pred):
    metrics=['NDCG5','HR5','MRR','NDCG10','HR10'];rows=[]
    for (split,subset,model,city),g in pred.groupby(['split','subset','model','city']):
        means=g.groupby('user')[metrics].mean().mean()
        rows.append(dict(split=split,subset=subset,model=model,city=city,events=len(g),users=g.user.nunique(),**means.to_dict()))
    df=pd.DataFrame(rows)
    for (split,subset,model),g in df[df.city.isin(MAIN)].groupby(['split','subset','model']):
        if len(g)==len(MAIN):
            selected=pred[(pred.split==split)&(pred.subset==subset)&(pred.model==model)&pred.city.isin(MAIN)]
            rows.append(dict(split=split,subset=subset,model=model,city='MAIN_MACRO',events=int(g.events.sum()),users=selected.user.nunique(),user_city_pairs=int(g.users.sum()),**g[metrics].mean().to_dict()))
    return pd.DataFrame(rows)

def main():
    start=time.perf_counter();items=pd.read_csv(ROOT/'data/items.csv');edges=pd.read_csv(ROOT/'data/cultural_edges.csv')
    contexts=readjl(ROOT/'data/contexts_main.jsonl');sequences=readjl(ROOT/'data/sequences_main.jsonl')
    cutoff=json.loads((ROOT/'data/temporal_cutoffs.json').read_text())
    # The protocol is frozen here; later modifications will invalidate this checksum.
    lock={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'scope_sha256':hashlib.sha256((ROOT/'data/scope_decisions.csv').read_bytes()).hexdigest(),'created_before_test_scoring':True}
    (ROOT/'results/protocol_lock.json').write_text(json.dumps(lock,indent=2))
    models={};kernels={};graphstats=[];batches={}
    for city in CITIES:
        tr=[s for s in sequences if s['city']==city and s['end']<=cutoff[city]['train_end']]
        m=CityModel.fit(city,items[items.city==city],edges[edges.city==city],tr,CONF['knn_shrinkage']);models[city]=m;kernels[city]={}
        F=m.features
        for graphkind,culture in [('category',False),('cultural',True)]:
            FF,keys,rows,_=make_features(m.items,edges[edges.city==city],cultural=culture)
            for restart in G['restart_probability']:
                K,R,P,W=ppr_kernel(FF,restart,m.pop,return_full=True);kernels[city][(graphkind,restart)]=K
            nc,labels=connected_components(W,directed=False)
            graphstats.append(dict(city=city,graph=graphkind,items=len(m.ids),feature_nodes=FF.shape[1],edges=int((FF>0).sum()),components=nc,items_with_cultural_attributes=int((m.features[:,np.array([r!='class' for r,q in m.feature_keys])].sum(1)>0).sum())))
            # Export graph used by the actual algorithm with typed relations and exact source IDs.
            nodes=[{'id':f'poi:{city}:{int(r.poiID)}','kind':'poi','label':r['name'],'qid':r.qid} for _,r in m.items.iterrows()]
            nodes +=[{'id':f'{r}:{q}','kind':r,'label':q} for r,q in keys]
            el=[{'source':f'poi:{city}:{int(m.ids[i])}','target':f'{r}:{q}','relation':r,'weight':float(FF[i,keys.index((r,q))])} for r,q,i in sorted(rows)]
            (ROOT/f'models/graph_{city}_{graphkind}.json').write_text(json.dumps({'nodes':nodes,'edges':el,'weighted_undirected':True},ensure_ascii=False,indent=2))
        kernels[city]['flat']=flat_kernel(F,m.pop)
        batches[city]={}
        for split in ['train','validation','test']:
            for subset in ['all','warm']:
                cs=[c for c in contexts if c['city']==city and c['split']==split and (subset=='all' or c['warm'])]
                batches[city][(split,subset)]=m.batch(cs)
    pd.DataFrame(graphstats).to_csv(ROOT/'results/graph_statistics.csv',index=False)
    # Drift cutoff uses functional categories of training prefixes only, independent of cultural gate features.
    drifts=np.concatenate([batches[c][('train','warm')]['category_drift'] for c in MAIN])
    threshold=float(np.quantile(drifts,.75));(ROOT/'results/drift_threshold.json').write_text(json.dumps({'training_75th_percentile':threshold,'training_contexts':len(drifts),'definition':'functional-category TV computed from observed old/prefix only'},indent=2))
    vb={c:batches[c][('validation','warm')] for c in MAIN}
    assert all(len(b['contexts']) for b in vb.values())
    records=[];params={}
    fixed=[{'restart':r,'eta':e,'alpha':a} for r in G['restart_probability'] for e in G['metadata_weight'] for a in G['fixed_short_weight']]
    for name,kind in [('CategoryFixed','category'),('CulturalFixed','cultural')]:
        params[name]=best_search(name,fixed,lambda city,b,p:models[city].score_kernel(b,combine(models[city].collaborative,kernels[city][(kind,p['restart'])],p['eta']),p['alpha']),models,vb,records)
    flatconfigs=[{'eta':e,'alpha':a} for e in G['metadata_weight'] for a in G['fixed_short_weight']]
    params['CulturalFlatFixed']=best_search('CulturalFlatFixed',flatconfigs,lambda city,b,p:models[city].score_kernel(b,combine(models[city].collaborative,kernels[city]['flat'],p['eta']),p['alpha']),models,vb,records)
    params['ItemKNN']=best_search('ItemKNN',[{'alpha':a} for a in G['fixed_short_weight']],lambda city,b,p:models[city].score_kernel(b,models[city].collaborative,p['alpha']),models,vb,records)
    params['Markov']=best_search('Markov',[{'prior':x} for x in G['markov_prior_strength']],lambda city,b,p:normalize(models[city].transitions[b['last']]+p['prior']*models[city].pop),models,vb,records)
    params['EASE']=best_search('EASE',[{'l2':x} for x in G['ease_l2']],lambda city,b,p:b['A']@models[city].ease(p['l2']),models,vb,records)
    geo={}
    for city,m in models.items():
        lat=np.radians(m.items.lat.to_numpy());lon=np.radians(m.items.lon.to_numpy());a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
        dist=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)));geo[city]=dist
    params['GeoFixed']=best_search('GeoFixed',[{'bandwidth_km':w,'alpha':a} for w in [.25,1,3,10] for a in G['fixed_short_weight']],lambda city,b,p:models[city].score_kernel(b,row_kernel(np.exp(-geo[city]/p['bandwidth_km']),models[city].pop),p['alpha']),models,vb,records)
    gconfigs=[{'minimum':a,'maximum':z,'c':c} for a in G['gate_minimum'] for z in G['gate_maximum'] if z>a for c in G['gate_confidence_c']]
    cp=params['CulturalFixed']
    fullhybrid={city:combine(m.collaborative,kernels[city][('cultural',cp['restart'])],cp['eta']) for city,m in models.items()}
    params['CulturalAdaptive']=best_search('CulturalAdaptive',gconfigs,lambda city,b,p:models[city].score_kernel(b,fullhybrid[city],gate_alpha(b['drift'],b['length'],p['minimum'],p['maximum'],p['c'])),models,vb,records)
    # Save calibration before any test ranks are produced.
    pd.DataFrame(records).to_csv(ROOT/'results/validation_search.csv',index=False)
    (ROOT/'results/selected_parameters.json').write_text(json.dumps(params,indent=2))
    predictions=[];toplists=[]
    def save_scores(city,split,subset,name,b,score,alpha=None):
        rank,order=ranks(score,b['target'],b['last'],models[city].pop,models[city].ids);met=metric_arrays(rank)
        for j,c in enumerate(b['contexts']):
            base={k:c[k] for k in ['context_id','city','user','seqID','target','target_time']}
            high=bool(b['category_drift'][j]>=threshold)
            row={**base,'split':split,'subset':subset,'model':name,'rank':int(rank[j]),'prefix_length':len(c['prefix']),'old_visits':len(c['old_items']),'category_drift':float(b['category_drift'][j]),'cultural_drift':float(b['drift'][j]),'high_drift':high,'alpha':float(np.broadcast_to(alpha,(len(rank),))[j]) if alpha is not None else None,**{k:float(v[j]) for k,v in met.items()}}
            predictions.append(row)
            if split=='test' and subset=='warm':toplists.append({**base,'model':name,'top5':json.dumps([int(x) for x in models[city].ids[order[j,:5]]]),'scores_top5':json.dumps([float(x) for x in score[j,order[j,:5]]])})
    for city,m in models.items():
        cp=params['CulturalFixed'];cat=params['CategoryFixed'];flat=params['CulturalFlatFixed'];ga=params['CulturalAdaptive']
        for split in ['validation','test']:
            for subset in ['all','warm']:
                b=batches[city][(split,subset)]
                if not len(b['contexts']):continue
                scores={'Popularity':np.tile(m.pop,(len(b['L']),1)),
                        'ItemKNN':m.score_kernel(b,m.collaborative,params['ItemKNN']['alpha']),
                        'Markov':normalize(m.transitions[b['last']]+params['Markov']['prior']*m.pop),
                        'EASE':b['A']@m.ease(params['EASE']['l2']),
                        'GeoFixed':m.score_kernel(b,row_kernel(np.exp(-geo[city]/params['GeoFixed']['bandwidth_km']),m.pop),params['GeoFixed']['alpha']),
                        'CategoryFixed':m.score_kernel(b,combine(m.collaborative,kernels[city][('category',cat['restart'])],cat['eta']),cat['alpha']),
                        'CulturalFlatFixed':m.score_kernel(b,combine(m.collaborative,kernels[city]['flat'],flat['eta']),flat['alpha']),
                        'CulturalFixed':m.score_kernel(b,fullhybrid[city],cp['alpha']),
                        'CulturalLong':m.score_kernel(b,fullhybrid[city],0),
                        'CulturalShort':m.score_kernel(b,fullhybrid[city],1)}
                al=gate_alpha(b['drift'],b['length'],ga['minimum'],ga['maximum'],ga['c']);scores['CulturalAdaptive']=m.score_kernel(b,fullhybrid[city],al)
                # Exact same full-model parameters, only cultural edges removed.
                scores['AblationNoCulture']=m.score_kernel(b,combine(m.collaborative,kernels[city][('category',cp['restart'])],cp['eta']),cp['alpha'])
                for name,score in scores.items():save_scores(city,split,subset,name,b,score,al if name=='CulturalAdaptive' else (cp['alpha'] if name=='CulturalFixed' else None))
        m.save(ROOT/f'models/{city}',params,{'cultural_hybrid':fullhybrid[city], 'cultural_kernel':kernels[city][('cultural',cp['restart'])], 'category_kernel':kernels[city][('category',cat['restart'])], 'flat_kernel':kernels[city]['flat'],'ease_weights':m.ease(params['EASE']['l2'])})
    pred=pd.DataFrame(predictions)
    # High/low drift are recorded as additional paired views of exactly the same predictions.
    for label,mask in [('high_drift',pred.high_drift),('low_drift',~pred.high_drift)]:
        z=pred[(pred.subset=='warm')&mask].copy();z['subset']=label;pred=pd.concat([pred,z],ignore_index=True)
    atomic_csv(pred,ROOT/'results/predictions.csv')
    atomic_csv(pd.DataFrame(toplists),ROOT/'results/top5_predictions.csv')
    summary=aggregate(pred);summary.to_csv(ROOT/'results/metrics.csv',index=False)
    pairs=[('H1','CulturalFixed','CategoryFixed','warm'),('H2','CulturalAdaptive','CulturalFixed','high_drift'),('secondary_all','CulturalAdaptive','CulturalFixed','warm'),('information_control','CulturalFixed','CulturalFlatFixed','warm'),('exact_ablation','CulturalFixed','AblationNoCulture','warm')]
    inf=[]
    for name,a,b,subset in pairs:
        # paired inference only on the untouched test portion
        r=paired_inference(pred[pred.split=='test'],a,b,subset);r['comparison']=name;inf.append(r)
    for city in CITIES:
        for a,b in [('CulturalFixed','CategoryFixed'),('CulturalAdaptive','CulturalFixed')]:
            r=paired_inference(pred[pred.split=='test'],a,b,'warm',cities=[city]);r['comparison']='city_'+city;inf.append(r)
    primary=[r for r in inf if r.get('comparison') in ['H1','H2'] and 'p_two_sided' in r]
    order=sorted(primary,key=lambda r:r['p_two_sided']);prev=0
    for i,r in enumerate(order):prev=max(prev,min(1,(len(order)-i)*r['p_two_sided']));r['p_holm']=prev
    pd.DataFrame(inf).to_csv(ROOT/'results/paired_inference.csv',index=False)
    # Robustness: fixed selected parameters; no seed or corruption selected for publication.
    rob=[]
    for kind,values in [('shuffle',range(20)),('drop10',range(10)),('drop25',range(10)),('drop50',range(10)),('removeP84',[0]),('removeP149',[0]),('removeP138',[0])]:
        for seed in values:
            vals={};swaps_total=0
            for city in MAIN:
                m=models[city];b=batches[city][('test','warm')]
                frac={'drop10':.1,'drop25':.25,'drop50':.5}.get(kind,0)
                F,_,_,sw=make_features(m.items,edges[edges.city==city],cultural=True,drop_property=kind[6:] if kind.startswith('remove') else None,shuffle_seed=seed if kind in ['shuffle','drop10','drop25','drop50'] else None,drop_fraction=frac)
                # Dropout must not additionally shuffle. Rebuild without swaps for dropout.
                if frac:
                    ee=edges[edges.city==city].copy();rng=np.random.default_rng(SEED+seed);ee=ee[rng.random(len(ee))>=frac]
                    F,_,_,sw=make_features(m.items,ee,cultural=True)
                K=ppr_kernel(F,cp['restart'],m.pop);score=m.score_kernel(b,combine(m.collaborative,K,cp['eta']),cp['alpha'])
                rank,_=ranks(score,b['target'],b['last'],m.pop,m.ids);vals[city]=metric_arrays(rank)['NDCG5'];swaps_total+=sum(sw.values())
            rob.append({'condition':kind,'seed':int(seed),'NDCG5_main_macro':macro(vals,{c:batches[c][('test','warm')] for c in MAIN}),'accepted_swaps':swaps_total})
    pd.DataFrame(rob).to_csv(ROOT/'results/robustness.csv',index=False)
    # Date-threshold sensitivity, with parameters frozen and identical fixed cutoffs.
    sensitivity=[]
    for name in ['from1990','from2004']:
        cc=readjl(ROOT/f'data/contexts_{name}.jsonl');ss=readjl(ROOT/f'data/sequences_{name}.jsonl')
        for city in CITIES:
            tr=[s for s in ss if s['city']==city and s['end']<=cutoff[city]['train_end']]
            m=CityModel.fit(city,items[items.city==city],edges[edges.city==city],tr,CONF['knn_shrinkage'])
            b=m.batch([c for c in cc if c['city']==city and c['split']=='test' and c['warm']]);
            K=ppr_kernel(m.features,cp['restart'],m.pop);H=combine(m.collaborative,K,cp['eta'])
            for mod,alpha in [('CulturalFixed',cp['alpha']),('CulturalAdaptive',gate_alpha(b['drift'],b['length'],ga['minimum'],ga['maximum'],ga['c']))]:
                score=m.score_kernel(b,H,alpha);rank,_=ranks(score,b['target'],b['last'],m.pop,m.ids)
                df=pd.DataFrame({'user':b['users'],**metric_arrays(rank)});sensitivity.append({'date_window':name,'city':city,'model':mod,'contexts':len(df),'users':df.user.nunique(),**df.groupby('user')[['NDCG5','HR5','MRR']].mean().mean().to_dict()})
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/date_sensitivity.csv',index=False)
    runtime={'wall_seconds':time.perf_counter()-start,'deterministic':True,'seed':SEED,'protocol_lock':lock,'test_read_after_validation_parameters_saved':True}
    (ROOT/'results/run_metadata.json').write_text(json.dumps(runtime,indent=2))
    print('\nTEST WARM\n',summary[(summary.split=='test')&(summary.subset=='warm')].to_string(index=False),flush=True)
    print('\nINFERENCE\n',pd.DataFrame(inf).to_string(index=False),flush=True)
    print('runtime',runtime['wall_seconds'],flush=True)
if __name__=='__main__':main()
