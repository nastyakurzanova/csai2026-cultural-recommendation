"""Descriptive validity checks, novel-attraction sensitivity, intervals and figures."""
from pathlib import Path
import json,numpy as np,pandas as pd,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import HeritageRecommender,make_features,ppr_kernel,flat_kernel,normalize,row_kernel,ranks,metric_arrays,gate_alpha
from experiment import readjl,aggregate,paired_inference,combine
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
NAMES={'Popularity':'Популярность','Markov':'Переходы Маркова','ItemKNN':'ItemKNN','EASE':'EASE','GeoFixed':'Географическая модель','CategoryFixed':'Категории, свободный вес','CulturalFixed':'Граф, свободный вес','CulturalAdaptive':'Граф и адаптация','CulturalFixedForced':'Граф, вклад ≥ 0,25','CulturalFlatFixedForced':'Те же сведения без диффузии','CategoryFixedForced':'Категории, вклад ≥ 0,25','CulturalAdaptiveForced':'Граф ≥ 0,25 и адаптация'}
def score_for(m,z,b,method,params):
    if method=='Popularity':return np.tile(m.pop,(len(b['L']),1))
    if method=='Markov':return normalize(m.transitions[b['last']]+params['Markov']['prior']*m.pop)
    if method=='ItemKNN':return m.score_kernel(b,m.collaborative,params['ItemKNN']['alpha'])
    if method=='EASE':return b['A']@z['ease_weights']
    if method=='CulturalFixed':return m.score_kernel(b,z['cultural_hybrid'],params[method]['alpha'])
    if method=='CulturalFixedForced':return m.score_kernel(b,z['cultural_forced_hybrid'],params[method]['alpha'])
    if method in ['CulturalAdaptive','CulturalAdaptiveForced']:
        p=params[method];a=gate_alpha(b['drift'],b['length'],p['minimum'],p['maximum'],p['c'])
        return m.score_kernel(b,z['cultural_hybrid' if method=='CulturalAdaptive' else 'cultural_forced_hybrid'],a)
    raise KeyError(method)
def main():
    mainpred=pd.read_csv(ROOT/'results/predictions.csv');ex=pd.read_csv(ROOT/'results/exploratory_predictions.csv');allpred=pd.concat([mainpred,ex],ignore_index=True)
    metrics=aggregate(allpred);metrics.to_csv(ROOT/'results/metrics_all_variants.csv',index=False)
    params=json.loads((ROOT/'results/selected_parameters.json').read_text());params.update(json.loads((ROOT/'results/exploratory_parameters.json').read_text()))
    contexts=readjl(ROOT/'data/contexts_main.jsonl');byid={c['context_id']:c for c in contexts}
    items=pd.read_csv(ROOT/'data/items.csv');idx=items.set_index(['city','poiID'])
    characteristics=[]
    for c in contexts:
        a=idx.loc[(c['city'],c['prefix'][-1])];b=idx.loc[(c['city'],c['target'])]
        la,lb=np.radians(a.lat),np.radians(b.lat);dl=np.radians(a.lon-b.lon)
        distance=6371*2*np.arcsin(np.sqrt(np.clip(np.sin((la-lb)/2)**2+np.cos(la)*np.cos(lb)*np.sin(dl/2)**2,0,1)))
        characteristics.append({'context_id':c['context_id'],'city':c['city'],'user':c['user'],'split':c['split'],'warm':c['warm'],'repeat_target':c['target'] in c['old_items']+c['prefix'],'gap_days':(c['sequence_start']-max(c['old_end_times']))/86400 if c['old_end_times'] else None,'distance_km':distance})
    ch=pd.DataFrame(characteristics);ch.to_csv(ROOT/'results/context_characteristics.csv',index=False)
    desc=[]
    for city,g in ch[(ch.split=='test')&ch.warm].groupby('city'):
        desc.append({'city':city,'contexts':len(g),'users':g.user.nunique(),'repeat_share':g.repeat_target.mean(),'median_previous_sequence_gap_days':g.gap_days.median(),'share_gap_atleast7days':(g.gap_days>=7).mean(),'share_gap_atleast30days':(g.gap_days>=30).mean(),'median_next_distance_km':g.distance_km.median(),'share_next_within1km':(g.distance_km<=1).mean()})
    pd.DataFrame(desc).to_csv(ROOT/'results/descriptive_validity.csv',index=False)
    # Absolute user-macro confidence intervals; zero is only an algebraic reference,
    # never represented as an actual recommender in the output tables.
    absol=[];test=allpred[(allpred.split=='test')&(allpred.subset=='warm')].copy()
    for name in NAMES:
        x=test[test.model==name].copy();zero=x.copy();zero['model']='algebraic_zero_reference';zero['NDCG5']=0
        ci=paired_inference(pd.concat([x,zero]),name,'algebraic_zero_reference')
        absol.append({'model':name,'NDCG5':ci['delta'],'ci_low':ci['ci_low'],'ci_high':ci['ci_high'],'users_global':ci['users_global'],'contexts':ci['contexts']})
    absolute=pd.DataFrame(absol);absolute.to_csv(ROOT/'results/absolute_intervals.csv',index=False)
    # Novel targets only; all previously observed POIs are removed from the candidate set.
    # Selection is target-based for this explicitly secondary task, never used for tuning.
    novel=[];timings=[];examples=[]
    methods=['Popularity','Markov','ItemKNN','EASE','CulturalFixed','CulturalAdaptive','CulturalFixedForced','CulturalAdaptiveForced']
    for city in ['Buda','Vien','Edin']:
        rec=HeritageRecommender(ROOT/'models'/city);m=rec.model;z=rec.z
        cs=[c for c in contexts if c['city']==city and c['split']=='test' and c['warm'] and c['target'] not in c['old_items']+c['prefix']]
        b=m.batch(cs);counts=[]
        for c in cs:counts.append(len(m.ids)-len(set(c['old_items']+c['prefix'])))
        for method in methods:
            scores=score_for(m,z,b,method,params)
            for j,c in enumerate(cs):
                for p in set(c['old_items']+c['prefix']):scores[j,m.index[p]]=-np.inf
            rank,_=ranks(scores,b['target'],b['last'],m.pop,m.ids);mets=metric_arrays(rank)
            for j,c in enumerate(cs):novel.append({k:c[k] for k in ['context_id','city','user']}|dict(split='test',subset='novel_only',model=method,candidates=counts[j],**{k:float(v[j]) for k,v in mets.items()}))
        # End-to-end in-process inference; model loading is excluded, no network.
        sample=next(c for c in contexts if c['city']==city and c['split']=='test' and c['warm'])
        for variant in ['optimized','graph_positive']:
            rr=HeritageRecommender(ROOT/'models'/city,variant)
            for _ in range(5):rr.recommend(sample['prefix'],sample['old_items'])
            elapsed=[]
            for _ in range(100):
                t=time.perf_counter_ns();out=rr.recommend(sample['prefix'],sample['old_items']);elapsed.append((time.perf_counter_ns()-t)/1e6)
            timings.append({'city':city,'variant':variant,'median_ms':float(np.median(elapsed)),'p95_ms':float(np.quantile(elapsed,.95)),'runs':100,'definition':'CPU in-process recommendation including constructing profiles and formatting top5; excludes model loading, network and UI'})
            examples.append({'city':city,'context_id':sample['context_id'],'user':sample['user'],'prefix':sample['prefix'],'old_items':sample['old_items'],'observed_target':sample['target'],'variant':variant,'result':out,'selection':'first chronological eligible test context by input file order; not selected for success'})
    novel=pd.DataFrame(novel);novel.to_csv(ROOT/'results/novel_target_predictions.csv',index=False)
    aggregate(novel).to_csv(ROOT/'results/novel_target_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(ROOT/'results/inference_timing.csv',index=False)
    (ROOT/'results/inference_examples.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False))
    inf=[]
    for a,b in [('CulturalAdaptive','CulturalFixed'),('CulturalFixedForced','CulturalFixed'),('CulturalFixedForced','Markov')]:
        inf.append(paired_inference(novel,a,b,'novel_only'))
    pd.DataFrame(inf).to_csv(ROOT/'results/novel_target_inference.csv',index=False)
    # Long-gap secondary stratum is derived without considering model success.
    gap=ch[(ch.gap_days>=7)&ch.warm][['context_id']]
    gappred=test.merge(gap,on='context_id',validate='many_to_one');gappred['subset']='previous_sequence_gap7days'
    aggregate(gappred).to_csv(ROOT/'results/long_gap_metrics.csv',index=False)
    # Publication figures use all specified comparisons; no selection of best seed.
    fig,ax=plt.subplots(figsize=(7.1,3.6));order=['Popularity','ItemKNN','EASE','GeoFixed','Markov','CulturalFixedForced','CulturalFixed','CulturalAdaptive']
    a=absolute.set_index('model').loc[order];y=np.arange(len(order))
    ax.errorbar(a.NDCG5,y,xerr=np.vstack([a.NDCG5-a.ci_low,a.ci_high-a.NDCG5]),fmt='o',color='#164f82',capsize=3)
    ax.set_yticks(y,[NAMES[x] for x in order]);ax.invert_yaxis();ax.set_xlabel('NDCG@5, среднее по пользователям и городам');ax.set_xlim(0,.62);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/quality_intervals.png',dpi=240);fig.savefig(ROOT/'figures/quality_intervals.pdf');plt.close(fig)
    rob=pd.read_csv(ROOT/'results/exploratory_robustness.csv');base=float(metrics.query("split=='test' and subset=='warm' and city=='MAIN_MACRO' and model=='CulturalFixedForced'").iloc[0].NDCG5)
    fig,ax=plt.subplots(figsize=(6.6,3.2));labs=['Без удаления','Удалено 10%','Удалено 25%','Удалено 50%'];means=[base];lo=[base];hi=[base]
    for cond in ['drop10','drop25','drop50']:
        v=rob[rob.condition==cond].NDCG5_main_macro;means.append(v.mean());lo.append(v.min());hi.append(v.max())
    ax.errorbar(range(4),means,yerr=[np.array(means)-lo,np.array(hi)-means],fmt='o-',color='#164f82',capsize=4,label='Среднее и диапазон 10 запусков')
    ax.set_xticks(range(4),labs);ax.set_ylabel('NDCG@5');ax.set_ylim(.18,.32);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(ROOT/'figures/edge_dropout.png',dpi=240);fig.savefig(ROOT/'figures/edge_dropout.pdf');plt.close(fig)
    print('VALIDITY\n',pd.DataFrame(desc).to_string(index=False));print('NOVEL\n',aggregate(novel).query("city=='MAIN_MACRO'").to_string(index=False));print('TIMING\n',pd.DataFrame(timings).to_string(index=False))
if __name__=='__main__':main()
