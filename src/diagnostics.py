"""Exploratory diagnostics added after the nested hybrid selected zero KG weight.

Primary protocol, primary predictions and hypothesis tests are preserved. All
positive-weight configurations are selected on validation only, not on test.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
from model import CityModel,make_features,ppr_kernel,flat_kernel,gate_alpha,ranks,metric_arrays
from experiment import readjl,macro,best_search,aggregate,paired_inference,combine
from research_io import atomic_csv
ROOT=Path(__file__).resolve().parents[1]

def main():
    note={'reason':'The original validation-optimal nested hybrid selected eta=0 for all metadata representations. Its corruption/ablation robustness is mathematically constant. The following exploratory analyses assess an explicitly positive metadata component. They do not replace primary H1/H2 or constitute independent confirmation.','selection':'reuse the saved validation grid; require eta>0, no new test-based tuning','reported':'all forced variants and all seeds, including losses','primary_protocol_unchanged':True}
    (ROOT/'results/exploratory_diagnostic_note.json').write_text(json.dumps(note,indent=2))
    conf=json.loads((ROOT/'protocol.json').read_text());params=json.loads((ROOT/'results/selected_parameters.json').read_text());grid=pd.read_csv(ROOT/'results/validation_search.csv')
    selected={}
    for base in ['CulturalFixed','CategoryFixed','CulturalFlatFixed']:
        r=grid[(grid.model==base)&(grid.eta>0)].sort_values('validation_user_macro_NDCG5',ascending=False,kind='stable').iloc[0]
        selected[base+'Forced']={k:float(r[k]) for k in ['restart','eta','alpha'] if k in r and pd.notna(r[k])}
    items=pd.read_csv(ROOT/'data/items.csv');edges=pd.read_csv(ROOT/'data/cultural_edges.csv');seq=readjl(ROOT/'data/sequences_main.jsonl');contexts=readjl(ROOT/'data/contexts_main.jsonl');cut=json.loads((ROOT/'data/temporal_cutoffs.json').read_text())
    models={};batches={};K={};rows=[];trial=[]
    for city in ['Buda','Vien','Edin']:
        m=CityModel.fit(city,items[items.city==city],edges[edges.city==city],[s for s in seq if s['city']==city and s['end']<=cut[city]['train_end']]);models[city]=m
        batches[city]={}
        for sp in ['validation','test']:
            for su in ['all','warm']:batches[city][(sp,su)]=m.batch([c for c in contexts if c['city']==city and c['split']==sp and (su=='all' or c['warm'])])
        for name,p in selected.items():
            if 'Flat' in name:k=flat_kernel(m.features,m.pop)
            else:
                F,_,_,_=make_features(m.items,edges[edges.city==city],cultural=('Cultural' in name));k=ppr_kernel(F,p['restart'],m.pop)
            K[(city,name)]=combine(m.collaborative,k,p['eta'])
    vb={c:batches[c][('validation','warm')] for c in ['Buda','Vien']}
    G=conf['grids'];configs=[{'minimum':a,'maximum':b,'c':c} for a in G['gate_minimum'] for b in G['gate_maximum'] if b>a for c in G['gate_confidence_c']]
    selected['CulturalAdaptiveForced']=best_search('CulturalAdaptiveForced',configs,lambda city,b,p:models[city].score_kernel(b,K[(city,'CulturalFixedForced')],gate_alpha(b['drift'],b['length'],p['minimum'],p['maximum'],p['c'])),models,vb,trial)
    # Same positive component and profile weight as cultural variant, with cultural edges removed.
    cp=selected['CulturalFixedForced']
    selected['NoCultureMatchedForced']=cp.copy()
    for city,m in models.items():
        F,_,_,_=make_features(m.items,edges[edges.city==city],cultural=False)
        K[(city,'NoCultureMatchedForced')]=combine(m.collaborative,ppr_kernel(F,cp['restart'],m.pop),cp['eta'])
        md=ROOT/'models'/city;old=np.load(md/'weights.npz');weights={k:old[k] for k in old.files};old.close();weights['cultural_forced_hybrid']=K[(city,'CulturalFixedForced')]
        np.savez_compressed(md/'weights.npz',**weights)
        meta=json.loads((md/'metadata.json').read_text());meta['parameters'].update(selected);meta['forced_variant_status']='exploratory positive-graph component chosen on validation; primary unrestricted result preserved'
        (md/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    threshold=json.loads((ROOT/'results/drift_threshold.json').read_text())['training_75th_percentile']
    for city,m in models.items():
        for sp in ['validation','test']:
            for su in ['all','warm']:
                b=batches[city][(sp,su)]
                for name,p in selected.items():
                    if name=='CulturalAdaptiveForced':alpha=gate_alpha(b['drift'],b['length'],p['minimum'],p['maximum'],p['c']);kernel=K[(city,'CulturalFixedForced')]
                    else:alpha=p['alpha'];kernel=K[(city,name)]
                    score=m.score_kernel(b,kernel,alpha);rank,_=ranks(score,b['target'],b['last'],m.pop,m.ids);met=metric_arrays(rank)
                    for j,c in enumerate(b['contexts']):
                        rows.append({k:c[k] for k in ['context_id','city','user','seqID','target','target_time']}|dict(split=sp,subset=su,model=name,rank=int(rank[j]),prefix_length=len(c['prefix']),old_visits=len(c['old_items']),category_drift=float(b['category_drift'][j]),cultural_drift=float(b['drift'][j]),high_drift=bool(b['category_drift'][j]>=threshold),alpha=float(np.broadcast_to(alpha,(len(rank),))[j]),**{k:float(v[j]) for k,v in met.items()}))
    d=pd.DataFrame(rows);extra=[]
    for label,flag in [('high_drift',True),('low_drift',False)]:
        z=d[(d.subset=='warm')&(d.high_drift==flag)].copy();z['subset']=label;extra.append(z)
    d=pd.concat([d]+extra,ignore_index=True);atomic_csv(d,ROOT/'results/exploratory_predictions.csv')
    pred=pd.read_csv(ROOT/'results/predictions.csv');combined=pd.concat([pred,d],ignore_index=True)
    aggregate(combined).to_csv(ROOT/'results/metrics_all_variants.csv',index=False)
    (ROOT/'results/exploratory_parameters.json').write_text(json.dumps(selected,indent=2))
    pd.DataFrame(trial).to_csv(ROOT/'results/exploratory_validation_search.csv',index=False)
    comparisons=[]
    for a,b,su in [('CulturalFixedForced','CategoryFixedForced','warm'),('CulturalFixedForced','NoCultureMatchedForced','warm'),('CulturalFixedForced','CulturalFlatFixedForced','warm'),('CulturalAdaptiveForced','CulturalFixedForced','warm'),('CulturalAdaptiveForced','CulturalFixedForced','high_drift'),('CulturalFixedForced','Markov','warm')]:
        comparisons.append(paired_inference(combined[combined.split=='test'],a,b,su))
    prev=0
    for i,row in enumerate(sorted(comparisons,key=lambda r:r.get('p_two_sided',1))):
        prev=max(prev,min(1,(len(comparisons)-i)*row['p_two_sided']));row['p_holm_exploratory_family']=prev
    pd.DataFrame(comparisons).to_csv(ROOT/'results/exploratory_inference.csv',index=False)
    # Corruptions use the strictly positive cultural component chosen on validation.
    cp=selected['CulturalFixedForced'];rob=[]
    for condition in ['shuffle','drop10','drop25','drop50','removeP84','removeP149','removeP138']:
        for seed in range(20 if condition=='shuffle' else (10 if condition.startswith('drop') else 1)):
            values={};swaps=0
            for city in ['Buda','Vien']:
                m=models[city];e=edges[edges.city==city].copy();drop={'drop10':.1,'drop25':.25,'drop50':.5}.get(condition,0)
                if drop:e=e[np.random.default_rng(conf['seed']+seed).random(len(e))>=drop]
                F,_,_,sw=make_features(m.items,e,cultural=True,drop_property=condition[6:] if condition.startswith('remove') else None,shuffle_seed=seed if condition=='shuffle' else None)
                k=ppr_kernel(F,cp['restart'],m.pop);b=batches[city][('test','warm')];score=m.score_kernel(b,combine(m.collaborative,k,cp['eta']),cp['alpha'])
                rank,_=ranks(score,b['target'],b['last'],m.pop,m.ids);values[city]=metric_arrays(rank)['NDCG5'];swaps+=sum(sw.values())
            rob.append({'condition':condition,'seed':seed,'NDCG5_main_macro':macro(values,{c:batches[c][('test','warm')] for c in ['Buda','Vien']}),'accepted_swaps':swaps,'exploratory':True})
    pd.DataFrame(rob).to_csv(ROOT/'results/exploratory_robustness.csv',index=False)
    # Exploratory stability across two fixed chronological origins with transferred settings.
    origins=[]
    for city,m0 in models.items():
        vs=sorted([s for s in seq if s['city']==city],key=lambda s:(s['end'],s['start'],s['seqID']))
        for fraction in [.5,.7]:
            boundary=vs[int(len(vs)*fraction)-1]['end'];end=vs[min(len(vs)-1,int(len(vs)*(fraction+.2))-1)]['end']
            m=CityModel.fit(city,m0.items,edges[edges.city==city],[s for s in vs if s['end']<=boundary])
            cs=[c for c in contexts if c['city']==city and c['warm'] and boundary<c['sequence_end']<=end]
            if not cs:continue
            b=m.batch(cs);H=combine(m.collaborative,ppr_kernel(m.features,cp['restart'],m.pop),cp['eta']);ga=selected['CulturalAdaptiveForced']
            for name,score in [('CulturalFixedForced',m.score_kernel(b,H,cp['alpha'])),('CulturalAdaptiveForced',m.score_kernel(b,H,gate_alpha(b['drift'],b['length'],ga['minimum'],ga['maximum'],ga['c']))),('Markov',m.transitions[b['last']])]:
                rank,_=ranks(score,b['target'],b['last'],m.pop,m.ids);df=pd.DataFrame({'user':b['users'],**metric_arrays(rank)})
                origins.append({'city':city,'origin_fraction':fraction,'model':name,'contexts':len(df),'users':df.user.nunique(),**df.groupby('user')[['NDCG5','HR5','MRR']].mean().mean().to_dict(),'exploratory':True,'overlap_with_original_periods':True})
    pd.DataFrame(origins).to_csv(ROOT/'results/exploratory_temporal_origins.csv',index=False)
    print(json.dumps(selected,indent=2));print(pd.DataFrame(comparisons).to_string(index=False))
if __name__=='__main__':main()
