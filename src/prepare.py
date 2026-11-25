"""Audited cultural scope, canonicalisation and temporally valid event contexts."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,shutil

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source'
# Decisions below use a tourist-attraction identity, not separate buildings of an institution.
# Explicit exclusions protect against network/area identities and ambiguous contemporary entities.
GROUPS={
 'Buda':{
  'palace_castle':[1,3,4,9,12], 'performing_arts':[5,13,15,31],
  'museum_library':[6,7,20,23,24,26,27,36], 'religious':[8,18,29,33],
  'monument':[19,35], 'historic_civic':[11,16,21,32], 'historic_infrastructure':[22,37,39]},
 'Vien':{
  'palace_castle':[1,5,7], 'museum_library':[4,6,9,11,13,14,16,19,21,22,25,27,28],
  'performing_arts':[15,23], 'religious':[17,18,20],
  'historic_civic':[24,26], 'historic_infrastructure':[3,8]},
 'Edin':{
  'palace_castle':[1,2,22,25], 'museum_library':[3,15,16,24,27],
  'performing_arts':[12,13], 'religious':[9], 'monument':[8,11],
  'historic_civic':[4,17,19,23,28,29], 'historic_infrastructure':[5,6],
  'historic_garden':[20]}
}
EXCLUDE={
 'Buda':{2:'Source identity is a library network, not the identifiable palace building; unresolved physical entity.',10:'Broad urban park; no explicit heritage designation in retained snapshot; scope kept to named cultural institutions/built attractions.',14:'Current replacement bridge without a retained cultural/heritage designation; not conflated with destroyed historic bridge.',17:'Natural hill; constituent memorials are separate named attractions.',25:'River island is a broad area; no individual heritage identity validated.',28:'Urban square with no retained heritage designation; constituent monuments are not automatically attributed to entire square.',30:'Neighbourhood/municipality is not a single named attraction.',34:'Natural summit, outside current cultural scope.',38:'Shopping/pedestrian street; no heritage designation retained for entire street.'},
 'Vien':{2:'Zoo-focused institution; animal collections outside the present cultural visitor task.',10:'Aquarium/zoo organisation; not conflated with historic flak-tower building.',29:'Built structure in desert garden; cultural status or historical identity not established in retained snapshot.'},
 'Edin':{6:'included',7:'Natural mountain; outside present cultural scope.',10:'Broad hill with several distinct monuments; constituent monuments are separate candidates.',14:'Conference venue without verified heritage/cultural-collection function.',18:'Urban garden with no retained specific heritage designation; not merged with nearby memorial.',21:'Public-administration building without retained cultural/heritage designation.',26:'Merged into National Museum of Scotland attraction complex, canonical POI 3.'}
}
def norm(x):
    s=np.asarray(x,dtype=float);return s/s.sum() if s.sum()>0 else s
def main():
    if not SOURCE.exists():
        src=ROOT.parent/'csai2026/source/heritage_step6'
        if not src.exists():raise FileNotFoundError('Materialize Heritage_Step6_Data.zip and set source folder.')
        shutil.copytree(src,SOURCE)
    p=SOURCE/'processed';cat=pd.read_csv(p/'poi_catalog_prepared.csv')
    facts=pd.read_csv(p/'cultural_statements.csv')
    labels=pd.read_csv(p/'cultural_entity_labels.csv').set_index('wikidata_id').label_en.to_dict()
    decisions=[];included=[]
    for city,groups in GROUPS.items():
        lookup={int(i):g for g,ids in groups.items() for i in ids}
        for r in cat[cat.city.eq(city)].itertuples():
            inc=int(r.poiID) in lookup
            why=('Named cultural attraction, classified from identity and snapshot P31; institutional venue interpreted at attraction-complex level.' if inc else EXCLUDE[city].get(int(r.poiID),'Outside predefined scope.'))
            if city=='Vien' and r.poiID in [3,8]:why='Listed object in Austria (P1435) and a physically identified landmark; not excluded merely because modern.'
            if city=='Edin' and r.poiID==20:why='Inventory Garden and Designed Landscape (P1435); historical garden included explicitly.'
            if city=='Edin' and r.poiID==3:why='Unified National Museum of Scotland attraction at Chambers Street; original Royal Museum 26 is part of this complex. Supported by NMS architecture trail and Historic Environment Scotland LB27748.'
            decisions.append(dict(city=city,poiID=int(r.poiID),name=r.name,qid=r.wikidata_id,include=inc,category=lookup.get(int(r.poiID),''),reason=why,snapshot_date='2026-10-26',source_url='https://www.wikidata.org/wiki/'+r.wikidata_id,identity_basis='exact English sitelink or official redirect and coordinate screen; author-rule audit, no independent expert annotation'))
            if inc:included.append(dict(city=city,poiID=int(r.poiID),name=r.name,qid=r.wikidata_id,category=lookup[int(r.poiID)],lat=float(r.lat),lon=float(r.lon)))
    manifest=pd.DataFrame(decisions);manifest.to_csv(ROOT/'data/scope_decisions.csv',index=False)
    items=pd.DataFrame(included);items.to_csv(ROOT/'data/items.csv',index=False)
    v=pd.read_csv(p/'visits_prepared.csv',dtype={'userID':str})
    v.loc[(v.city=='Edin')&(v.poiID==26),'poiID']=3
    v=v[v.city.isin(GROUPS)].copy()
    # Sequence-level source flags ensure ambiguous original photo ordering is not bridged.
    s=pd.read_csv(p/'sequence_audit.csv',dtype={'userID':str})
    s=s[s.city.isin(GROUPS)].copy()
    s['base_eligible']=~s.ambiguous&~s.overlaps_previous
    validkeys=set(zip(s[s.base_eligible].city,s[s.base_eligible].userID,s[s.base_eligible].seqID))
    itemkeys=set(zip(items.city,items.poiID))
    v=v[[tuple(x) in validkeys for x in v[['city','userID','seqID']].itertuples(index=False,name=None)]]
    v=v[[tuple(x) in itemkeys for x in v[['city','poiID']].itertuples(index=False,name=None)]]
    # Re-collapse only immediate identical attraction targets after cultural filtering and museum merge.
    v=v.sort_values(['city','userID','seqID','start_time','end_time','poiID'],kind='stable')
    prev=v[['city','userID','seqID','poiID']].shift()
    v=v[v[['city','userID','seqID','poiID']].ne(prev).any(axis=1)].copy()
    v.to_csv(ROOT/'data/cultural_visits_all_dates.csv',index=False)
    # Keep raw proof for every cultural feature; snapshot claims are never renamed into invented facts.
    includedkeys=set(zip(items.city,items.poiID));edges=[]
    canonical=items.set_index(['city','poiID'])
    for r in facts.itertuples():
        poi=3 if r.city=='Edin' and r.poiID==26 else int(r.poiID)
        if (r.city,poi) not in includedkeys or r.property not in ['P84','P149','P138'] or pd.isna(r.target_entity):continue
        edges.append(dict(city=r.city,poiID=poi,property=r.property,target=r.target_entity,label=labels.get(r.target_entity,r.target_entity),statement_id=r.statement_id,rank=r.rank,reference_count=int(r.references_count),qualifiers_json=r.qualifiers_json,snapshot_revision=int(r.revision_id),source_url='https://www.wikidata.org/wiki/'+r.wikidata_id))
    edges=pd.DataFrame(edges).drop_duplicates(['city','poiID','property','target'])
    edges.to_csv(ROOT/'data/cultural_edges.csv',index=False)
    # Catalogue-level cutoffs are locked independently of relevance scores and performance.
    # Reference cutoffs use main date window and full surviving source sequences (not only long histories).
    date0=946684800
    seq=s[s.base_eligible&(s.start>=date0)].copy()
    cutoffs={}
    for city,g in seq.groupby('city'):
        ends=np.sort(g.end.to_numpy());a=int(ends[max(0,int(len(ends)*.6)-1)]);b=int(ends[max(0,int(len(ends)*.8)-1)])
        cutoffs[city]={'train_end':a,'validation_end':b}
    (ROOT/'data/temporal_cutoffs.json').write_text(json.dumps(cutoffs,indent=2))
    results=[]
    for date_min,name in [(946684800,'main'),(631152000,'from1990'),(1072915200,'from2004')]:
        contexts=[];sequences=[]
        for (city,user,seqid),g in v.groupby(['city','userID','seqID'],sort=False):
            orig=s[(s.city==city)&(s.userID==user)&(s.seqID==seqid)].iloc[0]
            if orig.start<date_min:continue
            g=g.sort_values(['start_time','end_time','poiID'],kind='stable')
            ids=g.poiID.astype(int).tolist();times=g.start_time.astype(int).tolist()
            sequences.append(dict(city=city,user=user,seqID=int(seqid),start=int(orig.start),end=int(orig.end),items=ids,times=times))
        sequences.sort(key=lambda x:(x['city'],x['user'],x['start'],x['end'],x['seqID']))
        histories={}
        for seq in sequences:
            city,user=seq['city'],seq['user'];prior=histories.setdefault((city,user),[])
            assert not prior or max(x[1] for x in prior)<seq['start']
            cutoff=cutoffs[city];split='train' if seq['end']<=cutoff['train_end'] else ('validation' if seq['end']<=cutoff['validation_end'] else 'test')
            for t in range(1,len(seq['items'])):
                ctx={'context_id':f'{city}:{user}:{seq["seqID"]}:{t}', 'city':city,'user':user,'seqID':seq['seqID'],'sequence_start':seq['start'],'sequence_end':seq['end'],'target_time':seq['times'][t], 'split':split,'old_items':[x[0] for x in prior],'old_end_times':[x[1] for x in prior], 'prefix':seq['items'][:t],'prefix_times':seq['times'][:t], 'target':seq['items'][t],'warm':len(prior)>=3 and t>=2}
                assert ctx['target']!=ctx['prefix'][-1]
                assert max(ctx['prefix_times'])<ctx['target_time']
                contexts.append(ctx)
            prior.extend((i,seq['end']) for i in seq['items'])
        with (ROOT/f'data/contexts_{name}.jsonl').open('w') as f:
            for c in contexts:f.write(json.dumps(c,ensure_ascii=False)+'\n')
        with (ROOT/f'data/sequences_{name}.jsonl').open('w') as f:
            for c in sequences:f.write(json.dumps(c,ensure_ascii=False)+'\n')
        frame=pd.DataFrame([{k:c[k] for k in ['city','user','split','warm']} for c in contexts])
        for (city,split),g in frame.groupby(['city','split']):results.append(dict(date_window=name,city=city,split=split,all_contexts=len(g),all_users=g.user.nunique(),warm_contexts=int(g.warm.sum()),warm_users=g[g.warm].user.nunique()))
    pd.DataFrame(results).to_csv(ROOT/'data/split_audit.csv',index=False)
    conf=ROOT/'protocol.json';digest=hashlib.sha256(conf.read_bytes()).hexdigest()
    (ROOT/'results/protocol_lock.json').write_text(json.dumps({'protocol_sha256':digest,'scope_sha256':hashlib.sha256((ROOT/'data/scope_decisions.csv').read_bytes()).hexdigest(),'note':'computed before first holdout scoring; local timestamp is not third-party preregistration'},indent=2))
    print(items.groupby('city').size().to_string());print(pd.DataFrame(results).to_string(index=False));print('cultural edge counts',edges.groupby('city').size().to_dict())
if __name__=='__main__':main()
