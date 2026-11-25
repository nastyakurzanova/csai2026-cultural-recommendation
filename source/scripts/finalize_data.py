"""Create conservative screening tables, lineage, and diagnostics, not evaluation splits."""
from pathlib import Path
import pandas as pd,numpy as np,json
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'processed'
def save(df,name):df.to_csv(P/name,index=False,encoding='utf-8')
def main():
    cat=pd.read_csv(P/'poi_catalog_audited.csv')
    maps=[]
    for _,g in cat.groupby(['city','decoded_name','lat','lon'],dropna=False):
        canonical=int(g.poiID.min())
        for _,r in g.iterrows():maps.append({'city':r.city,'original_poiID':int(r.poiID),'poiID':canonical,'reason':'exact duplicate name and coordinates' if len(g)>1 else 'unchanged'})
    alias=pd.DataFrame(maps);save(alias,'poi_aliases.csv')
    c=cat.rename(columns={'poiID':'original_poiID'}).merge(alias,on=['city','original_poiID'],validate='one_to_one')
    canonical=[]
    for (city,poi),g in c.groupby(['city','poiID']):
        r=g.iloc[0]
        canonical.append({'city':city,'poiID':poi,'name':r.decoded_name,'lat':r.lat,'lon':r.lon,'source_poi_ids':'|'.join(g.original_poiID.astype(str)),'source_categories':'|'.join(sorted(g.theme.unique())),'category_candidate':bool((g.cultural_status=='candidate_by_category').any()),'scope_verified':False,'coordinate_outlier_50km':bool(g.coordinate_outlier_50km.any())})
    canon=pd.DataFrame(canonical)
    alignment=pd.read_csv(P/'wikidata_alignment.csv')
    a=alignment.rename(columns={'poiID':'original_poiID'}).merge(alias,on=['city','original_poiID'],validate='one_to_one')
    a=a.drop_duplicates(['city','poiID'])
    canon=canon.merge(a[['city','poiID','wikidata_id','wikidata_label','match_status','distance_km','match_method','revision_id']],on=['city','poiID'],how='left',validate='one_to_one')
    canon['coordinate_screen_passed']=canon.match_status.eq('exact_sitelink_coordinate_consistent')&~canon.coordinate_outlier_50km
    facts=pd.read_csv(P/'cultural_statements.csv')
    disambig=facts[(facts.property=='P31')&facts.target_entity.isin(['Q4167410','Q4167836'])]
    disambig_ids=set(disambig.wikidata_id)
    canon['is_disambiguation_or_category']=canon.wikidata_id.isin(disambig_ids)
    canon['shared_entity_requires_review']=canon.wikidata_id.notna()&canon.duplicated(['city','wikidata_id'],keep=False)
    canon['entity_usable_for_screening']=canon.coordinate_screen_passed&~canon.is_disambiguation_or_category&~canon.shared_entity_requires_review
    save(canon,'poi_catalog_prepared.csv')
    shared=canon[canon.wikidata_id.notna()&canon.duplicated(['city','wikidata_id'],keep=False)]
    save(shared,'shared_entity_review.csv')
    d=pd.read_csv(P/'photos_clean.csv',dtype={'photoID':str,'userID':str})
    d=d.rename(columns={'poiID':'original_poiID'}).merge(alias,on=['city','original_poiID'],validate='many_to_one')
    d=d.sort_values(['city','userID','seqID','dateTaken','photoID'],kind='stable')
    d['time_ambiguous']=d.groupby(['city','userID','seqID','dateTaken']).poiID.transform('nunique')>1
    old=d[['city','userID','seqID','poiID']].shift()
    start=d[['city','userID','seqID','poiID']].ne(old).any(axis=1)
    d['stop_no']=start.cumsum()
    v=d.groupby('stop_no',sort=False).agg(city=('city','first'),userID=('userID','first'),seqID=('seqID','first'),poiID=('poiID','first'),start_time=('dateTaken','min'),end_time=('dateTaken','max'),photo_count=('photoID','size'),time_ambiguous=('time_ambiguous','any')).reset_index(drop=True)
    v=v.merge(canon[['city','poiID','category_candidate','coordinate_screen_passed','entity_usable_for_screening']],on=['city','poiID'],validate='many_to_one')
    seq=v.groupby(['city','userID','seqID']).agg(start=('start_time','min'),end=('end_time','max'),ambiguous=('time_ambiguous','any'),visits=('poiID','size'))
    seq['date_review_required']=seq.start<946684800 # Explicit diagnostic cutoff, not a claim all earlier dates are false.
    seq['screening_eligible']=~seq.ambiguous&~seq.date_review_required
    seq=seq.reset_index().sort_values(['city','userID','start','end','seqID'])
    seq['previous_max_end']=seq.groupby(['city','userID']).end.cummax().groupby([seq.city,seq.userID]).shift()
    seq['overlaps_previous']=seq.start<=seq.previous_max_end
    seq['screening_eligible'] &=~seq.overlaps_previous
    v=v.merge(seq[['city','userID','seqID','screening_eligible','date_review_required']],on=['city','userID','seqID'],validate='many_to_one')
    v['visit_id']=[f'V{i+1:07d}' for i in range(len(v))]
    save(v,'visits_prepared.csv');save(seq,'sequence_audit.csv')
    # Cultural subset is deliberately labelled provisional; catalogue source categories are not final heritage truth.
    cv=v[v.screening_eligible&v.category_candidate&v.entity_usable_for_screening].copy()
    save(cv,'cultural_visits_provisional.csv')
    s=cv.groupby(['city','userID','seqID']).agg(n=('poiID','size'),start=('start_time','min'),end=('end_time','max')).reset_index().sort_values(['city','userID','start','end','seqID'])
    s['previous_candidate_visits']=s.groupby(['city','userID']).n.cumsum()-s.n
    s['h2_diagnostic_eligible']=(s.n>=3)&(s.previous_candidate_visits>=3)
    save(s,'history_feasibility_provisional.csv')
    # Rows with unresolved IDs must remain visible as pending, never counted as zero cultural knowledge.
    cov=[]
    for city,g in canon.groupby('city'):
        us=v[v.city==city];sq=seq[seq.city==city];cvs=cv[cv.city==city];cs=s[s.city==city]
        eligible=cs[cs.h2_diagnostic_eligible]
        row={'city':city,'canonical_pois':len(g),'linked_pois':int(g.wikidata_id.notna().sum()),'coordinate_screen_passed_pois':int(g.coordinate_screen_passed.sum()),'entity_usable_for_screening_pois':int(g.entity_usable_for_screening.sum()),'category_candidate_pois':int(g.category_candidate.sum()),'category_candidate_linked_coordinate_passed_pois':int((g.category_candidate&g.coordinate_screen_passed).sum()),'visits_after_duplicate_merge':len(us),'ambiguous_sequences_excluded_from_screening':int(sq.ambiguous.sum()),'old_date_sequences_pending_review':int(sq.date_review_required.sum()),'screening_sequences':int(sq.screening_eligible.sum()),'provisional_cultural_visits':len(cvs),'provisional_cultural_sequences_2plus':int((cs.n>=2).sum()),'h2_diagnostic_users':eligible.userID.nunique(),'h2_diagnostic_sequences':len(eligible),'h2_diagnostic_prediction_positions':int((eligible.n-2).sum())}
        for prop in ['P149','P84','P571','P1435','P793','P138','P921']:
            q=set(facts[(facts.city==city)&(facts.property==prop)].poiID)
            row[f'original_pois_with_{prop}']=len(q)
        cov.append(row)
    save(pd.DataFrame(cov),'data_readiness.csv')
    totals={'raw_photo_rows':int(pd.read_csv(P/'city_audit.csv').raw_photo_rows.sum()),'quarantined_photo_rows':int(pd.read_csv(P/'city_audit.csv').invalid_rows_quarantined.sum()),'clean_photo_rows':len(d),'raw_catalog_pois':len(cat),'canonical_pois':len(canon),'merged_duplicate_rows':len(cat)-len(canon),'prepared_visits':len(v),'aligned_original_pois':int(alignment.wikidata_id.notna().sum()),'aligned_canonical_pois':int(canon.wikidata_id.notna().sum()),'scope_verified':False,'h2_diagnostic_definition':'>=3 earlier provisional cultural visits in earlier sequences and >=3 provisional cultural visits in current sequence; short-term prefix >=2; illustrative availability diagnostic, not an approved evaluation split','caveat':'Exact-identity and coordinate screens are provisional; source-category cultural status requires individual review; no training or model evaluation was performed.'}
    (ROOT/'reports/preparation_summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(d)==int(pd.read_csv(P/'city_audit.csv').clean_photo_rows.sum())
    assert v.photo_count.sum()==len(d)
    assert not canon.duplicated(['city','poiID']).any()
    assert not v[['city','userID','seqID','poiID','start_time']].isna().any().any()
    print(json.dumps(totals,ensure_ascii=False,indent=2));print(pd.DataFrame(cov).to_string(index=False))
if __name__=='__main__':main()
