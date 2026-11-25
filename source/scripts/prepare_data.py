"""Reproducible audit and visit reconstruction. No model fitting or KG training."""
from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib, datetime
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
CITIES={'Buda':'Budapest','Delh':'Delhi','Edin':'Edinburgh','Glas':'Glasgow','Osak':'Osaka','Pert':'Perth','Toro':'Toronto','Vien':'Vienna'}
STRONG={'Historical','Cultural','Museum','Palace','Architectural','Religious','Religion','Building'}
REVIEW={'Structure','Precinct','Park','Education','Entertainment'}
def write(df,name):
    df.to_csv(ROOT/'processed'/name,index=False,encoding='utf-8')

def main():
    (ROOT/'processed').mkdir(exist_ok=True)
    cats=[];allvis=[];allphotos=[];summaries=[];issues=[]
    for city,name in CITIES.items():
        base=ROOT/'raw/data-ijcai15'
        p=pd.read_csv(base/'poiList-ijcai15'/f'POI-{city}.csv',sep=';')
        p=p.rename(columns={'long':'lon'});p.insert(0,'city',city)
        p['decoded_name']=p.poiName.map(unquote).str.replace('_',' ',regex=False)
        p['cultural_status']=p.theme.map(lambda x:'candidate_by_category' if x in STRONG else ('requires_individual_review' if x in REVIEW else 'outside_initial_scope'))
        for col in ['lat','lon']:p[col]=pd.to_numeric(p[col],errors='coerce')
        p['coordinate_valid']=p.lat.between(-90,90)&p.lon.between(-180,180)
        dup=p.duplicated(['poiID'],keep=False)
        if dup.any():raise ValueError(f'Duplicate POI IDs {city}')
        for _,g in p.groupby('decoded_name'):
            if len(g)>1:issues.append({'city':city,'type':'duplicate_name','detail':'; '.join(g.poiID.astype(str))+': '+g.decoded_name.iloc[0]})
        # Outliers are flags, not an automatic correction or deletion.
        medlat,medlon=p.lat.median(),p.lon.median()
        dlat=np.radians(p.lat-medlat);dlon=np.radians(p.lon-medlon)
        a=np.sin(dlat/2)**2+np.cos(np.radians(p.lat))*np.cos(np.radians(medlat))*np.sin(dlon/2)**2
        p['km_from_catalog_median']=6371*2*np.arcsin(np.sqrt(a.clip(0,1)))
        p['coordinate_outlier_50km']=p.km_from_catalog_median>50
        for _,r in p[p.coordinate_outlier_50km].iterrows():issues.append({'city':city,'type':'coordinate_outlier_50km','detail':f'{r.poiID}: {r.decoded_name}; {r.lat}, {r.lon}'})
        vf=next((base/'userVisits-ijcai15').glob(f'userVisits-{city}*.csv'))
        raw=pd.read_csv(vf,sep=';',dtype={'photoID':str,'userID':str})
        nraw=len(raw);duplicates=int(raw.duplicated().sum());d=raw.drop_duplicates().copy()
        for c in ['dateTaken','poiID','seqID']:d[c]=pd.to_numeric(d[c],errors='coerce')
        # A 2015 published source cannot contain real visits dated after 2015.
        # Earlier dates are retained for explicit review instead of guessed repairs.
        valid=(d.userID.notna()&d.photoID.notna()&d.dateTaken.between(1,1451606399)&d.poiID.isin(p.poiID)&d.seqID.notna())
        bad=d[~valid].copy()
        if len(bad):
            bad['reason']='missing identifier, unknown POI, or timestamp outside 1970 through 2015'
            write(bad,f'quarantine_{city}.csv')
        d=d[valid].copy();d[['dateTaken','poiID','seqID']]=d[['dateTaken','poiID','seqID']].astype('int64')
        # Conflicting photo identifiers are not silently counted twice.
        conflicts=d.groupby('photoID').size();conflict_ids=set(conflicts[conflicts>1].index)
        if conflict_ids:
            write(d[d.photoID.isin(conflict_ids)],f'conflicting_photos_{city}.csv')
            d=d[~d.photoID.isin(conflict_ids)].copy()
        d.insert(0,'city',city)
        # Global poiFreq is deliberately excluded: recompute popularity from training data later.
        d=d.drop(columns=['poiFreq'])
        d=d.sort_values(['userID','seqID','dateTaken','photoID'],kind='stable').reset_index(drop=True)
        d['timestamp_multi_poi']=d.groupby(['userID','seqID','dateTaken']).poiID.transform('nunique')>1
        prev=d[['userID','seqID','poiID']].shift()
        new=(d.userID.ne(prev.userID)|d.seqID.ne(prev.seqID)|d.poiID.ne(prev.poiID))
        d['stop_no']=new.cumsum()
        v=d.groupby('stop_no',sort=False).agg(city=('city','first'),userID=('userID','first'),seqID=('seqID','first'),poiID=('poiID','first'),start_time=('dateTaken','min'),end_time=('dateTaken','max'),photo_count=('photoID','size'),ambiguous_timestamp=('timestamp_multi_poi','any')).reset_index(drop=True)
        v['visit_id']=[f'{city}-V{i+1:06d}' for i in range(len(v))]
        v=v.merge(p[['poiID','theme','cultural_status','coordinate_outlier_50km']],on='poiID',how='left',validate='many_to_one')
        d=d.drop(columns='stop_no');allphotos.append(d);allvis.append(v);cats.append(p)
        seq=v.groupby(['userID','seqID']).agg(start=('start_time','min'),end=('end_time','max'),n=('poiID','size'),unique_pois=('poiID','nunique'))
        seq=seq.sort_values(['userID','start','end'])
        eligible=seq.groupby(level=0).size()
        prior=seq.groupby(level=0).cumcount()>0
        # Pure history-availability diagnostic, not train/test sample count.
        prior_next=int(((seq.n>=2)&prior).sum())
        strict=v[v.cultural_status=='candidate_by_category']
        strict_seq=strict.groupby(['userID','seqID']).size()
        chronological=seq.reset_index();chronological['prior_end']=chronological.groupby('userID').end.cummax().groupby(chronological.userID).shift()
        overlap=int((chronological.start<=chronological.prior_end).sum())
        summaries.append(dict(city=city,city_name=name,raw_photo_rows=nraw,exact_duplicate_rows_removed=duplicates,invalid_rows_quarantined=len(bad),conflicting_photo_rows_quarantined=sum(raw.photoID.isin(conflict_ids)),clean_photo_rows=len(d),users=d.userID.nunique(),catalog_pois=len(p),observed_pois=d.poiID.nunique(),sequences=len(seq),reconstructed_visits=len(v),sequences_2plus_visits=int((seq.n>=2).sum()),sequences_3plus_visits=int((seq.n>=3).sum()),users_2plus_sequences=int((eligible>=2).sum()),later_sequences_2plus_visits=prior_next,overlapping_sequences=overlap,category_candidate_pois=int((p.cultural_status=='candidate_by_category').sum()),candidate_category_visits=len(strict),candidate_category_sequences_2plus=int((strict_seq>=2).sum()),ambiguous_photo_rows=int(d.timestamp_multi_poi.sum()),coordinate_outliers=int(p.coordinate_outlier_50km.sum()),start_utc=pd.to_datetime(d.dateTaken.min(),unit='s',utc=True).isoformat(),end_utc=pd.to_datetime(d.dateTaken.max(),unit='s',utc=True).isoformat()))
    catalog=pd.concat(cats,ignore_index=True);vis=pd.concat(allvis,ignore_index=True)
    write(catalog,'poi_catalog_audited.csv');write(vis,'visits_clean.csv');write(pd.concat(allphotos,ignore_index=True),'photos_clean.csv')
    summary=pd.DataFrame(summaries);write(summary,'city_audit.csv')
    write(pd.DataFrame(issues,columns=['city','type','detail']),'quality_flags.csv')
    cat=catalog.groupby(['city','theme'],dropna=False).size().rename('poi_count').reset_index();write(cat,'category_counts.csv')
    (ROOT/'reports/audit.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cities':summaries,'definitions':{'visit':'maximal consecutive run of the same POI within an original user-sequence ordered by timestamp; repeat returns retained','category_candidate':'preliminary category-based candidate, not verified heritage inclusion','later_sequences_2plus_visits':'original sequences with at least two reconstructed visits and any earlier original sequence for that user; diagnostic before cultural and time-overlap filtering','no_model_results':'No train/test performance was measured'}},indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    print('\nCATALOG\n'+catalog[['city','poiID','decoded_name','theme']].to_string(index=False))
    print('\nFLAGS\n'+str(issues))

if __name__=='__main__':main()
