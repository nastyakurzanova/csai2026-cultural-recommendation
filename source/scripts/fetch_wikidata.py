"""Fetch Wikidata snapshots by exact English Wikipedia sitelinks.

No fuzzy matches are auto-confirmed. Names/coordinates are never silently repaired.
The output is source data for a future graph, not a trained recommender.
"""
from pathlib import Path
import csv,json,urllib.request,urllib.parse,concurrent.futures,math,datetime,time
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'raw/wikidata';RAW.mkdir(exist_ok=True)
API='https://www.wikidata.org/w/api.php'
PROPS={'P31':'instance_of','P149':'architectural_style','P84':'architect','P571':'inception','P1435':'heritage_designation','P793':'significant_event','P138':'named_after','P921':'main_subject','P361':'part_of','P527':'has_part'}
def fetch(params,path,endpoint=API):
    if path.exists():return json.loads(path.read_text())
    url=endpoint+'?'+urllib.parse.urlencode(dict(params,format='json',origin='*'))
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'HeritageRecommendationResearch/0.1 (academic source audit)','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=35) as r:data=json.load(r)
            if 'error' in data:raise ValueError(data['error'])
            path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
            return data
        except Exception as exc:
            if getattr(exc,'code',None)==429:
                # Stop this operation instead of hammering the public service.
                raise
            if attempt==2:raise
            time.sleep(1+attempt)
def distance(lat,lon,a,b):
    la,lb=math.radians(lat),math.radians(a)
    h=math.sin((lb-la)/2)**2+math.cos(la)*math.cos(lb)*math.sin(math.radians(b-lon)/2)**2
    return 6371*2*math.asin(math.sqrt(min(1,h)))
def write(name,rows,fields):
    with (ROOT/'processed'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    catalog=list(csv.DictReader((ROOT/'processed/poi_catalog_audited.csv').open()))
    titles=list(dict.fromkeys(r['decoded_name'] for r in catalog))
    batches=[titles[i:i+20] for i in range(0,len(titles),20)]
    def batch(item):
        i,ts=item
        d=fetch({'action':'wbgetentities','sites':'enwiki','titles':'|'.join(ts),'props':'info|labels|claims|sitelinks','languages':'en','sitefilter':'enwiki'},RAW/f'titles_{i:02d}.json')
        print('batch',i,'entities',len(d.get('entities',{})),flush=True)
        return d
    responses=list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(batch,enumerate(batches)))
    ents={};bytitle={}
    for response in responses:
        for q,e in response.get('entities',{}).items():
            if 'missing' in e:continue
            ents[q]=e
            if 'enwiki' in e.get('sitelinks',{}):bytitle[e['sitelinks']['enwiki']['title']]=q
    # Resolve historical English Wikipedia article names through official redirects.
    redirect_titles=set()
    unresolved=[t for t in titles if t not in bytitle]
    for bi in range(0,len(unresolved),20):
        ts=unresolved[bi:bi+20]
        try:
            response=fetch({'action':'query','prop':'pageprops','ppprop':'wikibase_item','redirects':'1','titles':'|'.join(ts)},RAW/f'wikipedia_redirects_{bi//20:02d}.json',endpoint='https://en.wikipedia.org/w/api.php')
        except Exception as exc:
            print('redirect lookup unavailable',str(exc),flush=True)
            continue
        query=response.get('query',{});changes={r['from']:r['to'] for r in query.get('normalized',[])+query.get('redirects',[])}
        targetids={p.get('title'):p.get('pageprops',{}).get('wikibase_item') for p in query.get('pages',{}).values()}
        for title in ts:
            dest=title;seen=set()
            while dest in changes and dest not in seen:seen.add(dest);dest=changes[dest]
            q=targetids.get(dest)
            if q:bytitle[title]=q;redirect_titles.add(title)
    missingids=sorted(set(bytitle.values())-set(ents))
    for bi in range(0,len(missingids),20):
        d=fetch({'action':'wbgetentities','ids':'|'.join(missingids[bi:bi+20]),'props':'info|labels|claims|sitelinks','languages':'en','sitefilter':'enwiki'},RAW/f'redirect_entities_{bi//20:02d}.json')
        ents.update({q:e for q,e in d.get('entities',{}).items() if 'missing' not in e})
    mapping=[];facts=[]
    for r in catalog:
        q=bytitle.get(r['decoded_name'],'');e=ents.get(q,{})
        coords=[]
        for s in e.get('claims',{}).get('P625',[]):
            if s.get('rank')=='deprecated':continue
            value=s.get('mainsnak',{}).get('datavalue',{}).get('value',{})
            if isinstance(value,dict) and 'latitude' in value and str(value.get('globe','')).endswith('Q2'):coords.append(value)
        dist=min((distance(float(r['lat']),float(r['lon']),x['latitude'],x['longitude']) for x in coords),default=None)
        status='exact_sitelink_coordinate_consistent' if q and dist is not None and dist<=2 else ('exact_sitelink_coordinate_mismatch' if q and dist is not None else ('exact_sitelink_no_coordinate' if q else 'unresolved_title'))
        mapping.append({'city':r['city'],'poiID':r['poiID'],'source_name':r['decoded_name'],'wikidata_id':q,'wikidata_label':e.get('labels',{}).get('en',{}).get('value',''),'match_status':status,'match_method':'wikipedia_redirect' if r['decoded_name'] in redirect_titles else 'exact_sitelink','distance_km':round(dist,3) if dist is not None else '', 'revision_id':e.get('lastrevid',''),'modified':e.get('modified',''),'scope_status':r['cultural_status']})
        for prop,meaning in PROPS.items():
            for s in e.get('claims',{}).get(prop,[]):
                if s.get('rank')=='deprecated':continue
                sn=s.get('mainsnak',{});dv=sn.get('datavalue')
                if not dv:continue
                value=dv['value'];target=value.get('id','') if isinstance(value,dict) else ''
                facts.append({'city':r['city'],'poiID':r['poiID'],'wikidata_id':q,'property':prop,'property_label':meaning,'target_entity':target,'value_json':json.dumps(value,ensure_ascii=False),'statement_id':s.get('id',''),'rank':s.get('rank',''),'references_count':len(s.get('references',[])),'qualifiers_json':json.dumps(s.get('qualifiers',{}),ensure_ascii=False),'revision_id':e.get('lastrevid',''),'match_status':status})
    write('wikidata_alignment.csv',mapping,list(mapping[0]))
    if facts:write('cultural_statements.csv',facts,list(facts[0]))
    labelids=sorted({f['target_entity'] for f in facts if f['target_entity']})
    labels=[];label_lookup_complete=True
    for i in range(0,len(labelids),50):
        # Cache keyed by actual IDs because redirects may add additional entities.
        import hashlib
        key=hashlib.sha256('|'.join(labelids[i:i+50]).encode()).hexdigest()[:12]
        try:
            d=fetch({'action':'wbgetentities','ids':'|'.join(labelids[i:i+50]),'props':'labels','languages':'en'},RAW/f'labels_{key}.json')
        except Exception as exc:
            print('Optional related-entity labels unavailable:',str(exc),flush=True)
            label_lookup_complete=False
            break
        for q,e in d.get('entities',{}).items():labels.append({'wikidata_id':q,'label_en':e.get('labels',{}).get('en',{}).get('value','')})
    write('cultural_entity_labels.csv',labels,['wikidata_id','label_en'])
    manifest={'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'exact English Wikipedia sitelinks plus official Wikipedia redirects; no fuzzy matching','properties':PROPS,'catalog_rows':len(catalog),'matched_rows':sum(bool(r['wikidata_id']) for r in mapping),'statement_rows':len(facts),'related_entity_label_lookup_complete':label_lookup_complete,'files':[p.name for p in sorted(RAW.glob('*.json'))]}
    (RAW/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
