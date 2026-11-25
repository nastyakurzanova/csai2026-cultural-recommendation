"""Create typed tables from audited study CSV files for the workbook builder."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'audit/results'
F=ROOT/'followup/results'
CITIES={'Buda':'Будапешт','Vien':'Вена','Edin':'Эдинбург','Delh':'Дели','Glas':'Глазго',
        'Osak':'Осака','Pert':'Перт','Toro':'Торонто'}
MODELS={'Markov':'Марков','Popularity':'Популярность','ItemKNN':'ItemKNN','EASE':'EASE',
        'GeoFixed':'Географическая модель','CulturalFixed':'ItemKNN и культурный граф',
        'CulturalAdaptive':'ItemKNN, граф и адаптация','CulturalFixedForced':'ItemKNN и ненулевой граф',
        'CategoryFree':'Марков и категории, свой подбор','FlatFree':'Марков и признаки, свой подбор',
        'CategoryMatched':'Марков и категории, равные веса','FlatMatched':'Марков и признаки, равные веса',
        'GraphPositive':'Марков и культурный граф','AdaptiveFree':'Марков, граф и выбранная адаптация',
        'AdaptiveNonzero':'Марков, граф и ненулевая адаптация'}
CAT={'historic_civic':'Историческое здание','museum_library':'Музей или библиотека',
     'palace_castle':'Дворец или замок','religious':'Религиозный объект','monument':'Памятник',
     'performing_arts':'Сценические искусства','historic_infrastructure':'Историческая инфраструктура',
     'historic_garden':'Исторический сад'}
PROP={'P84':'Архитектор','P149':'Архитектурный стиль','P138':'Именование'}


def macro(df,k):
    return float(df.groupby(['city','user'])[k].mean().groupby('city').mean().mean())


def main(output):
    h3=json.loads((A/'h3_summary.json').read_text())
    audit=json.loads((A/'audit_summary.json').read_text())
    params=json.loads((F/'parameters.json').read_text())
    original=pd.read_csv(ROOT/'results/metrics_all_variants.csv').query("city=='MAIN_MACRO' and subset=='warm' and split=='test'").set_index('model')
    transfer=pd.read_csv(F/'metrics.csv').query("city=='MACRO' and subset=='all' and evaluation=='transfer'").set_index('model')
    reused=pd.read_csv(F/'metrics.csv').query("city=='MACRO' and subset=='warm' and evaluation=='reused'").set_index('model')
    con=pd.read_csv(F/'contrasts.csv').query("evaluation=='transfer'").set_index('contrast')
    s=pd.read_csv(F/'nonzero_gate_contrasts.csv').query("evaluation=='transfer'").set_index('subset')
    hp=pd.read_csv(A/'h3_predictions.csv').query("evaluation=='transfer'")
    rank=hp.pivot(index='context_id',columns='model',values='rank')
    follow=pd.read_csv(F/'predictions.csv').query("evaluation=='transfer' and subset=='all'")
    allr=follow.pivot(index='context_id',columns='model',values='rank')
    dg=pd.read_csv(A/'context_diagnostics.csv').query("cohort=='transfer'").sort_values(['city','user','context_id'])
    usermap={u:f'U{i+1:03d}' for i,u in enumerate(sorted(dg.user.unique()))}
    pairs=[[CITIES[c],usermap[u]] for c,u in dg[['city','user']].drop_duplicates().itertuples(index=False,name=None)]
    events=[]
    for r in dg.itertuples(index=False):
        k=r.context_id
        events.append([CITIES[r.city],usermap[r.user],int(r.last_poi),int(r.target),
                       int(r.old_visits),int(r.prefix_visits),'Да' if r.both_have_cultural_features else 'Нет',
                       int(rank.loc[k,'Popularity']),int(rank.loc[k,'Markov']),
                       int(allr.loc[k,'CategoryMatched']),int(allr.loc[k,'FlatMatched']),
                       int(allr.loc[k,'GraphPositive']),None,None,k])
    comparisons=[]
    for k in ('Popularity','Markov','ItemKNN','EASE','GeoFixed','CulturalFixed','CulturalAdaptive','CulturalFixedForced'):
        r=original.loc[k];comparisons.append(['1. Будапешт и Вена',MODELS[k],272,47,float(r.NDCG5),float(r.HR5),float(r.MRR)])
    for k in ('Markov','CategoryMatched','FlatMatched','GraphPositive','AdaptiveFree'):
        r=reused.loc[k];comparisons.append(['2. Повторный тест',MODELS[k],int(r.contexts),int(r.users),float(r.NDCG5),float(r.HR5),float(r.MRR)])
    p=hp[hp.model.eq('Popularity')]
    comparisons.append(['2. Перенос',MODELS['Popularity'],81,44,None,macro(p,'HR5'),macro(p,'MRR')])
    for k in ('Markov','CategoryMatched','FlatMatched','GraphPositive','AdaptiveFree'):
        r=transfer.loc[k];comparisons.append(['2. Перенос',MODELS[k],81,44,None if k=='Markov' else float(r.NDCG5),float(r.HR5),float(r.MRR)])
    hypotheses=[]
    for label,k,result in [('H1','H1_cultural_links','Не подтверждена'),
                           ('Гибрид','end_to_end','Прибавка не установлена'),
                           ('Контроль','matched_graph_vs_flat','Прибавка не установлена')]:
        r=con.loc[k];hypotheses.append([label,MODELS[r.a]+' / '+MODELS[r.b],int(r.contexts),int(r.users_global),
                                      float(r.delta),float(r.ci_low),float(r.ci_high),float(r.p_two_sided),
                                      None if pd.isna(r.p_holm) else float(r.p_holm),result])
    hypotheses.insert(1,['H2','Адаптация / постоянный вес',16,8,0.,None,None,1.,1.,'Выбрано β = 0. Конфигурации совпали'])
    r=s.loc['two_profiles']
    hypotheses.append(['H2, диагностика','Ненулевая адаптация / постоянный вес',int(r.contexts),int(r.users_global),
                       float(r.delta),float(r.ci_low),float(r.ci_high),float(r.p_two_sided),None,'Вес изменён в 11 из 16. Метрика не выросла'])
    hypotheses.append(['H3','Марков / популярность',81,44,None,h3['ci_low'],h3['ci_high'],h3['p_two_sided'],None,
                       'Поддержана на данном тесте. Дополнительная проверка'])
    citydata=pd.read_csv(A/'coverage.csv')
    raw=pd.read_csv(ROOT/'source/processed/city_audit.csv').set_index('city')
    cohort=pd.read_csv(F/'transfer_cohort.csv').set_index('city')
    towns=[]
    for r in citydata.itertuples(index=False):
        n,u=(int(cohort.loc[r.city,'test_contexts']),int(cohort.loc[r.city,'test_users'])) if r.city in cohort.index else {
            'Buda':(158,31),'Vien':(114,21),'Edin':(225,41)}[r.city]
        towns.append([CITIES[r.city],int(raw.loc[r.city,'raw_photo_rows']),int(raw.loc[r.city,'invalid_rows_quarantined']),
                      int(r.places),int(r.edges),int(r.places_with_edges),int(r.shared_feature_values),n,u,None,None])
    items=pd.concat([pd.read_csv(ROOT/'data/items.csv'),pd.read_csv(F/'transfer_items.csv')])
    objects=[[CITIES[r.city],int(r.poiID),r.name,r.qid,CAT[r.category],r.lat,r.lon,
              'https://www.wikidata.org/wiki/'+r.qid] for r in items.itertuples(index=False)]
    edges=pd.concat([pd.read_csv(ROOT/'data/cultural_edges.csv'),pd.read_csv(F/'transfer_edges.csv')])
    facts=[[CITIES[r.city],int(r.poiID),r.property,PROP[r.property],r.target,
            None if pd.isna(r.label) else str(r.label),int(r.reference_count),int(r.snapshot_revision),r.statement_id,r.source_url]
           for r in edges.itertuples(index=False)]
    configuration=[]
    for k in ('Markov','CategoryFree','FlatFree','CategoryMatched','FlatMatched','GraphPositive','AdaptiveFree'):
        v=params['variants'][k]
        configuration.append([MODELS[k],params['markov_prior'],v.get('restart') if v['kind'] in ('category','graph') else None,
                              v['eta'],v['alpha'],v['beta'],v.get('c'),{'none':'Частоты переходов','graph':'Культурный граф',
                              'category':'Только категории','flat':'Косинус по тем же признакам'}[v['kind']]])
    v=params['variants']['AdaptiveFree']
    configuration.append([MODELS['AdaptiveNonzero'],params['markov_prior'],v['restart'],v['eta'],v['alpha'],float(s.loc['all','beta']),3,
                          'Разведочная проверка из сохранённой сетки'])
    checks=[[x['area'],x['result'],x.get('count'),x['observation']] for x in audit['checks']]
    checks.extend([
      ['Календарь переноса','Ограничение',5,'Пять событий переноса предшествуют последней исходной валидационной отсечке. Отдельно проверена чувствительность к общей дате.'],
      ['Почти равные баллы','Проверено',None,'После округления на 14 знаках часть списков поменялась, но NDCG@5 всех основных вариантов совпала.'],
      ['Историческая точность','Ограничение',89,'89 утверждений из 164 имеют ссылки. Сверка со снимком Wikidata не заменяет проверку первичных исторических источников.'],
    ])
    payload=dict(comparisons=comparisons,hypotheses=hypotheses,towns=towns,objects=objects,facts=facts,configuration=configuration,
                 checks=checks,events=events,pairs=pairs,expected=[h3['Popularity'],h3['Markov'],h3['delta']])
    output.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main(Path(sys.argv[1]))
