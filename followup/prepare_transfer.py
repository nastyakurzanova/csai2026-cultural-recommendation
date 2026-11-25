"""Build all five new city cohorts from the existing source snapshot."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'source' / 'processed'


def readjl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def writejl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare(out):
    """Scope and splits depend only on metadata and chronology."""
    out.mkdir(parents=True, exist_ok=True)
    scope = json.loads((HERE / 'scope.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert list(scope) == protocol['new_cities']
    catalog = pd.read_csv(SRC / 'poi_catalog_prepared.csv')
    facts = pd.read_csv(SRC / 'cultural_statements.csv')
    labels = pd.read_csv(SRC / 'cultural_entity_labels.csv').set_index('wikidata_id').label_en
    source_sequences = pd.read_csv(SRC / 'sequence_audit.csv', dtype={'userID': str})
    old_users = set(source_sequences[source_sequences.city.isin(protocol['reused_test'])].userID)
    source_sequences = source_sequences[
        source_sequences.city.isin(scope) & ~source_sequences.ambiguous &
        ~source_sequences.overlaps_previous & (source_sequences.start >= 946684800)
    ].copy()
    visits = pd.read_csv(SRC / 'visits_prepared.csv', dtype={'userID': str})
    items, decisions, edges, sequences, contexts, audit = [], [], [], [], [], []
    cutoffs = {}
    specific = {
        ('Delh', 2): 'Parliament identity and coordinates fail the existing entity screen.',
        ('Delh', 12): 'Broad Qutb complex overlaps the separately recorded Qutub Minar; exclude the broader candidate.',
        ('Glas', 21): 'Gallery identity and coordinates fail the existing entity screen.',
        ('Glas', 24): 'Sports venue; not treated as the designated heritage structure of a component.',
        ('Osak', 26): 'District, with existing entity/coordinate screen failure.',
        ('Osak', 27): 'Venue/network identity is unresolved by the existing entity screen.',
    }
    for city, groups in scope.items():
        lookup = {i: kind for kind, ids in groups.items() for i in ids}
        assert len(lookup) == sum(map(len, groups.values()))
        citycat = catalog[catalog.city.eq(city)]
        assert set(lookup) <= set(citycat.poiID)
        for r in citycat.itertuples(index=False):
            poi = int(r.poiID)
            included = poi in lookup
            why = specific.get((city, poi), 'Outside the specified cultural attraction types or physical scope.')
            if included:
                assert r.entity_usable_for_screening and r.coordinate_screen_passed
                why = 'Identified cultural attraction: ' + lookup[poi]
                items.append(dict(city=city, poiID=poi, name=r.name, qid=r.wikidata_id,
                                  category=lookup[poi], lat=r.lat, lon=r.lon))
            decisions.append(dict(city=city, poiID=poi, name=r.name, qid=r.wikidata_id,
                                  included=included, category=lookup.get(poi, ''), reason=why))
        cultural = facts[facts.city.eq(city) & facts.poiID.isin(lookup) &
                         facts.property.isin(['P84', 'P149', 'P138']) & facts.target_entity.notna()]
        for r in cultural.itertuples(index=False):
            edges.append(dict(city=city, poiID=int(r.poiID), property=r.property,
                              target=r.target_entity, label=labels.get(r.target_entity, r.target_entity),
                              statement_id=r.statement_id, rank=r.rank,
                              reference_count=int(r.references_count), qualifiers_json=r.qualifiers_json,
                              snapshot_revision=int(r.revision_id),
                              source_url='https://www.wikidata.org/wiki/' + r.wikidata_id))
        source = source_sequences[source_sequences.city.eq(city)]
        end_times = sorted(map(int, source.end))
        train_end = end_times[int(len(end_times) * .6) - 1]
        val_end = end_times[int(len(end_times) * .8) - 1]
        cutoffs[city] = dict(train_end=train_end, validation_end=val_end)
        selected = visits[visits.city.eq(city) & visits.poiID.isin(lookup)]
        joined = selected.merge(source[['userID', 'seqID', 'start', 'end']],
                                on=['userID', 'seqID'], validate='many_to_one')
        cityseq = []
        for (user, sid), g in joined.groupby(['userID', 'seqID']):
            g = g.sort_values(['start_time', 'end_time', 'poiID'], kind='stable')
            g = g[g.poiID.ne(g.poiID.shift())]
            start, end = int(g.start.iloc[0]), int(g.end.iloc[0])
            if end <= train_end:
                split = 'train'
            elif start > train_end and end <= val_end:
                split = 'validation'
            elif start > val_end:
                split = 'test'
            else:
                continue
            cityseq.append(dict(city=city, user=user, seqID=int(sid), start=start, end=end,
                                items=list(map(int, g.poiID)), times=list(map(int, g.start_time)), split=split))
        cityseq.sort(key=lambda s: (s['user'], s['start'], s['seqID']))
        sequences.extend(cityseq)
        prior = {}
        for seq in cityseq:
            user, sid = seq['user'], seq['seqID']
            older = prior.setdefault(user, [])
            assert not older or older[-1][1] < seq['start']
            places, ts = seq['items'], seq['times']
            for i in range(1, len(places)):
                assert ts[i - 1] < ts[i] and places[i - 1] != places[i]
                contexts.append(dict(
                    context_id=f'{city}:{user}:{sid}:{i}', city=city, user=user, seqID=sid,
                    split=seq['split'], sequence_start=seq['start'], sequence_end=seq['end'],
                    prefix=places[:i], prefix_times=ts[:i], old_items=[q for q, _ in older],
                    old_end_times=[e for _, e in older], target=places[i], target_time=ts[i],
                    warm=(len(older) >= 3 and i >= 2), transfer_user=(user not in old_users),
                ))
            older.extend((q, seq['end']) for q in places)
        test = [c for c in contexts if c['city'] == city and c['split'] == 'test']
        cohort = [c for c in test if c['transfer_user']]
        audit.append(dict(city=city, places=len(lookup), train_sequences=sum(s['split']=='train' for s in cityseq),
                          test_contexts_all=len(test), test_contexts=len(cohort),
                          test_users=len({c['user'] for c in cohort}),
                          two_profiles=sum(bool(c['old_items']) and len(c['prefix'])>=2 for c in cohort),
                          warm=sum(c['warm'] for c in cohort)))
    pd.DataFrame(items).to_csv(out / 'transfer_items.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'transfer_scope_decisions.csv', index=False)
    pd.DataFrame(edges).drop_duplicates(['city', 'poiID', 'property', 'target']).to_csv(
        out / 'transfer_edges.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'transfer_cohort.csv', index=False)
    (out / 'transfer_cutoffs.json').write_text(json.dumps(cutoffs, indent=2))
    writejl(out / 'transfer_sequences.jsonl', sequences)
    writejl(out / 'transfer_contexts.jsonl', contexts)
