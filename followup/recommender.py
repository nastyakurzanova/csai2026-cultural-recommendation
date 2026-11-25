"""Markov-anchored hybrid. CLI loads the exact matrices used in evaluation."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from model import CityModel, flat_kernel, make_features, normalize, ppr_kernel, ranks, row_kernel


def gate(batch, alpha, beta, c=3):
    n = batch['length']
    d = np.where(batch['metadata_available'], batch['drift'], 0.0)
    a = np.clip(alpha + beta * d * n / (n + c), 0, 1)
    return np.where(batch['long_available'], a, 1.0)


def masked_distribution(scores, last, pop):
    q = np.array(scores, dtype=float, copy=True)
    i = np.arange(len(q))
    q[i, last] = 0
    empty = q.sum(1) <= 1e-15
    q[empty] = pop
    q[i, last] = 0
    return normalize(q)


def predict_scores(batch, markov, graph, pop, params):
    a = gate(batch, params.get('alpha', 1.), params.get('beta', 0.), params.get('c', 3.))
    b = markov[batch['last']]
    eta = params.get('eta', 0.)
    if eta == 0:
        return b, a
    g = (1 - a[:, None]) * (batch['L'] @ graph) + a[:, None] * (batch['S'] @ graph)
    g = masked_distribution(g, batch['last'], pop)
    return (1 - eta) * b + eta * g, a


def markov_kernel(model, prior):
    return row_kernel(model.transitions + prior * model.pop[None, :], model.pop)


def graph_kernel(model, edges, kind, restart, seed=None):
    F, _, _, swaps = make_features(model.items, edges, cultural=kind != 'category', shuffle_seed=seed)
    kernel = flat_kernel(F, model.pop) if kind == 'flat' else ppr_kernel(F, restart, model.pop)
    return kernel, sum(swaps.values())


class MarkovHeritageRecommender:
    def __init__(self, path, variant='AdaptiveFree'):
        path = Path(path)
        meta = json.loads((path / 'metadata.json').read_text())
        if variant not in meta['variants']:
            raise ValueError('Unknown variant; use one of ' + ', '.join(meta['variants']))
        with np.load(path / 'weights.npz', allow_pickle=False) as data:
            z = {key: data[key] for key in data.files}
        self.model = CityModel(meta['city'], pd.read_csv(path / 'items.csv'), z['pop'],
                               z['collaborative'], z['transitions'], z['gram'], z['features'],
                               [tuple(x) for x in meta['feature_keys']])
        self.variant = variant
        self.params = meta['variants'][variant]
        self.markov = z['markov']
        self.graph = z.get('kernel_' + variant)

    def recommend(self, prefix, history=(), k=5):
        prefix, history = list(map(int, prefix)), list(map(int, history))
        if not prefix or k < 1:
            raise ValueError('Provide at least one prefix POI and k >= 1.')
        if any(i not in self.model.index for i in prefix + history):
            raise ValueError('Unknown POI for this city; see items.csv.')
        if any(a == b for a, b in zip(prefix, prefix[1:])):
            raise ValueError('Collapse adjacent identical places before inference.')
        candidates = [int(i) for i in self.model.ids if i != prefix[-1]]
        # The dummy target satisfies the shared batch interface. Scores never use it.
        b = self.model.batch([dict(prefix=prefix, old_items=history, target=candidates[0], user='inference')])
        scores, alpha = predict_scores(b, self.markov, self.graph, self.model.pop, self.params)
        _, order = ranks(scores, b['target'], b['last'], self.model.pop, self.model.ids)
        return dict(city=self.model.city, variant=self.variant, eta=self.params['eta'], alpha=float(alpha[0]),
                    recommendations=[dict(poiID=int(self.model.ids[i]), name=self.model.items.iloc[i]['name'],
                                          score=float(scores[0, i]))
                                     for i in order[0, :min(k, len(candidates))]])


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--variant', default='AdaptiveFree')
    p.add_argument('--prefix', required=True, type=int, nargs='+')
    p.add_argument('--history', type=int, nargs='*', default=[])
    p.add_argument('--k', type=int, default=5)
    a = p.parse_args()
    print(json.dumps(MarkovHeritageRecommender(a.model, a.variant).recommend(a.prefix, a.history, a.k),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    cli()
