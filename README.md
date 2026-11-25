# CSAI 2026 BJ-1242 reproducibility package

Adaptive Recommendation of Cultural Attractions Using a Knowledge Graph and Dynamic User Preferences

Authors: Anastasiia Kurzanova, Iurii N. Krotov, Svetlana S. Loseva, Dmitrii O. Shchepetov.
Bauman Moscow State Technical University, Department IU5.

This repository accompanies the six-page camera-ready paper for CSAI 2026, Track 3.
It contains code, original and prepared research data, model matrices, validation settings,
prediction-level rankings, aggregate results, and independent checks.

## Findings and scope

The transfer test uses 81 predictions from 44 users in Delhi, Glasgow, Osaka, Perth,
and Toronto. The full graph hybrid obtained city-macro NDCG@5 = 0.6837 versus
0.7064 for Markov alone. No improvement was established over the category graph.
Validation selected beta = 0; adaptation and its non-zero-beta sensitivity analysis
are exploratory. Markov outperformed popularity in a separate exploratory comparison.
Osaka contributes only two predictions. A degenerate within-city bootstrap does not
provide an informative estimate of population uncertainty.

## Environment and numerical reproduction

Use Python 3.12 and install the pinned dependencies:
python -m pip install -r requirements.txt

From the repository root:
    python run_all.py --from-raw --skip-article
    python followup/run.py
    python followup/gate_sensitivity.py
    python followup/test_followup.py
    python audit/check_study.py
    python audit/per_city_review.py

The original runs use one BLAS thread (OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1).
Numerical outputs are bundled. To check them without retraining, use the last two
commands. Source downloads and Wikidata responses are frozen, so no network is needed.
Legacy document-generation scripts may reference omitted publisher templates;
they are not numerical reproduction entry points.

## File map

- source/raw/: original Flickr archive, extracted tables and source README files,
  retrieval manifest and Wikidata responses from 26 October 2026.
- source/processed/ and data/: places, aliases, scope decisions, visits, sequences,
  contexts, cultural graph relations and temporal cutoffs.
- src/ and tests/: stage 1 methods, evaluation, diagnostics and tests.
- followup/: stage 2 Markov/graph model, transfer preparation and validation.
- models/ and followup/results/models/: saved matrices and settings.
- results/ and followup/results/: validation grids, metrics, prediction ranks,
  planned contrasts and explicitly exploratory sensitivity checks.
- audit/: independent data/rank checks and their outputs.
- audit/results/per_city_comparisons.csv: descriptive comparisons added after
  review, with 10,000 within-city user bootstrap draws, seed 20261125.
  These are not newly prespecified tests and do not change pooled results.
- SHA256SUMS: integrity hashes for this release.

## Third-party data and attribution

The Flickr User–POI Visits dataset is by Kwan Hui Lim, Jeffrey Chan,
Christopher Leckie and Shanika Karunasekera:
https://sites.google.com/site/limkwanhui/datacode
Cite their IJCAI 2015 study (paper reference [8]) when using these data.
The original archive has SHA-256:
7a5767a0c86e8f2213872c7b4aacc5c62d71b923f011f01a54b1e81024266410
Source README files are retained. User and photo identifiers are pseudonymous
research dataset identifiers; photographs are not included.

Wikidata facts are supplied under CC0:
https://www.wikidata.org/wiki/Wikidata:Licensing
Included references preserve claim provenance.

This repository does not relicense third-party data or confer additional rights
over source materials. The publisher's template and earlier manuscript drafts are
excluded. No participant recruitment or new private personal data collection
was undertaken.
