from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]
p=pd.read_csv(root/'followup/results/predictions.csv').query("evaluation=='transfer' and subset=='all'")
rows=[]
for city,g in p.groupby('city',sort=True):
 pivot=g.pivot(index=['user','context_id'],columns='model',values='NDCG5')
 for a,b in [('GraphPositive','Markov'),('GraphPositive','CategoryMatched')]:
  means=(pivot[a]-pivot[b]).groupby('user').mean()
  rng=np.random.default_rng(20261125)
  boot=means.to_numpy()[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1)
  lo,hi=np.quantile(boot,[.025,.975])
  rows.append(dict(city=city,contexts=len(pivot),users=len(means),a=a,b=b,delta=means.mean(),ci_low=lo,ci_high=hi,draws=10000,seed=20261125,
    markov=pivot.Markov.groupby('user').mean().mean(),graph=pivot.GraphPositive.groupby('user').mean().mean()))
out=root/'audit/results'
pd.DataFrame(rows).to_csv(out/'per_city_comparisons.csv',index=False)
print(pd.DataFrame(rows).query("b=='Markov'").to_string(index=False))
