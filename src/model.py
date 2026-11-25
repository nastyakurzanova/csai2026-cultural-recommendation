"""Cultural-graph recommendation with two explicit preference profiles.

The model is CPU-only and deterministic. Fitting estimates behavioural kernels;
validation chooses graph and gate hyperparameters. No tensor framework required.
"""
from pathlib import Path
from dataclasses import dataclass
import json,numpy as np,pandas as pd
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components

def normalize(x,axis=-1):
    x=np.asarray(x,dtype=np.float64)
    z=x.sum(axis=axis,keepdims=True)
    return np.divide(x,z,out=np.zeros_like(x),where=z>0)

def row_kernel(k,pop):
    k=np.maximum(np.asarray(k,dtype=float),0).copy();np.fill_diagonal(k,0)
    for i in np.where(k.sum(1)<=1e-15)[0]:
        k[i]=np.asarray(pop).copy();k[i,i]=0
        if k[i].sum()==0:k[i]=1;k[i,i]=0
    return normalize(k)

def make_features(items,edges,cultural=True,drop_property=None,shuffle_seed=None,drop_fraction=0):
    ids=items.poiID.astype(int).tolist();index={p:i for i,p in enumerate(ids)}
    rows={('class',str(r.category),i) for i,r in enumerate(items.itertuples())}
    cultural_sets={}
    if cultural:
        for r in edges.itertuples():
            if r.property==drop_property or r.poiID not in index:continue
            cultural_sets.setdefault(r.property,set()).add((index[r.poiID],str(r.target)))
    # Within-relation double-edge swaps preserve item degrees and entity degrees.
    swaplog={}
    for prop,pairs in cultural_sets.items():
        if shuffle_seed is not None:
            rng=np.random.default_rng(shuffle_seed+sum(map(ord,prop)))
            arr=sorted(pairs);done=0
            for _ in range(max(100,len(arr)*40)):
                if len(arr)<2:break
                a,b=rng.choice(len(arr),2,replace=False);i,x=arr[a];j,y=arr[b]
                if i==j or x==y or (i,y) in pairs or (j,x) in pairs:continue
                pairs.remove((i,x));pairs.remove((j,y));pairs.add((i,y));pairs.add((j,x))
                arr[a]=(i,y);arr[b]=(j,x);done+=1
            swaplog[prop]=done
        if drop_fraction:
            rng=np.random.default_rng((shuffle_seed or 0)+54321+sum(map(ord,prop)))
            arr=sorted(pairs);keep=rng.random(len(arr))>=drop_fraction;pairs={v for v,k in zip(arr,keep) if k}
        rows|={(prop,q,i) for i,q in pairs}
    keys=sorted({(r,q) for r,q,i in rows});ki={k:i for i,k in enumerate(keys)}
    F=np.zeros((len(ids),len(keys)))
    degrees={}
    for r,q,i in rows:degrees[(i,r)]=degrees.get((i,r),0)+1
    for r,q,i in rows:F[i,ki[(r,q)]]=1/degrees[(i,r)]
    return F,keys,rows,swaplog

def ppr_kernel(F,restart,pop,return_full=False):
    n,d=F.shape;W=np.zeros((n+d,n+d));W[:n,n:]=F;W[n:,:n]=F.T
    P=normalize(W)
    dangling=np.flatnonzero(P.sum(1)==0);P[dangling,dangling]=1
    # Each row of R is the stationary row distribution for a one-item restart.
    R=restart*solve(np.eye(n+d)-(1-restart)*P,np.eye(n+d),assume_a='gen')
    assert np.max(np.abs(R.sum(1)-1))<1e-8
    K=row_kernel(R[:n,:n],pop)
    if return_full:return K,R,P,W
    return K

def flat_kernel(F,pop):
    z=np.linalg.norm(F,axis=1,keepdims=True);L=np.divide(F,z,out=np.zeros_like(F),where=z>0)
    return row_kernel(L@L.T,pop)

def gate_alpha(d,n,minimum,maximum,c):
    assert 0<=minimum<=maximum<=1 and c>0
    return minimum+(maximum-minimum)*np.asarray(d)*np.asarray(n)/(np.asarray(n)+c)

def ranks(scores,targets,last,pop,ids):
    s=np.asarray(scores,dtype=float).copy();s[np.arange(len(s)),last]=-np.inf
    # Numpy lexsort: last key is primary. All tie keys are training-only or static IDs.
    order=np.lexsort((np.broadcast_to(np.asarray(ids),s.shape),np.broadcast_to(-np.asarray(pop),s.shape),-s),axis=1)
    pos=np.argmax(order==np.asarray(targets)[:,None],axis=1)+1
    return pos,order

def metric_arrays(rank):
    r=np.asarray(rank)
    return {'NDCG5':np.where(r<=5,1/np.log2(r+1),0),'HR5':(r<=5).astype(float),'MRR':1/r,
            'NDCG10':np.where(r<=10,1/np.log2(r+1),0),'HR10':(r<=10).astype(float)}

@dataclass
class CityModel:
    city:str
    items:pd.DataFrame
    pop:np.ndarray
    collaborative:np.ndarray
    transitions:np.ndarray
    gram:np.ndarray
    features:np.ndarray
    feature_keys:list
    def __post_init__(self):
        self.ids=self.items.poiID.astype(int).to_numpy();self.index={int(p):i for i,p in enumerate(self.ids)}
    @classmethod
    def fit(cls,city,items,edges,sequences,shrinkage=5):
        items=items.sort_values('poiID').reset_index(drop=True);index={int(p):i for i,p in enumerate(items.poiID)};n=len(index)
        users=sorted({s['user'] for s in sequences});ui={u:i for i,u in enumerate(users)}
        X=np.zeros((len(users),n));counts=np.ones(n)*1e-9;trans=np.zeros((n,n))
        for seq in sequences:
            inds=[index[int(p)] for p in seq['items']]
            X[ui[seq['user']],inds]=1
            np.add.at(counts,inds,1)
            for a,b in zip(inds,inds[1:]):trans[a,b]+=1
        G=X.T@X;diag=np.diag(G);den=np.sqrt(diag[:,None]*diag[None,:])
        C=np.divide(G,den,out=np.zeros_like(G),where=den>0)*(G/(G+shrinkage))
        pop=normalize(counts);C=row_kernel(C,pop)
        F,keys,_,_=make_features(items,edges)
        return cls(city,items,pop,C,trans,G,F,keys)
    def batch(self,contexts):
        n=len(self.ids);m=len(contexts);L=np.zeros((m,n));S=np.zeros((m,n));A=np.zeros((m,n))
        target=[];last=[];length=[];users=[]
        catkeys=sorted(self.items.category.unique());cats=np.zeros((n,len(catkeys)))
        for i,c in enumerate(self.items.category):cats[i,catkeys.index(c)]=1
        for j,c in enumerate(contexts):
            for p in c['old_items']:L[j,self.index[int(p)]]+=1
            for p in c['prefix']:S[j,self.index[int(p)]]+=1
            A[j]=((L[j]+S[j])>0).astype(float)
            target.append(self.index[int(c['target'])]);last.append(self.index[int(c['prefix'][-1])]);length.append(len(c['prefix']));users.append(c['user'])
        L=normalize(L);S=normalize(S)
        cult=np.array([r!='class' for r,q in self.feature_keys])
        fl=L@self.features[:,cult];fs=S@self.features[:,cult]
        available=(fl.sum(1)>0)&(fs.sum(1)>0)
        drift=.5*np.abs(normalize(fl)-normalize(fs)).sum(1);drift[~available]=0
        cdrift=.5*np.abs(normalize(L@cats)-normalize(S@cats)).sum(1)
        return dict(contexts=contexts,L=L,S=S,A=A,target=np.array(target),last=np.array(last),length=np.array(length),users=np.array(users),drift=drift,category_drift=cdrift,long_available=(L.sum(1)>0),metadata_available=available)
    def score_kernel(self,batch,kernel,alpha):
        a=np.broadcast_to(np.asarray(alpha),(len(batch['L']),)).copy();a[~batch['long_available']]=1
        return (1-a[:,None])*(batch['L']@kernel)+a[:,None]*(batch['S']@kernel)
    def ease(self,l2):
        precision=solve(self.gram+l2*np.eye(len(self.ids)),np.eye(len(self.ids)),assume_a='pos')
        B=-precision/np.diag(precision)[None,:];np.fill_diagonal(B,0)
        return B
    def save(self,path,parameters,kernels):
        path=Path(path);path.mkdir(parents=True,exist_ok=True);self.items.to_csv(path/'items.csv',index=False)
        np.savez_compressed(path/'weights.npz',pop=self.pop,collaborative=self.collaborative,transitions=self.transitions,gram=self.gram,features=self.features,**kernels)
        (path/'metadata.json').write_text(json.dumps({'city':self.city,'feature_keys':self.feature_keys,'parameters':parameters,'snapshot':'2026-10-26'},indent=2,ensure_ascii=False))

class HeritageRecommender:
    """Load saved city weights and rank a supplied observed prefix and older history."""
    def __init__(self,path,variant='optimized'):
        path=Path(path);meta=json.loads((path/'metadata.json').read_text())
        with np.load(path/'weights.npz') as saved:z={k:saved[k] for k in saved.files}
        if variant not in ['optimized','graph_positive']:raise ValueError('variant must be optimized or graph_positive')
        self.parameters=meta['parameters'];self.variant=variant;self.kernel=z['cultural_hybrid' if variant=='optimized' else 'cultural_forced_hybrid'];self.z=z
        self.model=CityModel(meta['city'],pd.read_csv(path/'items.csv'),z['pop'],z['collaborative'],z['transitions'],z['gram'],z['features'],[tuple(x) for x in meta['feature_keys']])
    def recommend(self,prefix,old_items=None,k=5):
        if not prefix:raise ValueError('At least one observed current attraction is required')
        for i in prefix+(old_items or []):
            if int(i) not in self.model.index:raise ValueError(f'Unknown city POI {i}')
        # target is a dummy for constructing a batch; never used for scoring.
        candidates=[int(i) for i in self.model.ids if i!=prefix[-1]]
        c={'old_items':old_items or [],'prefix':prefix,'target':candidates[0],'user':'inference'}
        b=self.model.batch([c]);g=self.parameters['CulturalAdaptive' if self.variant=='optimized' else 'CulturalAdaptiveForced'];a=gate_alpha(b['drift'],b['length'],g['minimum'],g['maximum'],g['c'])
        score=self.model.score_kernel(b,self.kernel,a)
        _,order=ranks(score,b['target'],b['last'],self.model.pop,self.model.ids)
        out=[]
        for i in order[0,:min(k,len(candidates))]:
            row=self.model.items.iloc[i]
            out.append({'poiID':int(row.poiID),'name':row['name'],'score':float(score[0,i])})
        return {'variant':self.variant,'short_weight':float(a[0]) if old_items else 1.0,'recommendations':out}
