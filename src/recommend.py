"""Command-line inference, using saved weights and no network requests."""
from pathlib import Path
import argparse,json
from model import HeritageRecommender
p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--prefix',nargs='+',type=int,required=True);p.add_argument('--history',nargs='*',type=int,default=[]);p.add_argument('--k',type=int,default=5);p.add_argument('--variant',choices=['optimized','graph_positive'],default='optimized')
a=p.parse_args();print(json.dumps(HeritageRecommender(a.model,a.variant).recommend(a.prefix,a.history,a.k),ensure_ascii=False,indent=2))
