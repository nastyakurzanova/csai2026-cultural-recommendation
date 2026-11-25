"""Rebuild the complete offline study in dependency order, with durable logs."""
from pathlib import Path
import argparse,subprocess,sys,os,json,hashlib,time,platform,importlib.metadata

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--from-raw',action='store_true',help='Also reconstruct original photo visits from bundled raw files. Wikidata processed snapshot remains fixed.')
    ap.add_argument('--skip-article',action='store_true',help='Run numerical study without generating DOCX')
    ap.add_argument('--start-at',choices=['prepare','run','diagnostics','analysis','tests','article'],help='Resume a previously prepared run at this stage')
    args=ap.parse_args()
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1';env['MPLBACKEND']='Agg'
    for name in ['data','models','results','figures']:(ROOT/name).mkdir(exist_ok=True)
    raw=ROOT/'source/raw/data-ijcai15.zip'
    expected='7a5767a0c86e8f2213872c7b4aacc5c62d71b923f011f01a54b1e81024266410'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=expected:raise ValueError('Original Flickr archive checksum mismatch')
    stages=[]
    if args.from_raw:
        stages.extend([('source_prepare',['source/scripts/prepare_data.py']),('source_finalize',['source/scripts/finalize_data.py'])])
    stages.extend([('prepare',['src/prepare.py']),('run',['src/experiment.py']),('diagnostics',['src/diagnostics.py']),('analysis',['src/analyze.py']),('tests',['-m','unittest','discover','-s','tests','-v'])])
    if not args.skip_article:stages.append(('article',['src/build_article.py']))
    if args.start_at:
        stages=stages[next(i for i,x in enumerate(stages) if x[0]==args.start_at):]
    outputs=[];total=time.perf_counter()
    for name,arguments in stages:
        print('Running',name,flush=True);start=time.perf_counter()
        p=subprocess.run([sys.executable,*arguments],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (ROOT/f'results/{name}.log').write_text(p.stdout,encoding='utf-8')
        outputs.append({'stage':name,'returncode':p.returncode,'seconds':time.perf_counter()-start})
        if p.returncode:
            print(p.stdout);raise SystemExit(p.returncode)
        print('Passed',name,flush=True)
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib','python-docx']}
    result={'status':'passed','python':sys.version,'platform':platform.platform(),'versions':versions,'blas_threads':1,'from_raw':args.from_raw,'raw_archive_sha256':expected,'total_seconds':time.perf_counter()-total,'stages':outputs,'tests':15,'scope':'numerical and artifact-generation pipeline; DOCX visual rendering reviewed separately'}
    (ROOT/'results/integration_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
