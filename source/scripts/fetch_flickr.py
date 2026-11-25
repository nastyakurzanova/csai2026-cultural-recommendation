"""Download and preserve the author's unmodified Flickr archive (stdlib only)."""
from pathlib import Path
import urllib.request, hashlib, json, zipfile, datetime

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://drive.google.com/uc?export=download&id=1ryshHuWKnBijXJgvaSCv-KsSUhVGhpIJ'
out = ROOT / 'raw/data-ijcai15.zip'
if not out.exists():
    req = urllib.request.Request(URL, headers={'User-Agent':'HeritageRecommendationResearch/0.1'})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    if not zipfile.is_zipfile(__import__('io').BytesIO(blob)):
        raise ValueError('Downloaded response is not a ZIP archive')
    out.write_bytes(blob)
blob = out.read_bytes()
with zipfile.ZipFile(out) as z:
    for item in z.infolist():
        target = (ROOT/'raw'/item.filename).resolve()
        if not target.is_relative_to((ROOT/'raw').resolve()):
            raise ValueError('Unsafe archive path')
    z.extractall(ROOT/'raw')
    entries = z.namelist()
manifest = {'source_page':'https://sites.google.com/site/limkwanhui/datacode',
            'download_url':URL, 'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'size_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'entries':entries}
(ROOT/'raw/flickr_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
