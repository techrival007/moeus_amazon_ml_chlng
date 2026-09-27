import json, numpy as np
from pathlib import Path
from ber import dense as dn
from ber.dense_run import _release, _release_scores, _models_dir
data=Path('data'); v='v8'
rel=_release(data,v)
ctx=dn.build_context(data,'W',tuple(rel['features']))
s=_release_scores(data,v,rel,ctx.edges.X,ctx.edges.ref,ctx.edges.src,ctx.edges.tgt)
g=rel['policy'].get('gamma',0.02)
target=3.136  # W-D pred/ref of the 0.947 LB file (single model, tau .85)
rows=[]
for tau in np.round(np.arange(0.6,0.96,0.025),3):
    r=dn.evaluate(ctx,'D',s,{'kind':'owner','gamma':g,'tau':float(tau)})
    rows.append((float(tau),r['macro_f'],r['pred_per_ref']))
    print(tau,'D',round(r['macro_f'],4),'US',round(r['country:US'],4),'India',round(r['country:India'],4),'sing',round(r['singletons'],4),'pred/ref',round(r['pred_per_ref'],3),flush=True)
tau_m=min(rows,key=lambda x:abs(x[2]-target))[0]
k=dn.evaluate(ctx,'K',s,{'kind':'owner','gamma':g,'tau':tau_m})
print('MATCHED tau',tau_m,'K',round(k['macro_f'],4),'US',round(k['country:US'],4),'India',round(k['country:India'],4),flush=True)
md=_models_dir(data,'v8_matched')
import shutil
for f in _models_dir(data,v).glob('*'): shutil.copy(f, md/f.name)
rel2=dict(rel); rel2['variant']='v8_matched'; rel2['policy']={'kind':'owner','gamma':g,'tau':tau_m}
(md/'release.json').write_text(json.dumps(rel2,indent=2))
