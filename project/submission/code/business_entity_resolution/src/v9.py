"""v9: v8 stage-1 folds reused; stage 2 retrained with e5 cosine + cosine competition features."""
import sys, json, time, shutil, numpy as np
from pathlib import Path
from ber import dense as dn
from ber.dense_run import _release, _models_dir
from ber.training import load_model
data = Path('data'); t0 = time.time()
rel8 = _release(data, 'v8'); names = tuple(rel8['features'])
md8 = _models_dir(data, 'v8'); md9 = _models_dir(data, 'v9')
folds = [load_model(md8 / f'stage1_fold{k}.txt') for k in range(2)]
GAMMA = rel8['policy'].get('gamma', 0.0)
TARGET_PRED = 3.14   # W-D predicted/ref of the LB-best operating point

def cosine(split, ref, src, tgt):
    E1 = np.load(data / 'emb' / f'{split}_s1.npy', mmap_mode='r')
    E = {s: np.load(data / 'emb' / f'{split}_s{s}.npy', mmap_mode='r') for s in (2, 3)}
    out = np.empty(ref.size, np.float32)
    for s in (2, 3):
        idx = np.flatnonzero(src == s)
        for a in range(0, idx.size, 1_000_000):
            j = idx[a:a + 1_000_000]
            out[j] = np.einsum('ij,ij->i', np.asarray(E1[ref[j]], np.float32), np.asarray(E[s][tgt[j]], np.float32))
    return out

def cos_feats(ref, src, tgt, c):
    tkey = src.astype(np.int64) * dn.BIG + tgt.astype(np.int64)
    _, t_rank, t_other, _, _ = dn._rank_stats(tkey, c, hi=0.9)
    _, r_rank, _, r_max, _ = dn._rank_stats(ref.astype(np.int64), c, hi=0.9)
    return np.stack([c, t_rank, c - t_other, r_rank, c - r_max], axis=1).astype(np.float32)

def stage2_X(X, ref, src, tgt, s1, split):
    c = cosine(split, ref, src, tgt)
    return np.concatenate([X, dn.group_features(ref, src, tgt, s1), cos_feats(ref, src, tgt, c)], axis=1)

if sys.argv[1] == 'train':
    ctx = dn.build_context(data, 'W', names)
    e = ctx.edges; tm = dn.train_mask(ctx)
    preds = [m.predict(e.X, num_threads=30) for m in folds]
    fold_of_ref = (e.ref * 2654435761 % 1000003) % 2
    s1 = np.where(tm, np.where(fold_of_ref == 0, preds[0], preds[1]), (preds[0] + preds[1]) / 2)
    print(f'[v9] stage-1 rescored {time.time()-t0:.0f}s', flush=True)
    X2 = stage2_X(e.X, e.ref, e.src, e.tgt, s1, 'train')
    print(f'[v9] X2 {X2.shape} {time.time()-t0:.0f}s', flush=True)
    bst = dn.fit_lgbm(X2[tm], ctx.pos[tm].astype(np.int8), 400, 30, seed=7)
    bst.save_model(str(md9 / 'stage2cos.txt'))
    s = bst.predict(X2, num_threads=30)
    print(f'[v9] stage-2 trained {time.time()-t0:.0f}s', flush=True)
    rows = []
    for tau in np.round(np.arange(0.6, 0.96, 0.025), 3):
        r = dn.evaluate(ctx, 'D', s, {'kind': 'owner', 'gamma': GAMMA, 'tau': float(tau)})
        rows.append((float(tau), r['macro_f'], r['pred_per_ref']))
        print(tau, 'D', round(r['macro_f'], 4), 'US', round(r['country:US'], 4), 'India', round(r['country:India'], 4),
              'sing', round(r['singletons'], 4), 'pred/ref', round(r['pred_per_ref'], 3), flush=True)
    tau_m = min(rows, key=lambda x: abs(x[2] - TARGET_PRED))[0]
    k = dn.evaluate(ctx, 'K', s, {'kind': 'owner', 'gamma': GAMMA, 'tau': tau_m})
    print('MATCHED tau', tau_m, 'D', [r for r in rows if r[0] == tau_m], 'K', round(k['macro_f'], 4),
          'US', round(k['country:US'], 4), 'India', round(k['country:India'], 4), flush=True)
    json.dump({'gamma': GAMMA, 'tau': tau_m, 'features': list(names)}, open(md9 / 'release.json', 'w'), indent=2)
    for k_ in range(2):
        shutil.copy(md8 / f'stage1_fold{k_}.txt', md9 / f'stage1_fold{k_}.txt')
else:
    out = Path(sys.argv[2]); rel = json.load(open(md9 / 'release.json'))
    e = dn.load_edges(data, 'test', names)
    s1 = np.mean([m.predict(e.X, num_threads=30) for m in folds], axis=0)
    X2 = stage2_X(e.X, e.ref, e.src, e.tgt, s1, 'test')
    s = load_model(md9 / 'stage2cos.txt').predict(X2, num_threads=30)
    pol = {'kind': 'owner', 'gamma': rel['gamma'], 'tau': rel['tau']}
    sel = dn.apply_policy(e.src, e.tgt, s, pol, e.ref)
    rep = dn.export_test(data, out, e, sel); rep['policy'] = pol
    dn.report(data, f'test_{out.name}', rep)
