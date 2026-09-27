"""Embed name+address with multilingual-e5-small (MIT) for W and test records."""
import sys, time, numpy as np, pyarrow.parquet as pq, glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
data = Path('data'); out = data / 'emb'; out.mkdir(exist_ok=True)
model = SentenceTransformer('intfloat/multilingual-e5-small', device='cuda')
model.half(); model.max_seq_length = 64

def needed(src, cohort):
    if cohort == 'test':
        return None
    if src == 1:
        return np.concatenate([pq.read_table(p, columns=['ref_ord']).column('ref_ord').to_numpy()
                               for p in glob.glob(f'data/env/{cohort}/catalog_*.parquet')])
    return np.concatenate([pq.read_table(p, columns=['tgt_ord']).column('tgt_ord').to_numpy()
                           for p in glob.glob(f'data/env/{cohort}/traffic_*/src{src}_*.parquet')])

def run(split, cohort, limit=None):
    for src in (1, 2, 3):
        t = pq.read_table(data / 'records' / f'{split}_s{src}.parquet', columns=['name_raw', 'addr_raw'])
        names = t.column('name_raw').to_pylist(); addrs = t.column('addr_raw').to_pylist()
        n = len(names)
        ords = needed(src, cohort)
        ords = np.arange(n) if ords is None else np.unique(ords)
        if limit: ords = ords[:limit]
        texts = [f"query: {names[i]} | {addrs[i] or ''}" for i in ords]
        t0 = time.time()
        E = model.encode(texts, batch_size=1024, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float16)
        dt = time.time() - t0
        print(f'{split} s{src}: {len(texts)} texts in {dt:.0f}s ({len(texts)/dt:.0f}/s)', flush=True)
        if limit: return
        full = np.zeros((n, E.shape[1]), np.float16); full[ords] = E
        np.save(out / f'{split}_s{src}.npy', full)
        del full, E, texts

if sys.argv[1] == 'bench':
    run('test', 'test', limit=200_000)
else:
    run(sys.argv[1], sys.argv[2])
    print('EMBED_DONE', sys.argv[1], flush=True)
