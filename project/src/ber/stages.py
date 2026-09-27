"""Pipeline stages II: index -> retrieve -> features -> train -> select ->
calibrate -> audit -> test outputs.

Multiprocessing workers are top-level functions (spawn-safe). Every shard
carries a sidecar fingerprint; mismatched shards are recomputed, never mixed.
Target identity is (source, tgt_ord) — S2 and S3 ordinal spaces are distinct.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber import pipeline as pl
from ber.candidates import compact_strict, compact_with_joint, fused_margin, fuse_lanes
from ber.decisions import apply_owner_policy, apply_threshold, select_owner_policy, select_threshold
from ber.export import write_scored_outputs
from ber.features import FEATURE_NAMES, pair_features
from ber.metrics import entity_score, macro_oracle, paired_bootstrap_lcb, score_report
from ber.records import RecordViews, iter_truth_parquet, sha256_file
from ber.retrieval import INDEX_FIELDS, QueryViews, Retriever
from ber.retrieval import build_index as build_tantivy_index
from ber.training import TrainConfig, load_model, train_model

RETRIEVAL_VERSION = "v4"
MAX_TERMS = 16
FEATURE_VERSION = "v6"
DFMAP_VERSION = "v2"
LANE_K = 8
BUDGET = 2

EDGE_SCHEMA = pa.schema([
    ("tgt_ord", pa.int64()), ("ref_ord", pa.int64()), ("source", pa.uint8()),
    ("fused_v", pa.float32()), ("name_rank", pa.int16()), ("addr_rank", pa.int16()),
    ("name_score", pa.float32()), ("addr_score", pa.float32()),
    ("margin", pa.float32()), ("pool_size", pa.int16()),
    ("joint_rank", pa.int16()), ("joint_score", pa.float32()),
    ("is_name_top1", pa.bool_()), ("is_addr_top1", pa.bool_()),
])

SCORE_SCHEMA = pa.schema([
    ("tgt_ord", pa.int64()), ("ref_ord", pa.int64()), ("source", pa.uint8()),
    ("score", pa.float64()),
])

BIG = 1 << 40


def _join_key(source: int, tgt_ord: int) -> int:
    return source * BIG + tgt_ord


# ================================================================ index


def stage_index(data_dir: Path, cohort: str, threads: int = 6) -> dict:
    """Build the tantivy index per country from the cohort catalog."""
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / cohort
    idx_dir = data_dir / "index" / cohort
    idx_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"cohort": cohort, "countries": {}}
    t0 = time.perf_counter()
    from ber.retrieval import IndexDoc

    for cat in sorted(env_dir.glob("catalog_*.parquet")):
        country = cat.stem[len("catalog_"):]
        out = idx_dir / country
        cat_fp = sha256_file(cat)
        if (out / "index.json").is_file():
            fp = json.loads((out / "index.json").read_text(encoding="utf-8"))
            if (fp.get("catalog_fingerprint") == cat_fp
                    and fp.get("index_fields") == list(INDEX_FIELDS)):
                stats["countries"][country] = {"refs": fp["refs"], "reused": True}
                continue
        t = pq.read_table(cat)
        cols = {c: t.column(c).to_pylist() for c in
                ("ref_ord", "name_tri", "name_word", "name_foldtri",
                 "addr_tri", "addr_word", "addr_foldtri")}
        docs = []
        for i in range(t.num_rows):
            name_all = " ".join(sorted(
                set(cols["name_tri"][i].split())
                | set(cols["name_word"][i].split())
                | set(cols["name_foldtri"][i].split())
            ))
            addr_all = " ".join(sorted(
                set(cols["addr_tri"][i].split())
                | set(cols["addr_word"][i].split())
                | set(cols["addr_foldtri"][i].split())
            ))
            docs.append(IndexDoc(cols["ref_ord"][i], name_all, addr_all))
        n = t.num_rows
        with tempfile.TemporaryDirectory(prefix=f".{country}-build-", dir=idx_dir) as temporary:
            new_index = Path(temporary) / "index"
            build_tantivy_index(new_index, docs, threads=threads)
            (new_index / "index.json").write_text(
                json.dumps({"catalog_fingerprint": cat_fp, "refs": n,
                            "index_fields": list(INDEX_FIELDS)}),
                encoding="utf-8",
            )
            backup = idx_dir / f".{country}-previous-{uuid.uuid4().hex}"
            if out.exists():
                out.rename(backup)
            try:
                new_index.rename(out)
            except BaseException:
                if backup.exists():
                    backup.rename(out)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        del t, docs, cols
        stats["countries"][country] = {"refs": n, "reused": False}
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    pl._report(data_dir / "reports" / f"index_{cohort}.json", stats)
    return stats


def stage_dfmap(data_dir: Path, cohort: str) -> dict:
    """Per-field, per-document frequencies of the actual merged index terms."""
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / cohort
    out_dir = data_dir / "dfmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"cohort": cohort, "countries": {}}
    for cat in sorted(env_dir.glob("catalog_*.parquet")):
        country = cat.stem[len("catalog_"):]
        out = out_dir / f"{cohort}_{country}.parquet"
        fp = pl._sha(sha256_file(cat), DFMAP_VERSION)
        if pl.done(out, fp):
            stats["countries"][country] = "reused"
            continue
        fields = ("name_tri", "name_word", "name_foldtri",
                  "addr_tri", "addr_word", "addr_foldtri")
        df: dict[str, Counter[str]] = {"name_all": Counter(), "addr_all": Counter()}
        pf = pq.ParquetFile(cat)
        n_docs = pf.metadata.num_rows
        for batch in pf.iter_batches(batch_size=50_000, columns=list(fields)):
            cols = [batch.column(i).to_pylist() for i in range(6)]
            for name_tri, name_word, name_fold, addr_tri, addr_word, addr_fold in zip(*cols):
                df["name_all"].update(set(name_tri.split()) | set(name_word.split()) | set(name_fold.split()))
                df["addr_all"].update(set(addr_tri.split()) | set(addr_word.split()) | set(addr_fold.split()))
        items = [(field, term, count) for field, values in df.items()
                 for term, count in sorted(values.items())]
        pq.write_table(
            pa.table({"field": [f for f, _, _ in items],
                      "token": [k for _, k, _ in items],
                      "df": pa.array([v for _, _, v in items], pa.int64())}),
            out, compression="zstd",
        )
        pl.mark(out, fp, n_docs=n_docs, df_version=DFMAP_VERSION)
        stats["countries"][country] = {"distinct_terms": len(items), "n_docs": n_docs}
    pl._report(data_dir / "reports" / f"dfmap_{cohort}.json", stats)
    return stats


# ================================================================ retrieve


def _retrieval_fingerprint(data_dir: Path, cohort: str, country: str,
                           shard: Path, budget: int, lane_k: int = LANE_K,
                           max_terms: int = MAX_TERMS, joint_k: int = 0) -> str:
    idx = data_dir / "index" / cohort / country / "index.json"
    dm = data_dir / "dfmap" / f"{cohort}_{country}.parquet"
    if not dm.is_file():
        raise FileNotFoundError(f"missing document frequencies for retrieval: {dm}")
    return pl._sha(sha256_file(idx), sha256_file(shard), sha256_file(dm),
                   budget, lane_k, max_terms, joint_k, RETRIEVAL_VERSION)


def retrieve_worker(args: tuple) -> tuple:
    """One traffic shard -> one edge shard (fused, compacted, with context)."""
    index_dir, shard_path, out_path, fingerprint, budget, lane_k, dfmap_path, max_terms, joint_k = args
    if pl.done(Path(out_path), fingerprint):
        return (out_path, 0, True)
    import sys
    import time as _t

    t0 = _t.perf_counter()
    retr = Retriever(index_dir)
    df_map = {}
    if dfmap_path:
        d = pq.read_table(dfmap_path)
        if "field" not in d.column_names:
            raise ValueError(f"stale frequency map schema: {dfmap_path}")
        for field in INDEX_FIELDS:
            part = d.filter(pc.equal(d.column("field"), field))
            df_map[field] = dict(zip(part.column("token").to_pylist(),
                                     part.column("df").to_pylist()))
        del d
    t = pq.read_table(shard_path)
    cols = {c: t.column(c).to_pylist() for c in
            ("tgt_ord", "source", "name_norm", "addr_norm", "name_fold", "addr_fold")}
    rows: dict[str, list] = {f.name: [] for f in EDGE_SCHEMA}
    n_edges = 0
    for i in range(t.num_rows):
        q = QueryViews(cols["name_norm"][i], cols["addr_norm"][i],
                       cols["name_fold"][i], cols["addr_fold"][i])
        name_hits, addr_hits = retr.search(q, lane_k=lane_k, df_map=df_map, max_terms=max_terms)
        fused = fuse_lanes(name_hits, addr_hits)
        joint_hits = retr.search_joint(q, df_map=df_map, joint_k=joint_k) if joint_k else []
        kept = compact_with_joint(fused, budget, joint_hits, joint_k)
        margin = fused_margin(fused)
        m = float(margin) if margin is not None else float("nan")
        top_name = {h.ref_id for h in name_hits if h.rank == 1}
        top_addr = {h.ref_id for h in addr_hits if h.rank == 1}
        for f in kept:
            rows["tgt_ord"].append(cols["tgt_ord"][i])
            rows["ref_ord"].append(f.ref_id)
            rows["source"].append(cols["source"][i])
            rows["fused_v"].append(f.v)
            rows["name_rank"].append(f.name_rank or 0)
            rows["addr_rank"].append(f.addr_rank or 0)
            rows["name_score"].append(f.lane_scores.get("name", 0.0))
            rows["addr_score"].append(f.lane_scores.get("addr", 0.0))
            rows["margin"].append(m)
            rows["pool_size"].append(len(fused))
            rows["joint_rank"].append(f.joint_rank or 0)
            rows["joint_score"].append(f.lane_scores.get("joint", 0.0))
            rows["is_name_top1"].append(f.ref_id in top_name)
            rows["is_addr_top1"].append(f.ref_id in top_addr)
            n_edges += 1
    pq.write_table(pa.table(rows, schema=EDGE_SCHEMA), out_path, compression="zstd")
    pl.mark(Path(out_path), fingerprint, n_edges=n_edges)
    print(f"[retrieve] {Path(out_path).name}: {t.num_rows} queries -> {n_edges} edges "
          f"in {_t.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
    return (out_path, n_edges, False)


def _prune_shards(directory: Path, keep: set[str], patterns: tuple[str, ...]) -> int:
    """Delete shard files (and sidecars) whose names are not in `keep`.

    Keeps the artifact directories consistent with the current traffic/catalog
    set so stale shards from older configurations can never be reprocessed.
    """
    removed = 0
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.suffix == ".json" or path.name in keep:
                continue
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            removed += 1
    return removed


def stage_retrieve(data_dir: Path, cohort: str, budget: int = BUDGET,
                    workers: int = 8, lane_k: int = LANE_K,
                    max_terms: int = MAX_TERMS, joint_k: int = 0) -> dict:
    """Run retrieval workers over all traffic shards, per country."""
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / cohort
    out_dir = data_dir / "edges" / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"cohort": cohort, "budget": budget, "lane_k": lane_k,
                             "max_terms": max_terms, "joint_k": joint_k, "countries": {}}
    t0 = time.perf_counter()

    import multiprocessing as mp

    for country_dir in sorted(env_dir.glob("traffic_*")):
        country = country_dir.stem[len("traffic_"):]
        cdir = out_dir / country
        cdir.mkdir(parents=True, exist_ok=True)
        jobs = []
        dm = data_dir / "dfmap" / f"{cohort}_{country}.parquet"
        keep = set()
        for shard in sorted(country_dir.glob("src*_*.parquet")):
            fp = _retrieval_fingerprint(data_dir, cohort, country, shard, budget,
                                        lane_k=lane_k, max_terms=max_terms, joint_k=joint_k)
            out = cdir / shard.name
            keep.add(out.name)
            jobs.append((str(data_dir / "index" / cohort / country), str(shard), str(out), fp,
                         budget, lane_k, str(dm) if dm.is_file() else None, max_terms, joint_k))
        if not jobs:
            continue
        _prune_shards(cdir, keep, ("src*_*.parquet",))
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            results = pool.map(retrieve_worker, jobs)
        n_edges = sum(r[1] for r in results)
        reused = sum(1 for r in results if r[2])
        stats["countries"][country] = {"shards": len(jobs), "edges": n_edges, "reused": reused}
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    pl._report(data_dir / "reports" / f"retrieve_{cohort}.json", stats)
    return stats


# ================================================================ features


def _feature_fingerprint(data_dir: Path, cohort: str, country: str,
                          edge_shard: Path) -> str:
    cat = data_dir / "env" / cohort / f"catalog_{country}.parquet"
    traffic = data_dir / "env" / cohort / f"traffic_{country}" / edge_shard.name
    dm = data_dir / "dfmap" / f"{cohort}_{country}.parquet"
    return pl._sha(sha256_file(edge_shard), sha256_file(traffic), sha256_file(cat),
                    sha256_file(dm) if dm.is_file() else "nodfmap",
                    FEATURE_VERSION)


def _enrich_shard(edges: pa.Table, traffic: pa.Table, catalog: pa.Table) -> pa.Table:
    """Join ref + target views onto an edge shard (vectorized, parent-side)."""
    tgt = traffic.select(["source", "tgt_ord", "name_norm", "addr_norm",
                          "name_indic", "addr_indic", "addr_missing"])
    tgt = tgt.rename_columns(["source", "tgt_ord", "tgt_name_norm", "tgt_addr_norm",
                              "tgt_name_indic", "tgt_addr_indic", "tgt_addr_missing"])
    ref = catalog.select(["ref_ord", "name_norm", "addr_norm",
                          "name_indic", "addr_indic", "addr_missing"])
    ref = ref.rename_columns(["ref_ord", "ref_name_norm", "ref_addr_norm",
                              "ref_name_indic", "ref_addr_indic", "ref_addr_missing"])
    joined = edges.join(tgt, keys=["source", "tgt_ord"], join_type="inner")
    joined = joined.join(ref, keys=["ref_ord"], join_type="inner")
    return joined.sort_by([("source", "ascending"), ("tgt_ord", "ascending")])


def _ensure_enriched(edge_shard: Path, traffic: pa.Table, catalog: pa.Table,
                     enriched: Path, fingerprint: str) -> None:
    """Keep the joined feature input bound to the same edge/record revision."""
    if pl.done(enriched, fingerprint):
        return
    pq.write_table(_enrich_shard(pq.read_table(edge_shard), traffic, catalog),
                   enriched, compression="zstd")
    pl.mark(enriched, fingerprint)


def feature_worker(args: tuple) -> tuple:
    """One enriched shard -> one feature shard (pure function over columns)."""
    enriched_path, out_path, fingerprint, dfmap_path = args
    enriched_path, out_path = Path(enriched_path), Path(out_path)
    if pl.done(out_path, fingerprint):
        return (str(out_path), 0, True)
    t0 = time.perf_counter()
    dfmap: dict[str, int] = {}
    n_docs = None
    if dfmap_path:
        dfmap_path = Path(dfmap_path)
        d = pq.read_table(dfmap_path)
        if "field" not in d.column_names:
            raise ValueError(f"stale frequency map schema: {dfmap_path}")
        name = d.filter(pc.equal(d.column("field"), "name_all"))
        dfmap = dict(zip(name.column("token").to_pylist(), name.column("df").to_pylist()))
        n_docs = json.loads(dfmap_path.with_suffix(".json").read_text())["n_docs"]
        del d

    t = pq.read_table(enriched_path)
    n = t.num_rows
    feats = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    cols = t.to_pydict()
    for i in range(n):
        ref = RecordViews(
            entity_id="S1-x", name_raw="", addr_raw="",
            name_norm=cols["ref_name_norm"][i], addr_norm=cols["ref_addr_norm"][i],
            name_fold="", addr_fold="", country="",
            name_indic=cols["ref_name_indic"][i], addr_indic=cols["ref_addr_indic"][i],
            addr_missing=cols["ref_addr_missing"][i],
        )
        tgt = RecordViews(
            entity_id=f"S{cols['source'][i]}-x", name_raw="", addr_raw="",
            name_norm=cols["tgt_name_norm"][i], addr_norm=cols["tgt_addr_norm"][i],
            name_fold="", addr_fold="", country="",
            name_indic=cols["tgt_name_indic"][i], addr_indic=cols["tgt_addr_indic"][i],
            addr_missing=cols["tgt_addr_missing"][i],
        )
        margin = cols["margin"][i]
        feats[i] = pair_features(
            ref, tgt,
            fused_v=cols["fused_v"][i], rank=cols["name_rank"][i] or 0,
            is_name_top1=cols["is_name_top1"][i], is_addr_top1=cols["is_addr_top1"][i],
            name_score=cols["name_score"][i], addr_score=cols["addr_score"][i],
            margin=None if margin != margin else margin,  # NaN -> None
            pool_size=cols["pool_size"][i], token_df=dfmap,
            token_df_n_docs=n_docs,
            joint_rank=cols["joint_rank"][i], joint_score=cols["joint_score"][i],
        )
    data = {"tgt_ord": cols["tgt_ord"], "ref_ord": cols["ref_ord"], "source": cols["source"]}
    for j, name in enumerate(FEATURE_NAMES):
        data[name] = feats[:, j]
    schema = pa.schema(
        [("tgt_ord", pa.int64()), ("ref_ord", pa.int64()), ("source", pa.uint8())]
        + [(name, pa.float32()) for name in FEATURE_NAMES]
    )
    pq.write_table(pa.table(data, schema=schema), out_path, compression="zstd")
    pl.mark(out_path, fingerprint, n_edges=n)
    print(f"[features] {out_path.name}: {n} edges in {time.perf_counter() - t0:.1f}s",
          file=sys.stderr, flush=True)
    return (str(out_path), n, False)


def stage_features(data_dir: Path, cohort: str, workers: int = 8) -> dict:
    """Join views onto edges (parent, vectorized) then compute features (mp)."""
    data_dir = Path(data_dir)
    edges_dir = data_dir / "edges" / cohort
    out_dir = data_dir / "features" / cohort
    stats: dict[str, Any] = {"cohort": cohort, "countries": {}}
    t0 = time.perf_counter()

    import multiprocessing as mp

    for country_dir in sorted(edges_dir.glob("*")):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        cdir = out_dir / country
        cdir.mkdir(parents=True, exist_ok=True)
        env_dir = data_dir / "env" / cohort
        catalog = pq.read_table(env_dir / f"catalog_{country}.parquet")
        traffic_paths = sorted(env_dir.glob(f"traffic_{country}/src*_*.parquet"))
        keep = {p.name for p in traffic_paths}
        pruned = _prune_shards(country_dir, keep, ("src*_*.parquet",))
        if pruned:
            print(f"[features] pruned {pruned} stale edge shards for {cohort}/{country}",
                  file=sys.stderr, flush=True)
        traffic_parts = [pq.read_table(td) for td in traffic_paths]
        traffic = pa.concat_tables(traffic_parts)
        del traffic_parts
        dfmap_path = data_dir / "dfmap" / f"{cohort}_{country}.parquet"
        jobs = []
        enriched_paths = []
        for shard in sorted(country_dir.glob("src*_*.parquet")):
            fp = _feature_fingerprint(data_dir, cohort, country, shard)
            enriched = cdir / f"enriched_{shard.name}"
            _ensure_enriched(shard, traffic, catalog, enriched, fp)
            enriched_paths.append((shard, enriched, fp))
        del catalog, traffic
        for shard, enriched, fp in enriched_paths:
            out = cdir / shard.name
            jobs.append((str(enriched), str(out), fp,
                         str(dfmap_path) if dfmap_path.is_file() else None))
        if not jobs:
            continue
        keep_outputs = ({shard.name for shard, _, _ in enriched_paths}
                        | {enriched.name for _, enriched, _ in enriched_paths})
        _prune_shards(cdir, keep_outputs, ("src*_*.parquet", "enriched_src*_*.parquet"))
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            results = pool.map(feature_worker, jobs)
        stats["countries"][country] = {
            "shards": len(jobs), "edges": sum(r[1] for r in results),
            "reused": sum(1 for r in results if r[2]),
        }
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    pl._report(data_dir / "reports" / f"features_{cohort}.json", stats)
    return stats


# ================================================================ train


def _fit_owner_map(data_dir: Path) -> dict[str, str]:
    """tgt_id -> owning anchor id, restricted to F-role anchors."""
    man = pq.read_table(data_dir / "manifest" / "split.parquet",
                        columns=["entity_id", "role", "is_anchor"])
    f_anchors = {e for e, r, a in zip(man.column("entity_id").to_pylist(),
                                      man.column("role").to_pylist(),
                                      man.column("is_anchor").to_pylist()) if a and r == "F"}
    del man
    owner: dict[str, str] = {}
    for sid, ids in iter_truth_parquet(data_dir / "records" / "truth.parquet"):
        if sid in f_anchors:
            for t in ids:
                owner[t] = sid
    return owner


def _fit_label(ref_id: str, tgt_id: str, supervised_targets: set[str],
               fit_owners: dict[str, str]) -> int | None:
    """Supervise F-role queries only; B traffic has withheld owner labels."""
    if tgt_id not in supervised_targets:
        return None
    return int(fit_owners.get(tgt_id) == ref_id)


def stage_train(data_dir: Path, cfg: TrainConfig | None = None) -> dict:
    """Fit LightGBM on the fit cohort's feature shards."""
    data_dir = Path(data_dir)
    feats_dir = data_dir / "features" / "fit"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cfg = cfg or TrainConfig(num_threads=8, seed=0)

    owner = _fit_owner_map(data_dir)
    man = pq.read_table(data_dir / "manifest" / "split.parquet",
                        columns=["entity_id", "role", "source"])
    supervised_targets = set(man.filter(pc.and_(
        pc.equal(man.column("role"), "F"), pc.not_equal(man.column("source"), 1)
    )).column("entity_id").to_pylist())
    del man
    Xs, ys = [], []
    n_edges = 0
    n_background_skipped = 0
    ref_ord_to_id = {}
    for country in sorted(p.name for p in feats_dir.iterdir() if p.is_dir()):
        country_dir = feats_dir / country
        cat = pq.read_table(data_dir / "env" / "fit" / f"catalog_{country}.parquet",
                            columns=["ref_ord", "entity_id"])
        ref_ord_to_id.update(zip(cat.column("ref_ord").to_pylist(), cat.column("entity_id").to_pylist()))
        del cat
        tgt_key_to_id = {}
        for td in sorted((data_dir / "env" / "fit" / f"traffic_{country}").glob("src*_*.parquet")):
            tt = pq.read_table(td, columns=["source", "tgt_ord", "entity_id"])
            for i in range(tt.num_rows):
                tgt_key_to_id[_join_key(tt.column("source")[i].as_py(),
                                        tt.column("tgt_ord")[i].as_py())] = tt.column("entity_id")[i].as_py()
            del tt
        for shard in sorted(country_dir.glob("src*_*.parquet")):
            t = pq.read_table(shard)
            keys = [_join_key(s, o) for s, o in zip(t.column("source").to_pylist(),
                                                    t.column("tgt_ord").to_pylist())]
            fmat = np.stack([t.column(name).to_numpy().astype(np.float32)
                             for name in FEATURE_NAMES], axis=1)
            labels = [
                _fit_label(ref_ord_to_id[r], tgt_key_to_id[k], supervised_targets, owner)
                for k, r in zip(keys, t.column("ref_ord").to_pylist())
            ]
            mask = np.fromiter((label is not None for label in labels), dtype=bool,
                               count=len(labels))
            y = np.fromiter((label for label in labels if label is not None),
                            dtype=np.int8, count=int(mask.sum()))
            if len(y):
                Xs.append(fmat[mask])
                ys.append(y)
            n_edges += len(y)
            n_background_skipped += t.num_rows - len(y)
            del t, keys, fmat, y
        del tgt_key_to_id
    X = np.concatenate(Xs) if Xs else np.empty((0, len(FEATURE_NAMES)), np.float32)
    y = np.concatenate(ys) if ys else np.empty((0,), np.int8)
    del Xs, ys
    n_pos = int(y.sum())
    bst = train_model(X, y, cfg)
    model_path = models_dir / "lgbm_v1.txt"
    bst.save_model(str(model_path), num_iteration=bst.current_iteration())
    importance = sorted(
        zip(FEATURE_NAMES, bst.feature_importance("gain").tolist()),
        key=lambda kv: -kv[1],
    )[:20]
    report = {
        "n_edges": n_edges, "n_background_skipped": n_background_skipped,
        "n_positive": n_pos,
        "positive_rate": n_pos / max(n_edges, 1),
        "model_fingerprint": sha256_file(model_path),
        "config": cfg.__dict__,
        "top_features": [[k, float(v)] for k, v in importance],
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    pl._report(data_dir / "reports" / "train.json", report)
    return report


# ================================================================ evaluation


def _cohort_edge_tuples(data_dir: Path, cohort: str, model_path: Path) -> tuple[list, dict, dict]:
    """Score a train cohort's edges; return (edges, truth, id maps) in-memory.

    edges: list of (ref_id, tgt_id, score). truth: ref_id -> set(tgt_id) for
    the cohort's focal anchors. Also returns candidate maps for reporting.
    """
    data_dir = Path(data_dir)
    bst = load_model(model_path)
    feats_dir = data_dir / "features" / cohort
    env_dir = data_dir / "env" / cohort

    ref_ord_to_id: dict[int, str] = {}
    for cat in sorted(env_dir.glob("catalog_*.parquet")):
        t = pq.read_table(cat, columns=["ref_ord", "entity_id"])
        ref_ord_to_id.update(zip(t.column("ref_ord").to_pylist(), t.column("entity_id").to_pylist()))
        del t
    tgt_key_to_id: dict[int, str] = {}
    for country_dir in sorted(env_dir.glob("traffic_*")):
        for td in sorted(country_dir.glob("src*_*.parquet")):
            tt = pq.read_table(td, columns=["source", "tgt_ord", "entity_id"])
            for i in range(tt.num_rows):
                tgt_key_to_id[_join_key(tt.column("source")[i].as_py(),
                                        tt.column("tgt_ord")[i].as_py())] = tt.column("entity_id")[i].as_py()
            del tt

    edges: list[tuple[str, str, float]] = []
    cand_by_ref: dict[str, set[str]] = {}
    for country_dir in sorted(p for p in feats_dir.iterdir() if p.is_dir()):
        for shard in sorted(country_dir.glob("src*_*.parquet")):
            t = pq.read_table(shard)
            fmat = np.stack([t.column(name).to_numpy().astype(np.float32)
                             for name in FEATURE_NAMES], axis=1)
            scores = bst.predict(fmat)
            srcs = t.column("source").to_pylist()
            tgts = t.column("tgt_ord").to_pylist()
            refs = t.column("ref_ord").to_pylist()
            for i in range(t.num_rows):
                rid = ref_ord_to_id[refs[i]]
                tid = tgt_key_to_id[_join_key(srcs[i], tgts[i])]
                edges.append((rid, tid, float(scores[i])))
                cand_by_ref.setdefault(rid, set()).add(tid)
            del t, fmat, scores
    return edges, cand_by_ref, ref_ord_to_id


def _cohort_truth(data_dir: Path, cohort_role: str) -> dict[str, set[str]]:
    man = pq.read_table(data_dir / "manifest" / "split.parquet",
                        columns=["entity_id", "role", "is_anchor"])
    role_of = dict(zip(man.column("entity_id").to_pylist(), man.column("role").to_pylist()))
    del man
    truth: dict[str, set[str]] = {}
    for sid, ids in iter_truth_parquet(data_dir / "records" / "truth.parquet"):
        if role_of.get(sid) == cohort_role:
            truth[sid] = set(ids)
    return truth


def stage_select(data_dir: Path, cohort: str = "D", role: str = "D",
                 model_path: Path | None = None) -> dict:
    """Policy selection on D: threshold baseline vs owner challenger."""
    data_dir = Path(data_dir)
    model_path = Path(model_path or data_dir / "models" / "lgbm_v1.txt")
    t0 = time.perf_counter()
    edges, cand_by_ref, _ = _cohort_edge_tuples(data_dir, cohort, model_path)
    truth = _cohort_truth(data_dir, role)
    required = sorted(truth)

    thresh = select_threshold(edges, truth, required)
    owner = select_owner_policy(edges, truth, required)

    preds_t = apply_threshold(edges, thresh.tau)
    preds_o = apply_owner_policy(edges, owner.gamma, owner.tau)
    st = np.array([entity_score(truth[r], preds_t.get(r, set())) for r in required])
    so = np.array([entity_score(truth[r], preds_o.get(r, set())) for r in required])
    lcb = paired_bootstrap_lcb(so, st, replicates=2000, seed=1)
    prefer_owner = lcb >= 0.001

    rep_t = score_report(truth, preds_t, cand_by_ref, required)
    rep_o = score_report(truth, preds_o, cand_by_ref, required)
    oracle = macro_oracle(truth, cand_by_ref, required)

    report = {
        "cohort": cohort,
        "n_edges": len(edges),
        "n_required": len(required),
        "oracle_macro_f": oracle,
        "threshold_policy": {"tau": thresh.tau, "macro_f": thresh.macro, **rep_t},
        "owner_policy": {"gamma": owner.gamma, "tau": owner.tau, "macro_f": owner.macro, **rep_o},
        "owner_vs_threshold_lcb": lcb,
        "selected_policy": "owner" if prefer_owner else "threshold",
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    pl._report(data_dir / "reports" / f"select_{cohort}.json", report)
    return report


# ================================================================ calibrate


def _platt_fit(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Monotone sigmoid calibration p = 1/(1+exp(a*s+b)); a <= 0 enforced."""
    from scipy.optimize import minimize

    def nll(theta):
        a, b = -np.exp(theta[0]), theta[1]
        z = np.clip(a * scores + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(z))
        eps = 1e-12
        return -np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))

    res = minimize(nll, x0=np.array([0.0, 0.0]), method="L-BFGS-B")
    return float(-np.exp(res.x[0])), float(res.x[1])


def platt_apply(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.clip(a * scores + b, -30, 30)
    return 1.0 / (1.0 + np.exp(z))


def stage_calibrate(data_dir: Path, cohort: str = "K", role: str = "K",
                    model_path: Path | None = None,
                    selected_policy: str = "threshold") -> dict:
    """Freeze the release policy: Platt on K + tau/gamma on complete K rows."""
    data_dir = Path(data_dir)
    model_path = Path(model_path or data_dir / "models" / "lgbm_v1.txt")
    t0 = time.perf_counter()
    edges, cand_by_ref, _ = _cohort_edge_tuples(data_dir, cohort, model_path)
    truth = _cohort_truth(data_dir, role)
    required = sorted(truth)
    truth_map = {r: set(v) for r, v in truth.items()}

    # Platt supervision: candidate edges whose reference is in K, across the
    # complete traffic (incoming background -> K negatives included)
    k_refs = set(required)
    sup = [(s, 1.0 if t in truth_map.get(r, set()) else 0.0)
           for r, t, s in edges if r in k_refs]
    a_arr = np.array([x[0] for x in sup], dtype=np.float64)
    b_arr = np.array([x[1] for x in sup], dtype=np.float64)
    cal_a, cal_b = _platt_fit(a_arr, b_arr)
    del sup, a_arr, b_arr

    cal_edges = [(r, t, float(p)) for (r, t, s), p in
                 zip(edges, platt_apply(np.array([e[2] for e in edges]), cal_a, cal_b))]

    thresh = select_threshold(cal_edges, truth, required)
    if selected_policy == "owner":
        owner = select_owner_policy(cal_edges, truth, required)
        policy = {"kind": "owner", "tau": owner.tau, "gamma": owner.gamma}
        k_macro = owner.macro
    else:
        policy = {"kind": "threshold", "tau": thresh.tau}
        k_macro = thresh.macro

    recipe = _retrieval_recipe(data_dir, cohort)
    if recipe != _retrieval_recipe(data_dir, "D") or recipe != _retrieval_recipe(data_dir, "fit"):
        raise ValueError("retrieval recipe differs between fit, development, and calibration")

    policy_bundle = {
        "policy": policy,
        "calibrator": {"a": cal_a, "b": cal_b},
        "model_fingerprint": sha256_file(model_path),
        "feature_version": FEATURE_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_recipe": recipe,
        "selected_on": cohort,
    }
    pol_dir = data_dir / "policy"
    pol_dir.mkdir(parents=True, exist_ok=True)
    (pol_dir / "release.json").write_text(json.dumps(policy_bundle, indent=2), encoding="utf-8")

    rep = score_report(truth, _apply_policy(cal_edges, policy), cand_by_ref, required)
    report = {
        "cohort": cohort,
        "n_edges": len(edges),
        "n_required": len(required),
        "platt": {"a": cal_a, "b": cal_b},
        "policy": policy,
        "k_macro_f": k_macro,
        **rep,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    pl._report(data_dir / "reports" / f"calibrate_{cohort}.json", report)
    return report


def _apply_policy(edges, policy: dict) -> dict[str, set[str]]:
    if policy["kind"] == "owner":
        return apply_owner_policy(edges, policy["gamma"], policy["tau"])
    return apply_threshold(edges, policy["tau"])


def _retrieval_recipe(data_dir: Path, cohort: str) -> dict:
    report = json.loads((data_dir / "reports" / f"retrieve_{cohort}.json").read_text())
    fields = ("budget", "lane_k", "max_terms", "joint_k")
    if any(field not in report for field in fields):
        raise ValueError(f"incomplete retrieval recipe for {cohort}; regenerate edges")
    return {field: report[field] for field in fields}


def _verify_release_bundle(bundle: dict, model_path: Path, expected_recipe: dict) -> None:
    if bundle.get("model_fingerprint") != sha256_file(model_path):
        raise ValueError("release model fingerprint does not match the actual model")
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("release feature version does not match the current feature code")
    if bundle.get("retrieval_version") != RETRIEVAL_VERSION:
        raise ValueError("release retrieval version does not match the current retrieval code")
    if bundle.get("retrieval_recipe") != expected_recipe:
        raise ValueError("release retrieval recipe does not match the scored candidates")


# ================================================================ audit


def stage_audit(data_dir: Path, cohort: str = "A", role: str = "A",
                model_path: Path | None = None) -> dict:
    """One-shot frozen-policy audit on A. No tuning on this cohort."""
    data_dir = Path(data_dir)
    model_path = Path(model_path or data_dir / "models" / "lgbm_v1.txt")
    bundle = json.loads((data_dir / "policy" / "release.json").read_text(encoding="utf-8"))
    _verify_release_bundle(bundle, model_path, _retrieval_recipe(data_dir, cohort))
    t0 = time.perf_counter()
    edges, cand_by_ref, _ = _cohort_edge_tuples(data_dir, cohort, model_path)
    truth = _cohort_truth(data_dir, role)
    required = sorted(truth)

    raw = np.array([e[2] for e in edges], dtype=np.float64)
    cal = platt_apply(raw, bundle["calibrator"]["a"], bundle["calibrator"]["b"])
    cal_edges = [(r, t, float(p)) for (r, t, _), p in zip(edges, cal)]
    preds = _apply_policy(cal_edges, bundle["policy"])

    rep = score_report(truth, preds, cand_by_ref, required)
    oracle = macro_oracle(truth, cand_by_ref, required)

    # slices: country + singleton + indic-script (from record tables)
    s1 = pq.read_table(data_dir / "records" / "train_s1.parquet",
                       columns=["entity_id", "country"])
    country_of = dict(zip(s1.column("entity_id").to_pylist(), s1.column("country").to_pylist()))
    del s1
    slices: dict[str, dict[str, Any]] = {}
    for cname in sorted(set(country_of.values())):
        req = [r for r in required if country_of.get(r) == cname]
        if len(req) >= 500:
            slices[f"country:{cname}"] = {
                "n": len(req),
                "macro_f": _macro_for(preds, truth, req),
            }
    req_singletons = [r for r in required if not truth[r]]
    req_linked = [r for r in required if truth[r]]
    slices["singletons"] = {"n": len(req_singletons), "macro_f": _macro_for(preds, truth, req_singletons)}
    slices["linked"] = {"n": len(req_linked), "macro_f": _macro_for(preds, truth, req_linked)}

    report = {
        "cohort": cohort,
        "policy": bundle["policy"],
        "n_edges": len(edges),
        "n_required": len(required),
        "macro_f": rep["macro_f"],
        "oracle_macro_f": oracle,
        **{k: v for k, v in rep.items() if k != "macro_f"},
        "slices": slices,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    pl._report(data_dir / "reports" / f"audit_{cohort}.json", report)
    return report


def _macro_for(preds, truth, required):
    return float(np.mean([entity_score(truth[r], preds.get(r, set())) for r in required]))


# ================================================================ test


def stage_test(data_dir: Path, output_dir: Path, budget: int = BUDGET,
               workers: int = 8, lane_k: int = LANE_K,
               max_terms: int = MAX_TERMS, joint_k: int = 0) -> dict:
    """Full test run: index, retrieve, features, score, decide, export."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    t0 = time.perf_counter()
    bundle = json.loads((data_dir / "policy" / "release.json").read_text(encoding="utf-8"))
    model_path = data_dir / "models" / "lgbm_v1.txt"
    _verify_release_bundle(bundle, model_path, {"budget": budget, "lane_k": lane_k,
                                                 "max_terms": max_terms, "joint_k": joint_k})
    model = load_model(model_path)
    a, b = bundle["calibrator"]["a"], bundle["calibrator"]["b"]

    stage_index(data_dir, "test")
    stage_dfmap(data_dir, "test")
    stage_retrieve(data_dir, "test", budget=budget, workers=workers,
                   lane_k=lane_k, max_terms=max_terms, joint_k=joint_k)
    stage_features(data_dir, "test", workers=workers)

    # score all test feature shards; build edge arrays per country
    feats_dir = data_dir / "features" / "test"
    ref_arrays, tgt_arrays, src_arrays, score_arrays = [], [], [], []
    for country_dir in sorted(p for p in feats_dir.iterdir() if p.is_dir()):
        for shard in sorted(country_dir.glob("src*_*.parquet")):
            t = pq.read_table(shard)
            fmat = np.stack([t.column(name).to_numpy().astype(np.float32)
                             for name in FEATURE_NAMES], axis=1)
            scores = platt_apply(model.predict(fmat), a, b)
            ref_arrays.append(t.column("ref_ord").to_numpy())
            tgt_arrays.append(t.column("tgt_ord").to_numpy())
            src_arrays.append(t.column("source").to_numpy())
            score_arrays.append(scores.astype(np.float64))
            del t, fmat
    ref_ord = np.concatenate(ref_arrays) if ref_arrays else np.array([], np.int64)
    tgt_ord = np.concatenate(tgt_arrays) if tgt_arrays else np.array([], np.int64)
    src_u8 = np.concatenate(src_arrays) if src_arrays else np.array([], np.uint8)
    score = np.concatenate(score_arrays) if score_arrays else np.array([], np.float64)
    del ref_arrays, tgt_arrays, src_arrays, score_arrays
    n_edges = int(ref_ord.size)

    # decisions (vectorized, frozen policy)
    if bundle["policy"]["kind"] == "threshold":
        selected = score >= bundle["policy"]["tau"]
    else:
        gamma = bundle["policy"]["gamma"]
        tau = bundle["policy"]["tau"]
        # group edges per (source, tgt): sort by (src, tgt, -score); the first
        # row of each group is the unique-top candidate, the second is the
        # runner-up. Exact top ties abstain even at gamma zero; a lone
        # candidate has a missing margin and is tau-eligible only.
        order = np.lexsort((-score, tgt_ord, src_u8))
        s = src_u8[order]
        t = tgt_ord[order]
        sc = score[order]
        new_group = np.ones(n_edges, bool)
        new_group[1:] = (s[1:] != s[:-1]) | (t[1:] != t[:-1])
        starts = np.flatnonzero(new_group)
        ends = np.concatenate([starts[1:], [n_edges]])
        first = sc[starts]
        has_second = (ends - starts) >= 2
        second = np.zeros(starts.size, dtype=np.float64)
        second[has_second] = sc[starts[has_second] + 1]
        tie = has_second & (first == second)
        margin_ok = ~has_second | ((first - second) >= gamma)
        eligible = margin_ok & ~tie & (first >= tau)
        selected = np.zeros(n_edges, bool)
        selected[order[starts[eligible]]] = True

    # id maps (arrays indexed by ordinal; test ordinals are row indices)
    s1 = pq.read_table(data_dir / "records" / "test_s1.parquet", columns=["ordinal", "entity_id"])
    ref_ids = s1.column("entity_id").to_pylist()
    n_refs = len(ref_ids)
    del s1
    s2 = pq.read_table(data_dir / "records" / "test_s2.parquet", columns=["ordinal", "entity_id"])
    s2_ids = s2.column("entity_id").to_pylist()
    del s2
    s3 = pq.read_table(data_dir / "records" / "test_s3.parquet", columns=["ordinal", "entity_id"])
    s3_ids = s3.column("entity_id").to_pylist()
    del s3

    match_lines = write_scored_outputs(output_dir, ref_ids, s2_ids, s3_ids,
                                       ref_ord, tgt_ord, src_u8, selected)

    report = {
        "n_edges": n_edges,
        "n_refs": n_refs,
        "n_refs_with_matches": match_lines,
        "selected_edges": int(selected.sum()),
        "policy": bundle["policy"],
        "model_fingerprint": bundle["model_fingerprint"],
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    pl._report(data_dir / "reports" / "test.json", report)
    return report


# ================================================================ validate


def run_validation(data_dir: Path, output_dir: Path, dataset_dir: Path,
                   official_validator: Path) -> dict:
    """Strict audit + official validator with --check-ids."""
    from ber.audit import audit_outputs

    output_dir = Path(output_dir)
    m, c = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    s1 = pq.read_table(data_dir / "records" / "test_s1.parquet", columns=["entity_id"])
    required = s1.column("entity_id").to_pylist()
    del s1
    valid_targets: set[str] = set()
    for source in (2, 3):
        t = pq.read_table(data_dir / "records" / f"test_s{source}.parquet", columns=["entity_id"])
        valid_targets.update(t.column("entity_id").to_pylist())
        del t

    failures = audit_outputs(m, c, required, valid_targets)
    official = None
    if Path(official_validator).is_file():
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, str(official_validator),
             "--matching", str(m), "--candidate", str(c),
             "--test-dir", str(dataset_dir / "test"), "--check-ids"],
            capture_output=True, text=True, timeout=3600,
        )
        official = {"exit_code": proc.returncode, "stdout_tail": proc.stdout[-2000:]}
    result = {"strict_failures": failures, "official": official,
               "pass": not failures and (official is None or official["exit_code"] == 0)}
    pl._report(Path(data_dir) / "reports" / "validation.json", result)
    return result
