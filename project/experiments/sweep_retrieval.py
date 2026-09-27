"""Compare candidate recall on reproducibly sampled D truth links.

Uses test-like D indexes, the full D reference catalog, and only training
labels. This small screening panel is not an official or held-out score.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from anyascii import anyascii

from ber.candidates import LaneHit, compact_strict, compact_with_joint, fuse_lanes
from ber.records import iter_truth_parquet
from ber.normalize import is_indic, normalize_text, tokens
from ber.retrieval import QueryViews, Retriever, plan_lane_terms, query_term_streams
from diagnose_retrieval import sample_rows


def legacy_map(catalog: Path) -> dict[str, int]:
    """Exact original v1 term-prioritization formula, for comparison only."""
    df: Counter[str] = Counter()
    for batch in pq.ParquetFile(catalog).iter_batches(
        batch_size=50_000,
        columns=["name_tri", "name_word", "name_foldtri",
                 "addr_tri", "addr_word", "addr_foldtri"],
    ):
        for col in batch.columns:
            for row in col.to_pylist():
                df.update(row.split())
    return dict(df)


def field_map(path: Path) -> dict[str, dict[str, int]]:
    data = pq.read_table(path)
    result = {}
    for field in ("name_all", "addr_all"):
        sub = data.filter(pc.equal(data.column("field"), field))
        result[field] = dict(zip(sub.column("token").to_pylist(), sub.column("df").to_pylist()))
    return result


def extra_hits(retr: Retriever, q: QueryViews, df: dict[str, dict[str, int]],
               kind: str, size: int = 8) -> list[int]:
    name_df, addr_df = df["name_all"], df["addr_all"]
    if kind == "joint":
        streams = query_term_streams(q)
        name = plan_lane_terms(streams["name_all"], name_df, 8)
        addr = plan_lane_terms(streams["addr_all"], addr_df, 8)
        if not name or not addr:
            return []
        parts = ["(" + " OR ".join(f"name_all:{term}" for term in name) + ")",
                 "(" + " OR ".join(f"addr_all:{term}" for term in addr) + ")"]
        expression = " AND ".join(parts)
    elif kind == "address_pair":
        words = sorted({w for w in tokens(q.addr_norm) if w in addr_df},
                       key=lambda w: (addr_df[w], w))
        numbers = [w for w in words if any(ch.isdigit() for ch in w)]
        if len(words) < 2:
            return []
        chosen = [numbers[0], next((w for w in words if w != numbers[0]), "")] if numbers else words[:2]
        if not all(chosen):
            return []
        expression = " AND ".join(f"addr_all:{w}" for w in chosen)
    else:
        raise ValueError(kind)
    query = retr.index.parse_query(expression, ["name_all", "addr_all"])
    return [int(retr.searcher.doc(addr).to_dict()["ref_ord"][0])
            for _, addr in retr.searcher.search(query, size).hits]


def run(sample_size: int) -> None:
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[1] / "data"
    man = pq.read_table(root / "manifest" / "split.parquet", columns=["entity_id", "role", "is_anchor"])
    focal = set(man.filter(pc.and_(pc.equal(man.column("role"), "D"), man.column("is_anchor")))
                .column("entity_id").to_pylist())
    del man
    s1 = pq.read_table(root / "records" / "train_s1.parquet", columns=["entity_id", "country"])
    focal_ref = {rid: (i, country) for i, (rid, country) in
                 enumerate(zip(s1.column("entity_id").to_pylist(), s1.column("country").to_pylist()))
                 if rid in focal}
    del s1
    truth = [(rid, tid, *focal_ref[rid])
             for rid, tids in iter_truth_parquet(root / "records" / "truth.parquet")
             if rid in focal for tid in tids]
    pairs = random.Random(2026).sample(truth, min(sample_size, len(truth)))
    print(f"Screening {len(pairs)} truth links from D (full-index retrieval)", flush=True)
    t2 = sample_rows(root / "records" / "train_s2.parquet", {p[1] for p in pairs if p[1].startswith("S2-")})
    t3 = sample_rows(root / "records" / "train_s3.parquet", {p[1] for p in pairs if p[1].startswith("S3-")})
    rows = t2 | t3
    indexes = {c: Retriever(root / "index" / "D" / c) for c in ("India", "US")}
    maps = {c: {"field": field_map(root / "dfmap" / f"D_{c}.parquet"),
                "legacy": legacy_map(root / "env" / "D" / f"catalog_{c}.parquet")}
            for c in indexes}
    settings = [("legacy", 8, 16), ("legacy", 32, 16),
                ("field", 8, 16), ("field", 32, 16), ("field", 32, 24)]
    budgets = (2, 3, 4, 8)
    results = {(*s, b): Counter() for s in settings for b in budgets}
    difficult: list[dict] = []
    production = [(8, 16, 2, 2), (16, 16, 2, 4), (32, 24, 2, 4),
                  (32, 24, 4, 4), (32, 24, 4, 8), (16, 16, 4, 8)]
    prod_stats = {c: Counter() for c in production}
    for i, (_rid, tid, owner, country) in enumerate(pairs):
        row = rows[tid]
        q = QueryViews(row["name_norm"], row["addr_norm"], row["name_fold"], row["addr_fold"])
        base: dict[tuple[str, int, int], list] = {}
        for mode, lane_k, max_terms in settings:
            name, addr = indexes[country].search(q, lane_k=lane_k,
                                                df_map=maps[country][mode], max_terms=max_terms)
            fused = fuse_lanes(name, addr)
            base[(mode, lane_k, max_terms)] = fused
            for budget in budgets:
                stat = results[(mode, lane_k, max_terms, budget)]
                kept = compact_strict(fused, budget)
                stat["found"] += any(f.ref_id == owner for f in kept)
                stat["edges"] += len(kept)
                stat[f"{country}_total"] += 1
                stat[f"{country}_found"] += any(f.ref_id == owner for f in kept)
        joint = extra_hits(indexes[country], q, maps[country]["field"], "joint", 8)
        address_pair = extra_hits(indexes[country], q, maps[country]["field"], "address_pair", 8)
        base_legacy = base[("legacy", 8, 16)]
        base_wide = base[("field", 32, 24)]
        strongest = {h.ref_id for h in compact_strict(base_wide, 8)} | set(joint[:4]) | set(address_pair[:4])
        for lane_k, max_terms, budget, joint_k in production:
            lane_key = ("field", lane_k, max_terms)
            fused_base = base.get(lane_key)
            if fused_base is None:
                name_hits, addr_hits = indexes[country].search(
                    q, lane_k=lane_k, df_map=maps[country]["field"], max_terms=max_terms)
                fused_base = fuse_lanes(name_hits, addr_hits)
                base[lane_key] = fused_base
            joint_hits = [LaneHit(ref_id=ref, rank=rank, score=1.0)
                          for rank, ref in enumerate(joint[:joint_k], start=1)]
            kept = compact_with_joint(fused_base, budget, joint_hits, joint_k)
            stat = prod_stats[(lane_k, max_terms, budget, joint_k)]
            hit = any(f.ref_id == owner for f in kept)
            stat["found"] += hit
            stat["edges"] += len(kept)
            stat[f"{country}_total"] += 1
            stat[f"{country}_found"] += hit
        if owner not in strongest:
            difficult.append({"reference": _rid, "target": tid, "country": country,
                              "target_name": row["name_raw"], "target_address": row["addr_raw"],
                              "indic_name": is_indic(row["name_raw"]),
                              "missing_address": not row["addr_norm"],
                              "joint_refs": joint[:4], "address_pair_refs": address_pair[:4]})
        if (i + 1) % 500 == 0:
            print(f"screened {i + 1}/{len(pairs)}", flush=True)
    summary = []
    for config, stat in results.items():
        n = len(pairs)
        row = {"mode": config[0], "lane_k": config[1], "max_terms": config[2],
               "budget": config[3], "found": stat["found"], "total": n,
               "recall": stat["found"] / n, "mean_pairs_per_query": stat["edges"] / n,
               "US_recall": stat["US_found"] / max(1, stat["US_total"]),
               "India_recall": stat["India_found"] / max(1, stat["India_total"])}
        summary.append(row)
        if config[3] in (2, 4, 8):
            print(config, "recall", round(row["recall"], 5),
                  "mean_pairs/query", round(row["mean_pairs_per_query"], 2), flush=True)
    for label, stat in prod_stats.items():
        n = len(pairs)
        row = {"name": "production", "lane_k": label[0], "max_terms": label[1],
               "budget": label[2], "joint_k": label[3],
               "found": stat["found"], "total": n,
               "recall": stat["found"] / n, "mean_pairs_per_query": stat["edges"] / n,
               "US_recall": stat["US_found"] / max(1, stat["US_total"]),
               "India_recall": stat["India_found"] / max(1, stat["India_total"])}
        summary.append(row)
        print(label, "recall", round(row["recall"], 5),
              "mean_pairs/query", round(row["mean_pairs_per_query"], 2), flush=True)
    out = root / "reports" / "retrieval_screen_D.json"
    out.write_text(json.dumps(summary, indent=2))
    if difficult:
        ref_rows = sample_rows(root / "records" / "train_s1.parquet",
                               {m["reference"] for m in difficult})
        for item in difficult:
            item["reference_name"] = ref_rows[item["reference"]]["name_raw"]
            item["reference_address"] = ref_rows[item["reference"]]["addr_raw"]
    (root / "reports" / "retrieval_screen_misses_D.json").write_text(json.dumps(difficult, indent=2))
    print("saved", out, "elapsed_s", round(time.perf_counter() - t0, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=2000)
    run(parser.parse_args().sample_size)
