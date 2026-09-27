"""Inspect candidate-support misses on an identity-disjoint training cohort.

Uses only supplied training truth and the existing catalog/index/edge shards.
Does not train a model or touch the test set's hidden labels.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.candidates import fuse_lanes
from ber.records import iter_truth_parquet
from ber.retrieval import QueryViews, Retriever


def sample_rows(path: Path, wanted: set[str]) -> dict[str, dict]:
    rows = {}
    if not wanted:
        return rows
    pf = pq.ParquetFile(path)
    columns = ["entity_id", "name_raw", "addr_raw", "name_norm", "addr_norm",
               "name_fold", "addr_fold"]
    wanted_arr = pa.array(sorted(wanted), type=pa.string())
    for batch in pf.iter_batches(batch_size=200_000, columns=columns):
        selected = pa.Table.from_batches([batch]).filter(pc.is_in(batch.column(0), value_set=wanted_arr))
        for r in selected.to_pylist():
            rows[r["entity_id"]] = r
    return rows


def run(cohort: str, sample_size: int) -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    split = pq.read_table(root / "manifest" / "split.parquet",
                          columns=["entity_id", "role", "is_anchor"])
    focal = set(split.filter(pc.and_(pc.equal(split.column("role"), cohort),
                                    split.column("is_anchor"))).column("entity_id").to_pylist())
    del split
    truth = {rid: tids for rid, tids in iter_truth_parquet(root / "records" / "truth.parquet")
             if rid in focal}
    wanted_targets = {tid for tids in truth.values() for tid in tids}

    s1 = pq.read_table(root / "records" / "train_s1.parquet",
                       columns=["entity_id", "country"])
    focal_ref = {rid: (ordinal, country) for ordinal, (rid, country)
                 in enumerate(zip(s1.column("entity_id").to_pylist(),
                                  s1.column("country").to_pylist())) if rid in focal}
    del s1
    targets: dict[str, tuple[int, int]] = {}
    for source in (2, 3):
        table = pq.read_table(root / "records" / f"train_s{source}.parquet",
                              columns=["entity_id"])
        targets.update({tid: (source, i) for i, tid in enumerate(table.column("entity_id").to_pylist())
                        if tid in wanted_targets})
        del table

    scored = set()
    for edge_file in sorted((root / "edges" / cohort).glob("*/src*.parquet")):
        table = pq.read_table(edge_file, columns=["ref_ord", "source", "tgt_ord"])
        scored.update((int(ref) << 26) | (int(src) << 24) | int(tgt)
                      for ref, src, tgt in zip(table.column("ref_ord").to_numpy(),
                                                table.column("source").to_numpy(),
                                                table.column("tgt_ord").to_numpy()))
    del table
    misses = []
    total = 0
    oracle_total = 0.0
    by_country: dict[str, Counter] = {"US": Counter(), "India": Counter()}
    for rid, tids in truth.items():
        ref_ord, country = focal_ref[rid]
        found = 0
        for tid in tids:
            source, tgt_ord = targets[tid]
            total += 1
            by_country[country]["all"] += 1
            if (ref_ord << 26) | (source << 24) | tgt_ord not in scored:
                by_country[country]["miss"] += 1
                misses.append((rid, tid, ref_ord, country))
            else:
                found += 1
        oracle_total += 1.0 if not tids else 5.0 * found / (len(tids) + 4.0 * found)
    print("focal truth links", total, "candidate misses", len(misses),
          "oracle macro-F0.5", oracle_total / len(truth), "by country", by_country)

    # A deterministic bounded sample avoids searching every missing target.
    chosen = random.Random(7).sample(misses, min(sample_size, len(misses)))
    refs = sample_rows(root / "records" / "train_s1.parquet", {r for r, _, _, _ in chosen})
    t2 = sample_rows(root / "records" / "train_s2.parquet", {t for _, t, _, _ in chosen if t.startswith("S2-")})
    t3 = sample_rows(root / "records" / "train_s3.parquet", {t for _, t, _, _ in chosen if t.startswith("S3-")})
    tgt_rows = t2 | t3
    counts = Counter()
    for index, (rid, tid, ref_ord, country) in enumerate(chosen):
        target = tgt_rows[tid]
        q = QueryViews(target["name_norm"], target["addr_norm"],
                       target["name_fold"], target["addr_fold"])
        if country not in run.cache:
            dt = pq.read_table(root / "dfmap" / f"{cohort}_{country}.parquet")
            if "field" in dt.column_names:
                run.cache[country] = ({f: dict(zip(dt.filter(pc.equal(dt.column("field"), f)).column("token").to_pylist(),
                                               dt.filter(pc.equal(dt.column("field"), f)).column("df").to_pylist()))
                                      for f in ("name_all", "addr_all")}, Retriever(root / "index" / cohort / country))
            else:
                run.cache[country] = (dict(zip(dt.column("token").to_pylist(), dt.column("df").to_pylist())),
                                      Retriever(root / "index" / cohort / country))
        df, retr = run.cache[country]
        name8, addr8 = retr.search(q, lane_k=8, df_map=df, max_terms=16)
        fr8 = next((i for i, h in enumerate(fuse_lanes(name8, addr8), 1) if h.ref_id == ref_ord), 0)
        name, addr = retr.search(q, lane_k=64, df_map=df, max_terms=16)
        fused = fuse_lanes(name, addr)
        nr = next((h.rank for h in name if h.ref_id == ref_ord), 0)
        ar = next((h.rank for h in addr if h.ref_id == ref_ord), 0)
        fr = next((i for i, h in enumerate(fused, 1) if h.ref_id == ref_ord), 0)
        counts["not_top64" if fr == 0 else "rank_1_2" if fr <= 2 else "rank_3_4" if fr <= 4
               else "rank_5_8" if fr <= 8 else "rank_9_16" if fr <= 16 else "rank_gt16"] += 1
        counts["not_top8" if fr8 == 0 else "rank_1_2_top8" if fr8 <= 2 else "rank_3_4_top8"
               if fr8 <= 4 else "rank_gt4_top8"] += 1
        if index < 25:
            ref = refs[rid]
            print(f"{country} {rid}->{tid} name_rank={nr} addr_rank={ar} fused8={fr8} fused64={fr}")
            print("  ref:", ref["name_raw"], "|", ref["addr_raw"])
            print("  tgt:", target["name_raw"], "|", target["addr_raw"])
    print("sampled true-owner rank distribution", counts)


run.cache = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="D", choices=("D", "K", "A"))
    parser.add_argument("--sample-size", type=int, default=200)
    arguments = parser.parse_args()
    run(arguments.cohort, arguments.sample_size)
