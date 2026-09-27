"""Screen forward reference-to-target retrieval on D's actual target traffic.

The indexed side is the S2/S3 traffic used by the held-out D cohort. This
experiment never uses test labels and does not produce submission candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.candidates import compact_strict, fuse_lanes
from ber.records import iter_truth_parquet
from ber.retrieval import QueryViews, Retriever, build_index, index_doc_from_views
from diagnose_retrieval import sample_rows

DATA = Path(__file__).resolve().parents[1] / "data"
TARGET_OFFSET = 1 << 40


def target_documents(country: str, df: dict[str, Counter]):
    for path in sorted((DATA / "env" / "D" / f"traffic_{country}").glob("src*.parquet")):
        pf = pq.ParquetFile(path)
        cols = ("source", "tgt_ord", "name_norm", "addr_norm", "name_fold", "addr_fold")
        for batch in pf.iter_batches(batch_size=20_000, columns=list(cols)):
            for src, ordinal, name, addr, name_fold, addr_fold in zip(
                *(batch.column(c).to_pylist() for c in cols)
            ):
                doc = index_doc_from_views(src * TARGET_OFFSET + ordinal,
                                           name, addr, name_fold, addr_fold)
                df["name_all"].update(set(doc.name_all.split()))
                df["addr_all"].update(set(doc.addr_all.split()))
                yield doc


def forward_index(country: str) -> tuple[Retriever, dict[str, dict[str, int]]]:
    out = DATA / "index-forward" / "D" / country
    marker = out / "complete.json"
    if not marker.exists():
        if out.exists():
            raise ValueError(f"forward index exists without complete marker: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = {"name_all": Counter(), "addr_all": Counter()}
        t0 = time.perf_counter()
        build_index(out, target_documents(country, df), threads=6)
        (out / "frequencies.json").write_text(json.dumps(df))
        marker.write_text(json.dumps({"country": country, "seconds": time.perf_counter() - t0}))
        print("built forward target index", country, "in", round(time.perf_counter() - t0, 1), "s", flush=True)
    df = json.loads((out / "frequencies.json").read_text())
    return Retriever(out), df


def run(sample_size: int) -> None:
    t0 = time.perf_counter()
    man = pq.read_table(DATA / "manifest" / "split.parquet",
                        columns=["entity_id", "role", "is_anchor"])
    focal = set(man.filter(pc.and_(pc.equal(man.column("role"), "D"),
                                   man.column("is_anchor"))).column("entity_id").to_pylist())
    del man
    s1 = pq.read_table(DATA / "records" / "train_s1.parquet", columns=["entity_id", "country"])
    ref_info = {rid: (i, country) for i, (rid, country) in enumerate(
        zip(s1.column("entity_id").to_pylist(), s1.column("country").to_pylist())) if rid in focal}
    del s1
    truth = [(rid, tid, *ref_info[rid])
             for rid, tids in iter_truth_parquet(DATA / "records" / "truth.parquet")
             if rid in focal for tid in tids]
    wanted = {tid for _, tid, _, _ in truth}
    target_codes = {}
    for source in (2, 3):
        table = pq.read_table(DATA / "records" / f"train_s{source}.parquet", columns=["entity_id"])
        target_codes.update({tid: (source, i) for i, tid in enumerate(table.column("entity_id").to_pylist())
                             if tid in wanted})
        del table
    support = set()
    for path in sorted((DATA / "edges" / "D").glob("*/src*.parquet")):
        batch = pq.read_table(path, columns=["ref_ord", "source", "tgt_ord"])
        support.update((int(ref) << 26) | (int(source) << 24) | int(tgt)
                       for ref, source, tgt in zip(batch.column("ref_ord").to_numpy(),
                                                    batch.column("source").to_numpy(),
                                                    batch.column("tgt_ord").to_numpy()))
    misses = [(rid, tid, ref, country) for rid, tid, ref, country in truth
              if ((ref << 26) | (target_codes[tid][0] << 24) | target_codes[tid][1]) not in support]
    rng = random.Random(2026)
    panel = rng.sample(misses, min(sample_size, len(misses)))
    print("joint-lane support misses", len(misses), "screening", len(panel), flush=True)
    refs = sample_rows(DATA / "records" / "train_s1.parquet", {rid for rid, _, _, _ in panel})
    indexes = {country: forward_index(country) for country in ("US", "India")}
    budgets = (2, 4, 8, 16, 32, 64)
    counters = {b: Counter() for b in budgets}
    examples = []
    for i, (rid, tid, _ref, country) in enumerate(panel):
        row = refs[rid]
        q = QueryViews(row["name_norm"], row["addr_norm"], row["name_fold"], row["addr_fold"])
        retr, df = indexes[country]
        name, addr = retr.search(q, lane_k=64, df_map=df, max_terms=16)
        fused = fuse_lanes(name, addr)
        source, target_ordinal = target_codes[tid]
        target_key = source * TARGET_OFFSET + target_ordinal
        found_rank = next((j for j, f in enumerate(fused, 1) if f.ref_id == target_key), 0)
        for budget in budgets:
            stat = counters[budget]
            kept = compact_strict(fused, budget)
            hit = any(f.ref_id == target_key for f in kept)
            stat["found"] += hit
            stat["edges"] += len(kept)
            stat[f"{country}_total"] += 1
            stat[f"{country}_found"] += hit
        if i < 50:
            examples.append({"reference": rid, "target": tid, "country": country,
                             "name": row["name_raw"], "address": row["addr_raw"],
                             "forward_fused_rank": found_rank})
        if (i + 1) % 500 == 0:
            print("queried", i + 1, "of", len(panel), flush=True)
    report = [{"budget": b, "recall_on_reverse_misses": stat["found"] / len(panel),
               "mean_forward_edges_per_reference": stat["edges"] / len(panel),
               "US_recall": stat["US_found"] / max(1, stat["US_total"]),
               "India_recall": stat["India_found"] / max(1, stat["India_total"])}
              for b, stat in counters.items()]
    (DATA / "reports" / "forward_screen_D.json").write_text(json.dumps({
        "reverse_missed_links": len(misses), "sampled": len(panel),
        "budget_frontier": report, "examples": examples,
        "elapsed_seconds": time.perf_counter() - t0,
    }, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1200)
    run(parser.parse_args().sample_size)
