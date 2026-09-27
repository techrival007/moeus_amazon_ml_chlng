"""Pipeline stages I: ingest -> split -> environment.

Scale design (master.md 15-16): vectorized parquet/numpy operations only;
no O(n^2) dict rebuilds; countries discovered dynamically.

Environment facts used here (master.md 7.2-7.3):
- Roles follow identity groups, so a cohort's mandatory traffic is exactly
  the set of targets carrying that role (owned or assigned-unlinked).
- Catalog: focal refs always kept + F/B background fill to the per-country
  quota (observed test sizes); shortfalls reported, never hidden.
- Traffic: mandatory role-R targets first, then a country-stratified B-role
  fill up to density * focal_ref_count; focal truth targets never dropped.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.records import build_source_parquet, build_truth_parquet, iter_truth_parquet, sha256_file
from ber.splits import SHARES, assign_roles, build_components, verify_split

DATA_SUBDIRS = ("records", "manifest", "env", "index", "dfmap", "edges", "features", "models", "policy", "reports")

CATALOG_SCHEMA = pa.schema([
    ("ref_ord", pa.int64()), ("entity_id", pa.string()), ("country", pa.string()),
    ("name_norm", pa.string()), ("addr_norm", pa.string()),
    ("name_fold", pa.string()), ("addr_fold", pa.string()),
    ("name_tri", pa.string()), ("name_word", pa.string()), ("name_foldtri", pa.string()),
    ("addr_tri", pa.string()), ("addr_word", pa.string()), ("addr_foldtri", pa.string()),
    ("name_indic", pa.bool_()), ("addr_indic", pa.bool_()), ("addr_missing", pa.bool_()),
])

TRAFFIC_SCHEMA = pa.schema([
    ("tgt_ord", pa.int64()), ("entity_id", pa.string()), ("source", pa.uint8()),
    ("country", pa.string()),
    ("name_norm", pa.string()), ("addr_norm", pa.string()),
    ("name_fold", pa.string()), ("addr_fold", pa.string()),
    ("name_indic", pa.bool_()), ("addr_indic", pa.bool_()), ("addr_missing", pa.bool_()),
])

SPLIT_SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("source", pa.uint8()), ("role", pa.string()),
    ("group_id", pa.int64()), ("country", pa.string()),
    ("truth_size", pa.int32()), ("is_anchor", pa.bool_()),
])


# ---------------------------------------------------------------- helpers


def _sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def ensure_dirs(data_dir: Path) -> None:
    for d in DATA_SUBDIRS:
        (data_dir / d).mkdir(parents=True, exist_ok=True)


def mark(shard: Path, fingerprint: str, **meta: Any) -> None:
    shard.with_suffix(".json").write_text(
        json.dumps({"fingerprint": fingerprint, **meta}), encoding="utf-8"
    )


def done(shard: Path, fingerprint: str) -> bool:
    side = shard.with_suffix(".json")
    if not shard.is_file() or not side.is_file():
        return False
    try:
        return json.loads(side.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
    except Exception:
        return False


def countries_of(table) -> list[str]:
    return sorted(set(table.column("country").to_pylist()))


def _id_array(ids: set[str]) -> pa.Array:
    return pa.array(sorted(ids), type=pa.string())


def _sha_of_file(path: Path) -> str:
    return sha256_file(path)


# ================================================================ ingest


def stage_ingest(dataset_dir: Path, data_dir: Path, workers: int = 6) -> dict:
    """Build the seven canonical parquet tables + fingerprint/density report."""
    dataset_dir, data_dir = Path(dataset_dir), Path(data_dir)
    ensure_dirs(data_dir)
    t0 = time.perf_counter()

    jobs = []
    for split in ("train", "test"):
        for source, tag in ((1, "S1"), (2, "S2"), (3, "S3")):
            src = dataset_dir / split / f"{split}_source{source}.tsv"
            dst = data_dir / "records" / f"{split}_s{source}.parquet"
            jobs.append((src, dst, tag))

    import multiprocessing as mp

    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        results = pool.starmap(build_source_parquet, jobs)
    counts = {Path(dst).name: n for (_, dst, _), (n, _) in zip(jobs, results)}

    truth_dst = data_dir / "records" / "truth.parquet"
    counts["truth.parquet"] = build_truth_parquet(
        dataset_dir / "train" / "train_ground_truth.tsv", truth_dst
    )

    fingerprints = {p.name: sha256_file(p) for p in sorted(dataset_dir.rglob("*.tsv"))}

    dist: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            name = f"{split}_s{source}"
            t = pq.read_table(data_dir / "records" / f"{name}.parquet", columns=["country"])
            cc = t.column("country").to_pylist()
            vals, cnts = np.unique(cc, return_counts=True)
            dist[name] = {str(v): int(c) for v, c in zip(vals, cnts)}
            del t, cc
    densities = {}
    for country in set(dist["test_s2"]) | set(dist["test_s3"]):
        tg = dist["test_s2"].get(country, 0) + dist["test_s3"].get(country, 0)
        densities[country] = tg / max(dist["test_s1"].get(country, 1), 1)

    report = {
        "counts": counts,
        "fingerprints": fingerprints,
        "country_distribution": dist,
        "test_target_density_per_ref": densities,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    _report(data_dir / "reports" / "ingest.json", report)
    return report


# ================================================================ split


def _duplicate_groups(t) -> list[list[str]]:
    """Entity-id groups for exact duplicate payloads (multi-row groups only)."""
    keep = ["name_raw", "addr_raw", "country"]
    srt = t.sort_by([(c, "ascending") for c in keep])
    n = srt.num_rows
    if n < 2:
        return []
    cols = [srt.column(c).to_pylist() for c in keep]
    ids = srt.column("entity_id").to_pylist()
    groups: list[list[str]] = []
    current = [ids[0]]
    for i in range(1, n):
        if all(cols[k][i] == cols[k][i - 1] for k in range(len(keep))):
            current.append(ids[i])
        else:
            if len(current) > 1:
                groups.append(current)
            current = [ids[i]]
    if len(current) > 1:
        groups.append(current)
    return groups


def stage_split(data_dir: Path, seed: int = 7, shares=SHARES) -> dict:
    """Components, F/D/K/A/B roles, leakage verification, manifest parquet."""
    data_dir = Path(data_dir)
    ensure_dirs(data_dir)
    rec = data_dir / "records"
    t0 = time.perf_counter()

    truth: dict[str, list[str]] = {}
    for sid, ids in iter_truth_parquet(rec / "truth.parquet"):
        truth[sid] = ids
    group_of, groups = build_components(truth)

    s1 = pq.read_table(rec / "train_s1.parquet", columns=["entity_id", "country"])
    anchor_ids = s1.column("entity_id").to_pylist()
    anchor_country = dict(zip(anchor_ids, s1.column("country").to_pylist()))
    anchor_meta = {a: (anchor_country[a], len(truth.get(a, ()))) for a in anchor_ids}
    del s1

    owned = {t for v in truth.values() for t in v}
    unlinked_groups: list[list[str]] = []
    unlinked_all: set[str] = set()
    for source in (2, 3):
        t = pq.read_table(rec / f"train_s{source}.parquet")
        ids_set = set(t.column("entity_id").to_pylist()) - owned
        unlinked_all |= ids_set
        sub = t.filter(pc.is_in(t.column("entity_id"), value_set=_id_array(ids_set)))
        unlinked_groups.extend(_duplicate_groups(sub))
        del t, sub
    # every unlinked target must receive a role: duplicate-payload groups stay
    # together, and every remaining unlinked id forms its own singleton group
    dup_members = {i for g in unlinked_groups for i in g}
    unlinked_groups.extend([i] for i in sorted(unlinked_all - dup_members))
    del unlinked_all, dup_members

    unlinked_meta: dict[str, tuple[str, str]] = {}
    for source in (2, 3):
        t = pq.read_table(rec / f"train_s{source}.parquet", columns=["entity_id", "country"])
        idx = dict(zip(t.column("entity_id").to_pylist(), t.column("country").to_pylist()))
        for g in unlinked_groups:
            head = g[0]
            if head.startswith(f"S{source}-"):
                unlinked_meta[head] = (idx[head], f"S{source}")
        del t, idx

    role_of = assign_roles(
        anchor_ids,
        anchor_meta,
        {i: g for i, g in enumerate(unlinked_groups)},
        group_of=group_of,
        target_meta=unlinked_meta,
        shares=shares,
        seed=seed,
    )
    verify_split(truth, role_of, group_of)

    # manifest: batched rows per source file (country from each record's file)
    n = 0
    role_counts: dict[str, int] = {}
    writer = pq.ParquetWriter(
        data_dir / "manifest" / "split.parquet", SPLIT_SCHEMA, compression="zstd"
    )
    try:
        for source, tag in ((1, "S1"), (2, "S2"), (3, "S3")):
            t = pq.read_table(rec / f"train_s{source}.parquet",
                              columns=["entity_id", "country"])
            ids = t.column("entity_id").to_pylist()
            ctry = t.column("country").to_pylist()
            batch: dict[str, list] = {f.name: [] for f in SPLIT_SCHEMA}
            for i in range(len(ids)):
                eid = ids[i]
                role = role_of[eid]
                role_counts[role] = role_counts.get(role, 0) + 1
                batch["entity_id"].append(eid)
                batch["source"].append(source)
                batch["role"].append(role)
                batch["group_id"].append(group_of.get(eid, -1))
                batch["country"].append(ctry[i])
                batch["truth_size"].append(len(truth.get(eid, ())) if tag == "S1" else 0)
                batch["is_anchor"].append(tag == "S1")
                n += 1
                if len(batch["entity_id"]) >= 200_000:
                    writer.write_table(pa.table(batch, schema=SPLIT_SCHEMA))
                    batch = {f.name: [] for f in SPLIT_SCHEMA}
            if batch["entity_id"]:
                writer.write_table(pa.table(batch, schema=SPLIT_SCHEMA))
            del t, ids, ctry
    finally:
        writer.close()
    del truth, group_of, groups

    report = {
        "seed": seed,
        "shares": list(shares),
        "n_entities": n,
        "role_counts": role_counts,
        "leakage_check": "passed",
        "manifest_fingerprint": sha256_file(data_dir / "manifest" / "split.parquet"),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    _report(data_dir / "reports" / "split.json", report)
    return report


def stage_reserve_audit(data_dir: Path, fraction: float = 0.05, seed: int = 13) -> dict:
    """Reserve a fresh audit cohort from whole B anchor/target components.

    The original manifest is retained so this operation is reproducible and
    idempotent. Unlinked background targets remain B-role distractors.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("audit fraction must be in (0, 1]")
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest" / "split.parquet"
    original = manifest.with_name("split_before_a2.parquet")
    if not original.exists():
        shutil.copy2(manifest, original)
    table = pq.read_table(original)
    roles = table.column("role")
    anchors = table.filter(pc.and_(pc.equal(roles, "B"), table.column("is_anchor")))
    limit = round(fraction * (1 << 64))
    selected = {
        int(gid) for sid, gid in zip(anchors.column("entity_id").to_pylist(),
                                    anchors.column("group_id").to_pylist())
        if int.from_bytes(hashlib.sha256(f"{seed}:{sid}".encode()).digest()[:8], "big") < limit
    }
    mask = pc.and_(pc.equal(roles, "B"), pc.is_in(
        table.column("group_id"), value_set=pa.array(sorted(selected), type=pa.int64())))
    updated = table.set_column(table.schema.get_field_index("role"), "role",
                               pc.if_else(mask, "A2", roles))
    temporary = manifest.with_name("split_with_a2.parquet")
    pq.write_table(updated, temporary, compression="zstd", row_group_size=200_000)
    temporary.replace(manifest)
    report = {"seed": seed, "fraction": fraction, "n_anchor_groups": len(selected),
              "n_A2_rows": int(pc.sum(mask).as_py()),
              "original_fingerprint": sha256_file(original),
              "manifest_fingerprint": sha256_file(manifest)}
    _report(data_dir / "reports" / "reserve_audit.json", report)
    return report


# ================================================================ environment


def _write_catalog(env_dir: Path, country: str, sub) -> Path:
    """Write one country's catalog parquet with precomputed term streams."""
    from ber.normalize import tokens, trigrams

    cols = {c: sub.column(c).to_pylist() for c in
            ("ordinal", "entity_id", "country", "name_norm", "addr_norm",
             "name_fold", "addr_fold", "name_indic", "addr_indic", "addr_missing")}
    data: dict[str, list] = {k: list(v) for k, v in cols.items()}
    data["ref_ord"] = data.pop("ordinal")
    data["name_tri"] = [" ".join(trigrams(s)) for s in cols["name_norm"]]
    data["name_word"] = [" ".join(tokens(s)) for s in cols["name_norm"]]
    data["name_foldtri"] = [" ".join(trigrams(s)) for s in cols["name_fold"]]
    data["addr_tri"] = [" ".join(trigrams(s)) for s in cols["addr_norm"]]
    data["addr_word"] = [" ".join(tokens(s)) for s in cols["addr_norm"]]
    data["addr_foldtri"] = [" ".join(trigrams(s)) for s in cols["addr_fold"]]
    out = env_dir / f"catalog_{country}.parquet"
    pq.write_table(pa.table(data, schema=CATALOG_SCHEMA), out, compression="zstd")
    return out


def _write_traffic_shards(env_dir: Path, country: str, sub, source_tag: int,
                          shard_size: int) -> int:
    """Write one country's traffic shards for one source; returns shard count."""
    d = env_dir / f"traffic_{country}"
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob(f"src{source_tag}_*.parquet"):
        p.unlink()
        side = p.with_suffix(".json")
        if side.exists():
            side.unlink()
    n = sub.num_rows
    written = 0
    for si in range(0, n, shard_size):
        chunk = sub.slice(si, min(shard_size, n - si))
        cols = {c: chunk.column(c).to_pylist() for c in
                ("ordinal", "entity_id", "country", "name_norm", "addr_norm",
                 "name_fold", "addr_fold", "name_indic", "addr_indic", "addr_missing")}
        data = {
            "tgt_ord": cols["ordinal"], "entity_id": cols["entity_id"],
            "source": [source_tag] * chunk.num_rows, "country": cols["country"],
            "name_norm": cols["name_norm"], "addr_norm": cols["addr_norm"],
            "name_fold": cols["name_fold"], "addr_fold": cols["addr_fold"],
            "name_indic": cols["name_indic"], "addr_indic": cols["addr_indic"],
            "addr_missing": cols["addr_missing"],
        }
        pq.write_table(pa.table(data, schema=TRAFFIC_SCHEMA),
                        d / f"src{source_tag}_{si // shard_size:04d}.parquet",
                        compression="zstd")
        written += 1
    return written


def stage_env_test(data_dir: Path, shard_size: int = 100_000) -> dict:
    """Test environment: catalog = all test refs; traffic = all test targets."""
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / "test"
    env_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"countries": {}, "cohort": "test"}
    t0 = time.perf_counter()

    s1 = pq.read_table(data_dir / "records" / "test_s1.parquet")
    total_refs = 0
    for country in countries_of(s1):
        sub = s1.filter(pc.equal(s1.column("country"), country)).sort_by("ordinal")
        _write_catalog(env_dir, country, sub)
        stats["countries"][country] = {
            "refs": sub.num_rows, "quota": sub.num_rows, "shortfall": 0,
        }
        total_refs += sub.num_rows
    del s1
    stats["total_refs"] = total_refs

    total_targets = 0
    for source in (2, 3):
        t = pq.read_table(data_dir / "records" / f"test_s{source}.parquet")
        for country in countries_of(t):
            sub = t.filter(pc.equal(t.column("country"), country)).sort_by("ordinal")
            shards = _write_traffic_shards(env_dir, country, sub, source, shard_size)
            c = stats["countries"].setdefault(country, {})
            c[f"s{source}_targets"] = sub.num_rows
            c[f"s{source}_shards"] = shards
            total_targets += sub.num_rows
        del t
    stats["total_targets"] = total_targets
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    _report(env_dir / "env.json", stats)
    return stats


def dense_catalog_ids(ref_ids: list[str], ref_roles: list[str], quota: int,
                      rng: np.random.Generator,
                      held_roles: tuple[str, ...] | None = None) -> tuple[list[str], list[str]]:
    """(catalog, ghosts): every non-B ref kept, B refs sampled up to quota.

    Unchosen B refs become ghosts: their targets stay available as ownerless
    distractors, exactly like test targets whose business has no S1 record.
    """
    held = [r for r, role in zip(ref_ids, ref_roles)
            if role != "B" and (held_roles is None or role in held_roles)]
    pool = sorted(r for r, role in zip(ref_ids, ref_roles) if role == "B")
    need = max(0, quota - len(held))
    pick = set(rng.permutation(len(pool))[:need].tolist()) if pool else set()
    chosen = [pool[i] for i in sorted(pick)]
    ghosts = [pool[i] for i in range(len(pool)) if i not in pick]
    return sorted(held + chosen), ghosts


def dense_traffic_ids(target_ids: list[str], owner_of: dict[str, str],
                      catalog: set[str], quota: int,
                      rng: np.random.Generator) -> tuple[list[str], int, int]:
    """(traffic, n_owned, n_orphan): all catalog-owned targets + orphan fill.

    Orphans are unlinked targets and targets owned by refs outside the
    catalog; they fill the per-source quota (test density * catalog size).
    """
    owned = [t for t in target_ids if owner_of.get(t) in catalog]
    orphans = sorted(t for t in target_ids if owner_of.get(t) not in catalog)
    need = max(0, quota - len(owned))
    take = min(need, len(orphans))
    pick = np.sort(rng.permutation(len(orphans))[:take])
    return sorted(owned + [orphans[i] for i in pick]), len(owned), int(take)


def stage_env_dense(data_dir: Path, cohort: str, quotas: dict[str, int],
                    densities: dict[str, dict[int, float]], seed: int = 7,
                    shard_size: int = 100_000, scale: float = 1.0,
                    held_roles: tuple[str, ...] | None = None) -> dict:
    """Test-faithful dense environment over the whole training set.

    Catalog per country: all D/K/A/A2/F refs + B fill to the test catalog
    size. Traffic: every target owned by a catalog ref plus orphans up to
    the observed test target density *per catalog ref* (not per focal ref),
    so each reference sees its siblings' and ownerless records as on test.
    """
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / cohort
    env_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"cohort": cohort, "dense": True, "countries": {}}
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    man = pq.read_table(data_dir / "manifest" / "split.parquet",
                        columns=["entity_id", "source", "role", "country"])
    anchors = man.filter(pc.equal(man.column("source"), 1))
    del man
    owner_of: dict[str, str] = {}
    for sid, ids in iter_truth_parquet(data_dir / "records" / "truth.parquet"):
        for t in ids:
            owner_of[t] = sid

    s1 = pq.read_table(data_dir / "records" / "train_s1.parquet")
    catalog_all: set[str] = set()
    n_catalog: dict[str, int] = {}
    for country in countries_of(anchors):
        sub_a = anchors.filter(pc.equal(anchors.column("country"), country))
        chosen, ghosts = dense_catalog_ids(sub_a.column("entity_id").to_pylist(),
                                           sub_a.column("role").to_pylist(),
                                           int(round(quotas.get(country, 0) * scale)), rng,
                                           held_roles)
        catalog_all.update(chosen)
        sub = s1.filter(pc.is_in(s1.column("entity_id"), value_set=_id_array(set(chosen))))
        sub = sub.sort_by("ordinal")
        _write_catalog(env_dir, country, sub)
        n_catalog[country] = sub.num_rows
        stats["countries"][country] = {"refs": sub.num_rows, "ghost_refs": len(ghosts),
                                       "quota": quotas.get(country, 0)}
    del s1, anchors
    stats["total_refs"] = sum(n_catalog.values())

    total_targets = 0
    for source in (2, 3):
        t = pq.read_table(data_dir / "records" / f"train_s{source}.parquet")
        for country in sorted(n_catalog):
            mask = pc.equal(t.column("country"), country)
            country_ids = t.column("entity_id").filter(mask).to_pylist()
            quota = int(round(densities.get(country, {}).get(source, 0.0) * n_catalog[country]))
            traffic, n_owned, n_orphan = dense_traffic_ids(country_ids, owner_of,
                                                           catalog_all, quota, rng)
            sub = t.filter(pc.is_in(t.column("entity_id"), value_set=_id_array(set(traffic))))
            sub = sub.sort_by("ordinal")
            shards = _write_traffic_shards(env_dir, country, sub, source, shard_size)
            c = stats["countries"][country]
            c[f"s{source}_quota"] = quota
            c[f"s{source}_owned"] = n_owned
            c[f"s{source}_orphans"] = n_orphan
            c[f"s{source}_shards"] = shards
            total_targets += len(traffic)
        del t
    for c in stats["countries"].values():
        n = c.get("s2_owned", 0) + c.get("s2_orphans", 0) + c.get("s3_owned", 0) + c.get("s3_orphans", 0)
        c["targets_per_ref"] = n / max(c["refs"], 1)
        c["ownerless_fraction"] = (c.get("s2_orphans", 0) + c.get("s3_orphans", 0)) / max(n, 1)
    stats["total_targets"] = total_targets
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    _report(env_dir / "env.json", stats)
    return stats


def stage_env_train(data_dir: Path, cohort: str, role: str,
                    quotas: dict[str, int], densities: dict[str, dict[int, float]],
                    seed: int = 7, shard_size: int = 100_000) -> dict:
    """Train-cohort environment: catalog + traffic per master.md 7.2-7.3."""
    data_dir = Path(data_dir)
    env_dir = data_dir / "env" / cohort
    env_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"cohort": cohort, "role": role, "countries": {}}
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    man = pq.read_table(data_dir / "manifest" / "split.parquet")
    ids = man.column("entity_id")
    roles = man.column("role")
    sources = man.column("source")
    countries = man.column("country")

    focal_mask = pc.and_(pc.equal(roles, role), pc.equal(sources, 1))
    bg_mask = pc.and_(
        pc.is_in(roles, value_set=pa.array(["F", "B"], type=pa.string())),
        pc.equal(sources, 1),
    )
    country_np = countries.to_numpy()
    focal_np = focal_mask.to_numpy()
    focal_countries, focal_counts = np.unique(country_np[focal_np], return_counts=True)
    focal_ref_counts = {str(c): int(n) for c, n in zip(focal_countries, focal_counts)}

    s1 = pq.read_table(data_dir / "records" / "train_s1.parquet")
    total_refs = 0
    for country in sorted(set(countries.to_pylist())):
        focal_ids = set(ids.filter(pc.and_(focal_mask, pc.equal(countries, country))).to_pylist())
        bg_ids = set(ids.filter(pc.and_(bg_mask, pc.equal(countries, country))).to_pylist())
        bg_ids -= focal_ids
        quota = quotas.get(country, 0)
        chosen = sorted(focal_ids)
        need = quota - len(focal_ids)
        if need > 0 and bg_ids:
            pool = sorted(bg_ids)
            take = min(need, len(pool))
            pick = rng.permutation(len(pool))[:take]
            chosen = sorted(set(chosen) | {pool[i] for i in pick})
        sub = s1.filter(pc.is_in(s1.column("entity_id"), value_set=_id_array(set(chosen))))
        sub = sub.sort_by("ordinal")
        _write_catalog(env_dir, country, sub)
        stats["countries"][country] = {
            "refs": sub.num_rows, "focal": len(focal_ids),
            "background": sub.num_rows - len(focal_ids),
            "quota": quota, "shortfall": max(0, len(focal_ids) - quota),
        }
        total_refs += sub.num_rows
    del s1
    stats["total_refs"] = total_refs

    # traffic: mandatory = every target carrying the cohort role (roles
    # follow groups), then B-role fill to density * focal_ref_count
    total_targets = 0
    for source in (2, 3):
        t = pq.read_table(data_dir / "records" / f"train_s{source}.parquet")
        role_ids = set(ids.filter(pc.and_(pc.equal(roles, role), pc.equal(sources, source))).to_pylist())
        b_ids = set(ids.filter(pc.and_(pc.equal(roles, "B"), pc.equal(sources, source))).to_pylist())
        for country in sorted(stats["countries"]):
            mask = pc.equal(t.column("country"), country)
            country_ids = set(t.column("entity_id").filter(mask).to_pylist())
            mand = sorted(role_ids & country_ids)
            b_pool = sorted(b_ids & country_ids)
            quota = int(round(densities.get(country, {}).get(source, 0.0)
                              * focal_ref_counts.get(country, 0)))
            traffic = list(mand)
            if quota > len(mand) and b_pool:
                take = min(quota - len(mand), len(b_pool))
                pick = rng.permutation(len(b_pool))[:take]
                traffic.extend(b_pool[i] for i in sorted(pick))
            sub = t.filter(pc.is_in(t.column("entity_id"), value_set=_id_array(set(traffic))))
            sub = sub.sort_by("ordinal")
            shards = _write_traffic_shards(env_dir, country, sub, source, shard_size)
            c = stats["countries"][country]
            c[f"s{source}_mandatory"] = len(mand)
            c[f"s{source}_quota"] = quota
            c[f"s{source}_background_fill"] = len(traffic) - len(mand)
            c[f"s{source}_shards"] = shards
            total_targets += len(traffic)
        del t
    stats["total_targets"] = total_targets
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    _report(env_dir / "env.json", stats)
    return stats
