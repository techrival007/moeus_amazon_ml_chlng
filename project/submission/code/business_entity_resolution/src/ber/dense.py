"""Dense-environment training, policy selection, audit and test decisions.

The dense cohort (pipeline.stage_env_dense) reproduces test conditions: every
catalog reference is surrounded by its siblings' and ownerless records. All
cohorts are evaluated inside that one environment, so labels only choose
which rows are trained (F/B refs and F/B targets) or scored (D/K/A2 refs).

Everything is vectorized over numpy edge arrays (tens of millions of edges).
Target identity is (source, tgt_ord); reference identity is the S1 ordinal.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ber import pipeline as pl
from ber.features import FEATURE_NAMES
from ber.records import iter_truth_parquet, sha256_file
from ber.training import TrainConfig, load_model, train_model

ROLE_CODES = {"F": 0, "D": 1, "K": 2, "A": 3, "B": 4, "A2": 5}
TRAIN_ROLES = (ROLE_CODES["F"], ROLE_CODES["B"])
BIG = np.int64(1 << 40)

STAGE2_NAMES = (
    "s1", "t_n", "t_rank", "t_gap_other", "t_n_hi",
    "r_n", "r_rank", "r_gap_max", "r_n_hi",
)


# ================================================================ labels


@dataclass
class Labels:
    ref_role: np.ndarray            # S1 ordinal -> role code
    ref_truth_n: np.ndarray         # S1 ordinal -> number of true targets
    tgt_owner: dict[int, np.ndarray]  # source -> target ordinal -> S1 ordinal / -1
    tgt_role: dict[int, np.ndarray]   # source -> target ordinal -> role code


def load_labels(data_dir: Path) -> Labels:
    data_dir = Path(data_dir)
    ord_of: dict[int, dict[str, int]] = {}
    for source in (1, 2, 3):
        ids = pq.read_table(data_dir / "records" / f"train_s{source}.parquet",
                            columns=["entity_id"]).column("entity_id").to_pylist()
        ord_of[source] = {e: i for i, e in enumerate(ids)}
    roles = {s: np.full(len(ord_of[s]), -1, np.int8) for s in (1, 2, 3)}
    man = pq.read_table(data_dir / "manifest" / "split.parquet",
                        columns=["entity_id", "source", "role"])
    for e, s, r in zip(man.column("entity_id").to_pylist(), man.column("source").to_pylist(),
                       man.column("role").to_pylist()):
        roles[s][ord_of[s][e]] = ROLE_CODES[r]
    del man
    owner = {s: np.full(len(ord_of[s]), -1, np.int64) for s in (2, 3)}
    truth_n = np.zeros(len(ord_of[1]), np.int32)
    for sid, ids in iter_truth_parquet(data_dir / "records" / "truth.parquet"):
        o = ord_of[1][sid]
        truth_n[o] = len(ids)
        for t in ids:
            s = int(t[1])
            owner[s][ord_of[s][t]] = o
    return Labels(roles[1], truth_n, owner, {2: roles[2], 3: roles[3]})


def _pick(arrays: dict[int, np.ndarray], src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    out = np.empty(src.size, dtype=arrays[2].dtype)
    for s in (2, 3):
        m = src == s
        out[m] = arrays[s][tgt[m]]
    return out


# ================================================================ edges


@dataclass
class Edges:
    ref: np.ndarray
    src: np.ndarray
    tgt: np.ndarray
    X: np.ndarray
    country: np.ndarray  # small int code per edge
    countries: list[str]


def load_edges(data_dir: Path, cohort: str, names: tuple[str, ...] | None = None) -> Edges:
    names = names or FEATURE_NAMES
    feats_dir = Path(data_dir) / "features" / cohort
    refs, srcs, tgts, xs, cc = [], [], [], [], []
    countries = sorted(p.name for p in feats_dir.iterdir() if p.is_dir())
    for ci, country in enumerate(countries):
        for shard in sorted((feats_dir / country).glob("src*_*.parquet")):
            t = pq.read_table(shard, columns=["ref_ord", "source", "tgt_ord", *names])
            refs.append(t.column("ref_ord").to_numpy())
            srcs.append(t.column("source").to_numpy())
            tgts.append(t.column("tgt_ord").to_numpy())
            xs.append(np.stack([t.column(n).to_numpy().astype(np.float32)
                                for n in names], axis=1))
            cc.append(np.full(t.num_rows, ci, np.int8))
            del t
    return Edges(np.concatenate(refs), np.concatenate(srcs), np.concatenate(tgts),
                 np.concatenate(xs), np.concatenate(cc), countries)


def catalog_refs(data_dir: Path, cohort: str) -> np.ndarray:
    parts = [pq.read_table(p, columns=["ref_ord"]).column("ref_ord").to_numpy()
             for p in sorted((Path(data_dir) / "env" / cohort).glob("catalog_*.parquet"))]
    return np.concatenate(parts)


# ================================================================ group features


def _group_bounds(keys_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = keys_sorted.size
    new = np.ones(n, bool)
    new[1:] = keys_sorted[1:] != keys_sorted[:-1]
    starts = np.flatnonzero(new)
    ends = np.concatenate([starts[1:], [n]])
    return starts, ends


def _rank_stats(keys: np.ndarray, s: np.ndarray, hi: float = 0.5):
    """Per-edge (group size, rank by -s, best other score, group max, #>=hi)."""
    order = np.lexsort((-s, keys))
    ks, ss = keys[order], s[order]
    starts, ends = _group_bounds(ks)
    sizes = ends - starts
    gid = np.repeat(np.arange(starts.size), sizes)
    rank = np.arange(ks.size) - starts[gid]
    top = ss[starts]
    second = np.full(starts.size, -1.0)
    has2 = sizes >= 2
    second[has2] = ss[starts[has2] + 1]
    best_other = np.where(rank == 0, second[gid], top[gid])
    n_hi = np.add.reduceat((ss >= hi).astype(np.int32), starts) if ks.size else np.array([], np.int32)
    out = [np.empty(s.size) for _ in range(5)]
    for arr, val in zip(out, (sizes[gid], rank, best_other, top[gid], n_hi[gid])):
        arr[order] = val
    return out


def group_features(ref: np.ndarray, src: np.ndarray, tgt: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Stage-2 competition features from stage-1 scores (STAGE2_NAMES order)."""
    tkey = src.astype(np.int64) * BIG + tgt.astype(np.int64)
    t_n, t_rank, t_other, _, t_hi = _rank_stats(tkey, s)
    r_n, r_rank, _, r_max, r_hi = _rank_stats(ref.astype(np.int64), s)
    return np.stack([s, t_n, t_rank, s - t_other, t_hi,
                     r_n, r_rank, s - r_max, r_hi], axis=1).astype(np.float32)


# ================================================================ policies


def winner_mask(src: np.ndarray, tgt: np.ndarray, s: np.ndarray, gamma: float) -> np.ndarray:
    """Per-target unique top edge whose margin over the runner-up >= gamma.

    Exact top ties abstain; a lone candidate always passes the margin gate.
    """
    tkey = src.astype(np.int64) * BIG + tgt.astype(np.int64)
    order = np.lexsort((-s, tkey))
    ks, ss = tkey[order], s[order]
    starts, ends = _group_bounds(ks)
    has2 = (ends - starts) >= 2
    first = ss[starts]
    second = np.full(starts.size, -np.inf)
    second[has2] = ss[starts[has2] + 1]
    ok = ~has2 | ((first > second) & ((first - second) >= gamma))
    mask = np.zeros(s.size, bool)
    mask[order[starts[ok]]] = True
    return mask


def macro_f(sel: np.ndarray, pos: np.ndarray, eref: np.ndarray,
            required: np.ndarray, truth_n: np.ndarray) -> float:
    """Exact macro-F0.5 over `required` refs; eref maps edges to required index (-1 = other)."""
    m = eref >= 0
    s_cnt = np.bincount(eref[m], weights=sel[m], minlength=required.size)
    tp = np.bincount(eref[m], weights=(sel & pos)[m], minlength=required.size)
    t = truth_n[required].astype(np.float64)
    f = np.where((t == 0) & (s_cnt == 0), 1.0,
                 np.where((t == 0) | (s_cnt == 0), 0.0, 5.0 * tp / np.maximum(t + 4.0 * s_cnt, 1e-9)))
    return float(f.mean())


def per_ref_f(sel, pos, eref, required, truth_n) -> np.ndarray:
    m = eref >= 0
    s_cnt = np.bincount(eref[m], weights=sel[m], minlength=required.size)
    tp = np.bincount(eref[m], weights=(sel & pos)[m], minlength=required.size)
    t = truth_n[required].astype(np.float64)
    return np.where((t == 0) & (s_cnt == 0), 1.0,
                    np.where((t == 0) | (s_cnt == 0), 0.0, 5.0 * tp / np.maximum(t + 4.0 * s_cnt, 1e-9)))


def sweep_tau(score: np.ndarray, base: np.ndarray, pos: np.ndarray, eref: np.ndarray,
              required: np.ndarray, truth_n: np.ndarray, n_grid: int = 600) -> tuple[float, float]:
    """Best tau over a quantile grid of required-edge scores (edges with base=True only)."""
    m = (eref >= 0) & base
    sc, ps, er = score[m], pos[m], eref[m]
    grid = np.unique(np.quantile(sc, np.linspace(0.0, 1.0, n_grid))) if sc.size else np.array([0.5])
    best = (macro_f(np.zeros(sc.size, bool), ps, er, required, truth_n), float("inf"))
    for tau in grid:
        f = macro_f(sc >= tau, ps, er, required, truth_n)
        if f > best[0]:
            best = (f, float(tau))
    # refine around the best grid point
    if np.isfinite(best[1]):
        for tau in np.linspace(best[1] - 0.02, best[1] + 0.02, 81):
            f = macro_f(sc >= tau, ps, er, required, truth_n)
            if f > best[0]:
                best = (f, float(tau))
    return best[1], best[0]


# ================================================================ context


@dataclass
class DenseContext:
    edges: Edges
    pos: np.ndarray        # edge is a true link
    ref_role: np.ndarray   # per-edge ref role
    tgt_role: np.ndarray   # per-edge target role
    labels: Labels
    catalog: np.ndarray


def build_context(data_dir: Path, cohort: str,
                  names: tuple[str, ...] | None = None) -> DenseContext:
    labels = load_labels(data_dir)
    e = load_edges(data_dir, cohort, names)
    owner = _pick(labels.tgt_owner, e.src, e.tgt)
    return DenseContext(e, owner == e.ref, labels.ref_role[e.ref],
                        _pick(labels.tgt_role, e.src, e.tgt), labels,
                        catalog_refs(data_dir, cohort))


def required_index(ctx: DenseContext, role: str) -> tuple[np.ndarray, np.ndarray]:
    """(required S1 ordinals of that role in the catalog, per-edge index or -1)."""
    code = ROLE_CODES[role]
    req = np.sort(ctx.catalog[ctx.labels.ref_role[ctx.catalog] == code])
    pos = np.searchsorted(req, ctx.edges.ref)
    pos = np.clip(pos, 0, max(req.size - 1, 0))
    eref = np.where((req.size > 0) & (req[pos] == ctx.edges.ref), pos, -1)
    return req, eref


def train_mask(ctx: DenseContext) -> np.ndarray:
    return np.isin(ctx.ref_role, TRAIN_ROLES) & np.isin(ctx.tgt_role, TRAIN_ROLES)


def evaluate(ctx: DenseContext, role: str, score: np.ndarray, policy: dict) -> dict[str, Any]:
    req, eref = required_index(ctx, role)
    sel = apply_policy(ctx.edges.src, ctx.edges.tgt, score, policy, ctx.edges.ref)
    tn = ctx.labels.ref_truth_n
    f = per_ref_f(sel, ctx.pos, eref, req, tn)
    m = eref >= 0
    tp = int((sel & ctx.pos & m).sum())
    fp = int((sel & ~ctx.pos & m).sum())
    n_truth = int(tn[req].sum())
    oracle = macro_f(ctx.pos.copy(), ctx.pos, eref, req, tn)
    out = {"role": role, "n_required": int(req.size), "macro_f": float(f.mean()),
           "oracle": oracle, "tp": tp, "fp": fp, "fn": n_truth - tp,
           "pred_per_ref": (tp + fp) / max(req.size, 1),
           "truth_per_ref": n_truth / max(req.size, 1)}
    for ci, c in enumerate(ctx.edges.countries):
        # a ref's country is the country of its edges; refs without edges are rare
        ref_c = np.full(req.size, -1)
        ref_c[eref[m]] = ctx.edges.country[m]
        sl = ref_c == ci
        if sl.sum() >= 200:
            out[f"country:{c}"] = float(f[sl].mean())
    sing = tn[req] == 0
    out["singletons"] = float(f[sing].mean()) if sing.any() else None
    return out


def expected_f_select(ref: np.ndarray, p: np.ndarray, base: np.ndarray,
                      beta: float = 0.0, floor: float = 0.0) -> np.ndarray:
    """Per-reference top-k maximizing expected F0.5 under calibrated p.

    For a ref with candidate probabilities sorted p1 >= p2 >= ..., expected
    truth size S ~ sum(p) + beta (beta covers links outside the candidates).
    E[F | top-k] ~ 5 * sum(p1..pk) / (S + 4k) for k >= 1 and prod(1 - p) for
    k = 0 (all candidates wrong: the empty row scores 1 only if truly empty).
    Only `base` edges are selectable (e.g. owner winners); edges below
    `floor` are never selected.
    """
    q = np.where(base, p, 0.0)
    order = np.lexsort((-q, ref))
    rs, qs = ref[order], q[order]
    starts, ends = _group_bounds(rs)
    sizes = ends - starts
    gid = np.repeat(np.arange(starts.size), sizes)
    csum = np.cumsum(qs)
    before = np.concatenate([[0.0], csum])[starts]
    ck = csum - before[gid]                       # sum of top-(rank+1)
    k = np.arange(qs.size) - starts[gid] + 1
    total = np.add.reduceat(qs, starts) if qs.size else np.array([])
    obj = 5.0 * ck / (total[gid] + beta + 4.0 * k)
    logq = np.log(np.clip(1.0 - qs, 1e-12, 1.0))
    p_empty = np.exp(np.add.reduceat(logq, starts)) if qs.size else np.array([])
    # best k per group (first max), compared against k = 0
    obj_masked = np.where(qs > max(floor, 0.0), obj, -1.0)
    best_obj = np.maximum.reduceat(obj_masked, starts) if qs.size else np.array([])
    is_best = obj_masked == best_obj[gid]
    first_best = np.full(starts.size, np.iinfo(np.int64).max)
    np.minimum.at(first_best, gid[is_best], k[is_best])
    kstar = np.where(best_obj > p_empty, first_best, 0)
    chosen_sorted = k <= kstar[gid]
    sel = np.zeros(p.size, bool)
    sel[order] = chosen_sorted & (qs > 0)
    return sel


def apply_policy(src, tgt, score, policy: dict, ref: np.ndarray | None = None) -> np.ndarray:
    if policy["kind"] == "efm":
        base = winner_mask(src, tgt, score, policy["gamma"])
        return expected_f_select(ref, score, base, policy["beta"], policy.get("floor", 0.0))
    sel = score >= policy["tau"]
    if policy["kind"] == "owner":
        sel &= winner_mask(src, tgt, score, policy["gamma"])
    return sel


def select_policy(ctx: DenseContext, role: str, score: np.ndarray,
                  gammas=(0.0, 0.02, 0.05, 0.1, 0.2)) -> dict[str, Any]:
    req, eref = required_index(ctx, role)
    tn = ctx.labels.ref_truth_n
    allm = np.ones(score.size, bool)
    tau, f = sweep_tau(score, allm, ctx.pos, eref, req, tn)
    results = [{"kind": "threshold", "tau": tau, "macro_f": f}]
    for g in gammas:
        w = winner_mask(ctx.edges.src, ctx.edges.tgt, score, g)
        tau_o, f_o = sweep_tau(score, w, ctx.pos, eref, req, tn)
        results.append({"kind": "owner", "gamma": g, "tau": tau_o, "macro_f": f_o})
    # expected-F0.5 per-reference selection (needs calibrated probabilities)
    ref = ctx.edges.ref
    for g in (0.0, 0.1):
        w = winner_mask(ctx.edges.src, ctx.edges.tgt, score, g)
        for beta in (0.0, 0.1, 0.3):
            for floor in (0.0, 0.1):
                sel = expected_f_select(ref, score, w, beta, floor)
                results.append({"kind": "efm", "gamma": g, "beta": beta, "floor": floor, "tau": 0.0,
                                "macro_f": macro_f(sel, ctx.pos, eref, req, tn)})
    best = max(results, key=lambda r: r["macro_f"])
    return {"best": best, "all": results}


# ================================================================ models


def fit_lgbm(X: np.ndarray, y: np.ndarray, rounds: int, threads: int, seed: int = 0,
             leaves: int = 127, lr: float = 0.05):
    cfg = TrainConfig(num_leaves=leaves, learning_rate=lr, num_rounds=rounds,
                      num_threads=threads, seed=seed, histogram_pool_mb=4096)
    return train_model(X, y, cfg)


def stage1_oof(ctx: DenseContext, rounds: int, threads: int, folds: int = 2) -> tuple[np.ndarray, list]:
    """Out-of-fold stage-1 scores on training edges; fold-average elsewhere."""
    tm = train_mask(ctx)
    fold_of_ref = (ctx.edges.ref * 2654435761 % 1000003) % folds
    score = np.zeros(ctx.edges.ref.size)
    models = []
    for k in range(folds):
        fit = tm & (fold_of_ref != k)
        bst = fit_lgbm(ctx.edges.X[fit], ctx.pos[fit].astype(np.int8), rounds, threads, seed=k)
        models.append(bst)
        pred = bst.predict(ctx.edges.X)
        hold = tm & (fold_of_ref == k)
        score[hold] = pred[hold]
        score[~tm] += pred[~tm] / folds
    return score, models


def report(data_dir: Path, name: str, payload: dict) -> None:
    pl._report(Path(data_dir) / "reports" / f"dense_{name}.json", payload)
    print(json.dumps(payload, indent=2, default=str), flush=True)


# ================================================================ test scoring


def score_test(data_dir: Path, stage1_models: list, stage2_model=None,
               calibrator: tuple[float, float] | None = None):
    """Stage-1 (fold-average) and optional stage-2 scores over cached test features."""
    e = load_edges(data_dir, "test")
    s1 = np.mean([m.predict(e.X) for m in stage1_models], axis=0)
    if calibrator is not None:
        from ber.stages import platt_apply
        s1 = platt_apply(s1, *calibrator)
    if stage2_model is None:
        return e, s1
    G = group_features(e.ref, e.src, e.tgt, s1)
    return e, stage2_model.predict(np.concatenate([e.X, G], axis=1))


def export_test(data_dir: Path, output_dir: Path, e: Edges, selected: np.ndarray) -> dict:
    from ber.export import write_scored_outputs

    data_dir = Path(data_dir)
    ids = {s: pq.read_table(data_dir / "records" / f"test_s{s}.parquet",
                            columns=["entity_id"]).column("entity_id").to_pylist() for s in (1, 2, 3)}
    lines = write_scored_outputs(Path(output_dir), ids[1], ids[2], ids[3],
                                 e.ref, e.tgt, e.src, selected)
    per_country = {}
    for ci, c in enumerate(e.countries):
        m = e.country == ci
        refs_c = np.unique(e.ref[m])
        sel_refs = np.unique(e.ref[m & selected])
        per_country[c] = {"pred_per_ref": float(selected[m].sum()) / max(refs_c.size, 1),
                          "nonempty_frac_of_refs_with_cands": sel_refs.size / max(refs_c.size, 1)}
    tkey = e.src.astype(np.int64) * BIG + e.tgt.astype(np.int64)
    _, cnt = np.unique(tkey[selected], return_counts=True)
    return {"n_edges": int(e.ref.size), "selected": int(selected.sum()),
            "rows_with_matches": lines, "targets_multi_owner": int((cnt > 1).sum()),
            "per_country": per_country}
