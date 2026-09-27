"""Exact competition metric: per-reference macro-F0.5, support oracle, and
paired bootstrap lower confidence bounds.

Conventions (master.md 5.2):
    F(T, S) = 1.0                      if T empty and S empty
            = 0.0                      if exactly one of T, S is empty
            = 5*|T&S| / (|T| + 4*|S|)  otherwise
    macro = mean of F over ALL required references.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np


def entity_score(truth: Collection[str], pred: Collection[str]) -> float:
    """Exact F0.5 for one reference row. Inputs are treated as sets."""
    t = len(set(truth))
    s = len(set(pred))
    if t == 0 and s == 0:
        return 1.0
    if t == 0 or s == 0:
        return 0.0
    tp = len(set(truth) & set(pred))
    return 5.0 * tp / (t + 4.0 * s)


def oracle_score(truth: Collection[str], candidates: Collection[str]) -> float:
    """Best achievable score restricted to the candidate support."""
    t = len(set(truth))
    if t == 0:
        return 1.0
    r = len(set(truth) & set(candidates))
    if r == 0:
        return 0.0
    return 5.0 * r / (t + 4.0 * r)


def _as_mapping(rows: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]]) -> dict[str, frozenset[str]]:
    if isinstance(rows, Mapping):
        return {k: frozenset(v) for k, v in rows.items()}
    return {k: frozenset(v) for k, v in rows}


def macro_score(
    truth: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    pred: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    required: Iterable[str],
) -> float:
    """Mean entity score over all required references (truth must cover them)."""
    t = _as_mapping(truth)
    p = _as_mapping(pred)
    total = 0.0
    n = 0
    for rid in required:
        if rid not in t:
            raise KeyError(f"truth is missing required reference {rid!r}")
        total += entity_score(t[rid], p.get(rid, frozenset()))
        n += 1
    return total / n


def macro_oracle(
    truth: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    candidates: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    required: Iterable[str],
) -> float:
    t = _as_mapping(truth)
    c = _as_mapping(candidates)
    total = 0.0
    n = 0
    for rid in required:
        if rid not in t:
            raise KeyError(f"truth is missing required reference {rid!r}")
        total += oracle_score(t[rid], c.get(rid, frozenset()))
        n += 1
    return total / n


def score_report(
    truth: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    pred: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    candidates: Mapping[str, Collection[str]] | Iterable[tuple[str, Collection[str]]],
    required: Iterable[str],
) -> dict[str, Any]:
    """Macro F, oracle ceiling, loss decomposition, and pair-level counts."""
    t = _as_mapping(truth)
    p = _as_mapping(pred)
    c = _as_mapping(candidates)
    req = list(required)
    for rid in req:
        if rid not in t:
            raise KeyError(f"truth is missing required reference {rid!r}")

    macro = 0.0
    oracle = 0.0
    tp = fp = fn = 0
    n_truth_empty = 0
    n_pred_empty = 0
    n_cand_empty = 0
    for rid in req:
        tt, pp, cc = t[rid], p.get(rid, frozenset()), c.get(rid, frozenset())
        macro += entity_score(tt, pp)
        oracle += oracle_score(tt, cc)
        tp += len(tt & pp)
        fp += len(pp - tt)
        fn += len(tt - pp)
        n_truth_empty += not tt
        n_pred_empty += not pp
        n_cand_empty += not cc
    n = len(req)
    macro /= n
    oracle /= n
    return {
        "n_required": n,
        "macro_f": macro,
        "oracle": oracle,
        "blocking_loss": 1.0 - oracle,
        "scoring_loss": oracle - macro,
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "n_truth_empty": n_truth_empty,
        "n_pred_empty": n_pred_empty,
        "n_cand_empty": n_cand_empty,
    }


def paired_bootstrap_lcb(
    scores_new: np.ndarray,
    scores_ref: np.ndarray,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> float:
    """One-sided lower confidence bound on mean(scores_new - scores_ref).

    Resamples reference rows (paired). Used for the non-inferiority gate in
    master.md 6.3: accept a compact policy when LCB >= -0.001.
    """
    a = np.asarray(scores_new, dtype=np.float64)
    b = np.asarray(scores_ref, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("score vectors must be same-length 1-D arrays")
    if a.size == 0:
        raise ValueError("empty score vectors")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    delta = a - b
    rng = np.random.default_rng(seed)
    n = delta.size
    idx = rng.integers(0, n, size=(replicates, n))
    means = delta[idx].mean(axis=1)
    # one-sided lower bound at `confidence`
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha))
