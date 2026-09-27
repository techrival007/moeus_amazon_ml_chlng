"""Pair comparison features over candidate edges (compact, auditable schema).

Every family answers a named error pattern (master.md 8.5): name similarity,
address similarity with missingness kept distinct from conflict, cross-field
composition signals, retrieval provenance, and script/meta indicators.
Entity-id numeric suffixes never enter features; only the source prefix does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rapidfuzz.distance import JaroWinkler, Levenshtein

from ber.normalize import tokens
from ber.records import RecordViews

FEATURE_NAMES: tuple[str, ...] = (
    # name family
    "name_jw",
    "name_lev_norm",
    "name_tok_jaccard",
    "name_tok_overlap",
    "name_exact",
    "name_sorted_exact",
    "name_sorted_jw",
    "name_len_ratio",
    "name_char_len_diff",
    "name_containment",
    "name_shared_df_weight",
    "name_target_indic",
    "name_ref_indic",
    # address family
    "addr_jw",
    "addr_lev_norm",
    "addr_tok_jaccard",
    "addr_exact",
    "addr_sorted_exact",
    "addr_shared_numeric",
    "addr_numeric_jaccard",
    "addr_len_ratio",
    "addr_char_len_diff",
    "addr_target_missing",
    "addr_ref_missing",
    # cross-field composition
    "name_exact_addr_conflict",
    "name_exact_addr_missing",
    # retrieval context
    "fused_v",
    "rank",
    "margin_g",
    "is_name_top1",
    "is_addr_top1",
    "name_lane_score",
    "addr_lane_score",
    "pool_size",
    "joint_rank",
    "joint_score",
    # meta
    "source_s2",
)

_BOOL = (True, False)


def _tokset(norm: str) -> frozenset[str]:
    return frozenset(tokens(norm))


def _numset(norm: str) -> frozenset[str]:
    return frozenset(t for t in tokens(norm) if any(c.isdigit() for c in t))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _len_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _df_weight(shared: frozenset[str], df: Mapping[str, int] | None,
               n_docs: int | None = None) -> float:
    """Mean smooth-idf weight of shared tokens (0.0 without a df map).

    idf(t) = log((N+1)/(df(t)+1)) + 1, with N derived from the map.
    """
    if not shared or not df:
        return 0.0
    n = n_docs if n_docs is not None else max(df.values())
    total = 0.0
    for t in shared:
        total += math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0
    return total / len(shared)


def pair_features(
    ref: RecordViews,
    tgt: RecordViews,
    fused_v: float,
    rank: int,
    is_name_top1: bool,
    is_addr_top1: bool,
    name_score: float,
    addr_score: float,
    margin: float | None,
    pool_size: int,
    token_df: Mapping[str, int] | None = None,
    token_df_n_docs: int | None = None,
    joint_rank: int = 0,
    joint_score: float = 0.0,
) -> list[float]:
    """One feature row aligned with FEATURE_NAMES."""
    rn, tn = ref.name_norm, tgt.name_norm
    ra, ta = ref.addr_norm, tgt.addr_norm
    rt, tt = _tokset(rn), _tokset(tn)
    rta, tta = _tokset(ra), _tokset(ta)
    rnum, tnum = _numset(ra), _numset(ta)

    name_missing = not rn or not tn
    addr_missing = not ra.strip() or not ta.strip()

    name_jw = 0.0 if name_missing else JaroWinkler.similarity(rn, tn)
    name_lev = 0.0 if name_missing else Levenshtein.normalized_similarity(rn, tn)
    addr_jw = 0.0 if addr_missing else JaroWinkler.similarity(ra, ta)
    addr_lev = 0.0 if addr_missing else Levenshtein.normalized_similarity(ra, ta)
    name_sorted_r = " ".join(sorted(rt)) if rt else ""
    name_sorted_t = " ".join(sorted(tt)) if tt else ""
    name_sorted_jw = (0.0 if name_missing
                      else JaroWinkler.similarity(name_sorted_r, name_sorted_t))

    shared_numeric = len(rnum & tnum)

    return [
        # name family
        float(name_jw),
        float(name_lev),
        _jaccard(rt, tt),
        float(len(rt & tt)),
        1.0 if rn == tn else 0.0,
        1.0 if rt == tt else 0.0,
        float(name_sorted_jw),
        _len_ratio(rn, tn),
        float(abs(len(rn) - len(tn))),
        1.0 if (rt and rt <= tt) or (tt and tt <= rt) else 0.0,
        _df_weight(rt & tt, token_df, token_df_n_docs),
        1.0 if tgt.name_indic else 0.0,
        1.0 if ref.name_indic else 0.0,
        # address family
        float(addr_jw),
        float(addr_lev),
        _jaccard(rta, tta),
        1.0 if (ra == ta and not addr_missing) else 0.0,
        1.0 if (rta == tta and not addr_missing) else 0.0,
        float(shared_numeric),
        _jaccard(rnum, tnum),
        _len_ratio(ra, ta),
        float(abs(len(ra) - len(ta))),
        1.0 if not tgt.addr_norm.strip() else 0.0,
        1.0 if not ref.addr_norm.strip() else 0.0,
        # cross-field composition: exact name with fully conflicting addresses
        1.0 if (rn == tn and not addr_missing and _jaccard(rta, tta) == 0.0) else 0.0,
        1.0 if (rn == tn and addr_missing) else 0.0,
        # retrieval context
        float(fused_v),
        float(rank),
        float(margin if margin is not None else 0.0),
        1.0 if is_name_top1 else 0.0,
        1.0 if is_addr_top1 else 0.0,
        float(name_score),
        float(addr_score),
        float(pool_size),
        float(joint_rank),
        float(joint_score),
        # meta
        1.0 if tgt.entity_id.startswith("S2-") else 0.0,
    ]
