"""Pair comparison features over candidate edges (compact, auditable schema).

Every family answers a named error pattern (master.md 8.5): name similarity,
address similarity with missingness kept distinct from conflict, cross-field
composition signals, retrieval provenance, and script/meta indicators.
Entity-id numeric suffixes never enter features; only the source prefix does.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from rapidfuzz.distance import JaroWinkler, Levenshtein

from ber.normalize import latin_accent_fold, tokens
from ber.translit import skeleton, skeleton_token
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
    # v5: sibling/near-twin discrimination (single-digit numbers, accents,
    # legal-suffix-free name core)
    "addr_num_jaccard_all",
    "addr_num_conflict",
    "addr_first_num_eq",
    "addr_num_symdiff",
    "name_fold_jw",
    "addr_fold_jw",
    "name_core_jaccard",
    "name_core_exact",
    "name_core_missing_tokens",
    # v6: cross-script consonant skeletons (Indic -> Latin transliteration)
    "name_skel_jaccard",
    "name_skel_jw",
    "name_skel_core_jaccard",
    "addr_skel_jaccard",
)

_BOOL = (True, False)

_NUM = re.compile(r"\d+")

# legal forms and generic business words, several languages; dropped to expose
# the distinctive name core ("advik farms" vs "advik properties")
_GENERIC = frozenset("""
llc inc ltd pvt private limited corp corporation co company companies
sarl sas sasu eurl sa sci snc selarl scop llp lp plc gmbh the and et
de la le les du des of group holding holdings services service center
centre international india industries enterprises partners associates
mr mrs smt shri sri dr ms m s dba formerly known as
""".split())


_GENERIC_SKEL = frozenset(skeleton_token(w) for w in _GENERIC) - {""}


def _nums(norm: str) -> list[str]:
    """All digit runs (single digits included), leading zeros stripped."""
    return [n.lstrip("0") or "0" for n in _NUM.findall(norm)]


def _core(tokset: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in tokset if t not in _GENERIC)


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

    rn_all, tn_all = _nums(ra), _nums(ta)
    rns, tns = frozenset(rn_all), frozenset(tn_all)
    if rn_all and tn_all:
        first_eq = 1.0 if rn_all[0] == tn_all[0] else 0.0
        num_conflict = 1.0 if not (rns & tns) else 0.0
    else:
        first_eq, num_conflict = -1.0, 0.0
    rf, tf = latin_accent_fold(rn), latin_accent_fold(tn)
    rfa, tfa = latin_accent_fold(ra), latin_accent_fold(ta)
    rc, tc = _core(rt), _core(tt)
    rsk, tsk = skeleton(rn), skeleton(tn)
    rsk_set, tsk_set = frozenset(rsk), frozenset(tsk)
    rsk_core, tsk_core = rsk_set - _GENERIC_SKEL, tsk_set - _GENERIC_SKEL
    rask, task = frozenset(skeleton(ra)), frozenset(skeleton(ta))

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
        # v5
        _jaccard(rns, tns),
        num_conflict,
        first_eq,
        float(len(rns ^ tns)),
        0.0 if name_missing else float(JaroWinkler.similarity(rf, tf)),
        0.0 if addr_missing else float(JaroWinkler.similarity(rfa, tfa)),
        _jaccard(rc, tc),
        1.0 if (rc and rc == tc) else 0.0,
        float(len(rc ^ tc)),
        # v6
        _jaccard(rsk_set, tsk_set),
        (float(JaroWinkler.similarity(" ".join(sorted(rsk_set)), " ".join(sorted(tsk_set))))
         if rsk_set and tsk_set else 0.0),
        _jaccard(rsk_core, tsk_core),
        0.0 if addr_missing else _jaccard(rask, task),
    ]
