"""Evaluation environment construction: catalogs and query traffic.

build_catalog (master.md 7.3 steps 1-2): focal references are always kept;
background is a label-blind, seeded, per-country sample from F/B roles only,
capped by the country quota; D/K/A records are never background.

build_traffic (steps 3-4): focal-owned targets and cohort-assigned unlinked
targets are reserved first; the remaining per-country quota (observed test
density times focal reference count) is filled by a country/source-stratified
sample of B-role targets. Focal truth targets are never dropped. Background
targets whose owner is not in the catalog are legitimate no-owner distractors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

_BACKGROUND_ROLES = ("F", "B")


def build_catalog(
    role_of: Mapping[str, str],
    ref_country: Mapping[str, str],
    focal_ids: Sequence[str],
    quotas: Mapping[str, int],
    seed: int = 0,
) -> list[str]:
    """Catalog ids: focal refs + per-country F/B background fill to quota."""
    rng = np.random.default_rng(seed)
    focal = list(dict.fromkeys(focal_ids))
    focal_by_country: dict[str, list[str]] = defaultdict(list)
    for r in focal:
        focal_by_country[ref_country[r]].append(r)
    background_by_country: dict[str, list[str]] = defaultdict(list)
    for r in sorted(role_of):
        if role_of[r] in _BACKGROUND_ROLES and r not in set(focal) and r in ref_country:
            background_by_country[ref_country[r]].append(r)

    catalog: list[str] = []
    for country in sorted(set(focal_by_country) | set(background_by_country)):
        focal_c = focal_by_country.get(country, [])
        quota = quotas.get(country, 0)
        catalog.extend(focal_c)
        need = quota - len(focal_c)
        if need > 0:
            pool = background_by_country.get(country, [])
            if pool:
                take = min(need, len(pool))
                idx = rng.permutation(len(pool))[:take]
                catalog.extend(pool[i] for i in sorted(idx))
    return catalog


def build_traffic(
    role_of: Mapping[str, str],
    focal_ref_ids: Sequence[str],
    truth: Mapping[str, Sequence[str]],
    ref_country: Mapping[str, str],
    tgt_country: Mapping[str, str],
    tgt_source: Mapping[str, str],
    densities: Mapping[str, float],
    seed: int = 0,
    catalog: Sequence[str] | None = None,
    return_report: bool = False,
):
    """Query traffic ids for one focal cohort, plus an optional report.

    Mandatory first: every target owned by a focal reference (with the
    focal reference's role), plus cohort-assigned unlinked targets.
    Fill: per-country quota = density * focal_ref_count(country), filled by
    a country/source-stratified B-role sample.
    """
    rng = np.random.default_rng(seed)
    focal_refs = list(dict.fromkeys(focal_ref_ids))
    focal_role = role_of.get(focal_refs[0]) if focal_refs else None

    mandatory: list[str] = []
    for r in focal_refs:
        for t in truth.get(r, ()):
            mandatory.append(t)
    # cohort-assigned unlinked targets share the focal role and have no owner
    owned_anywhere = {t for v in truth.values() for t in v}
    if focal_role is not None:
        for t in sorted(role_of):
            if role_of[t] == focal_role and t in tgt_country and t not in owned_anywhere:
                mandatory.append(t)
    mandatory = list(dict.fromkeys(mandatory))

    quota_by_country: dict[str, int] = {}
    focal_counts: dict[str, int] = defaultdict(int)
    for r in focal_refs:
        focal_counts[ref_country[r]] += 1
    for country, n_focal in focal_counts.items():
        quota_by_country[country] = int(round(densities.get(country, 0.0) * n_focal))

    mandatory_by_country: dict[str, list[str]] = defaultdict(list)
    for t in mandatory:
        mandatory_by_country[tgt_country[t]].append(t)

    b_pool_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in sorted(role_of):
        if role_of[t] == "B" and t in tgt_country:
            b_pool_by_stratum[(tgt_country[t], tgt_source[t])].append(t)

    traffic: list[str] = list(mandatory)
    for country in sorted(quota_by_country):
        need = quota_by_country[country] - len(mandatory_by_country.get(country, []))
        if need <= 0:
            continue
        strata = sorted(k for k in b_pool_by_stratum if k[0] == country)
        if not strata:
            continue
        total_pool = sum(len(b_pool_by_stratum[k]) for k in strata)
        # proportional allocation across (country, source) strata
        alloc = {k: need * len(b_pool_by_stratum[k]) // total_pool for k in strata}
        leftover = need - sum(alloc.values())
        for k in sorted(alloc)[:leftover]:
            alloc[k] += 1
        chosen: list[str] = []
        for k in sorted(alloc):
            pool = b_pool_by_stratum[k]
            if not pool or alloc[k] <= 0:
                continue
            take = min(alloc[k], len(pool))
            idx = rng.permutation(len(pool))[:take]
            chosen.extend(pool[i] for i in sorted(idx))
        traffic.extend(chosen)

    if not return_report:
        return traffic

    catalog_set = set(catalog) if catalog is not None else None
    b_targets = [t for t in traffic if role_of.get(t) == "B" and t in owned_anywhere]
    if b_targets and catalog_set is not None:
        owners_in_catalog = sum(
            1 for t in b_targets if any(t in truth.get(r, ()) for r in catalog_set)
        )
        owner_fraction = owners_in_catalog / len(b_targets) if b_targets else 0.0
    elif b_targets and catalog_set is None:
        owner_fraction = float("nan")
    else:
        owner_fraction = 0.0
    report = {
        "n_mandatory": len(mandatory),
        "n_total": len(traffic),
        "n_background_fill": len(traffic) - len(mandatory),
        "quota_by_country": quota_by_country,
        "owner_present_fraction": owner_fraction,
    }
    return traffic, report
