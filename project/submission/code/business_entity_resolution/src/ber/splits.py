"""Identity-component splits, F/D/K/A/B role assignment, leakage checks.

Design (master.md 7.1-7.2):
- Components join each S1 anchor with ALL of its labeled targets. Every truth
  edge's endpoints share one component and therefore one role.
- Unlinked targets form their own groups; exact duplicate payloads
  (source, raw name, raw address, country) stay together so one business
  cannot leak across roles through a perfect duplicate.
- Roles are assigned per stratum with a seeded RNG and largest-remainder
  quotas: anchors stratified by (country, truth-size bucket), unlinked
  groups by (country, source). Deterministic given the seed.
- verify_split enforces the protection contract; sample_background_refs
  builds catalogs where focal references are always kept and background
  comes only from F/B roles.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

ROLES = ("F", "D", "K", "A", "B")
SHARES = (0.05, 0.05, 0.05, 0.05, 0.80)
_QUOTA_ROLES = ("F", "D", "K", "A")
_PROTECTED = ("D", "K", "A", "A2")
_BACKGROUND_ROLES = ("F", "B")


# ---------------------------------------------------------------- union-find


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: lexicographically smaller id wins
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def build_components(truth: Mapping[str, Sequence[str]]) -> tuple[dict[str, int], list[frozenset[str]]]:
    """Union-find over truth edges; group ids ordered by smallest member id."""
    uf = _UnionFind()
    ids: set[str] = set(truth)
    for anchor, targets in truth.items():
        ids.add(anchor)
        for t in targets:
            ids.add(t)
            uf.union(anchor, t)
    clusters: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        clusters[uf.find(i)].append(i)
    ordered = sorted(clusters.values(), key=lambda members: min(members))
    group_of: dict[str, int] = {}
    groups: list[frozenset[str]] = []
    for gid, members in enumerate(ordered):
        groups.append(frozenset(members))
        for m in members:
            group_of[m] = gid
    return group_of, groups


def group_unlinked_targets(
    payloads: Mapping[str, tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], list[str]]:
    """Group unlinked target records by exact payload (source, name, addr, country).

    Only payloads occurring more than once produce multi-id groups; the
    returned mapping contains every payload key, singletons included.
    """
    by_payload: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for tid in sorted(payloads):
        by_payload[payloads[tid]].append(tid)
    return dict(by_payload)


def _bucket(truth_size: int) -> int:
    if truth_size <= 3:
        return truth_size
    return 4


def _largest_remainder(n: int, shares: Sequence[float], rotate: int = 0) -> dict[str, int]:
    """Exact integer quotas for the four assigned roles; B receives the rest.

    Ties in the largest-remainder distribution break in role order rotated by
    `rotate` (the stratum ordinal), so global role counts stay proportional
    instead of systematically favoring early roles.
    """
    raw = {r: shares[i] * n for i, r in enumerate(_QUOTA_ROLES)}
    quotas = {r: int(v) for r, v in raw.items()}
    missing = min(
        n - sum(quotas.values()),
        round(sum(raw.values())) - sum(quotas.values()),
    )
    missing = max(0, missing)
    rotated = _QUOTA_ROLES[rotate % len(_QUOTA_ROLES):] + _QUOTA_ROLES[: rotate % len(_QUOTA_ROLES)]
    order = sorted(_QUOTA_ROLES, key=lambda r: (-(raw[r] - quotas[r]), rotated.index(r)))
    for r in order[:missing]:
        quotas[r] += 1
    quotas["B"] = n - sum(quotas.values())
    return quotas


def assign_roles(
    anchor_ids: Iterable[str] | Mapping[str, Any],
    anchor_meta: Mapping[str, tuple[str, int]],
    unlinked_groups: Mapping[Any, Sequence[str]] | None = None,
    *,
    group_of: Mapping[str, int] | None = None,
    target_meta: Mapping[str, tuple[str, str]] | None = None,
    shares: Sequence[float] = SHARES,
    seed: int = 0,
) -> dict[str, str]:
    """Assign F/D/K/A/B roles to whole components (or bare anchors).

    When group_of is given, roles are assigned per component using the
    anchor's (country, truth-size bucket) stratum and every member inherits
    the component's role. Unlinked groups (if provided) are assigned in their
    own (country, source) strata.
    """
    anchor_ids = list(anchor_ids)
    if unlinked_groups is None:
        unlinked_groups = {}

    # Build the assignment units: (stratum, sorted member ids)
    if group_of is not None:
        comp_members: dict[int, list[str]] = defaultdict(list)
        for i, g in group_of.items():
            comp_members[g].append(i)
        units: list[tuple[tuple[str, int], list[str]]] = []
        for anchor in anchor_ids:
            g = group_of[anchor]
            country, size = anchor_meta[anchor]
            members = sorted(comp_members[g])
            units.append(((country, _bucket(size)), members))
        seen_groups: set[int] = {group_of[a] for a in anchor_ids}
        for g, members in comp_members.items():
            if g not in seen_groups and members:
                # target-only component cannot exist (targets link to anchors)
                continue
    else:
        units = [((anchor_meta[a][0], _bucket(anchor_meta[a][1])), [a]) for a in anchor_ids]

    # Unlinked groups: stratum from (country, source) of the first member
    for key in sorted(unlinked_groups, key=repr):
        ids = list(unlinked_groups[key])
        if not ids:
            continue
        if target_meta is not None and ids[0] in target_meta:
            country, source = target_meta[ids[0]]
        else:
            country, source = "?", "?"
        units.append(((country, source), sorted(ids)))

    # Per-stratum seeded assignment (strata processed in sorted order)
    rng = np.random.default_rng(seed)
    by_stratum: dict[tuple, list[list[str]]] = defaultdict(list)
    for stratum, members in units:
        by_stratum[stratum].append(members)

    role_of: dict[str, str] = {}
    for stratum_no, stratum in enumerate(sorted(by_stratum, key=repr)):
        members_list = by_stratum[stratum]
        n = len(members_list)
        quotas = _largest_remainder(n, shares, rotate=stratum_no)
        perm = rng.permutation(n)
        idx = 0
        for role in _QUOTA_ROLES:
            for _ in range(quotas[role]):
                if idx >= n:
                    break
                for m in members_list[int(perm[idx])]:
                    role_of[m] = role
                idx += 1
        for j in range(idx, n):
            for m in members_list[int(perm[j])]:
                role_of[m] = "B"
    return role_of


def verify_split(
    truth: Mapping[str, Sequence[str]],
    role_of: Mapping[str, str],
    group_of: Mapping[str, int],
    fitting_pairs: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Raise ValueError if the protection contract is violated."""
    for anchor, targets in truth.items():
        ra = role_of.get(anchor)
        ga = group_of.get(anchor)
        for t in targets:
            if role_of.get(t) != ra:
                raise ValueError(
                    f"truth edge ({anchor} -> {t}) crosses roles: "
                    f"{ra!r} vs {role_of.get(t)!r}"
                )
            if group_of.get(t) != ga:
                raise ValueError(f"truth edge ({anchor} -> {t}) crosses groups")
    if fitting_pairs is not None:
        for a, b in fitting_pairs:
            for endpoint in (a, b):
                if role_of.get(endpoint) in _PROTECTED:
                    raise ValueError(
                        f"supervised fitting pair references protected "
                        f"{role_of[endpoint]}-role record {endpoint!r}"
                    )


def sample_background_refs(
    role_of: Mapping[str, str],
    country_of: Mapping[str, str],
    focal_ids: Sequence[str],
    quota: int | Mapping[str, int],
    seed: int = 0,
) -> list[str]:
    """Catalog for one cohort: focal refs (always kept) + F/B background fill.

    Never drops a focal reference; background comes only from F/B roles;
    per-country quotas cap the fill, not the focal set.
    """
    rng = np.random.default_rng(seed)
    focal = list(dict.fromkeys(focal_ids))
    focal_by_country: dict[str, list[str]] = defaultdict(list)
    for f in focal:
        focal_by_country[country_of[f]].append(f)
    background_by_country: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(role_of):
        if role_of[rid] in _BACKGROUND_ROLES and rid not in set(focal):
            background_by_country[country_of[rid]].append(rid)

    chosen: list[str] = []
    for country in sorted(set(focal_by_country) | set(background_by_country)):
        focal_c = focal_by_country.get(country, [])
        q = quota[country] if isinstance(quota, Mapping) else quota
        need = q - len(focal_c)
        chosen.extend(focal_c)
        if need > 0:
            pool = background_by_country.get(country, [])
            if pool:
                take = min(need, len(pool))
                idx = rng.permutation(len(pool))[:take]
                chosen.extend(pool[i] for i in sorted(idx))
    return chosen
