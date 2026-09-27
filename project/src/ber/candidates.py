"""RRF fusion (offset 60) of retrieval lanes and strict candidate compaction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

RRF_OFFSET = 60


@dataclass(frozen=True)
class LaneHit:
    """One reference returned by one lane, with its within-lane rank."""

    ref_id: str
    rank: int  # 1-based within the lane
    score: float = 1.0


@dataclass(frozen=True)
class Fused:
    """A reference hypothesis after RRF fusion across active lanes."""

    ref_id: str
    v: float
    u: float
    name_rank: int | None
    addr_rank: int | None
    lane_scores: dict[str, float] = field(default_factory=dict)
    joint_rank: int | None = None


def fuse_lanes(name_hits: list[LaneHit], addr_hits: list[LaneHit]) -> list[Fused]:
    """Fuse the two field lanes with RRF (offset 60), sorted deterministically.

    u(a) = sum over active lanes containing a of 1/(60 + rank)
    z    = active_lanes / 61
    v    = u / z   (in (0, 1])
    Sort: v desc, addr_rank asc (missing last), name_rank asc (missing last),
    ref_id asc.
    """
    active = sum(1 for hits in (name_hits, addr_hits) if hits)
    if active == 0:
        return []
    z = active / (RRF_OFFSET + 1)
    u: dict[str, float] = {}
    name_rank: dict[str, int | None] = {}
    addr_rank: dict[str, int | None] = {}
    lane_scores: dict[str, dict[str, float]] = {}
    for lane, hits in (("name", name_hits), ("addr", addr_hits)):
        for h in hits:
            u[h.ref_id] = u.get(h.ref_id, 0.0) + 1.0 / (RRF_OFFSET + h.rank)
            lane_scores.setdefault(h.ref_id, {})[lane] = h.score
    for h in name_hits:
        name_rank[h.ref_id] = h.rank
    for h in addr_hits:
        addr_rank[h.ref_id] = h.rank
    all_refs = set(u) | set(name_rank) | set(addr_rank)
    big = 1 << 30
    fused = [
        Fused(
            ref_id=r,
            v=u.get(r, 0.0) / z,
            u=u.get(r, 0.0),
            name_rank=name_rank.get(r),
            addr_rank=addr_rank.get(r),
            lane_scores=lane_scores.get(r, {}),
        )
        for r in all_refs
    ]
    fused.sort(key=lambda f: (-f.v, f.addr_rank if f.addr_rank is not None else big,
                              f.name_rank if f.name_rank is not None else big, f.ref_id))
    return fused


def fused_margin(fused: list[Fused]) -> float | None:
    """v(best) - v(runner-up); None when there is no runner-up."""
    if len(fused) < 2:
        return None
    return fused[0].v - fused[1].v


def compact_strict(fused: list[Fused], budget: int) -> list[Fused]:
    """Strict uniform reverse budget over the deterministic fused order.

    Exact fused-score ties at the cutoff boundary are retained in full
    (they were already retrieved; silently dropping them would pretend a
    tie was resolved).
    """
    if not fused or budget <= 0:
        return []
    kept = list(fused[:budget])
    i = budget
    while i < len(fused) and fused[i].v == kept[-1].v:
        kept.append(fused[i])
        i += 1
    return kept


def compact_with_joint(fused: list[Fused], budget: int,
                       joint_hits: list[LaneHit], joint_k: int) -> list[Fused]:
    """Union the base reverse budget with a bounded joint-field rescue lane."""
    kept = compact_strict(fused, budget)
    positions = {item.ref_id: i for i, item in enumerate(kept)}
    pool = {item.ref_id: item for item in fused}
    for hit in joint_hits[:joint_k]:
        original = pool.get(hit.ref_id, Fused(hit.ref_id, 0.0, 0.0, None, None))
        joint = replace(original, joint_rank=hit.rank,
                        lane_scores={**original.lane_scores, "joint": hit.score})
        if hit.ref_id in positions:
            kept[positions[hit.ref_id]] = joint
        else:
            positions[hit.ref_id] = len(kept)
            kept.append(joint)
    return kept
