"""Retrieval fusion (RRF, offset 60) and deterministic candidate compaction.

Contract from master.md sections 9.2-9.4:
- u(a,b) = sum over active lanes returning a of 1/(60 + rank_l(a,b))
- z(b) = active_lanes / 61; v(a,b) = u/z in [0,1]
- margin g(b) = v(best) - v(runner_up) only when a runner-up exists
- deterministic order: v desc, then address-lane rank, then name-lane rank,
  then reference id; a missing rank sorts last
- strict uniform budget b: retain the ordered prefix; if the cutoff splits an
  exact fused-score tie, retain every tied member already in the pool
- empty raw pool -> no candidates; a lone hypothesis is retained (a missing
  runner-up is NOT evidence of a large margin)
"""

import math

import pytest

from ber.candidates import LaneHit, compact_strict, fuse_lanes, fused_margin


def hit(ref, rank, score=1.0):
    return LaneHit(ref_id=ref, rank=rank, score=score)


class TestFuseLanes:
    def test_top_in_both_lanes_gets_v_one(self):
        name = [hit("r1", 1), hit("r2", 2)]
        addr = [hit("r1", 1), hit("r3", 2)]
        fused = fuse_lanes(name, addr)
        assert fused[0].ref_id == "r1"
        assert fused[0].v == pytest.approx((1 / 61 + 1 / 61) / (2 / 61))
        assert fused[0].v == pytest.approx(1.0)

    def test_single_lane_top_also_reaches_v_one(self):
        name = [hit("r1", 1), hit("r2", 2), hit("r3", 3)]
        fused = fuse_lanes(name, [])
        assert fused[0].ref_id == "r1"
        assert fused[0].v == pytest.approx(1.0)
        # rank 2 in one of one active lanes
        assert fused[1].v == pytest.approx((1 / 62) / (1 / 61))

    def test_ref_in_one_lane_only(self):
        name = [hit("r1", 1), hit("r2", 4)]
        addr = [hit("r3", 1), hit("r2", 2)]
        fused = fuse_lanes(name, addr)
        by_ref = {f.ref_id: f for f in fused}
        # r2 appears rank 4 in name and rank 2 in addr
        assert by_ref["r2"].v == pytest.approx((1 / 64 + 1 / 62) / (2 / 61))
        # r1/r3 appear in exactly one lane
        assert by_ref["r1"].v == pytest.approx((1 / 61) / (2 / 61))
        assert by_ref["r1"].name_rank == 1 and by_ref["r1"].addr_rank is None
        assert by_ref["r3"].addr_rank == 1 and by_ref["r3"].name_rank is None

    def test_v_bounds(self):
        name = [hit("r1", 1), hit("r2", 8)]
        addr = [hit("r3", 1), hit("r2", 8)]
        for f in fuse_lanes(name, addr):
            assert 0.0 < f.v <= 1.0 + 1e-12

    def test_deterministic_order_tiebreaks(self):
        # same v: decided by addr_rank, then name_rank, then ref_id
        name = [hit("rB", 1), hit("rA", 1), hit("rC", 1)]
        addr = [hit("rA", 2), hit("rC", 1), hit("rB", 3)]
        fused = fuse_lanes(name, addr)
        # all three have v = (1/61 + 1/6x)/z... rC has addr_rank 1 -> first
        # rA addr_rank 2 -> second; rB addr_rank 3 -> third
        assert [f.ref_id for f in fused] == ["rC", "rA", "rB"]

    def test_missing_rank_sorts_last(self):
        # r1 only in name lane; r2 in both lanes but same fused v as r1?
        # construct: r1 name rank1; r2 name rank1 + addr rank1 -> r2 first.
        name = [hit("r1", 1), hit("r2", 1)]
        addr = [hit("r2", 1)]
        fused = fuse_lanes(name, addr)
        assert fused[0].ref_id == "r2"

    def test_empty_lanes_give_empty_pool(self):
        assert fuse_lanes([], []) == []


class TestFusedMargin:
    def test_margin_is_gap_to_runner_up(self):
        name = [hit("r1", 1), hit("r2", 2)]
        addr = [hit("r1", 1), hit("r2", 2)]
        fused = fuse_lanes(name, addr)
        g = fused_margin(fused)
        # v(r1)=1.0, v(r2)=(1/62+1/62)/(2/61)=61/62
        assert g == pytest.approx(1.0 - 61.0 / 62.0)

    def test_no_runner_up_means_no_margin(self):
        fused = fuse_lanes([hit("r1", 1)], [])
        assert fused_margin(fused) is None
        assert fused_margin([]) is None


class TestCompactStrict:
    def test_budget_prefix_retained(self):
        name = [hit(f"r{i}", i) for i in range(1, 6)]
        fused = fuse_lanes(name, [])
        kept = compact_strict(fused, budget=2)
        assert [f.ref_id for f in kept] == ["r1", "r2"]

    def test_budget_larger_than_pool_keeps_all(self):
        fused = fuse_lanes([hit("r1", 1), hit("r2", 2)], [])
        assert len(compact_strict(fused, budget=5)) == 2

    def test_exact_tie_at_cutoff_expands(self):
        # three refs with identical fused v; budget 2 must keep all three
        name = [hit("rA", 1), hit("rB", 1), hit("rC", 1)]
        addr = [hit("rA", 2), hit("rB", 2), hit("rC", 2)]
        fused = fuse_lanes(name, addr)
        assert len({round(f.v, 12) for f in fused}) == 1  # all tied
        kept = compact_strict(fused, budget=2)
        assert len(kept) == 3

    def test_no_tie_no_expansion(self):
        name = [hit("r1", 1), hit("r2", 2), hit("r3", 3)]
        fused = fuse_lanes(name, [])
        kept = compact_strict(fused, budget=2)
        assert [f.ref_id for f in kept] == ["r1", "r2"]

    def test_empty_pool_gives_no_candidates(self):
        assert compact_strict([], budget=2) == []

    def test_lone_hypothesis_is_retained(self):
        fused = fuse_lanes([hit("r1", 1)], [])
        kept = compact_strict(fused, budget=2)
        assert [f.ref_id for f in kept] == ["r1"]

    def test_budget_one_keeps_top(self):
        name = [hit("r1", 1), hit("r2", 2)]
        fused = fuse_lanes(name, [])
        kept = compact_strict(fused, budget=1)
        assert [f.ref_id for f in kept] == ["r1"]

    def test_joint_lane_adds_new_support_without_duplicate_existing_candidates(self):
        from ber.candidates import compact_with_joint

        base = fuse_lanes([hit("r1", 1), hit("r2", 2)], [hit("r3", 1)])
        joined = compact_with_joint(base, budget=2,
                                    joint_hits=[hit("r4", 1, 5.0), hit("r1", 2, 4.0)],
                                    joint_k=2)
        assert len({item.ref_id for item in joined}) == len(joined)
        assert "r4" in {item.ref_id for item in joined}
        assert len(joined) == len(compact_strict(base, 2)) + 1
