"""Identity-component splits with F/D/K/A/B roles and leakage verification.

Contract from master.md sections 7.1-7.2:
- Components: each S1 anchor plus ALL of its labeled S2/S3 targets (union-find
  over truth edges). Every truth edge's endpoints share one component and one
  role. Unlinked target records form their own groups; exact duplicate target
  payloads (same source, raw name, raw address, country) stay together.
- Roles F/D/K/A/B with shares 5/5/5/5/80 percent, stratified by country and
  truth-size bucket (anchors) or country and source (unlinked groups),
  assigned deterministically from a fixed seed, persisted as a manifest.
- verify_split fails if any truth edge crosses roles/groups or if D/K/A
  records appear in supervised fitting data.
- Catalog sampling (env construction) keeps every focal reference and never
  uses D/K/A records as background; background comes from F and B only.
"""

import numpy as np
import pytest

from ber.splits import (
    ROLES,
    SHARES,
    assign_roles,
    build_components,
    group_unlinked_targets,
    sample_background_refs,
    verify_split,
)


@pytest.fixture
def toy_truth():
    # 6 anchors; a3 shares a target with a5 (defensive: data never does this)
    return {
        "S1-a1": ["S2-x1", "S3-y1"],
        "S1-a2": ["S2-x2"],
        "S1-a3": ["S2-x3", "S3-y3"],
        "S1-a4": [],
        "S1-a5": ["S2-x3", "S3-y5"],  # x3 shared with a3
        "S1-a6": ["S2-x6", "S3-y6", "S2-x7"],
    }


class TestBuildComponents:
    def test_truth_edges_share_component(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        assert group_of["S2-x1"] == group_of["S1-a1"]
        assert group_of["S3-y3"] == group_of["S2-x3"]
        assert group_of["S1-a3"] == group_of["S1-a5"]  # shared target merges anchors

    def test_singleton_anchors_are_components(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        assert group_of["S1-a4"] in set(group_of.values())
        assert len(groups) == 5  # a1, a2, (a3+a5), a4, a6

    def test_every_id_is_grouped(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        ids = set(toy_truth) | {t for v in toy_truth.values() for t in v}
        assert set(group_of) == ids
        assert sum(len(g) for g in groups) == len(ids)


class TestGroupUnlinkedTargets:
    def test_exact_duplicate_payloads_group_together(self):
        # unlinked targets; payload = (source, name, addr, country)
        targets = {
            "S2-u1": ("S2", "Acme Corp", "1 Main St", "US"),
            "S2-u2": ("S2", "Acme Corp", "1 Main St", "US"),  # dup of u1
            "S2-u3": ("S2", "Acme Corp", "2 Main St", "US"),  # different addr
            "S3-u4": ("S3", "Acme Corp", "1 Main St", "US"),  # different source
        }
        groups = group_unlinked_targets(targets)
        by_id = {i: g for g, ids in groups.items() for i in ids}
        assert by_id["S2-u1"] == by_id["S2-u2"]
        assert by_id["S2-u1"] != by_id["S2-u3"]
        assert by_id["S2-u1"] != by_id["S3-u4"]

    def test_distinct_payloads_are_singletons(self):
        targets = {
            "S2-u1": ("S2", "A", "1", "US"),
            "S2-u2": ("S2", "B", "2", "US"),
        }
        groups = group_unlinked_targets(targets)
        assert all(len(ids) == 1 for ids in groups.values())


class TestAssignRoles:
    def _anchor_meta(self, n_us, n_in, sizes):
        meta = {}
        for i in range(n_us):
            meta[f"S1-a{i}"] = ("US", sizes[i % len(sizes)])
        for i in range(n_in):
            meta[f"S1-b{i}"] = ("India", sizes[i % len(sizes)])
        return meta

    def test_exact_shares_on_balanced_input(self):
        # 200 anchors, 4 strata of 50 (country x bucket); 5% of 50 = 2.5
        meta = {}
        for i in range(200):
            country = "US" if i % 2 == 0 else "India"
            bucket = (i // 2) % 2
            meta[f"S1-a{i}"] = (country, bucket)
        role_of = assign_roles({i: None for i in meta}, meta, {}, shares=(0.05, 0.05, 0.05, 0.05, 0.80), seed=11)
        counts = {r: sum(1 for v in role_of.values() if v == r) for r in ROLES}
        assert counts["F"] == 10 and counts["D"] == 10 and counts["K"] == 10 and counts["A"] == 10
        assert counts["B"] == 160
        # stratification: each stratum gets its own proportional assignment
        for country in ("US", "India"):
            for bucket in (0, 1):
                sub = [k for k, v in meta.items() if v == (country, bucket)]
                roles = [role_of[k] for k in sub]
                assert len(sub) == 50
                for r in ("F", "D", "K", "A"):
                    assert roles.count(r) in (2, 3)  # largest remainder of 2.5

    def test_deterministic_given_seed(self):
        meta = {f"S1-a{i}": ("US", i % 3) for i in range(60)}
        r1 = assign_roles({i: None for i in meta}, meta, {}, seed=42)
        r2 = assign_roles({i: None for i in meta}, meta, {}, seed=42)
        assert r1 == r2
        r3 = assign_roles({i: None for i in meta}, meta, {}, seed=43)
        assert r1 != r3

    def test_whole_component_gets_one_role(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        anchor_meta = {
            "S1-a1": ("US", 2), "S1-a2": ("US", 1), "S1-a3": ("US", 2),
            "S1-a4": ("US", 0), "S1-a5": ("US", 2), "S1-a6": ("India", 3),
        }
        # roles assigned per group; every member inherits the group's role
        role_of = assign_roles(
            {i: None for i in anchor_meta},
            anchor_meta,
            {},
            group_of=group_of,
            shares=(0.05, 0.05, 0.05, 0.05, 0.80),
            seed=7,
        )
        for anchor, targets in toy_truth.items():
            for t in targets:
                assert role_of[t] == role_of[anchor]
        assert role_of["S1-a3"] == role_of["S1-a5"]  # merged component


class TestVerifySplit:
    def test_fresh_audit_role_is_protected_from_fitting_pairs(self):
        truth = {"S1-a": ["S2-a"]}
        roles = {"S1-a": "A2", "S2-a": "A2"}
        groups = {"S1-a": 1, "S2-a": 1}
        with pytest.raises(ValueError, match="fitting"):
            verify_split(truth, roles, groups, fitting_pairs=[("S1-a", "S2-a")])

    def test_clean_split_passes(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        anchor_meta = {
            "S1-a1": ("US", 2), "S1-a2": ("US", 1), "S1-a3": ("US", 2),
            "S1-a4": ("US", 0), "S1-a5": ("US", 2), "S1-a6": ("India", 3),
        }
        role_of = assign_roles(
            {i: None for i in anchor_meta}, anchor_meta, {},
            group_of=group_of, shares=SHARES, seed=7,
        )
        verify_split(toy_truth, role_of, group_of)  # no exception

    def test_truth_edge_with_mismatched_role_fails(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        anchor_meta = {
            "S1-a1": ("US", 2), "S1-a2": ("US", 1), "S1-a3": ("US", 2),
            "S1-a4": ("US", 0), "S1-a5": ("US", 2), "S1-a6": ("India", 3),
        }
        role_of = assign_roles(
            {i: None for i in anchor_meta}, anchor_meta, {},
            group_of=group_of, shares=SHARES, seed=7,
        )
        # corrupt: move one target of a1 to another role
        first_target = toy_truth["S1-a1"][0]
        role_of[first_target] = "B" if role_of[first_target] != "B" else "F"
        with pytest.raises(ValueError, match="role"):
            verify_split(toy_truth, role_of, group_of)

    def test_fitting_pair_with_holdout_endpoint_fails(self, toy_truth):
        group_of, groups = build_components(toy_truth)
        anchor_meta = {
            "S1-a1": ("US", 2), "S1-a2": ("US", 1), "S1-a3": ("US", 2),
            "S1-a4": ("US", 0), "S1-a5": ("US", 2), "S1-a6": ("India", 3),
        }
        # shares chosen so every protected role exists among 6 units
        role_of = assign_roles(
            {i: None for i in anchor_meta}, anchor_meta, {},
            group_of=group_of, shares=(0.25, 0.25, 0.10, 0.10, 0.30), seed=7,
        )
        # simulate supervised fitting pairs referencing a protected-role record
        protected = [i for i, r in role_of.items() if r in ("D", "K", "A")]
        assert protected
        fitting_pairs = [(protected[0], "S2-anything")]
        with pytest.raises(ValueError, match="fitting"):
            verify_split(toy_truth, role_of, group_of, fitting_pairs=fitting_pairs)


class TestSampleBackgroundRefs:
    def test_background_excludes_protected_and_keeps_focal(self):
        # refs: 30 US refs, roles assigned
        refs = {f"S1-r{i}": ("US", 0) for i in range(30)}
        role_of = assign_roles({i: None for i in refs}, refs, {}, shares=(0.05, 0.05, 0.05, 0.05, 0.80), seed=5)
        focal = [i for i, r in role_of.items() if r == "D"]
        chosen = sample_background_refs(
            role_of, country_of={i: "US" for i in refs},
            focal_ids=focal, quota=10, seed=3,
        )
        # background fill is F/B only; focal D refs are kept unconditionally
        assert all(role_of[c] in ("F", "B") for c in chosen if c not in focal)
        assert not (set(chosen) - set(focal)) & set(focal)
        assert len(set(chosen)) == len(chosen)
        assert len(chosen) == 10

    def test_quota_smaller_than_focal_returns_focal_plus_fill(self):
        refs = {f"S1-r{i}": ("US", 0) for i in range(40)}
        role_of = assign_roles({i: None for i in refs}, refs, {}, shares=(0.05, 0.05, 0.05, 0.05, 0.80), seed=5)
        focal = [i for i, r in role_of.items() if r == "D"]
        chosen = sample_background_refs(
            role_of, country_of={i: "US" for i in refs},
            focal_ids=focal, quota=3, seed=3,
        )
        # focal always kept even when exceeding quota; rest filled with F/B
        assert set(focal) <= set(chosen)
        assert all(role_of[c] in ("F", "B") for c in chosen if c not in focal)

    def test_deterministic(self):
        refs = {f"S1-r{i}": ("US", 0) for i in range(40)}
        role_of = assign_roles({i: None for i in refs}, refs, {}, seed=5)
        focal = [i for i, r in role_of.items() if r == "D"]
        a = sample_background_refs(role_of, {i: "US" for i in refs}, focal, 20, seed=9)
        b = sample_background_refs(role_of, {i: "US" for i in refs}, focal, 20, seed=9)
        assert a == b
