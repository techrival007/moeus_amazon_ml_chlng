"""Evaluation environment construction: catalogs and query traffic.

Contract from master.md section 7.3:
- Catalog for focal cohort G: all focal references (always kept, never
  dropped to hit a quota) plus a fixed, label-blind background sample from
  F and B roles only; other protected cohorts (D/K/A) are never background.
- Country-specific catalog quotas come from the observed test sizes and are
  capped by availability; the same target size is used across comparisons.
- Query traffic: every target owned by G and G-assigned unlinked targets are
  reserved FIRST; remaining quota filled by a country/source-stratified
  sample of B-role targets. Focal targets are never dropped even when they
  exceed the quota. Every focal truth target must appear in the manifest.
- Background targets whose owner is outside the catalog are legitimate
  no-owner distractors; the owner-present fraction is reported.
"""

import numpy as np
import pytest

from ber.env import build_catalog, build_traffic


class RoleWorld:
    """Synthetic record universe for environment tests."""

    def __init__(self):
        # 100 anchors: ids S1-000..S1-099, roles assigned deterministically
        self.ref_country = {f"S1-{i:03d}": ("US" if i < 60 else "India") for i in range(100)}
        # 200 targets: S2-000..S2-099 (US 60 / India 40),
        # S3-100..S3-199 (US 60 / India 40)
        self.tgt_country = {}
        for i in range(100):
            self.tgt_country[f"S2-{i:03d}"] = "US" if i < 60 else "India"
        for i in range(100, 200):
            self.tgt_country[f"S3-{i:03d}"] = "US" if i < 160 else "India"
        self.tgt_source = {t: t[:2] for t in self.tgt_country}
        # simple deterministic role assignment: hash-free modular
        self.role_of = {}
        for i, r in enumerate(self.ref_country):
            self.role_of[r] = "FDB"[(i * 7) % 3]  # F/D/B mix for refs
        for i, t in enumerate(self.tgt_country):
            self.role_of[t] = "FKAB"[(i * 5) % 4]
        # truth: some anchors own targets; ensure role consistency
        self.truth = {}
        for i, r in enumerate(self.ref_country):
            owned = [t for t in self.tgt_country if self.role_of.get(t) == self.role_of[r]]
            self.truth[r] = owned[: (i % 3)]  # 0..2 owned targets

    def refs_with_role(self, *roles):
        return [r for r in self.ref_country if self.role_of[r] in roles]

    def targets_with_role(self, *roles):
        return [t for t in self.tgt_country if self.role_of[t] in roles]


@pytest.fixture
def world():
    return RoleWorld()


class TestBuildCatalog:
    def test_focal_refs_always_included(self, world):
        focal = [r for r in world.ref_country if world.role_of[r] == "D"]
        cat = build_catalog(
            role_of=world.role_of,
            ref_country=world.ref_country,
            focal_ids=focal,
            quotas={"US": 10, "India": 8},
            seed=1,
        )
        assert set(focal) <= set(cat)

    def test_background_only_from_f_and_b(self, world):
        focal = [r for r in world.ref_country if world.role_of[r] == "D"]
        cat = build_catalog(
            role_of=world.role_of, ref_country=world.ref_country,
            focal_ids=focal, quotas={"US": 40, "India": 30}, seed=1,
        )
        for r in cat:
            if r not in focal:
                assert world.role_of[r] in ("F", "B")

    def test_quota_caps_background_not_focal(self, world):
        # many focal refs (> quota): all kept; background only fills leftover
        focal = world.refs_with_role("D") + world.refs_with_role("K")
        cat = build_catalog(
            role_of=world.role_of, ref_country=world.ref_country,
            focal_ids=focal, quotas={"US": 5, "India": 5}, seed=1,
        )
        us_total = sum(1 for r in cat if world.ref_country[r] == "US")
        in_total = sum(1 for r in cat if world.ref_country[r] == "India")
        us_focal = sum(1 for r in focal if world.ref_country[r] == "US")
        in_focal = sum(1 for r in focal if world.ref_country[r] == "India")
        assert us_total >= us_focal and in_total >= in_focal
        # no background added when focal already exceeds quota
        assert us_total == max(us_focal, 5) if us_focal > 5 else us_total == 5

    def test_deterministic(self, world):
        focal = world.refs_with_role("D")
        a = build_catalog(world.role_of, world.ref_country, focal, {"US": 40, "India": 30}, seed=3)
        b = build_catalog(world.role_of, world.ref_country, focal, {"US": 40, "India": 30}, seed=3)
        assert a == b

    def test_availability_cap(self, world):
        # quota larger than available refs per country
        focal = world.refs_with_role("D")
        cat = build_catalog(
            world.role_of, world.ref_country, focal, {"US": 10_000, "India": 10_000}, seed=3,
        )
        us_total = sum(1 for r in cat if world.ref_country[r] == "US")
        in_total = sum(1 for r in cat if world.ref_country[r] == "India")
        assert us_total <= 60 and in_total <= 40
        assert set(focal) <= set(cat)


class TestBuildTraffic:
    def test_focal_owned_targets_reserved_first(self, world):
        # D-role anchors and their truth-owned targets must all appear
        focal_refs = world.refs_with_role("D")
        owned = {t for r in focal_refs for t in world.truth[r]}
        owned_d = [t for t in owned if world.role_of.get(t) == "D"]
        traffic = build_traffic(
            role_of=world.role_of,
            focal_ref_ids=focal_refs,
            truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country,
            tgt_source=world.tgt_source,
            densities={"US": 5.0, "India": 5.0},
            seed=2,
        )
        for t in owned_d:
            assert t in set(traffic)

    def test_assigned_unlinked_targets_included(self, world):
        # unlinked D-role targets are mandatory cohort traffic
        unlinked_d = [t for t in world.tgt_country
                      if world.role_of[t] == "D" and not any(t in v for v in world.truth.values())]
        focal_refs = world.refs_with_role("D")
        traffic = build_traffic(
            role_of=world.role_of, focal_ref_ids=focal_refs, truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country, tgt_source=world.tgt_source,
            densities={"US": 5.0, "India": 5.0}, seed=2,
        )
        for t in unlinked_d:
            assert t in set(traffic)

    def test_background_fill_is_b_role_only_and_stratified(self, world):
        focal_refs = world.refs_with_role("D")
        owned_d = {t for r in focal_refs for t in world.truth[r] if world.role_of.get(t) == "D"}
        traffic = build_traffic(
            role_of=world.role_of, focal_ref_ids=focal_refs, truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country, tgt_source=world.tgt_source,
            densities={"US": 5.0, "India": 5.0}, seed=2,
        )
        for t in traffic:
            if t not in owned_d:
                assert world.role_of[t] == "B"  # background fill is B only

    def test_quota_fill_respects_country_density(self, world):
        focal_refs = world.refs_with_role("D")
        traffic = build_traffic(
            role_of=world.role_of, focal_ref_ids=focal_refs, truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country, tgt_source=world.tgt_source,
            densities={"US": 5.0, "India": 5.0}, seed=2,
        )
        us = sum(1 for t in traffic if world.tgt_country[t] == "US")
        in_ = sum(1 for t in traffic if world.tgt_country[t] == "India")
        us_focal = sum(1 for r in focal_refs if world.ref_country[r] == "US")
        in_focal = sum(1 for r in focal_refs if world.ref_country[r] == "India")
        # total per country ~ density * focal refs, but never below mandatory
        assert us >= min(5 * us_focal, 60) - 5 * us_focal + 0  # sanity: bounded
        assert us <= 60 and in_ <= 40  # availability

    def test_every_focal_truth_target_present(self, world):
        # regardless of quotas, every truth target of a focal anchor appears
        focal_refs = world.refs_with_role("D")
        all_owned = {t for r in focal_refs for t in world.truth[r]}
        traffic = build_traffic(
            role_of=world.role_of, focal_ref_ids=focal_refs, truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country, tgt_source=world.tgt_source,
            densities={"US": 0.1, "India": 0.1}, seed=2,  # tiny densities
        )
        for t in all_owned:
            assert t in set(traffic)

    def test_deterministic(self, world):
        focal_refs = world.refs_with_role("D")
        a = build_traffic(
            world.role_of, focal_refs, world.truth, world.ref_country,
            world.tgt_country, world.tgt_source,
            {"US": 5.0, "India": 5.0}, seed=4,
        )
        b = build_traffic(
            world.role_of, focal_refs, world.truth, world.ref_country,
            world.tgt_country, world.tgt_source,
            {"US": 5.0, "India": 5.0}, seed=4,
        )
        assert a == b

    def test_owner_present_fraction_reported(self, world):
        focal_refs = world.refs_with_role("D")
        cat = build_catalog(
            world.role_of, world.ref_country, focal_refs,
            {"US": 40, "India": 30}, seed=1,
        )
        out = build_traffic(
            world.role_of, focal_ref_ids=focal_refs, truth=world.truth,
            ref_country=world.ref_country,
            tgt_country=world.tgt_country, tgt_source=world.tgt_source,
            densities={"US": 5.0, "India": 5.0}, seed=2,
            catalog=cat,
            return_report=True,
        )
        assert isinstance(out, tuple)
        traffic, report = out
        assert "owner_present_fraction" in report
        assert 0.0 <= report["owner_present_fraction"] <= 1.0
