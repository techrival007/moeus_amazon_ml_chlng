"""Decision policies: exact global-threshold sweep and owner-competition challenger.

Contract from master.md sections 11.1-11.4:
- Threshold selection sweeps the exact score breakpoints: a prediction at tau
  includes every edge with score >= tau; equal-score edges enter together
  (never partially). The sweep includes the empty prediction. Selection is
  exact, not a grid approximation.
- apply_threshold returns per-reference predicted target sets.
- The owner challenger groups every target's competitors, requires a unique
  top score, applies gamma to the calibrated margin (exact top ties abstain
  even at gamma zero), treats a lone candidate as margin-missing and
  tau-eligible, then applies tau to the surviving winner edges. No reference
  capacity limit exists.
"""

import random

import pytest

from ber.decisions import (
    apply_owner_policy,
    apply_threshold,
    select_owner_policy,
    select_threshold,
)


class TestSelectThreshold:
    def test_official_example_boundary(self):
        # truth r1 -> {b1, b2}; candidate scores b1=0.9, b2=0.8, x3=0.7
        # best tau in (0.7, 0.8]: predicts exactly {b1, b2} -> r1 = 1.0
        edges = [("r1", "b1", 0.9), ("r1", "b2", 0.8), ("r1", "x3", 0.7)]
        truth = {"r1": {"b1", "b2"}, "r2": set()}
        res = select_threshold(edges, truth, ["r1", "r2"])
        assert res.macro == pytest.approx((1.0 + 1.0) / 2)
        assert res.tau > 0.7 and res.tau <= 0.8
        # including the whole group at tau <= 0.7 reproduces the official 5/7
        from ber.metrics import macro_score

        p = apply_threshold(edges, 0.7)
        assert p["r1"] == {"b1", "b2", "x3"}
        assert macro_score(truth, p, ["r1", "r2"]) == pytest.approx((5 / 7 + 1.0) / 2)

    def test_empty_prediction_baseline_included(self):
        edges = [("r1", "x1", 0.9)]  # r1 truth {a1}; r2 truth {} (singleton)
        truth = {"r1": {"a1"}, "r2": set()}
        res = select_threshold(edges, truth, ["r1", "r2"])
        # best: predict nothing -> (0.0 + 1.0)/2 = 0.5
        assert res.macro == pytest.approx(0.5)

    def test_inclusive_threshold_semantics(self):
        edges = [("r1", "a1", 0.5)]
        truth = {"r1": {"a1"}}
        res = select_threshold(edges, truth, ["r1"])
        assert res.tau <= 0.5
        assert res.macro == pytest.approx(1.0)

    def test_equal_scores_enter_atomically(self):
        # r1 truth {a}; candidates a and x with EQUAL scores: no breakpoint
        # can include a alone (which would score 1.0); the tie group enters
        # together (5/9) or not at all (0.0). Best is the atomic 5/9.
        edges = [("r1", "a1", 0.8), ("r1", "x1", 0.8)]
        truth = {"r1": {"a1"}}
        res = select_threshold(edges, truth, ["r1"])
        assert res.macro == pytest.approx(5 / 9)
        p = apply_threshold(edges, res.tau)
        assert p["r1"] == {"a1", "x1"}  # never a partial tie

    def test_sweep_matches_brute_force(self):
        rng = random.Random(0)
        for trial in range(30):
            refs = [f"r{i}" for i in range(6)]
            truth = {r: set() for r in refs}
            edges = []
            pool = [f"t{j}" for j in range(10)]
            for r in refs:
                for t in rng.sample(pool, k=rng.randint(0, 4)):
                    if rng.random() < 0.4:
                        truth[r].add(t)
                    score = round(rng.choice([0.1, 0.2, 0.3, 0.5, 0.5, 0.9]), 2)
                    edges.append((r, t, score))
            res = select_threshold(edges, truth, refs)
            # brute force over midpoints of the sorted unique scores + extremes
            scores = sorted({e[2] for e in edges} | {0.0, 1.0})
            mids = [0.0] + [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)] + [1.1]
            from ber.metrics import macro_score

            def preds_at(tau):
                p = {}
                for r, t, s in edges:
                    if s >= tau:
                        p.setdefault(r, set()).add(t)
                return p

            best = max(macro_score(truth, preds_at(m), refs) for m in mids)
            assert res.macro == pytest.approx(best, abs=1e-12), trial

    def test_multiple_refs_and_singleton_mix(self):
        edges = [
            ("r1", "a", 0.9), ("r1", "b", 0.6),
            ("r2", "x", 0.95),
            ("r3", "c", 0.4),
        ]
        truth = {"r1": {"a", "b"}, "r2": set(), "r3": {"c"}}
        res = select_threshold(edges, truth, ["r1", "r2", "r3"])
        # best: tau in (0.6, 0.9] keeps a for r1 (5/6), drops x (r2 -> 1.0),
        # drops c (r3 -> 0.0): macro = (5/6 + 1 + 0)/3 = 0.611
        # vs tau in (0.4,0.6]: r1 {a,b}=1, r2 0, r3 {c}=1 -> (1+0+1)/3=0.667
        assert res.macro == pytest.approx((1.0 + 0.0 + 1.0) / 3)


class TestApplyThreshold:
    def test_inclusive_and_grouping(self):
        edges = [("r1", "a", 0.7), ("r1", "b", 0.7), ("r2", "c", 0.5)]
        p = apply_threshold(edges, 0.7)
        assert p == {"r1": {"a", "b"}, "r2": set()}

    def test_tau_above_all_gives_all_empty(self):
        edges = [("r1", "a", 0.1)]
        assert apply_threshold(edges, 0.5) == {"r1": set()}


class TestOwnerPolicy:
    def test_margin_gate_blocks_ambiguous_winner(self):
        edges = [("r1", "t1", 0.9), ("r2", "t1", 0.85)]
        truth = {"r1": {"t1"}, "r2": set()}
        res_hi = select_owner_policy(edges, truth, ["r1", "r2"], gammas=[0.1])
        # margin 0.05 < 0.1 -> t1 unassigned -> r1 misses (0), r2 empty (1.0)
        assert res_hi.macro == pytest.approx(0.5)
        assert apply_owner_policy(edges, res_hi.gamma, res_hi.tau) == {"r1": set(), "r2": set()}

        res_lo = select_owner_policy(edges, truth, ["r1", "r2"], gammas=[0.02])
        assert res_lo.macro == pytest.approx(1.0)
        assert apply_owner_policy(edges, res_lo.gamma, res_lo.tau) == {"r1": {"t1"}, "r2": set()}

    def test_exact_top_tie_abstains_even_at_gamma_zero(self):
        edges = [("r1", "t1", 0.9), ("r2", "t1", 0.9)]
        truth = {"r1": {"t1"}, "r2": set()}
        res = select_owner_policy(edges, truth, ["r1", "r2"], gammas=[0.0])
        # tie -> t1 unassigned -> (0 + 1)/2
        assert res.macro == pytest.approx(0.5)

    def test_lone_candidate_is_tau_eligible(self):
        edges = [("r1", "t1", 0.6)]
        truth = {"r1": {"t1"}}
        res = select_owner_policy(edges, truth, ["r1"], gammas=[0.05])
        assert res.macro == pytest.approx(1.0)
        assert apply_owner_policy(edges, res.gamma, res.tau) == {"r1": {"t1"}}

    def test_tau_still_applies_to_winners(self):
        # winner score below a FIXED tau -> unassigned
        edges = [("r1", "t1", 0.3), ("r2", "t1", 0.1)]
        truth = {"r1": {"t1"}, "r2": set()}
        p = apply_owner_policy(edges, 0.0, 0.5)
        assert p == {"r1": set(), "r2": set()}
        from ber.metrics import macro_score

        assert macro_score(truth, p, ["r1", "r2"]) == pytest.approx(0.5)

    def test_owner_exclusivity_removes_cross_reference_false_positive(self):
        # t1 truly belongs to r1; r2 also has t1 as candidate with lower score
        # plus t2 belongs to r2 with r1 as competitor
        edges = [
            ("r1", "t1", 0.95), ("r2", "t1", 0.60),
            ("r2", "t2", 0.90), ("r1", "t2", 0.55),
        ]
        truth = {"r1": {"t1"}, "r2": {"t2"}}
        res = select_owner_policy(edges, truth, ["r1", "r2"], gammas=[0.05])
        assert res.macro == pytest.approx(1.0)
        p = apply_owner_policy(edges, res.gamma, res.tau)
        assert p == {"r1": {"t1"}, "r2": {"t2"}}

    def test_gamma_sweep_reports_best(self):
        edges = [("r1", "t1", 0.9), ("r2", "t1", 0.88)]
        truth = {"r1": {"t1"}, "r2": set()}
        res = select_owner_policy(edges, truth, ["r1", "r2"], gammas=[0.05, 0.02, 0.1])
        assert res.gamma == 0.02  # only gamma below the 0.02 margin works
        assert res.macro == pytest.approx(1.0)

    def test_matches_brute_force_on_random_cases(self):
        rng = random.Random(7)
        for _ in range(20):
            refs = [f"r{i}" for i in range(5)]
            tgts = [f"t{j}" for j in range(6)]
            truth = {r: set(rng.sample(tgts, k=rng.randint(0, 2))) for r in refs}
            edges = []
            for t in tgts:
                owners = rng.sample(refs, k=rng.randint(1, 3))
                for r in owners:
                    score = round(rng.uniform(0.05, 0.99), 2)
                    edges.append((r, t, score))
            # de-duplicate truth targets never appearing as edges (missed truths)
            res = select_owner_policy(edges, truth, refs, gammas=[0.0, 0.03])
            # brute force: every (gamma, tau) over winner edges
            from ber.metrics import macro_score

            best = 0.0
            scores = sorted({e[2] for e in edges}) + [1.1]
            for gamma in (0.0, 0.03):
                for tau in scores:
                    p = apply_owner_policy(edges, gamma, tau)
                    best = max(best, macro_score(truth, p, refs))
            assert res.macro == pytest.approx(best, abs=1e-9)
