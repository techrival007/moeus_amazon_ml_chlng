"""Exact macro-F0.5 scorer, support oracle, and paired bootstrap LCB.

Contract from master.md sections 5.2-5.3, 6.3, 21.3:
- entity_score: 1.0 only when truth and prediction are both empty; 0.0 when
  exactly one is empty; otherwise 5*tp/(t + 4*s) with t=len(truth),
  s=len(pred), tp=len(truth & pred).
- macro_score: mean of entity scores over ALL required references. Truth must
  cover every required reference. A prediction row that is absent is an empty
  prediction, not an error. Extra prediction rows not in required are ignored
  by the scorer but are contract violations caught by audit.py.
- oracle: best achievable score restricted to candidate support; equals 1.0
  for empty truth, else 5*r/(t + 4*r) with r=len(truth & candidates).
- paired bootstrap over reference components for policy comparison with a
  one-sided 95% lower confidence bound.
"""

import numpy as np
import pytest

from ber.metrics import (
    entity_score,
    macro_oracle,
    macro_score,
    paired_bootstrap_lcb,
    score_report,
)


class TestEntityScore:
    def test_official_worked_example_is_five_sevenths(self):
        # organizer example: P=2/3, R=1 -> F0.5 = 5/7 = 0.714285714...
        assert entity_score({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}) == pytest.approx(5 / 7, abs=1e-12)

    def test_singleton_conventions(self):
        assert entity_score(set(), set()) == 1.0
        assert entity_score(set(), {"x"}) == 0.0
        assert entity_score({"a"}, set()) == 0.0

    def test_small_penalty_table(self):
        assert entity_score({"a"}, {"a"}) == 1.0
        assert entity_score({"a"}, {"a", "x"}) == pytest.approx(5 / 9)
        assert entity_score({"a"}, {"x"}) == 0.0
        assert entity_score({"a", "b"}, {"a"}) == pytest.approx(5 / 6)
        assert entity_score({"a", "b"}, {"a", "b"}) == 1.0
        assert entity_score({"a", "b"}, {"a", "b", "x"}) == pytest.approx(5 / 7)
        assert entity_score({"a", "b"}, {"a", "x"}) == pytest.approx(1 / 2)
        assert entity_score({"a", "b", "c"}, {"a", "b"}) == pytest.approx(10 / 11)
        assert entity_score({"a", "b", "c"}, {"a", "b", "x"}) == pytest.approx(2 / 3)

    def test_unretrieved_truth_still_counts_in_denominator(self):
        # truth has z outside support; predicting the retrieved a is 5/6
        assert entity_score({"a", "z"}, {"a"}) == pytest.approx(5 / 6)

    def test_repeated_prediction_ids_are_not_double_counted(self):
        # duplicates are an audit failure; scorer treats inputs as sets
        assert entity_score({"a", "b"}, {"a", "a", "b"}) == 1.0


class TestMacroScore:
    def test_mean_over_all_required_rows(self):
        truth = {"r1": set(), "r2": {"a"}, "r3": {"a", "b"}}
        pred = {"r1": set(), "r2": set(), "r3": {"a", "b"}}
        # (1.0 + 0.0 + 1.0) / 3
        assert macro_score(truth, pred, ["r1", "r2", "r3"]) == pytest.approx(2 / 3)

    def test_missing_prediction_row_is_empty_prediction(self):
        truth = {"r1": set(), "r2": {"a"}}
        pred = {"r1": set()}  # r2 missing -> empty -> 0.0
        assert macro_score(truth, pred, ["r1", "r2"]) == pytest.approx(0.5)

    def test_all_empty_prediction_scores_singleton_fraction(self):
        truth = {"r1": set(), "r2": {"a"}, "r3": {"a", "b"}}
        assert macro_score(truth, {}, ["r1", "r2", "r3"]) == pytest.approx(1 / 3)

    def test_truth_must_cover_required(self):
        with pytest.raises(KeyError):
            macro_score({"r1": set()}, {}, ["r1", "r2"])

    def test_input_order_does_not_matter(self):
        truth = {"r1": {"a"}, "r2": {"a", "b"}}
        pred = {"r1": {"a", "x"}, "r2": {"a"}}
        a = macro_score(truth, pred, ["r1", "r2"])
        b = macro_score(truth, pred, ["r2", "r1"])
        assert a == b


class TestOracle:
    def test_oracle_conventions(self):
        assert entity_score.__name__  # sanity: import worked
        from ber.metrics import oracle_score

        assert oracle_score(set(), {"a", "b"}) == 1.0
        assert oracle_score({"a"}, set()) == 0.0
        assert oracle_score({"a"}, {"a"}) == 1.0
        assert oracle_score({"a", "z"}, {"a"}) == pytest.approx(5 / 6)

    def test_oracle_is_best_restricted_score_over_all_actions(self):
        # exhaustive check on small universes: oracle == max over subsets
        from itertools import combinations
        from ber.metrics import oracle_score

        universe = ["a", "b", "c", "z"]
        for t in range(len(universe) + 1):
            for truth in combinations(universe, t):
                truth = frozenset(truth)
                for c in range(len(universe) + 1):
                    for cand in combinations(universe, c):
                        cand = frozenset(cand)
                        best = max(
                            entity_score(truth, frozenset(pred))
                            for k in range(len(cand) + 1)
                            for pred in combinations(cand, k)
                        )
                        assert oracle_score(truth, cand) == pytest.approx(best, abs=1e-12)

    def test_macro_oracle(self):
        truth = {"r1": set(), "r2": {"a", "z"}, "r3": {"a", "b"}}
        cand = {"r1": set(), "r2": {"a"}, "r3": {"a", "b"}}
        # r1: 1.0 (singleton), r2: 5/6, r3: 1.0
        assert macro_oracle(truth, cand, ["r1", "r2", "r3"]) == pytest.approx((1 + 5 / 6 + 1) / 3)


class TestScoreReport:
    def test_decomposition_is_consistent(self):
        truth = {"r1": set(), "r2": {"a", "z"}, "r3": {"a", "b"}}
        cand = {"r1": set(), "r2": {"a"}, "r3": {"a", "b"}}
        pred = {"r1": set(), "r2": {"a"}, "r3": {"a"}}
        rep = score_report(truth, pred, cand, ["r1", "r2", "r3"])
        assert rep["n_required"] == 3
        assert rep["macro_f"] == pytest.approx((1.0 + 5 / 6 + 5 / 6) / 3)
        assert rep["oracle"] == pytest.approx((1.0 + 5 / 6 + 1.0) / 3)
        assert rep["blocking_loss"] == pytest.approx(1.0 - rep["oracle"])
        assert rep["scoring_loss"] == pytest.approx(rep["oracle"] - rep["macro_f"])
        assert rep["blocking_loss"] >= 0.0 and rep["scoring_loss"] >= -1e-12
        # pair-level diagnostics
        assert rep["pair_tp"] == 2  # (r2,a) and (r3,a)
        assert rep["pair_fp"] == 0
        assert rep["pair_fn"] == 2  # (r2,z) unretrieved + (r3,b) missed

    def test_report_missing_prediction_rows(self):
        truth = {"r1": set(), "r2": {"a"}}
        rep = score_report(truth, {}, {"r1": set(), "r2": set()}, ["r1", "r2"])
        assert rep["macro_f"] == pytest.approx(0.5)
        assert rep["n_pred_empty"] == 2


class TestPairedBootstrapLcb:
    def test_identical_scores_have_zero_lcb(self):
        rng_scores = np.full(2000, 0.5)
        lcb = paired_bootstrap_lcb(rng_scores, rng_scores, replicates=500, seed=1)
        assert lcb == pytest.approx(0.0, abs=1e-9)

    def test_constant_gain_has_positive_lcb(self):
        a = np.full(500, 0.9)
        b = np.full(500, 0.8)
        lcb = paired_bootstrap_lcb(a, b, replicates=1000, seed=7)
        assert lcb > 0.09

    def test_lcb_below_mean_delta(self):
        rng = np.random.default_rng(3)
        a = rng.random(1000)
        b = rng.random(1000)
        lcb = paired_bootstrap_lcb(a, b, replicates=2000, seed=3)
        assert lcb < (a - b).mean()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_bootstrap_lcb(np.ones(5), np.ones(6))
