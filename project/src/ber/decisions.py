"""Decision policies: exact threshold sweep and owner-competition challenger.

select_threshold performs the exact breakpoint sweep: edges enter in
descending score order, equal-score groups enter atomically, and the running
macro-F is maintained incrementally per reference row. The empty prediction
(starting state) is a candidate breakpoint.

The owner challenger (master.md 11.2-11.3) assigns each target to at most one
reference: the unique top-scored candidate, gated by a calibrated margin
gamma over the runner-up, then by tau. Exact top-score ties abstain even at
gamma zero; a lone candidate has a missing margin and is tau-eligible only.
No reference capacity limit exists: a reference may win any number of targets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from ber.metrics import entity_score


@dataclass(frozen=True)
class PolicyResult:
    tau: float
    macro: float
    gamma: float | None = None  # None for the plain threshold policy


def _row_score(t: int, s: int, tp: int) -> float:
    """entity_score from running counts (t = truth size, s = selected size)."""
    if t == 0 and s == 0:
        return 1.0
    if t == 0 or s == 0:
        return 0.0
    return 5.0 * tp / (t + 4.0 * s)


def select_threshold(
    edges: Sequence[tuple[str, str, float]],
    truth: Mapping[str, Collection[str]],
    required: Sequence[str],
) -> PolicyResult:
    """Exact sweep over score breakpoints; returns the best (tau, macro)."""
    if not required:
        raise ValueError("required references must be non-empty")
    for rid in required:
        if rid not in truth:
            raise KeyError(f"truth is missing required reference {rid!r}")

    truth_counts = {rid: len(set(truth[rid])) for rid in required}
    true_sets = {rid: frozenset(truth[rid]) for rid in required}
    # start from all-empty predictions
    pred_counts: dict[str, int] = {rid: 0 for rid in required}
    tp_counts: dict[str, int] = {rid: 0 for rid in required}
    current = {rid: _row_score(truth_counts[rid], 0, 0) for rid in required}
    total = sum(current.values())
    n = len(required)

    best_tau = float("inf")  # empty prediction
    best_macro = total / n

    # edges grouped by equal score, descending; skip edges whose ref is not
    # required (they never affect the reported rows)
    req_set = set(required)
    filtered = [(r, t, s) for (r, t, s) in edges if r in req_set]
    filtered.sort(key=lambda e: -e[2])

    i = 0
    m = len(filtered)
    while i < m:
        j = i
        score = filtered[i][2]
        while j < m and filtered[j][2] == score:
            j += 1
        # enter group [i:j) atomically
        for r, t, _ in filtered[i:j]:
            old = current[r]
            pred_counts[r] += 1
            if t in true_sets[r]:
                tp_counts[r] += 1
            new = _row_score(truth_counts[r], pred_counts[r], tp_counts[r])
            current[r] = new
            total += new - old
        macro = total / n
        if macro > best_macro or (macro == best_macro and score < best_tau):
            # prefer the smallest tau achieving the best macro
            best_macro = macro
            best_tau = score
        i = j

    return PolicyResult(tau=best_tau, macro=best_macro)


def apply_threshold(
    edges: Sequence[tuple[str, str, float]],
    tau: float,
) -> dict[str, set[str]]:
    """Predicted sets at tau (inclusive); references with no qualifying edge
    still appear with an empty set."""
    preds: dict[str, set[str]] = {}
    for r, t, s in edges:
        preds.setdefault(r, set())
        if s >= tau:
            preds[r].add(t)
    return preds


def _winner_edges(
    edges: Sequence[tuple[str, str, float]],
    gamma: float,
) -> list[tuple[str, str, float]]:
    """Per-target unique top candidate passing the margin gate.

    Exact top-score ties abstain; a lone candidate has a missing margin and
    survives the gate (tau is its only barrier).
    """
    by_target: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for r, t, s in edges:
        by_target[t].append((r, t, s))
    winners: list[tuple[str, str, float]] = []
    for t in sorted(by_target):
        cands = sorted(by_target[t], key=lambda e: (-e[2], e[0]))
        top = cands[0]
        if len(cands) == 1:
            winners.append(top)
            continue
        if cands[1][2] == top[2]:
            continue  # exact tie: abstain even at gamma zero
        if (top[2] - cands[1][2]) >= gamma:
            winners.append(top)
    return winners


def select_owner_policy(
    edges: Sequence[tuple[str, str, float]],
    truth: Mapping[str, Collection[str]],
    required: Sequence[str],
    gammas: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
) -> PolicyResult:
    """Sweep (gamma, tau) over the winner-edge policies; best macro wins."""
    best: PolicyResult | None = None
    for gamma in gammas:
        winners = _winner_edges(edges, gamma)
        res = select_threshold(winners, truth, required)
        cand = PolicyResult(tau=res.tau, macro=res.macro, gamma=gamma)
        if best is None or cand.macro > best.macro + 1e-15:
            best = cand
    if best is None:
        raise ValueError("gammas must be non-empty")
    return best


def apply_owner_policy(
    edges: Sequence[tuple[str, str, float]],
    gamma: float,
    tau: float,
) -> dict[str, set[str]]:
    """Apply the frozen owner policy; every reference in `edges` appears."""
    preds: dict[str, set[str]] = {r: set() for r, _, _ in edges}
    for r, t, s in _winner_edges(edges, gamma):
        if s >= tau:
            preds.setdefault(r, set()).add(t)
    return preds
