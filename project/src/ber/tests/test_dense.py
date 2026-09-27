import numpy as np

from ber import dense as dn
from ber.pipeline import dense_catalog_ids, dense_traffic_ids


def test_dense_catalog_keeps_every_held_out_ref_and_fills_with_b():
    rng = np.random.default_rng(0)
    ids = [f"r{i}" for i in range(10)]
    roles = ["D", "K", "A2", "F", "B", "B", "B", "B", "B", "B"]
    cat, ghosts = dense_catalog_ids(ids, roles, quota=6, rng=rng)
    assert {"r0", "r1", "r2", "r3"} <= set(cat)
    assert len(cat) == 6 and len(ghosts) == 4
    assert not set(cat) & set(ghosts)


def test_dense_traffic_includes_all_owned_and_orphans_to_quota():
    rng = np.random.default_rng(0)
    owner = {"t1": "r1", "t2": "r1", "t3": "g1", "t4": "r2"}  # g1 is a ghost
    targets = ["t1", "t2", "t3", "t4", "u1", "u2", "u3"]
    traffic, n_owned, n_orphan = dense_traffic_ids(targets, owner, {"r1", "r2"}, quota=5, rng=rng)
    assert {"t1", "t2", "t4"} <= set(traffic)
    assert n_owned == 3 and n_orphan == 2 and len(traffic) == 5


def test_winner_mask_one_owner_per_target_and_ties_abstain():
    src = np.array([2, 2, 2, 3, 3, 2])
    tgt = np.array([1, 1, 2, 1, 1, 3])
    s = np.array([0.9, 0.5, 0.7, 0.6, 0.6, 0.8])
    w = dn.winner_mask(src, tgt, s, gamma=0.0)
    assert w.tolist() == [True, False, True, False, False, True]
    w2 = dn.winner_mask(src, tgt, s, gamma=0.5)
    assert w2.tolist() == [False, False, True, False, False, True]


def test_macro_f_matches_entity_score():
    from ber.metrics import entity_score

    # ref 0: truth {a,b}, pred {a,c}; ref 1: singleton, pred {} ; ref 2: singleton, pred {d}
    eref = np.array([0, 0, 0, 2])
    sel = np.array([True, False, True, True])
    pos = np.array([True, True, False, False])
    truth_n = np.array([2, 0, 0])
    required = np.array([0, 1, 2])
    expected = np.mean([entity_score({"a", "b"}, {"a", "c"}), 1.0, 0.0])
    assert abs(dn.macro_f(sel, pos, eref, required, truth_n) - expected) < 1e-12


def test_group_features_ranks_and_gaps():
    ref = np.array([0, 1, 0])
    src = np.array([2, 2, 2])
    tgt = np.array([5, 5, 6])
    s = np.array([0.9, 0.4, 0.7])
    G = dn.group_features(ref, src, tgt, s)
    names = dict(zip(dn.STAGE2_NAMES, G.T))
    assert names["t_n"].tolist() == [2, 2, 1]
    assert names["t_rank"].tolist() == [0, 1, 0]
    assert np.allclose(names["t_gap_other"], [0.5, -0.5, 0.7 + 1.0])
    assert names["r_rank"].tolist() == [0, 0, 1]
    assert np.allclose(names["r_gap_max"], [0.0, 0.0, -0.2])


def test_expected_f_select_picks_confident_topk_or_empty():
    # ref 0: two sure + one doubtful -> keep 2; ref 1: all doubtful -> empty
    ref = np.array([0, 0, 0, 1, 1])
    p = np.array([0.99, 0.98, 0.2, 0.1, 0.05])
    base = np.ones(5, bool)
    sel = dn.expected_f_select(ref, p, base)
    assert sel.tolist() == [True, True, False, False, False]
    # base mask removes a winner-excluded edge from consideration
    base2 = np.array([True, False, True, True, True])
    sel2 = dn.expected_f_select(ref, p, base2)
    assert sel2.tolist() == [True, False, False, False, False]
