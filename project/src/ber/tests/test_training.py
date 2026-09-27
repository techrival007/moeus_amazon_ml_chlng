"""Bounded-memory LightGBM training and scoring (master.md 15.4, 10.1)."""

import numpy as np
import pytest

from ber.training import TrainConfig, load_model, train_model


def make_data(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 6)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.05 * rng.random(n) > 0.5).astype(np.int8)
    return X, y


class TestTrainModel:
    def test_learns_separable_pattern(self):
        X, y = make_data()
        cfg = TrainConfig(num_threads=4, seed=0)
        bst = train_model(X, y, cfg)
        p = bst.predict(X)
        acc = float(((p > 0.5) == y).mean())
        assert acc > 0.9

    def test_deterministic_given_seed(self):
        X, y = make_data(20_000)
        a = train_model(X, y, TrainConfig(num_threads=4, seed=7))
        b = train_model(X, y, TrainConfig(num_threads=4, seed=7))
        assert np.array_equal(a.predict(X), b.predict(X))

    def test_bounded_memory_settings_present(self):
        cfg = TrainConfig()
        assert cfg.max_bin == 127
        assert cfg.histogram_pool_mb <= 512
        assert cfg.num_leaves == 63
        assert cfg.min_data_in_leaf == 100
        assert cfg.force_col_wise is True

    def test_save_load_roundtrip_identical_predictions(self, tmp_path):
        X, y = make_data(20_000)
        bst = train_model(X, y, TrainConfig(num_threads=4, seed=3))
        path = tmp_path / "model.txt"
        bst.save_model(str(path), num_iteration=bst.current_iteration())
        loaded = load_model(path)
        assert np.array_equal(bst.predict(X), loaded.predict(X))

    def test_predict_dtype_and_range(self):
        X, y = make_data(10_000)
        bst = train_model(X, y, TrainConfig(num_threads=4, seed=1))
        p = bst.predict(X)
        assert p.dtype == np.float64
        assert ((p >= 0.0) & (p <= 1.0)).all()


def test_supervised_fit_labels_do_not_relabel_background_owned_targets_as_negative():
    from ber.stages import _fit_label

    supervised_targets = {"S2-fit", "S2-unlinked-fit"}
    fit_owners = {"S2-fit": "S1-fit"}
    assert _fit_label("S1-fit", "S2-fit", supervised_targets, fit_owners) == 1
    assert _fit_label("S1-other", "S2-fit", supervised_targets, fit_owners) == 0
    assert _fit_label("S1-other", "S2-unlinked-fit", supervised_targets, fit_owners) == 0
    assert _fit_label("S1-background", "S2-background", supervised_targets, fit_owners) is None
