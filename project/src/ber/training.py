"""Bounded-memory LightGBM training and scoring.

Configuration from master.md 15.4 / 10.1: binary log loss, 63 leaves,
min 100 examples per leaf, max_bin 127, bounded histogram pool,
force_col_wise to avoid extra row-wise Dataset memory, fixed seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    objective: str = "binary"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    lambda_l2: float = 1.0
    max_bin: int = 127
    histogram_pool_mb: int = 512
    force_col_wise: bool = True
    num_threads: int = 8
    seed: int = 0
    num_rounds: int = 200
    early_stopping_rounds: int | None = None

    def as_params(self) -> dict:
        p = asdict(self)
        p.pop("num_rounds")
        p.pop("early_stopping_rounds")
        p["verbosity"] = -1
        p["deterministic"] = True
        return p


def train_model(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> lgb.Booster:
    """Fit one binary classifier on a float32 feature matrix."""
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    kwargs = {}
    valid_sets = None
    if cfg.early_stopping_rounds:
        valid_sets = [ds]
        kwargs["callbacks"] = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
    return lgb.train(
        cfg.as_params(),
        ds,
        num_boost_round=cfg.num_rounds,
        valid_sets=valid_sets,
        **kwargs,
    )


def load_model(path: Path | str) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def save_model(bst: lgb.Booster, path: Path | str) -> None:
    bst.save_model(str(path), num_iteration=bst.current_iteration())
