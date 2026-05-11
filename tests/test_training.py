from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from polymarket_gym.config import EnvConfig
from polymarket_gym.policy import PolicyFeatures
from polymarket_gym.spaces import build_observation_space
from polymarket_gym.training.callbacks import (
    EpisodeCounterCallback,
    VisualizationCallback,
)


# ---------- chronological_split ----------


class _StubLoader:
    def __init__(self, ids_with_dates: list[tuple[str, str]]) -> None:
        self._ids = [mid for mid, _ in ids_with_dates]
        self._meta = {
            mid: {
                "market_id": mid,
                "question": "?",
                "end_date": pd.Timestamp(d, tz="UTC"),
                "yes_payoff": 1.0,
            }
            for mid, d in ids_with_dates
        }

    def eligible_market_ids(self, cfg):
        return list(self._ids)

    def load_meta(self, market_id):
        return self._meta[market_id]


def test_chronological_split_partitions_by_end_date():
    from polymarket_gym.training.splits import chronological_split

    loader = _StubLoader(
        [
            ("c", "2024-03-01"),
            ("a", "2024-01-01"),
            ("d", "2024-04-01"),
            ("b", "2024-02-01"),
            ("e", "2024-05-01"),
        ]
    )
    cfg = EnvConfig()
    train_ids, eval_ids = chronological_split(loader, cfg, eval_frac=0.4)
    assert train_ids == ["a", "b", "c"]
    assert eval_ids == ["d", "e"]


def test_chronological_split_rejects_empty_train():
    from polymarket_gym.training.splits import chronological_split

    loader = _StubLoader([("a", "2024-01-01")])
    with pytest.raises(RuntimeError):
        chronological_split(loader, EnvConfig(), eval_frac=0.5)


# ---------- PolicyFeatures ----------


def test_policy_features_forward_shape():
    cfg = EnvConfig(lookback=16)
    obs_space = build_observation_space(cfg)
    extractor = PolicyFeatures(obs_space, features_dim=64)
    batch = {
        "window": torch.randn(4, 16, 7),
        "scalars": torch.randn(4, 4),
    }
    out = extractor(batch)
    assert out.shape == (4, 64)


# ---------- VisualizationCallback ----------


class _FakeEnv:
    """Drives the callback through a short rollout without touching SB3."""

    def __init__(self, n_steps: int = 6) -> None:
        self._n = n_steps
        self._i = 0

    def reset(self):
        self._i = 0
        return {}, {"market_id": "fake"}

    def step(self, action):
        self._i += 1
        info = {
            "bar_close": 0.5 + 0.01 * self._i,
            "position_tokens": 0.0 if action != 2 else 100.0,
            "pv": 1000.0 + self._i,
            "last_fill_price": 0.5 if action in (0, 2) else None,
        }
        terminated = self._i >= self._n
        return {}, 1.0, terminated, False, info

    def close(self):
        pass


class _FakeModel:
    def __init__(self) -> None:
        self.ep_info_buffer = [{"r": 1.0, "l": 10}, {"r": 2.0, "l": 12}]
        self.logger = SimpleNamespace(name_to_value={}, record=lambda *a, **k: None)

    def predict(self, obs, deterministic=True):
        return np.array(2), None


def test_visualization_callback_writes_png(tmp_path: Path):
    counter = EpisodeCounterCallback()
    cb = VisualizationCallback(
        eval_env_fn=lambda: _FakeEnv(n_steps=5),
        every_n_episodes=2,
        out_dir=tmp_path,
        counter=counter,
    )
    cb.model = _FakeModel()
    cb._init_callback()

    # Simulate VecEnv infos: feed 3 episode-end infos through the counter.
    counter.locals = {"infos": [{"episode": {"r": 1.0, "l": 5}}]}
    cb.locals = counter.locals
    for _ in range(3):
        counter._on_step()
        cb._on_step()

    pngs = list(tmp_path.glob("ep_*_fake.png"))
    assert len(pngs) >= 1, f"expected at least one PNG, got {list(tmp_path.iterdir())}"
