from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bars_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, n)), 0.01, 0.99)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.005,
            "low": close - 0.005,
            "close": close,
            "volume_usd": rng.uniform(0, 1000, n),
            "volume_tokens": rng.uniform(0, 5000, n),
            "n_trades": rng.integers(0, 50, n),
            "hl_range": 0.01,
            "rv": rng.uniform(0, 0.05, n),
        },
        index=idx,
    )


@pytest.fixture
def window_history(bars_df):
    history = bars_df
    window = history.tail(10).copy()
    return window, history


def _fresh_registry():
    """Re-import the features package so each test starts with a clean registry.

    Lets us assert isolation properties without interference between tests.
    """
    import sys

    import polymarket_gym.features as features

    for mod_name in list(sys.modules):
        if mod_name.startswith("polymarket_gym.features."):
            del sys.modules[mod_name]
    features._REGISTRY.clear()
    features._discovered = False
    features.discover()
    return features


def test_isolation_does_not_mutate_inputs(window_history):
    features = _fresh_registry()
    window, history = window_history
    win_before = window.copy(deep=True)
    hist_before = history.copy(deep=True)
    pipeline = features.build(["sma_close"])
    features.apply_features(pipeline, window, history)
    pd.testing.assert_frame_equal(window, win_before)
    pd.testing.assert_frame_equal(history, hist_before)


def test_two_features_independent_of_order(window_history):
    features = _fresh_registry()
    window, history = window_history
    a = features.apply_features(
        features.build(["sma_close", "log_return_momentum"]), window, history
    )
    b = features.apply_features(
        features.build(["log_return_momentum", "sma_close"]), window, history
    )
    assert set(a.columns) == set(b.columns)
    pd.testing.assert_series_equal(a["sma_close"], b["sma_close"])
    pd.testing.assert_series_equal(a["log_return_momentum"], b["log_return_momentum"])


def test_single_feature_in_isolation(window_history):
    features = _fresh_registry()
    window, history = window_history
    out = features.apply_features(features.build(["sma_close"]), window, history)
    assert "sma_close" in out.columns
    assert "log_return_momentum" not in out.columns
    assert len(out) == len(window)


def test_registry_rejects_duplicate_names():
    features = _fresh_registry()

    class Dummy:
        name = "sma_close"

        def apply(self, window_df, history_df):
            return window_df

    features.discover()
    with pytest.raises(ValueError, match="already registered"):
        features.register(Dummy())


def test_extra_features_extend_observation_space(in_memory_feed_factory):
    from gymnasium.utils.env_checker import check_env

    from polymarket_gym.config import EnvConfig
    from polymarket_gym.env import PolymarketDirectionalEnv
    from polymarket_gym.spaces import N_WINDOW_FEATURES
    from tests.conftest import make_two_phase_bars

    base = EnvConfig(bar_size="1h", lookback=8, min_bars_per_episode=24)
    enriched = EnvConfig(
        bar_size="1h",
        lookback=8,
        min_bars_per_episode=24,
        extra_features=("sma_close", "log_return_momentum"),
    )
    bars = make_two_phase_bars(120, open_price=0.4, close_price=0.6)

    env_a = PolymarketDirectionalEnv(config=base, feed=in_memory_feed_factory(bars))
    env_b = PolymarketDirectionalEnv(config=enriched, feed=in_memory_feed_factory(bars, cfg=enriched))
    assert env_a.observation_space["window"].shape == (8, N_WINDOW_FEATURES)
    assert env_b.observation_space["window"].shape == (8, N_WINDOW_FEATURES + 2)

    check_env(env_b, skip_render_check=True)
    obs, _ = env_b.reset(seed=0)
    assert obs["window"].shape == (8, N_WINDOW_FEATURES + 2)


def test_broken_feature_is_caught(window_history):
    features = _fresh_registry()
    window, history = window_history

    class DropsRows:
        name = "drops_rows"

        def apply(self, window_df, history_df):
            return window_df.iloc[:-1]

    with pytest.raises(ValueError, match="changed row count"):
        features.apply_features([DropsRows()], window, history)
