from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from polymarket_gym.config import EnvConfig
from polymarket_gym.env import PolymarketDirectionalEnv
from polymarket_gym.execution import SimulatedVenue
from polymarket_gym.feed import Bar, InMemoryFeed, MarketMeta
from tests.conftest import make_constant_bars, make_two_phase_bars


def _make_env(cfg: EnvConfig, feed) -> PolymarketDirectionalEnv:
    return PolymarketDirectionalEnv(config=cfg, feed=feed, venue=SimulatedVenue())


def test_check_env_historical(make_historical_feed, tiny_cfg):
    env = _make_env(tiny_cfg, make_historical_feed())
    check_env(env, skip_render_check=True)


def test_check_env_with_in_memory_feed(in_memory_feed_factory, tiny_cfg):
    """Live-swap contract test: env works with any MarketFeed implementation."""
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0)
    env = _make_env(tiny_cfg, feed)
    check_env(env, skip_render_check=True)


def test_determinism(make_historical_feed, tiny_cfg):
    feed1 = make_historical_feed()
    feed2 = make_historical_feed()
    env1 = _make_env(tiny_cfg, feed1)
    env2 = _make_env(tiny_cfg, feed2)
    env1.reset(seed=42, options={"market_id": "mkt-yes"})
    env2.reset(seed=42, options={"market_id": "mkt-yes"})
    rng = np.random.default_rng(7)
    actions = rng.integers(0, 3, size=30)
    rewards1 = []
    rewards2 = []
    for a in actions:
        _, r1, term1, trunc1, _ = env1.step(int(a))
        _, r2, term2, trunc2, _ = env2.step(int(a))
        rewards1.append(r1)
        rewards2.append(r2)
        assert term1 == term2
        if term1 or trunc1:
            break
    assert rewards1 == rewards2


def test_no_lookahead(in_memory_feed_factory, tiny_cfg):
    """Action chosen at decision time t fills at bar t's open, not its close."""
    bars = make_two_phase_bars(120, open_price=0.4, close_price=0.7)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0)
    env = _make_env(tiny_cfg, feed)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(2)
    expected_fill = 0.4 * (1.0 + tiny_cfg.fee_rate)
    assert info["last_fill_price"] == pytest.approx(expected_fill)
    expected_pv = tiny_cfg.initial_cash / expected_fill * 0.7
    expected_reward = expected_pv - tiny_cfg.initial_cash
    assert reward == pytest.approx(expected_reward)


def test_terminal_settlement_yes(in_memory_feed_factory, tiny_cfg):
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0)
    env = _make_env(tiny_cfg, feed)
    env.reset(seed=0)
    _, _, term, _, info = env.step(2)
    assert not term
    while True:
        _, _, term, trunc, info = env.step(1)
        if term or trunc:
            break
    assert term
    assert info["settled"] is True
    assert info["settlement_price"] == pytest.approx(1.0)
    expected = tiny_cfg.initial_cash * 1.0 / (0.5 * (1.0 + tiny_cfg.fee_rate)) * 1.0
    assert info["cash"] == pytest.approx(expected)


def test_terminal_settlement_no(in_memory_feed_factory, tiny_cfg):
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=0.0)
    env = _make_env(tiny_cfg, feed)
    env.reset(seed=0)
    env.step(2)
    term = False
    while not term:
        _, _, term, _, info = env.step(1)
    assert info["settled"] is True
    assert info["settlement_price"] == pytest.approx(0.0)
    assert info["cash"] == pytest.approx(0.0)
    assert info["position_tokens"] == 0.0


def test_invalid_actions_are_noops(in_memory_feed_factory, tiny_cfg):
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0)
    env = _make_env(tiny_cfg, feed)
    env.reset(seed=0)
    obs1, r1, _, _, info1 = env.step(0)
    assert info1["cash"] == tiny_cfg.initial_cash
    assert info1["position_tokens"] == 0.0
    assert r1 == 0.0
    env.step(2)
    pre_cash = env._cash  # type: ignore[attr-defined]
    pre_pos = env._position_tokens  # type: ignore[attr-defined]
    _, r3, _, _, info3 = env.step(2)
    assert info3["cash"] == pre_cash
    assert info3["position_tokens"] == pre_pos
    assert info3["last_fill_price"] is None


def test_invalid_action_penalty_when_enabled(in_memory_feed_factory):
    cfg = EnvConfig(
        bar_size="1h", lookback=8, min_bars_per_episode=24,
        initial_cash=1_000.0, fee_bps=0.0, terminal_settlement=True,
        invalid_action_penalty=0.5,
    )
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0, cfg=cfg)
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    _, r, _, _, _ = env.step(0)
    assert r == pytest.approx(-0.5)


def test_random_rollout_smoke(make_historical_feed, tiny_cfg):
    feed = make_historical_feed()
    env = _make_env(tiny_cfg, feed)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        rng = np.random.default_rng(ep)
        steps = 0
        while True:
            action = int(rng.integers(0, 3))
            obs, r, term, trunc, info = env.step(action)
            assert np.isfinite(r)
            assert np.all(np.isfinite(obs["window"]))
            assert np.all(np.isfinite(obs["scalars"]))
            total_reward += r
            steps += 1
            if term or trunc:
                break
        assert steps > 0
