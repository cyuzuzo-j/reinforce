from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from polymarket_gym.config import EnvConfig
from polymarket_gym.env import PolymarketDirectionalEnv
from polymarket_gym.execution import SimulatedVenue
from tests.conftest import make_constant_bars, make_two_phase_bars


def _make_env(cfg: EnvConfig, feed) -> PolymarketDirectionalEnv:
    return PolymarketDirectionalEnv(config=cfg, feed=feed, venue=SimulatedVenue())


def _zero_cost_cfg(**overrides) -> EnvConfig:
    """Tiny cfg with no spread/impact so reward math is analytic."""
    base = dict(
        bar_size="1h", lookback=8, min_bars_per_episode=24,
        initial_cash=1_000.0, fee_bps=0.0,
        min_spread_bps=0.0, spread_vol_factor=0.0, impact_factor=0.0,
    )
    base.update(overrides)
    return EnvConfig(**base)


def test_check_env_historical(make_historical_feed, tiny_cfg):
    env = _make_env(tiny_cfg, make_historical_feed())
    check_env(env, skip_render_check=True)


def test_check_env_with_in_memory_feed(in_memory_feed_factory, tiny_cfg):
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0)
    env = _make_env(tiny_cfg, feed)
    check_env(env, skip_render_check=True)


def test_determinism(make_historical_feed, tiny_cfg):
    env1 = _make_env(tiny_cfg, make_historical_feed())
    env2 = _make_env(tiny_cfg, make_historical_feed())
    env1.reset(seed=42, options={"market_id": "mkt-yes"})
    env2.reset(seed=42, options={"market_id": "mkt-yes"})
    rng = np.random.default_rng(7)
    actions = rng.integers(0, tiny_cfg.n_actions, size=30)
    rewards1, rewards2 = [], []
    for a in actions:
        _, r1, term1, trunc1, _ = env1.step(int(a))
        _, r2, term2, _, _ = env2.step(int(a))
        rewards1.append(r1); rewards2.append(r2)
        assert term1 == term2
        if term1 or trunc1:
            break
    assert rewards1 == rewards2


def test_no_lookahead(in_memory_feed_factory):
    """Action chosen at decision time t fills at bar t's open, not its close."""
    cfg = _zero_cost_cfg()
    bars = make_two_phase_bars(120, open_price=0.4, close_price=0.7)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0, cfg=cfg)
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(cfg.n_actions - 1)  # full long YES
    assert info["last_fill_price"] == pytest.approx(0.4)
    expected_pv = cfg.initial_cash / 0.4 * 0.7
    assert info["pv"] == pytest.approx(expected_pv)
    assert reward == pytest.approx(math.log(expected_pv / cfg.initial_cash))


def test_terminal_settlement_yes(in_memory_feed_factory):
    cfg = _zero_cost_cfg()
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0, cfg=cfg)
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    long_yes = cfg.n_actions - 1
    env.step(long_yes)
    while True:
        _, _, term, trunc, info = env.step(long_yes)  # maintain full YES
        if term or trunc:
            break
    assert term
    assert info["settled"] is True and info["settlement_price"] == pytest.approx(1.0)
    # 2000 YES tokens * 1.0 payoff = $2000 cash
    assert info["cash"] == pytest.approx(cfg.initial_cash / 0.5 * 1.0)


def test_no_side_settlement_when_yes_payoff_zero(in_memory_feed_factory):
    """Buying NO when market resolves NO → cash multiplies by 1/no_price."""
    cfg = _zero_cost_cfg()
    bars = make_constant_bars(120, price=0.3)  # YES at 0.3 → NO at 0.7
    feed = in_memory_feed_factory(bars, yes_payoff=0.0, cfg=cfg)
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    env.step(0)  # full long NO
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, info = env.step(0)  # maintain full NO
    assert info["settled"] is True and info["settlement_price"] == pytest.approx(0.0)
    expected = cfg.initial_cash / 0.7 * 1.0  # each NO token pays 1.0
    assert info["cash"] == pytest.approx(expected)
    assert info["no_tokens"] == 0.0


def test_truncation_settles_position(in_memory_feed_factory):
    cfg = _zero_cost_cfg(max_episode_steps=20)
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=1.0, cfg=cfg)
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    long_yes = cfg.n_actions - 1
    env.step(long_yes)
    term = trunc = False
    info: dict = {}
    while not (term or trunc):
        _, _, term, trunc, info = env.step(long_yes)  # hold YES
    assert trunc and not term
    assert info["settled"] is True
    # YES tokens settled at payoff=1.0 → cash = $2000
    assert info["cash"] == pytest.approx(cfg.initial_cash / 0.5 * 1.0)


def test_log_reward_zero_when_flat(in_memory_feed_factory):
    cfg = _zero_cost_cfg()
    bars = make_constant_bars(120, price=0.5)
    feed = in_memory_feed_factory(bars, yes_payoff=0.5, cfg=cfg)  # no payoff jump
    env = _make_env(cfg, feed)
    env.reset(seed=0)
    total = 0.0
    term = trunc = False
    while not (term or trunc):
        _, r, term, trunc, _ = env.step(cfg.flat_action)
        total += r
    assert total == pytest.approx(0.0, abs=1e-9)


def test_random_rollout_smoke(make_historical_feed, tiny_cfg):
    feed = make_historical_feed()
    env = _make_env(tiny_cfg, feed)
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        steps = 0
        while True:
            a = int(rng.integers(0, tiny_cfg.n_actions))
            obs, r, term, trunc, _ = env.step(a)
            assert np.isfinite(r)
            assert np.all(np.isfinite(obs["window"]))
            assert np.all(np.isfinite(obs["scalars"]))
            steps += 1
            if term or trunc:
                break
        assert steps > 0
