from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader
from polymarket_gym.env import PolymarketDirectionalEnv
from polymarket_gym.execution import SimulatedVenue
from polymarket_gym.feed import HistoricalFeed


from polymarket_gym.feed import _InsufficientBarsError


class _RandomMarketWrapper(gym.Wrapper):
    """On every reset, picks a random market_id from the allowed list."""

    def __init__(self, env: gym.Env, market_ids: list[str], seed: int) -> None:
        super().__init__(env)
        if not market_ids:
            raise ValueError("market_ids cannot be empty")
        self._market_ids = list(market_ids)
        self._rng = np.random.default_rng(seed)

    def set_market_ids(self, ids: list[str]) -> None:
        """Expand or replace the allowed market pool (takes effect at next reset)."""
        if not ids:
            raise ValueError("cannot set empty market_ids")
        self._market_ids = list(ids)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = dict(options) if options else {}
        if "market_id" in opts:
            return self.env.reset(seed=seed, options=opts)
        # Random selection with retry on insufficient bars.
        while self._market_ids:
            chosen = str(self._rng.choice(self._market_ids))
            opts["market_id"] = chosen
            try:
                return self.env.reset(seed=seed, options=opts)
            except _InsufficientBarsError:
                self._market_ids.remove(chosen)
        raise RuntimeError("all markets in split have insufficient bars")


class FeeScheduleWrapper(gym.Wrapper):
    """Applies reduced fees for the first `warmup_episodes` resets per sub-env.

    Overwrites `env.unwrapped.cfg` with a new EnvConfig on each reset,
    switching from `warmup_bps` to `full_bps` after the warmup window.

    Note: `warmup_episodes` is per-sub-env, not global. With n_envs=4 and
    warmup_episodes=10, approximately 40 total low-fee episodes will run.
    """

    def __init__(
        self,
        env: gym.Env,
        warmup_episodes: int,
        warmup_bps: float,
        full_bps: float,
    ) -> None:
        super().__init__(env)
        self._warmup_episodes = warmup_episodes
        self._warmup_bps = warmup_bps
        self._full_bps = full_bps
        self._reset_count = 0
        self._full_cfg = env.unwrapped.cfg

    def reset(self, **kwargs):
        bps = self._warmup_bps if self._reset_count < self._warmup_episodes else self._full_bps
        self.env.unwrapped.cfg = dataclasses.replace(self._full_cfg, min_spread_bps=bps)
        self._reset_count += 1
        return self.env.reset(**kwargs)


def make_env(
    markets_path: Path,
    quant_path: Path,
    cfg: EnvConfig,
    market_ids: list[str],
    seed: int,
    monitor_dir: str | Path | None = None,
    fee_warmup_episodes: int = 0,
    fee_warmup_bps: float = 5.0,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a fully-wrapped env for SB3."""

    def _thunk() -> gym.Env:
        loader = MarketLoader(markets_path, quant_path)
        feed = HistoricalFeed(loader, cfg)
        env = PolymarketDirectionalEnv(config=cfg, feed=feed, venue=SimulatedVenue())
        env = _RandomMarketWrapper(env, market_ids, seed=seed)
        if fee_warmup_episodes > 0:
            env = FeeScheduleWrapper(
                env,
                warmup_episodes=fee_warmup_episodes,
                warmup_bps=fee_warmup_bps,
                full_bps=cfg.min_spread_bps,
            )
        env = gym.wrappers.FlattenObservation(env)
        if monitor_dir is not None:
            Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{seed}"))
        else:
            env = Monitor(env)
        return env

    return _thunk


def make_vec_env(
    markets_path: Path,
    quant_path: Path,
    cfg: EnvConfig,
    market_ids: list[str],
    n_envs: int,
    seed: int,
    monitor_dir: str | Path | None = None,
    subproc: bool = False,
    fee_warmup_episodes: int = 0,
    fee_warmup_bps: float = 5.0,
) -> VecEnv:
    thunks = [
        make_env(
            markets_path,
            quant_path,
            cfg,
            market_ids,
            seed=seed + i,
            monitor_dir=monitor_dir,
            fee_warmup_episodes=fee_warmup_episodes,
            fee_warmup_bps=fee_warmup_bps,
        )
        for i in range(n_envs)
    ]
    cls = SubprocVecEnv if subproc and n_envs > 1 else DummyVecEnv
    return cls(thunks)
