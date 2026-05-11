from __future__ import annotations

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


class _RandomMarketWrapper(gym.Wrapper):
    """On every reset, picks a random market_id from the allowed list."""

    def __init__(self, env: gym.Env, market_ids: list[str], seed: int) -> None:
        super().__init__(env)
        if not market_ids:
            raise ValueError("market_ids cannot be empty")
        self._market_ids = list(market_ids)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = dict(options) if options else {}
        if "market_id" not in opts:
            opts["market_id"] = str(self._rng.choice(self._market_ids))
        return self.env.reset(seed=seed, options=opts)


def make_env(
    markets_path: Path,
    quant_path: Path,
    cfg: EnvConfig,
    market_ids: list[str],
    seed: int,
    monitor_dir: str | Path | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a fully-wrapped env for SB3."""

    def _thunk() -> gym.Env:
        loader = MarketLoader(markets_path, quant_path)
        feed = HistoricalFeed(loader, cfg)
        env = PolymarketDirectionalEnv(config=cfg, feed=feed, venue=SimulatedVenue())
        env = _RandomMarketWrapper(env, market_ids, seed=seed)
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
) -> VecEnv:
    thunks = [
        make_env(
            markets_path,
            quant_path,
            cfg,
            market_ids,
            seed=seed + i,
            monitor_dir=monitor_dir,
        )
        for i in range(n_envs)
    ]
    cls = SubprocVecEnv if subproc and n_envs > 1 else DummyVecEnv
    return cls(thunks)
