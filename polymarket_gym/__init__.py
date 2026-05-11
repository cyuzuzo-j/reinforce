from __future__ import annotations

from gymnasium.envs.registration import register

from polymarket_gym.config import EnvConfig

register(
    id="PolymarketDirectional-v0",
    entry_point="polymarket_gym.env:PolymarketDirectionalEnv",
)

__all__ = ["EnvConfig"]
