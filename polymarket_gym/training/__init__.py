from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np


def rollout(
    env: gym.Env,
    policy: Callable[[Any], int],
    *,
    seed: int | None = None,
) -> dict[str, list]:
    """Run one episode with `policy(obs) -> action`. Returns per-step traces."""
    obs, info = env.reset(seed=seed)
    out: dict[str, list] = {
        "actions": [], "rewards": [], "pvs": [], "prices": [],
        "fill_prices": [], "sides": [], "positions": [],
    }
    market_id = info.get("market_id", "unknown")
    done = False
    while not done:
        action = int(policy(obs))
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = bool(terminated or truncated)
        out["actions"].append(action)
        out["rewards"].append(float(reward))
        out["pvs"].append(float(step_info.get("pv", float("nan"))))
        out["prices"].append(float(step_info.get("bar_close", float("nan"))))
        fp = step_info.get("last_fill_price")
        out["fill_prices"].append(float(fp) if fp is not None else None)
        out["sides"].append(str(step_info.get("fill_side", "flat")))
        out["positions"].append(float(step_info.get("position_tokens", 0.0)))
    out["market_id"] = market_id  # type: ignore[assignment]
    out["return"] = float(np.nansum(out["rewards"]))  # type: ignore[assignment]
    return out
