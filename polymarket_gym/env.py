from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polymarket_gym.config import EnvConfig
from polymarket_gym.execution import ExecutionVenue, FillResult, SimulatedVenue
from polymarket_gym.feed import Bar, MarketFeed
from polymarket_gym.spaces import build_observation_space, pack_observation


class PolymarketDirectionalEnv(gym.Env):
    """Single-market YES-only directional trading env.

    Action space: ``Discrete(3)`` — 0 = sell all, 1 = hold, 2 = buy all.
    Reward: mark-to-market change in portfolio value, minus optional
    invalid-action penalty. Terminal settlement at ``feed.settlement_price()``.

    The env depends only on the ``MarketFeed`` and ``ExecutionVenue``
    protocols, so swapping a historical replay for a live websocket feed
    (and a simulated venue for a real CLOB venue) is purely wiring.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        feed: MarketFeed | None = None,
        venue: ExecutionVenue | None = None,
    ) -> None:
        super().__init__()
        if feed is None:
            raise ValueError("PolymarketDirectionalEnv requires a MarketFeed")
        self.cfg = config if config is not None else EnvConfig()
        self.feed = feed
        self.venue = venue if venue is not None else SimulatedVenue()
        self.action_space = spaces.Discrete(3)
        self.observation_space = build_observation_space(self.cfg)

        self._rng: np.random.Generator = np.random.default_rng(self.cfg.seed)
        self._cash: float = self.cfg.initial_cash
        self._position_tokens: float = 0.0
        self._pv_prev: float = self.cfg.initial_cash
        self._last_close: float = 0.0
        self._step_count: int = 0
        self._terminated: bool = False
        self._market_meta: Any = None
        self._total_bars: int = 0

    # --- gymnasium API -------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.cfg.seed is not None:
            self._rng = np.random.default_rng(self.cfg.seed)
        market_id = None
        if options is not None:
            market_id = options.get("market_id")

        meta = self.feed.reset(market_id=market_id, rng=self._rng)
        self._market_meta = meta
        self._total_bars = meta.n_bars
        self._cash = float(self.cfg.initial_cash)
        self._position_tokens = 0.0
        self._pv_prev = float(self.cfg.initial_cash)
        self._step_count = 0
        self._terminated = False

        history = self.feed.history()
        self._last_close = history[-1].close if history else 0.0
        obs = pack_observation(
            history,
            position_tokens=self._position_tokens,
            cash=self._cash,
            portfolio_value=self._pv_prev,
            bars_remaining=self._total_bars - len(history),
            total_bars=self._total_bars,
            cfg=self.cfg,
        )
        info = {
            "market_id": meta.market_id,
            "question": meta.question,
            "yes_payoff": meta.yes_payoff,
            "n_bars": meta.n_bars,
        }
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError(f"action must be in {{0,1,2}}, got {action}")

        next_bar = self.feed.advance()
        if next_bar is None:
            return self._finalize_episode()

        fill = self.venue.submit(
            action=action,
            next_bar=next_bar,
            position_tokens=self._position_tokens,
            cash=self._cash,
            cfg=self.cfg,
        )
        self._apply_fill(fill)

        penalty = self._invalid_action_penalty(action, fill)
        pv_new = self._cash + self._position_tokens * next_bar.close
        reward = (pv_new - self._pv_prev) - penalty
        self._pv_prev = pv_new
        self._last_close = next_bar.close
        self._step_count += 1

        truncated = (
            self.cfg.max_episode_steps is not None
            and self._step_count >= self.cfg.max_episode_steps
        )

        history = self.feed.history()
        bars_remaining = max(0, self._total_bars - len(history))
        obs = pack_observation(
            history,
            position_tokens=self._position_tokens,
            cash=self._cash,
            portfolio_value=pv_new,
            bars_remaining=bars_remaining,
            total_bars=self._total_bars,
            cfg=self.cfg,
        )
        info = {
            "pv": pv_new,
            "cash": self._cash,
            "position_tokens": self._position_tokens,
            "bar_close": next_bar.close,
            "bar_open": next_bar.open,
            "last_fill_price": fill.fill_price if fill.tokens_delta != 0 else None,
            "fee_paid": fill.fee_paid,
            "step": self._step_count,
        }
        terminated = False
        if truncated:
            self._terminated = True
        return obs, float(reward), terminated, bool(truncated), info

    def close(self) -> None:
        close_feed = getattr(self.feed, "close", None)
        if callable(close_feed):
            close_feed()
        close_venue = getattr(self.venue, "close", None)
        if callable(close_venue):
            close_venue()

    # --- internals -----------------------------------------------------

    def _apply_fill(self, fill: FillResult) -> None:
        self._cash += fill.cash_delta
        self._position_tokens += fill.tokens_delta
        if abs(self._position_tokens) < 1e-12:
            self._position_tokens = 0.0
        if abs(self._cash) < 1e-12:
            self._cash = 0.0

    def _invalid_action_penalty(self, action: int, fill: FillResult) -> float:
        if self.cfg.invalid_action_penalty == 0.0:
            return 0.0
        is_buy_when_long = action == 2 and self._position_tokens > 0.0 and fill.tokens_delta == 0
        is_sell_when_flat = action == 0 and self._position_tokens == 0.0 and fill.tokens_delta == 0
        if is_buy_when_long or is_sell_when_flat:
            return float(self.cfg.invalid_action_penalty)
        return 0.0

    def _finalize_episode(self) -> tuple[dict, float, bool, bool, dict]:
        self._terminated = True
        settlement = self.feed.settlement_price() if self.cfg.terminal_settlement else None
        reward = 0.0
        if settlement is not None and self._position_tokens > 0.0:
            settle_delta = (settlement - self._last_close) * self._position_tokens
            self._cash += self._position_tokens * settlement
            self._pv_prev = self._cash
            self._position_tokens = 0.0
            reward = settle_delta
        history = self.feed.history()
        obs = pack_observation(
            history,
            position_tokens=self._position_tokens,
            cash=self._cash,
            portfolio_value=self._pv_prev,
            bars_remaining=0,
            total_bars=self._total_bars,
            cfg=self.cfg,
        )
        info = {
            "pv": self._pv_prev,
            "cash": self._cash,
            "position_tokens": self._position_tokens,
            "settlement_price": settlement,
            "settled": settlement is not None,
            "step": self._step_count,
        }
        return obs, float(reward), True, False, info
