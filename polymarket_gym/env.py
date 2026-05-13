from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polymarket_gym.config import EnvConfig
from polymarket_gym.execution import ExecutionVenue, FillResult, SimulatedVenue
from polymarket_gym.feed import Bar, MarketFeed
from polymarket_gym.spaces import build_observation_space, pack_observation


class PolymarketDirectionalEnv(gym.Env):
    """Single-market YES/NO directional trading env.

    Action space: ``Discrete(2N-1)`` mapping to signed target fractions in
    ``[-1, 1]`` of portfolio value. Positive = long YES, negative = long NO,
    middle = flat. With ``n_action_levels=N``, action ``N-1`` is flat,
    ``2N-2`` is full YES, ``0`` is full NO.

    Reward: per-step log-return of portfolio value. Terminal settlement at
    ``feed.settlement_price()`` is just the last bar's mark — log-return
    naturally absorbs the payoff jump from final close to ``yes_payoff``.
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
        self.action_space = spaces.Discrete(self.cfg.n_actions)
        self._action_fracs = self.cfg.action_fracs
        self.observation_space = build_observation_space(self.cfg)

        self._rng: np.random.Generator = np.random.default_rng(self.cfg.seed)
        self._cash: float = self.cfg.initial_cash
        self._yes_tokens: float = 0.0
        self._no_tokens: float = 0.0
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
        market_id = options.get("market_id") if options else None

        meta = self.feed.reset(market_id=market_id, rng=self._rng)
        self._market_meta = meta
        self._total_bars = meta.n_bars
        self._cash = float(self.cfg.initial_cash)
        self._yes_tokens = 0.0
        self._no_tokens = 0.0
        self._pv_prev = float(self.cfg.initial_cash)
        self._step_count = 0
        self._terminated = False

        history = self.feed.history()
        self._last_close = history[-1].close if history else 0.0
        info = {
            "market_id": meta.market_id,
            "question": meta.question,
            "yes_payoff": meta.yes_payoff,
            "n_bars": meta.n_bars,
        }
        return self._obs(), info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        action = int(action)
        if not (0 <= action < self.cfg.n_actions):
            raise ValueError(f"action must be in [0, {self.cfg.n_actions}), got {action}")

        next_bar = self.feed.advance()
        if next_bar is None:
            return self._finalize_episode()

        target_frac = self._action_fracs[action]
        fill = self.venue.submit(
            target_frac=target_frac,
            next_bar=next_bar,
            yes_tokens=self._yes_tokens,
            no_tokens=self._no_tokens,
            cash=self._cash,
            cfg=self.cfg,
        )
        self._apply_fill(fill)

        pv_new = self._mark_to_market(next_bar.close)
        reward = self._log_return(pv_new)
        self._pv_prev = pv_new
        self._last_close = next_bar.close
        self._step_count += 1

        info = {
            "pv": pv_new,
            "cash": self._cash,
            "yes_tokens": self._yes_tokens,
            "no_tokens": self._no_tokens,
            "position_tokens": self._yes_tokens - self._no_tokens,  # legacy
            "bar_close": next_bar.close,
            "bar_open": next_bar.open,
            "last_fill_price": fill.fill_price,
            "fill_side": fill.side,
            "fee_paid": fill.fee_paid,
            "step": self._step_count,
        }

        truncated = (
            self.cfg.max_episode_steps is not None
            and self._step_count >= self.cfg.max_episode_steps
        )
        if truncated:
            obs, settle_reward, _, _, settle_info = self._finalize_episode(force_settle=True)
            return obs, float(reward + settle_reward), False, True, {**info, **settle_info}

        return self._obs(), float(reward), False, False, info

    def close(self) -> None:
        for target in (self.feed, self.venue):
            fn = getattr(target, "close", None)
            if callable(fn):
                fn()

    # --- internals -----------------------------------------------------

    def _apply_fill(self, fill: FillResult) -> None:
        self._cash += fill.cash_delta
        self._yes_tokens += fill.yes_delta
        self._no_tokens += fill.no_delta
        if abs(self._yes_tokens) < 1e-12:
            self._yes_tokens = 0.0
        if abs(self._no_tokens) < 1e-12:
            self._no_tokens = 0.0
        if abs(self._cash) < 1e-12:
            self._cash = 0.0

    def _mark_to_market(self, yes_close: float) -> float:
        return self._cash + self._yes_tokens * yes_close + self._no_tokens * (1.0 - yes_close)

    def _log_return(self, pv_new: float) -> float:
        return math.log(max(pv_new, 1e-9) / max(self._pv_prev, 1e-9))

    def _obs(self) -> dict:
        history = self.feed.history()
        bars_remaining = max(0, self._total_bars - len(history))
        yes_close = history[-1].close if history else 0.0
        return pack_observation(
            history,
            yes_tokens=self._yes_tokens,
            no_tokens=self._no_tokens,
            cash=self._cash,
            portfolio_value=self._mark_to_market(yes_close),
            bars_remaining=bars_remaining,
            total_bars=self._total_bars,
            cfg=self.cfg,
        )

    def _finalize_episode(self, *, force_settle: bool = False) -> tuple[dict, float, bool, bool, dict]:
        self._terminated = True
        settlement = self.feed.settlement_price() if self.cfg.terminal_settlement else None
        if settlement is None and force_settle and self.cfg.terminal_settlement and self._market_meta is not None:
            settlement = self._market_meta.yes_payoff
        if settlement is not None:
            yes_cash = self._yes_tokens * settlement
            no_cash = self._no_tokens * (1.0 - settlement)
            self._cash += yes_cash + no_cash
            self._yes_tokens = 0.0
            self._no_tokens = 0.0
            pv_new = self._cash
        else:
            pv_new = self._mark_to_market(self._last_close)
        reward = self._log_return(pv_new)
        self._pv_prev = pv_new
        info = {
            "pv": pv_new,
            "cash": self._cash,
            "yes_tokens": self._yes_tokens,
            "no_tokens": self._no_tokens,
            "position_tokens": 0.0,
            "settlement_price": settlement,
            "settled": settlement is not None,
            "step": self._step_count,
        }
        return self._obs(), float(reward), True, False, info
