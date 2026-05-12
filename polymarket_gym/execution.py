from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from polymarket_gym.config import EnvConfig
from polymarket_gym.feed import Bar


@dataclass(frozen=True)
class FillResult:
    fill_price: float
    tokens_delta: float
    cash_delta: float
    fee_paid: float

    @classmethod
    def noop(cls) -> "FillResult":
        return cls(fill_price=0.0, tokens_delta=0.0, cash_delta=0.0, fee_paid=0.0)


@runtime_checkable
class ExecutionVenue(Protocol):
    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult: ...


def _half_spread(volume_usd: float, cfg: EnvConfig) -> float:
    """Estimate half-spread from bar volume.

    On zero-volume (forward-filled) bars the spread is capped at 20%, which
    strongly penalises trading on stale prices. On liquid bars ($100K+) it
    approaches the min_spread_bps floor.
    """
    raw = cfg.spread_vol_factor / math.sqrt(volume_usd + 1.0)
    min_hs = cfg.min_spread_bps / 10_000.0
    return float(min(max(raw, min_hs), 0.20))


class SimulatedVenue:
    """Fills to a target portfolio fraction at next-bar open with a spread model.

    half_spread = spread_vol_factor / sqrt(volume_usd + 1), floored at
    min_spread_bps and capped at 20%. This models the bid-ask spread on a
    thin CLOB market — zero-volume bars (forward-filled prices) get maximum
    spread, making trading on stale prices expensive.

    Partial fills: the agent specifies a target position fraction in [0, 1]
    of current portfolio value. The venue executes the delta trade needed to
    reach that fraction, capped by available cash or tokens.
    """

    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        ref = next_bar.open
        hs = _half_spread(next_bar.volume_usd, cfg)

        pv = cash + position_tokens * ref
        if pv <= 1e-9:
            return FillResult.noop()

        target_frac = float(min(max(target_frac, 0.0), 1.0))
        buy_price = ref * (1.0 + hs)
        sell_price = ref * (1.0 - hs)

        # Target tokens at the buy price (conservative for buys)
        target_tokens = (target_frac * pv) / buy_price if buy_price > 1e-12 else 0.0
        delta = target_tokens - position_tokens

        if abs(delta) < 1e-9:
            return FillResult.noop()

        if delta > 0:  # buying
            max_tokens = cash / buy_price if buy_price > 1e-12 else 0.0
            delta = min(delta, max_tokens)
            if delta < 1e-9:
                return FillResult.noop()
            cash_spent = delta * buy_price
            fee_paid = delta * ref * hs
            return FillResult(
                fill_price=buy_price,
                tokens_delta=delta,
                cash_delta=-cash_spent,
                fee_paid=fee_paid,
            )
        else:  # selling
            delta = max(delta, -position_tokens)
            if abs(delta) < 1e-9:
                return FillResult.noop()
            cash_received = (-delta) * sell_price
            fee_paid = (-delta) * ref * hs
            return FillResult(
                fill_price=sell_price,
                tokens_delta=delta,
                cash_delta=cash_received,
                fee_paid=fee_paid,
            )


class PolymarketCLOBVenue:
    """Placeholder for a live CLOB execution venue.

    A real implementation would:
      - translate target_frac into a CLOB order (market or aggressive limit)
      - submit via Polymarket's REST API with idempotency keys
      - poll for fills, compute realized avg fill price and fees
      - return a `FillResult` populated with the actual fills
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        raise NotImplementedError("PolymarketCLOBVenue is a v1 stub")
