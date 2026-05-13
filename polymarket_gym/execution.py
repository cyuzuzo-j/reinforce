from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from polymarket_gym.config import EnvConfig
from polymarket_gym.feed import Bar

Side = Literal["yes", "no", "flat"]


@dataclass(frozen=True)
class FillResult:
    yes_delta: float = 0.0
    no_delta: float = 0.0
    cash_delta: float = 0.0
    fee_paid: float = 0.0
    fill_price: float | None = None  # representative price for logging
    side: Side = "flat"

    @classmethod
    def noop(cls) -> "FillResult":
        return cls()

    @property
    def tokens_delta(self) -> float:
        """Net signed position change (positive = more YES, negative = more NO)."""
        return self.yes_delta - self.no_delta


@runtime_checkable
class ExecutionVenue(Protocol):
    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        yes_tokens: float,
        no_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult: ...


def _half_spread(volume_usd: float, cfg: EnvConfig) -> float:
    raw = cfg.spread_vol_factor / math.sqrt(volume_usd + 1.0)
    min_hs = cfg.min_spread_bps / 10_000.0
    return float(min(max(raw, min_hs), 0.20))


def _trade_cost(notional: float, volume_usd: float, cfg: EnvConfig) -> float:
    """Half-spread plus linear market impact, each capped at 20%."""
    hs = _half_spread(volume_usd, cfg)
    impact = cfg.impact_factor * abs(notional) / max(volume_usd, 1.0)
    return hs + min(impact, cfg.impact_cap)


class SimulatedVenue:
    """Fills to a target signed portfolio fraction at next-bar open.

    target_frac in [-1, 1]: positive = long YES, negative = long NO.
    Switching sides is automatic: opposite-side holdings are sold to flat
    before the new side is opened. Costs combine a volume-based half-spread
    with a linear market-impact term, both capped.
    """

    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        yes_tokens: float,
        no_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        ref_yes = next_bar.open
        ref_no = 1.0 - ref_yes
        pv = cash + yes_tokens * ref_yes + no_tokens * ref_no
        if pv <= 1e-9 or ref_yes <= 1e-12 or ref_no <= 1e-12:
            return FillResult.noop()

        tf = float(min(max(target_frac, -1.0), 1.0))
        target_yes = (tf * pv) / ref_yes if tf > 0 else 0.0
        target_no = (-tf * pv) / ref_no if tf < 0 else 0.0

        yes_d = no_d = cash_d = fee = 0.0
        primary_price: float | None = None
        primary_side: Side = "flat"

        def _sell(tokens: float, ref: float) -> tuple[float, float]:
            cost = _trade_cost(tokens * ref, next_bar.volume_usd, cfg)
            price = ref * (1.0 - cost)
            return price, tokens * ref * cost

        def _buy(want: float, ref: float, cash_avail: float) -> tuple[float, float, float]:
            cost = _trade_cost(want * ref, next_bar.volume_usd, cfg)
            price = ref * (1.0 + cost)
            if price <= 1e-12:
                return 0.0, 0.0, 0.0
            tokens = min(want, cash_avail / price)
            return tokens, price, tokens * ref * cost

        # ---- sells first to free cash --------------------------------------
        for side, current, target, ref in (
            ("yes", yes_tokens, target_yes, ref_yes),
            ("no", no_tokens, target_no, ref_no),
        ):
            if current - target < 1e-9:
                continue
            sell_tokens = current - target
            price, fp = _sell(sell_tokens, ref)
            cash_d += sell_tokens * price
            cash += sell_tokens * price
            fee += fp
            if side == "yes":
                yes_d -= sell_tokens
            else:
                no_d -= sell_tokens
            if (side == "yes" and tf < 0) or (side == "no" and tf > 0) or tf == 0:
                primary_price, primary_side = price, side  # type: ignore[assignment]

        # ---- buys ---------------------------------------------------------
        for side, current, target, ref in (
            ("yes", yes_tokens, target_yes, ref_yes),
            ("no", no_tokens, target_no, ref_no),
        ):
            if target - current < 1e-9:
                continue
            tokens, price, fp = _buy(target - current, ref, cash)
            if tokens < 1e-9:
                continue
            spent = tokens * price
            cash_d -= spent
            cash -= spent
            fee += fp
            if side == "yes":
                yes_d += tokens
            else:
                no_d += tokens
            primary_price, primary_side = price, side  # type: ignore[assignment]

        if abs(yes_d) < 1e-9 and abs(no_d) < 1e-9:
            return FillResult.noop()
        return FillResult(
            yes_delta=yes_d,
            no_delta=no_d,
            cash_delta=cash_d,
            fee_paid=fee,
            fill_price=primary_price,
            side=primary_side,
        )


class PolymarketCLOBVenue:
    """Live CLOB execution stub — same signed-target_frac interface as `SimulatedVenue`."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def submit(
        self,
        target_frac: float,
        next_bar: Bar,
        yes_tokens: float,
        no_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        raise NotImplementedError("PolymarketCLOBVenue is a v1 stub")
