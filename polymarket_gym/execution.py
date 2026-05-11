from __future__ import annotations

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
        action: int,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult: ...


class SimulatedVenue:
    """Fills the agent's order at next-bar open ± fee_bps. Full bankroll semantics."""

    def submit(
        self,
        action: int,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        fee = cfg.fee_rate
        if action == 2 and position_tokens == 0.0 and cash > 0.0:
            fill_price = next_bar.open * (1.0 + fee)
            tokens = cash / fill_price
            fee_paid = cash * fee / (1.0 + fee)
            return FillResult(
                fill_price=fill_price,
                tokens_delta=tokens,
                cash_delta=-cash,
                fee_paid=fee_paid,
            )
        if action == 0 and position_tokens > 0.0:
            fill_price = next_bar.open * (1.0 - fee)
            gross = position_tokens * next_bar.open
            proceeds = position_tokens * fill_price
            fee_paid = gross - proceeds
            return FillResult(
                fill_price=fill_price,
                tokens_delta=-position_tokens,
                cash_delta=proceeds,
                fee_paid=fee_paid,
            )
        return FillResult.noop()


class PolymarketCLOBVenue:
    """Placeholder for a live CLOB execution venue.

    A real implementation would:
      - translate the discrete action into a CLOB order (market or aggressive limit)
      - submit via Polymarket's REST API with idempotency keys
      - poll for fills, compute realized avg fill price and fees
      - return a `FillResult` populated with the actual fills

    Because `FillResult` already carries the realized fill price and fee_paid,
    the env's reward math is unchanged when this replaces `SimulatedVenue`.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def submit(
        self,
        action: int,
        next_bar: Bar,
        position_tokens: float,
        cash: float,
        cfg: EnvConfig,
    ) -> FillResult:
        raise NotImplementedError("PolymarketCLOBVenue is a v1 stub")
