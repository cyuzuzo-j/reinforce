from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader, build_bars


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    volume_tokens: float
    n_trades: int
    hl_range: float
    rv: float


@dataclass(frozen=True)
class MarketMeta:
    market_id: str
    question: str
    end_date: pd.Timestamp
    yes_payoff: float | None  # None until resolved (live case)
    n_bars: int


@runtime_checkable
class MarketFeed(Protocol):
    def reset(
        self, *, market_id: str | None, rng: np.random.Generator
    ) -> MarketMeta: ...
    def history(self) -> list[Bar]: ...
    def advance(self) -> Bar | None: ...
    def is_resolved(self) -> bool: ...
    def settlement_price(self) -> float | None: ...
    def eligible_market_ids(self) -> list[str]: ...


def _bars_df_to_list(df: pd.DataFrame) -> list[Bar]:
    if df.empty:
        return []
    return [
        Bar(
            ts=ts,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume_usd=float(row.volume_usd),
            volume_tokens=float(row.volume_tokens),
            n_trades=int(row.n_trades),
            hl_range=float(row.hl_range),
            rv=float(row.rv),
        )
        for ts, row in df.iterrows()
    ]


class HistoricalFeed:
    """Replays bars built from a `MarketLoader`'s parquet slice."""

    def __init__(self, loader: MarketLoader, cfg: EnvConfig) -> None:
        self._loader = loader
        self._cfg = cfg
        self._bars: list[Bar] = []
        self._cursor: int = 0
        self._meta: MarketMeta | None = None
        self._eligible_cache: list[str] | None = None

    def eligible_market_ids(self) -> list[str]:
        if self._eligible_cache is None:
            self._eligible_cache = self._loader.eligible_market_ids(self._cfg)
        return list(self._eligible_cache)

    def reset(self, *, market_id: str | None, rng: np.random.Generator) -> MarketMeta:
        ids = self.eligible_market_ids()
        if not ids:
            raise RuntimeError("no eligible markets in loader")
        if market_id is None:
            market_id = str(rng.choice(ids))
        elif market_id not in ids:
            raise ValueError(f"market_id {market_id!r} not eligible")

        meta_dict = self._loader.load_meta(market_id)
        trades = self._loader.load_trades(market_id)
        bars_df = build_bars(
            trades,
            bar_size=self._cfg.bar_size,
            end_date=meta_dict["end_date"],
            price_eps=self._cfg.price_eps,
        )
        if len(bars_df) < self._cfg.min_bars_per_episode:
            raise RuntimeError(
                f"market {market_id!r} has {len(bars_df)} bars, "
                f"need >= {self._cfg.min_bars_per_episode}"
            )

        self._bars = _bars_df_to_list(bars_df)
        self._cursor = self._cfg.lookback
        self._meta = MarketMeta(
            market_id=meta_dict["market_id"],
            question=meta_dict["question"],
            end_date=meta_dict["end_date"],
            yes_payoff=meta_dict["yes_payoff"],
            n_bars=len(self._bars),
        )
        return self._meta

    def history(self) -> list[Bar]:
        return self._bars[: self._cursor]

    def advance(self) -> Bar | None:
        if self._cursor >= len(self._bars):
            return None
        bar = self._bars[self._cursor]
        self._cursor += 1
        return bar

    def is_resolved(self) -> bool:
        return self._cursor >= len(self._bars) and self._meta is not None

    def settlement_price(self) -> float | None:
        if not self.is_resolved() or self._meta is None:
            return None
        return self._meta.yes_payoff

    def close(self) -> None:
        self._loader.close()


class InMemoryFeed:
    """Test/utility feed that replays a pre-built list of bars.

    Useful for unit tests and for live-swap dry-runs: anything that produces
    a sequence of `Bar` records can drive the env through this class.
    """

    def __init__(
        self,
        bars: Iterable[Bar],
        meta: MarketMeta,
        cfg: EnvConfig,
    ) -> None:
        self._bars = list(bars)
        if len(self._bars) < cfg.min_bars_per_episode:
            raise ValueError(
                f"InMemoryFeed needs >= {cfg.min_bars_per_episode} bars, got {len(self._bars)}"
            )
        self._meta = meta
        self._cfg = cfg
        self._cursor = cfg.lookback

    def eligible_market_ids(self) -> list[str]:
        return [self._meta.market_id]

    def reset(self, *, market_id: str | None, rng: np.random.Generator) -> MarketMeta:
        if market_id is not None and market_id != self._meta.market_id:
            raise ValueError(f"InMemoryFeed only serves {self._meta.market_id!r}")
        self._cursor = self._cfg.lookback
        return self._meta

    def history(self) -> list[Bar]:
        return self._bars[: self._cursor]

    def advance(self) -> Bar | None:
        if self._cursor >= len(self._bars):
            return None
        bar = self._bars[self._cursor]
        self._cursor += 1
        return bar

    def is_resolved(self) -> bool:
        return self._cursor >= len(self._bars)

    def settlement_price(self) -> float | None:
        if not self.is_resolved():
            return None
        return self._meta.yes_payoff


class LiveFeed:
    """Placeholder for a websocket-driven Polymarket feed.

    The class is intentionally non-functional in v1: it locks the interface
    so a future implementation can plug in without touching the env. To
    flesh it out you would:
      - subscribe to Polymarket's CLOB WS trades for `market_id`
      - append into a rolling trade DataFrame
      - on each bar boundary, call `build_bars` on the rolling buffer and
        push the newest closed bar into an internal queue
      - poll Gamma API on episode end to fetch `outcomePrices` for settlement
    """

    def __init__(self, market_id: str, cfg: EnvConfig) -> None:
        self.market_id = market_id
        self.cfg = cfg

    def eligible_market_ids(self) -> list[str]:
        return [self.market_id]

    def reset(self, *, market_id: str | None, rng: np.random.Generator) -> MarketMeta:
        raise NotImplementedError("LiveFeed is a v1 stub; implement WS plumbing here")

    def history(self) -> list[Bar]:
        raise NotImplementedError

    def advance(self) -> Bar | None:
        raise NotImplementedError

    def is_resolved(self) -> bool:
        raise NotImplementedError

    def settlement_price(self) -> float | None:
        raise NotImplementedError
