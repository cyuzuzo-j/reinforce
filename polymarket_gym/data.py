from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from polymarket_gym.config import EnvConfig


def parse_outcome_prices(raw: str | list | tuple) -> tuple[float, float]:
    """Parse Polymarket outcome_prices into (yes_payoff, no_payoff).

    Accepts JSON strings, Python-repr strings, or already-parsed sequences.
    Raises ValueError for anything that doesn't reduce to a length-2 numeric list.
    """
    if raw is None:
        raise ValueError("outcome_prices is None")
    if isinstance(raw, (list, tuple)):
        parsed: Any = list(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("outcome_prices is empty string")
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"cannot parse outcome_prices={raw!r}") from e
    else:
        raise ValueError(f"unsupported outcome_prices type: {type(raw)!r}")

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"outcome_prices must be length-2 list, got {parsed!r}")
    try:
        yes = float(parsed[0])
        no = float(parsed[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"non-numeric outcome_prices: {parsed!r}") from e
    return yes, no


def is_cleanly_resolved_binary(payoffs: tuple[float, float], closed: bool | int | None) -> bool:
    """A market is cleanly resolved if it's closed and outcome_prices ∈ {(1,0),(0,1)}."""
    if not bool(closed):
        return False
    return payoffs in ((1.0, 0.0), (0.0, 1.0))


_BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume_usd",
    "volume_tokens",
    "n_trades",
    "hl_range",
    "rv",
]


def build_bars(
    trades: pd.DataFrame,
    bar_size: str,
    end_date: pd.Timestamp,
    price_eps: float = 1e-6,
) -> pd.DataFrame:
    """Resample a per-market trade DataFrame into OHLCV bars + features.

    Required columns on `trades`: ``datetime``, ``price``, ``usd_amount``, ``token_amount``.

    Behavior:
      - Resamples to ``bar_size`` aligned bars between the first trade and
        ``min(last_trade_ts, end_date)``.
      - Forward-fills ``close`` only within that active window; empty bars
        inherit ``close`` into ``open/high/low`` with zero volume/n_trades/rv.
      - Clips prices to ``[price_eps, 1 - price_eps]`` so log-returns are finite.
    """
    if trades.empty:
        return pd.DataFrame(columns=_BAR_COLUMNS)

    df = trades.loc[:, ["datetime", "price", "usd_amount", "token_amount"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["price"] = df["price"].astype(float).clip(price_eps, 1.0 - price_eps)
    df["usd_amount"] = df["usd_amount"].astype(float).abs()
    df["token_amount"] = df["token_amount"].astype(float).abs()

    end_ts = pd.to_datetime(end_date, utc=True)
    df = df[df["datetime"] <= end_ts]
    if df.empty:
        return pd.DataFrame(columns=_BAR_COLUMNS)

    df["log_price"] = np.log(df["price"].to_numpy())
    df["log_ret"] = df["log_price"].diff()

    grouper = pd.Grouper(key="datetime", freq=bar_size, label="left", closed="left")
    agg = df.groupby(grouper).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume_usd=("usd_amount", "sum"),
        volume_tokens=("token_amount", "sum"),
        n_trades=("price", "count"),
        rv=("log_ret", lambda s: float(np.nanstd(s.to_numpy(), ddof=0)) if s.notna().any() else 0.0),
    )

    first_bar = df["datetime"].iloc[0].floor(bar_size)
    last_trade_bar = df["datetime"].iloc[-1].floor(bar_size)
    end_bar = end_ts.floor(bar_size)
    last_bar = min(last_trade_bar, end_bar)
    if last_bar < first_bar:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    full_index = pd.date_range(first_bar, last_bar, freq=bar_size, tz="UTC")
    agg = agg.reindex(full_index)

    agg["close"] = agg["close"].ffill()
    agg["open"] = agg["open"].fillna(agg["close"])
    agg["high"] = agg["high"].fillna(agg["close"])
    agg["low"] = agg["low"].fillna(agg["close"])
    for col in ("volume_usd", "volume_tokens", "n_trades", "rv"):
        agg[col] = agg[col].fillna(0.0)
    agg["n_trades"] = agg["n_trades"].astype(np.int64)

    for col in ("open", "high", "low", "close"):
        agg[col] = agg[col].clip(price_eps, 1.0 - price_eps)
    agg["hl_range"] = (agg["high"] - agg["low"]).clip(lower=0.0)

    return agg.loc[:, _BAR_COLUMNS]


@dataclass(frozen=True)
class _MarketRow:
    market_id: str
    question: str
    end_date: pd.Timestamp
    yes_payoff: float
    closed: bool


class MarketLoader:
    """Loads markets metadata in-memory and trades on demand via DuckDB."""

    def __init__(self, markets_path: str | Path, quant_path: str | Path) -> None:
        self.markets_path = Path(markets_path)
        self.quant_path = Path(quant_path)
        self._con = duckdb.connect()
        self._markets = self._load_markets()

    def _load_markets(self) -> pd.DataFrame:
        df = pd.read_parquet(self.markets_path)
        if "market_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "market_id"})
        df["market_id"] = df["market_id"].astype(str)
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
        elif "endDate" in df.columns:
            df["end_date"] = pd.to_datetime(df["endDate"], utc=True, errors="coerce")
        else:
            df["end_date"] = pd.NaT
        closed_col = "closed" if "closed" in df.columns else None
        df["closed"] = df[closed_col].astype(bool) if closed_col else True
        outcome_col = "outcome_prices" if "outcome_prices" in df.columns else "outcomePrices"
        if outcome_col not in df.columns:
            raise ValueError(
                f"markets parquet missing outcome_prices/outcomePrices column: {list(df.columns)}"
            )
        payoffs: list[tuple[float, float] | None] = []
        for raw in df[outcome_col]:
            try:
                payoffs.append(parse_outcome_prices(raw))
            except ValueError:
                payoffs.append(None)
        df["_payoffs"] = payoffs
        df["yes_payoff"] = [p[0] if p is not None else np.nan for p in payoffs]
        if "question" not in df.columns:
            df["question"] = ""
        return df

    def eligible_market_ids(self, cfg: EnvConfig) -> list[str]:
        df = self._markets
        mask = df["_payoffs"].apply(
            lambda p: p in ((1.0, 0.0), (0.0, 1.0)) if p is not None else False
        )
        mask &= df["closed"].astype(bool)
        mask &= df["end_date"].notna()
        eligible = df.loc[mask, "market_id"].astype(str).tolist()
        # Additional bar-count gating happens lazily in load_trades+build_bars callers
        # because it requires the actual trade history.
        return eligible

    def load_trades(self, market_id: str) -> pd.DataFrame:
        query = (
            "SELECT datetime, price, usd_amount, token_amount "
            "FROM read_parquet(?) WHERE market_id = ? ORDER BY datetime"
        )
        df = self._con.execute(query, [str(self.quant_path), str(market_id)]).fetch_df()
        if df.empty:
            return df
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df

    def load_meta(self, market_id: str) -> dict:
        row = self._markets.loc[self._markets["market_id"] == str(market_id)]
        if row.empty:
            raise KeyError(f"market_id {market_id!r} not in markets parquet")
        r = row.iloc[0]
        return {
            "market_id": str(r["market_id"]),
            "question": str(r.get("question", "")),
            "end_date": pd.Timestamp(r["end_date"]),
            "yes_payoff": float(r["yes_payoff"]) if pd.notna(r["yes_payoff"]) else None,
        }

    def close(self) -> None:
        self._con.close()
