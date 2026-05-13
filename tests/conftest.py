from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader
from polymarket_gym.feed import Bar, HistoricalFeed, InMemoryFeed, MarketMeta


def _make_trades(
    market_id: str,
    start: pd.Timestamp,
    n_minutes: int,
    seed: int,
    base_price: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_minutes, freq="1min", tz="UTC")
    walk = np.cumsum(rng.normal(0, 0.01, size=n_minutes))
    prices = np.clip(base_price + walk, 0.02, 0.98)
    return pd.DataFrame(
        {
            # MarketLoader.load_trades reads `timestamp` (unix seconds) — matches prod schema.
            "timestamp": (timestamps.astype("int64") // 1_000_000_000).astype("uint64"),
            "market_id": market_id,
            "price": prices,
            "usd_amount": rng.uniform(10.0, 100.0, size=n_minutes),
            "token_amount": rng.uniform(20.0, 200.0, size=n_minutes),
        }
    )


@pytest.fixture
def tiny_markets_df() -> pd.DataFrame:
    end = pd.Timestamp("2024-01-10 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "market_id": ["mkt-yes", "mkt-no", "mkt-unresolved"],
            "question": ["Will YES?", "Will NO?", "Unresolved?"],
            "end_date": [end, end, end],
            "closed": [True, True, False],
            "outcome_prices": ['["1", "0"]', "['0','1']", '["0.4", "0.6"]'],
        }
    )


@pytest.fixture
def tiny_trades_df() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    yes_trades = _make_trades("mkt-yes", start, n_minutes=60 * 24 * 5, seed=1, base_price=0.6)
    no_trades = _make_trades("mkt-no", start, n_minutes=60 * 24 * 5, seed=2, base_price=0.4)
    return pd.concat([yes_trades, no_trades], ignore_index=True)


@pytest.fixture
def tiny_parquet_paths(
    tmp_path: Path, tiny_markets_df: pd.DataFrame, tiny_trades_df: pd.DataFrame
) -> dict:
    markets_path = tmp_path / "markets.parquet"
    quant_path = tmp_path / "quant_sample.parquet"
    tiny_markets_df.to_parquet(markets_path)
    tiny_trades_df.to_parquet(quant_path)
    return {"markets": markets_path, "quant": quant_path}


@pytest.fixture
def tiny_cfg() -> EnvConfig:
    return EnvConfig(
        bar_size="1h",
        lookback=8,
        min_bars_per_episode=24,
        initial_cash=1_000.0,
        fee_bps=10.0,
        terminal_settlement=True,
    )


@pytest.fixture
def tiny_loader(tiny_parquet_paths: dict) -> MarketLoader:
    loader = MarketLoader(tiny_parquet_paths["markets"], tiny_parquet_paths["quant"])
    yield loader
    loader.close()


@pytest.fixture
def make_historical_feed(tiny_loader: MarketLoader, tiny_cfg: EnvConfig):
    def _factory(cfg: EnvConfig | None = None) -> HistoricalFeed:
        return HistoricalFeed(tiny_loader, cfg if cfg is not None else tiny_cfg)

    return _factory


def make_constant_bars(
    n: int,
    *,
    price: float = 0.5,
    start: pd.Timestamp = pd.Timestamp("2024-01-01", tz="UTC"),
    freq: str = "1h",
) -> list[Bar]:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return [
        Bar(
            ts=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume_usd=100.0,
            volume_tokens=200.0,
            n_trades=10,
            hl_range=0.0,
            rv=0.0,
        )
        for ts in idx
    ]


def make_two_phase_bars(
    n: int,
    *,
    open_price: float,
    close_price: float,
    start: pd.Timestamp = pd.Timestamp("2024-01-01", tz="UTC"),
    freq: str = "1h",
) -> list[Bar]:
    """Bars where every bar has the same (open != close) so look-ahead tests are unambiguous."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return [
        Bar(
            ts=ts,
            open=open_price,
            high=max(open_price, close_price),
            low=min(open_price, close_price),
            close=close_price,
            volume_usd=100.0,
            volume_tokens=200.0,
            n_trades=10,
            hl_range=abs(close_price - open_price),
            rv=0.0,
        )
        for ts in idx
    ]


@pytest.fixture
def in_memory_feed_factory(tiny_cfg: EnvConfig):
    def _factory(
        bars: list[Bar],
        *,
        yes_payoff: float = 1.0,
        market_id: str = "fake-mkt",
        cfg: EnvConfig | None = None,
    ) -> InMemoryFeed:
        c = cfg if cfg is not None else tiny_cfg
        meta = MarketMeta(
            market_id=market_id,
            question="fake?",
            end_date=bars[-1].ts,
            yes_payoff=yes_payoff,
            n_bars=len(bars),
        )
        return InMemoryFeed(bars, meta, c)

    return _factory
