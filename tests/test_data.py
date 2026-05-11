from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polymarket_gym.data import (
    is_cleanly_resolved_binary,
    parse_outcome_prices,
    build_bars,
)


def test_parse_outcome_prices_json():
    assert parse_outcome_prices('["1", "0"]') == (1.0, 0.0)
    assert parse_outcome_prices('["0", "1"]') == (0.0, 1.0)
    assert parse_outcome_prices('["0.52", "0.48"]') == pytest.approx((0.52, 0.48))


def test_parse_outcome_prices_repr():
    assert parse_outcome_prices("['1','0']") == (1.0, 0.0)
    assert parse_outcome_prices("['0', '1']") == (0.0, 1.0)


def test_parse_outcome_prices_list():
    assert parse_outcome_prices([1, 0]) == (1.0, 0.0)
    assert parse_outcome_prices(("0.3", "0.7")) == pytest.approx((0.3, 0.7))


@pytest.mark.parametrize("bad", [None, "", "garbage", "['1']", "[1,2,3]"])
def test_parse_outcome_prices_bad(bad):
    with pytest.raises(ValueError):
        parse_outcome_prices(bad)


def test_is_cleanly_resolved_binary():
    assert is_cleanly_resolved_binary((1.0, 0.0), closed=True)
    assert is_cleanly_resolved_binary((0.0, 1.0), closed=True)
    assert not is_cleanly_resolved_binary((1.0, 0.0), closed=False)
    assert not is_cleanly_resolved_binary((0.5, 0.5), closed=True)


def _make_trades(n: int, base: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "price": np.clip(base + rng.normal(0, 0.005, n).cumsum(), 0.01, 0.99),
            "usd_amount": rng.uniform(1, 10, n),
            "token_amount": rng.uniform(1, 10, n),
        }
    )


def test_build_bars_empty_bar_fills():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 00:30", "2024-01-01 05:30"], utc=True
            ),
            "price": [0.40, 0.60],
            "usd_amount": [10.0, 20.0],
            "token_amount": [25.0, 30.0],
        }
    )
    end = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    bars = build_bars(df, bar_size="1h", end_date=end)
    assert len(bars) == 6  # bars at 00, 01, 02, 03, 04, 05
    for i in range(1, 5):
        assert bars.iloc[i]["volume_usd"] == 0.0
        assert bars.iloc[i]["n_trades"] == 0
        assert bars.iloc[i]["rv"] == 0.0
        assert bars.iloc[i]["close"] == pytest.approx(0.40)
        assert bars.iloc[i]["open"] == pytest.approx(0.40)


def test_build_bars_clips_prices():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00"], utc=True
            ),
            "price": [0.0, 1.0],
            "usd_amount": [1.0, 1.0],
            "token_amount": [1.0, 1.0],
        }
    )
    end = pd.Timestamp("2024-01-01 03:00", tz="UTC")
    bars = build_bars(df, bar_size="1h", end_date=end, price_eps=1e-3)
    assert (bars[["open", "high", "low", "close"]] >= 1e-3).all().all()
    assert (bars[["open", "high", "low", "close"]] <= 1 - 1e-3).all().all()


def test_build_bars_no_ffill_past_end_date():
    df = _make_trades(120)
    end = pd.Timestamp("2024-01-01 00:30", tz="UTC")
    bars = build_bars(df, bar_size="1h", end_date=end)
    assert bars.index.max() <= end.floor("1h")


def test_build_bars_handles_no_trades_before_end():
    df = pd.DataFrame(columns=["datetime", "price", "usd_amount", "token_amount"])
    bars = build_bars(df, bar_size="1h", end_date=pd.Timestamp("2024-01-01", tz="UTC"))
    assert bars.empty
