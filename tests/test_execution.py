from __future__ import annotations

import pytest

from polymarket_gym.config import EnvConfig
from polymarket_gym.execution import PolymarketCLOBVenue, SimulatedVenue
from polymarket_gym.feed import Bar


def _bar(open_p: float = 0.5, close_p: float | None = None) -> Bar:
    close_p = close_p if close_p is not None else open_p
    return Bar(
        ts=None,  # type: ignore[arg-type]
        open=open_p,
        high=max(open_p, close_p),
        low=min(open_p, close_p),
        close=close_p,
        volume_usd=0.0,
        volume_tokens=0.0,
        n_trades=0,
        hl_range=abs(open_p - close_p),
        rv=0.0,
    )


def test_simulated_venue_buy_fill_price_includes_fee():
    cfg = EnvConfig(fee_bps=10.0)
    venue = SimulatedVenue()
    bar = _bar(open_p=0.5)
    fill = venue.submit(action=2, next_bar=bar, position_tokens=0.0, cash=1_000.0, cfg=cfg)
    assert fill.fill_price == pytest.approx(0.5 * (1.0 + cfg.fee_rate))
    assert fill.cash_delta == pytest.approx(-1_000.0)
    assert fill.tokens_delta == pytest.approx(1_000.0 / (0.5 * (1.0 + cfg.fee_rate)))
    assert fill.fee_paid > 0.0


def test_simulated_venue_sell_fill_price_includes_fee():
    cfg = EnvConfig(fee_bps=10.0)
    venue = SimulatedVenue()
    bar = _bar(open_p=0.5)
    fill = venue.submit(action=0, next_bar=bar, position_tokens=100.0, cash=0.0, cfg=cfg)
    assert fill.fill_price == pytest.approx(0.5 * (1.0 - cfg.fee_rate))
    assert fill.tokens_delta == pytest.approx(-100.0)
    assert fill.cash_delta == pytest.approx(100.0 * 0.5 * (1.0 - cfg.fee_rate))
    assert fill.fee_paid > 0.0


def test_simulated_venue_invalid_actions_are_noops():
    cfg = EnvConfig(fee_bps=10.0)
    venue = SimulatedVenue()
    bar = _bar(open_p=0.5)
    fill = venue.submit(action=0, next_bar=bar, position_tokens=0.0, cash=1_000.0, cfg=cfg)
    assert fill.tokens_delta == 0.0 and fill.cash_delta == 0.0 and fill.fee_paid == 0.0
    fill = venue.submit(action=2, next_bar=bar, position_tokens=10.0, cash=0.0, cfg=cfg)
    assert fill.tokens_delta == 0.0 and fill.cash_delta == 0.0 and fill.fee_paid == 0.0
    fill = venue.submit(action=1, next_bar=bar, position_tokens=0.0, cash=1_000.0, cfg=cfg)
    assert fill.tokens_delta == 0.0 and fill.cash_delta == 0.0 and fill.fee_paid == 0.0


def test_simulated_venue_roundtrip_loses_two_legs_of_fee():
    """Buy at open=p, sell at open=p → cash_final = initial * (1 - f) / (1 + f)."""
    cfg = EnvConfig(fee_bps=50.0)
    venue = SimulatedVenue()
    bar = _bar(open_p=0.5)
    f = cfg.fee_rate
    buy = venue.submit(action=2, next_bar=bar, position_tokens=0.0, cash=1_000.0, cfg=cfg)
    cash_after_buy = 1_000.0 + buy.cash_delta
    pos_after_buy = buy.tokens_delta
    sell = venue.submit(
        action=0, next_bar=bar, position_tokens=pos_after_buy, cash=cash_after_buy, cfg=cfg
    )
    cash_final = cash_after_buy + sell.cash_delta
    assert cash_final == pytest.approx(1_000.0 * (1.0 - f) / (1.0 + f))


def test_clob_venue_is_a_stub():
    cfg = EnvConfig()
    venue = PolymarketCLOBVenue()
    with pytest.raises(NotImplementedError):
        venue.submit(action=2, next_bar=_bar(), position_tokens=0.0, cash=1.0, cfg=cfg)
