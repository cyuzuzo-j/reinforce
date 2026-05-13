from __future__ import annotations

import pytest

from polymarket_gym.config import EnvConfig
from polymarket_gym.execution import PolymarketCLOBVenue, SimulatedVenue
from polymarket_gym.feed import Bar


def _bar(open_p: float = 0.5, close_p: float | None = None, volume_usd: float = 1e9) -> Bar:
    """High-volume default bar so spread/impact are negligible (≈ fee_rate only paths)."""
    close_p = close_p if close_p is not None else open_p
    return Bar(
        ts=None,  # type: ignore[arg-type]
        open=open_p,
        high=max(open_p, close_p),
        low=min(open_p, close_p),
        close=close_p,
        volume_usd=volume_usd,
        volume_tokens=0.0,
        n_trades=0,
        hl_range=abs(open_p - close_p),
        rv=0.0,
    )


# A bar large enough that half_spread floors at min_spread_bps and impact is tiny.
def _zero_impact_cfg(**kw) -> EnvConfig:
    return EnvConfig(min_spread_bps=0.0, spread_vol_factor=0.0, impact_factor=0.0, **kw)


def test_buy_full_yes_at_open():
    cfg = _zero_impact_cfg()
    venue = SimulatedVenue()
    fill = venue.submit(
        target_frac=1.0, next_bar=_bar(open_p=0.5),
        yes_tokens=0.0, no_tokens=0.0, cash=1_000.0, cfg=cfg,
    )
    assert fill.side == "yes"
    assert fill.fill_price == pytest.approx(0.5)
    assert fill.yes_delta == pytest.approx(2_000.0)
    assert fill.no_delta == 0.0
    assert fill.cash_delta == pytest.approx(-1_000.0)


def test_buy_full_no_at_complementary_price():
    cfg = _zero_impact_cfg()
    venue = SimulatedVenue()
    fill = venue.submit(
        target_frac=-1.0, next_bar=_bar(open_p=0.3),
        yes_tokens=0.0, no_tokens=0.0, cash=1_000.0, cfg=cfg,
    )
    assert fill.side == "no"
    assert fill.fill_price == pytest.approx(0.7)
    assert fill.no_delta == pytest.approx(1_000.0 / 0.7)
    assert fill.cash_delta == pytest.approx(-1_000.0)


def test_flat_target_sells_existing_yes():
    cfg = _zero_impact_cfg()
    venue = SimulatedVenue()
    fill = venue.submit(
        target_frac=0.0, next_bar=_bar(open_p=0.5),
        yes_tokens=200.0, no_tokens=0.0, cash=0.0, cfg=cfg,
    )
    assert fill.yes_delta == pytest.approx(-200.0)
    assert fill.cash_delta == pytest.approx(100.0)


def test_side_switch_closes_then_opens():
    cfg = _zero_impact_cfg()
    venue = SimulatedVenue()
    fill = venue.submit(
        target_frac=-1.0, next_bar=_bar(open_p=0.5),
        yes_tokens=100.0, no_tokens=0.0, cash=0.0, cfg=cfg,
    )
    # 100 YES sold for $50, then $50 buys 100 NO at price (1-0.5)=0.5
    assert fill.yes_delta == pytest.approx(-100.0)
    assert fill.no_delta == pytest.approx(100.0)


def test_no_target_with_no_funds_is_noop():
    cfg = _zero_impact_cfg()
    venue = SimulatedVenue()
    fill = venue.submit(
        target_frac=1.0, next_bar=_bar(open_p=0.5),
        yes_tokens=0.0, no_tokens=0.0, cash=0.0, cfg=cfg,
    )
    assert fill.yes_delta == 0.0 and fill.no_delta == 0.0


def test_impact_scales_with_size_over_volume():
    """Larger trade-to-volume ratio pays a higher effective price."""
    cfg = EnvConfig(min_spread_bps=0.0, spread_vol_factor=0.0, impact_factor=0.5)
    venue = SimulatedVenue()
    high_vol = venue.submit(
        target_frac=1.0, next_bar=_bar(open_p=0.5, volume_usd=100_000.0),
        yes_tokens=0.0, no_tokens=0.0, cash=1_000.0, cfg=cfg,
    )
    low_vol = venue.submit(
        target_frac=1.0, next_bar=_bar(open_p=0.5, volume_usd=1_000.0),
        yes_tokens=0.0, no_tokens=0.0, cash=1_000.0, cfg=cfg,
    )
    assert low_vol.fill_price > high_vol.fill_price
    assert low_vol.fee_paid > high_vol.fee_paid


def test_clob_venue_is_a_stub():
    cfg = EnvConfig()
    venue = PolymarketCLOBVenue()
    with pytest.raises(NotImplementedError):
        venue.submit(
            target_frac=1.0, next_bar=_bar(),
            yes_tokens=0.0, no_tokens=0.0, cash=1.0, cfg=cfg,
        )
