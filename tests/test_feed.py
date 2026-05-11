from __future__ import annotations

import numpy as np
import pytest

from polymarket_gym.feed import HistoricalFeed, InMemoryFeed, LiveFeed, MarketFeed


def test_historical_feed_eligible_market_ids(make_historical_feed):
    feed = make_historical_feed()
    ids = feed.eligible_market_ids()
    assert set(ids) == {"mkt-yes", "mkt-no"}


def test_historical_feed_replays_in_order(make_historical_feed):
    feed = make_historical_feed()
    feed.reset(market_id="mkt-yes", rng=np.random.default_rng(0))
    last_ts = None
    while (bar := feed.advance()) is not None:
        if last_ts is not None:
            assert bar.ts > last_ts
        last_ts = bar.ts


def test_historical_feed_history_is_causal(make_historical_feed, tiny_cfg):
    feed = make_historical_feed()
    feed.reset(market_id="mkt-yes", rng=np.random.default_rng(0))
    initial = feed.history()
    assert len(initial) == tiny_cfg.lookback
    next_bar = feed.advance()
    history_after = feed.history()
    assert next_bar is not None
    assert len(history_after) == tiny_cfg.lookback + 1
    assert history_after[-1].ts == next_bar.ts


def test_historical_feed_settlement_price_after_exhaustion(make_historical_feed):
    feed = make_historical_feed()
    feed.reset(market_id="mkt-yes", rng=np.random.default_rng(0))
    assert feed.settlement_price() is None
    while feed.advance() is not None:
        pass
    assert feed.is_resolved()
    assert feed.settlement_price() == pytest.approx(1.0)


def test_historical_feed_settlement_for_no_market(make_historical_feed):
    feed = make_historical_feed()
    feed.reset(market_id="mkt-no", rng=np.random.default_rng(0))
    while feed.advance() is not None:
        pass
    assert feed.settlement_price() == pytest.approx(0.0)


def test_historical_feed_protocol_conformance(make_historical_feed):
    feed = make_historical_feed()
    assert isinstance(feed, MarketFeed)


def test_in_memory_feed_protocol_conformance(in_memory_feed_factory):
    from tests.conftest import make_constant_bars

    feed = in_memory_feed_factory(make_constant_bars(50))
    assert isinstance(feed, MarketFeed)


def test_live_feed_is_a_stub():
    feed = LiveFeed(market_id="fake", cfg=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        feed.reset(market_id=None, rng=np.random.default_rng(0))
