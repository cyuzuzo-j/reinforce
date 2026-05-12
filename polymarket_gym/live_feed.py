"""Live Polymarket feed: REST bootstrap + CLOB WS for go-forward bars.

The market-data WS (``wss://ws-subscriptions-clob.polymarket.com/ws/market``)
publishes ``book``, ``price_change``, and ``last_trade_price`` events for a
subscribed asset_id. Public trades (with size/fees) are not always emitted on
this channel — we fall back to the midpoint from ``price_change`` to drive bar
construction, and treat any ``last_trade_price`` events that do arrive as
trade fills.

Historical bootstrap uses the public ``data-api.polymarket.com/trades`` REST
endpoint, which exposes per-trade size/price/timestamp for the YES side.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import websockets

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import build_bars
from polymarket_gym.feed import Bar, MarketMeta, _bars_df_to_list

_log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class PolymarketMarketInfo:
    market_id: str  # conditionId, e.g. "0x0b4cc3b…"
    question: str
    end_date: pd.Timestamp
    yes_token_id: str
    no_token_id: str
    yes_payoff: Optional[float]  # None until the market resolves


def fetch_open_market(
    min_volume_24h: float = 100_000.0,
    limit: int = 100,
    price_range: tuple[float, float] = (0.15, 0.85),
) -> PolymarketMarketInfo:
    """Pick the highest-24h-volume open binary market within ``price_range``.

    The price filter avoids degenerate "already-decided" markets where YES sits
    at ~0.001 or ~0.999; the trained policy can't do much in those.
    """
    r = requests.get(
        f"{GAMMA_BASE}/markets",
        params={
            "closed": "false",
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
        },
        timeout=15,
    )
    r.raise_for_status()
    lo, hi = price_range
    for m in r.json():
        if not m.get("enableOrderBook"):
            continue
        if not m.get("acceptingOrders", True):
            continue
        outcomes_raw = m.get("outcomes")
        if isinstance(outcomes_raw, str):
            outcomes = json.loads(outcomes_raw)
        else:
            outcomes = outcomes_raw or []
        if len(outcomes) != 2:
            continue
        tok_raw = m.get("clobTokenIds")
        if not tok_raw:
            continue
        token_ids = json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
        if len(token_ids) != 2:
            continue
        if float(m.get("volume24hr", 0) or 0) < min_volume_24h:
            continue
        op_raw = m.get("outcomePrices")
        if op_raw is None:
            continue
        try:
            op = json.loads(op_raw) if isinstance(op_raw, str) else op_raw
            yes_price = float(op[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not (lo <= yes_price <= hi):
            continue
        return PolymarketMarketInfo(
            market_id=str(m["conditionId"]),
            question=str(m.get("question", "")),
            end_date=pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"),
            yes_token_id=str(token_ids[0]),
            no_token_id=str(token_ids[1]),
            yes_payoff=None,
        )
    raise RuntimeError(
        f"no open markets matched "
        f"(limit={limit}, min_volume_24h={min_volume_24h:.0f}, "
        f"price_range={price_range})"
    )


def fetch_market_by_condition_id(condition_id: str) -> PolymarketMarketInfo:
    r = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"condition_ids": condition_id},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise RuntimeError(f"market {condition_id!r} not found on Gamma")
    m = rows[0]
    tok_raw = m["clobTokenIds"]
    token_ids = json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
    return PolymarketMarketInfo(
        market_id=str(m["conditionId"]),
        question=str(m.get("question", "")),
        end_date=pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"),
        yes_token_id=str(token_ids[0]),
        no_token_id=str(token_ids[1]),
        yes_payoff=None,
    )


def fetch_historical_trades(
    condition_id: str,
    *,
    yes_token_id: str | None = None,
    min_hours_back: float = 48.0,
    max_pages: int = 100,
    page_size: int = 500,
) -> pd.DataFrame:
    """Page back through data-api ``/trades`` until trades span ``min_hours_back``.

    Returns a DataFrame with columns ``datetime``, ``price``, ``usd_amount``,
    ``token_amount`` — the schema expected by ``build_bars``.

    If ``yes_token_id`` is given, restricts to trades whose ``asset`` matches
    (i.e. YES-side trades), otherwise converts NO-side prices to YES via
    ``1 - p`` so the resulting series is on the YES side.
    """
    now_ts = time.time()
    cutoff_ts = now_ts - min_hours_back * 3600.0
    rows: list[dict] = []
    oldest_ts = now_ts

    for page in range(max_pages):
        r = requests.get(
            f"{DATA_API_BASE}/trades",
            params={
                "market": condition_id,
                "limit": page_size,
                "offset": page * page_size,
            },
            timeout=20,
        )
        if r.status_code == 400:
            # data-api caps pagination at offset=3000; stop gracefully.
            _log.info(
                "data-api offset cap hit at page %d (%s); stopping pagination",
                page,
                r.text[:100],
            )
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for t in batch:
            try:
                ts = int(t["timestamp"])
                price = float(t["price"])
                size = float(t["size"])
                asset = str(t.get("asset", ""))
            except (KeyError, TypeError, ValueError):
                continue
            if yes_token_id is not None:
                # Convert NO-side trades to YES-equivalent price (complement).
                if asset != yes_token_id:
                    price = 1.0 - price
            rows.append(
                {
                    "datetime": pd.to_datetime(ts, unit="s", utc=True),
                    "price": price,
                    "usd_amount": size * price,
                    "token_amount": size,
                }
            )
            oldest_ts = min(oldest_ts, ts)
        if oldest_ts <= cutoff_ts:
            break
        if len(batch) < page_size:
            break

    if not rows:
        return pd.DataFrame(
            columns=["datetime", "price", "usd_amount", "token_amount"]
        )
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Tick:
    ts: pd.Timestamp
    price: float
    volume_usd: float = 0.0
    volume_tokens: float = 0.0
    is_trade: bool = False  # True if from a real trade event, False if midpoint


class WebsocketLiveFeed:
    """Live Polymarket feed (MarketFeed protocol).

    Bootstrapping: REST trades → ``build_bars`` → drop the in-flight bar.
    Live: WS in a daemon thread maintains best_bid/best_ask (and any
    ``last_trade_price`` events). ``advance()`` blocks until the next bar
    boundary, then emits an OHLCV bar synthesized from ticks accumulated in
    that window.

    Note: the public market WS does not always emit per-trade events; when no
    trade ticks land in a window, the bar's volume features are zero and
    OHLC are derived from the midpoint stream.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        info: PolymarketMarketInfo,
        *,
        bootstrap_hours_back: float = 48.0,
        bar_close_timeout_sec: float = 30 * 60.0,
    ) -> None:
        self._cfg = cfg
        self._info = info
        self._bootstrap_hours_back = bootstrap_hours_back
        self._bar_close_timeout_sec = bar_close_timeout_sec

        self._bars: list[Bar] = []
        self._cursor: int = 0
        self._meta: MarketMeta | None = None

        self._tick_queue: queue.Queue[_Tick] = queue.Queue()
        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._last_known_price: float | None = None

        self._best_bid: float | None = None
        self._best_ask: float | None = None
        self._bid_lock = threading.Lock()

    # ── MarketFeed protocol ────────────────────────────────────────────────

    def eligible_market_ids(self) -> list[str]:
        return [self._info.market_id]

    def reset(
        self, *, market_id: str | None, rng: np.random.Generator
    ) -> MarketMeta:
        if market_id is not None and market_id != self._info.market_id:
            raise ValueError(
                f"WebsocketLiveFeed serves only {self._info.market_id!r}, "
                f"not {market_id!r}"
            )
        _log.info(
            "bootstrap: fetching trades for %s (~%.0fh back)",
            self._info.market_id,
            self._bootstrap_hours_back,
        )
        trades = fetch_historical_trades(
            self._info.market_id,
            yes_token_id=self._info.yes_token_id,
            min_hours_back=self._bootstrap_hours_back,
        )
        if trades.empty:
            raise RuntimeError(
                f"no historical trades returned for {self._info.market_id!r}"
            )

        bar_size_offset = pd.tseries.frequencies.to_offset(self._cfg.bar_size)
        now = pd.Timestamp.now(tz="UTC")
        current_bar_open = now.floor(self._cfg.bar_size)
        # Build bars only over fully-completed windows: end_date = current_bar_open.
        bars_df = build_bars(
            trades,
            self._cfg.bar_size,
            end_date=current_bar_open,
            price_eps=self._cfg.price_eps,
        )
        # Drop any in-flight bar that snuck in (open == current_bar_open).
        if not bars_df.empty and bars_df.index[-1] >= current_bar_open:
            bars_df = bars_df.iloc[:-1]
        if len(bars_df) < self._cfg.lookback:
            raise RuntimeError(
                f"only {len(bars_df)} complete bars from "
                f"{len(trades)} trades — need >= lookback={self._cfg.lookback}. "
                f"Increase --bootstrap-hours-back or pick a more active market."
            )

        self._bars = _bars_df_to_list(bars_df)
        # Cursor points one past the last bootstrapped bar so the initial
        # observation uses the full bootstrap history.
        self._cursor = len(self._bars)
        if self._bars:
            self._last_known_price = self._bars[-1].close

        self._meta = MarketMeta(
            market_id=self._info.market_id,
            question=self._info.question,
            end_date=self._info.end_date,
            yes_payoff=self._info.yes_payoff,
            n_bars=len(self._bars),
        )

        # Drain any pre-existing items in the queue (defensive).
        with self._tick_queue.mutex:
            self._tick_queue.queue.clear()

        self._start_ws_thread()
        _log.info(
            "bootstrap complete: %d bars (last=%s, close=%.4f)",
            len(self._bars),
            self._bars[-1].ts,
            self._bars[-1].close,
        )
        return self._meta

    def history(self) -> list[Bar]:
        return self._bars[: self._cursor]

    def advance(self) -> Bar | None:
        """Block until the next bar boundary closes, then emit it.

        Returns ``None`` if the timeout fires before the boundary or if a
        terminal condition is reached (currently never — live markets only
        terminate when the user stops the loop).
        """
        if not self._bars:
            return None
        bar_size_offset = pd.tseries.frequencies.to_offset(self._cfg.bar_size)
        # Next window opens immediately after the last bar's open time.
        window_start = self._bars[-1].ts + bar_size_offset
        window_end = window_start + bar_size_offset

        deadline = window_end + pd.Timedelta(seconds=2)  # tiny grace for clock skew
        max_deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(
            seconds=self._bar_close_timeout_sec
        )
        if deadline > max_deadline:
            _log.warning(
                "next bar closes at %s but timeout deadline is %s — will return early",
                deadline,
                max_deadline,
            )
            deadline = max_deadline

        ticks: list[_Tick] = []
        while True:
            now = pd.Timestamp.now(tz="UTC")
            if now >= deadline:
                break
            remaining = (deadline - now).total_seconds()
            try:
                tick = self._tick_queue.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if tick.ts < window_start:
                # Stale tick from before this window — ignore.
                continue
            if tick.ts >= window_end:
                # Tick belongs to the next window; push back and stop.
                # (No urgency for ordering; the queue is FIFO enough for ticks.)
                self._tick_queue.put(tick)
                break
            ticks.append(tick)

        bar = self._build_bar(window_start, ticks)
        self._bars.append(bar)
        self._cursor += 1
        self._last_known_price = bar.close
        if self._meta is not None:
            self._meta = MarketMeta(
                market_id=self._meta.market_id,
                question=self._meta.question,
                end_date=self._meta.end_date,
                yes_payoff=self._meta.yes_payoff,
                n_bars=len(self._bars),
            )
        return bar

    def is_resolved(self) -> bool:
        return False

    def settlement_price(self) -> float | None:
        return None

    def close(self) -> None:
        self._ws_stop.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=5)

    # ── internals ──────────────────────────────────────────────────────────

    def _build_bar(self, window_start: pd.Timestamp, ticks: list[_Tick]) -> Bar:
        if ticks:
            prices = np.array([t.price for t in ticks], dtype=np.float64)
            prices = np.clip(
                prices, self._cfg.price_eps, 1.0 - self._cfg.price_eps
            )
            log_rets = np.diff(np.log(prices))
            rv = float(np.nanstd(log_rets)) if log_rets.size else 0.0
            trade_ticks = [t for t in ticks if t.is_trade]
            return Bar(
                ts=window_start,
                open=float(prices[0]),
                high=float(prices.max()),
                low=float(prices.min()),
                close=float(prices[-1]),
                volume_usd=float(sum(t.volume_usd for t in trade_ticks)),
                volume_tokens=float(sum(t.volume_tokens for t in trade_ticks)),
                n_trades=len(trade_ticks),
                hl_range=float(prices.max() - prices.min()),
                rv=rv,
            )
        # No ticks landed in the window — carry the last known price forward.
        close = self._last_known_price or self._bars[-1].close
        return Bar(
            ts=window_start,
            open=close,
            high=close,
            low=close,
            close=close,
            volume_usd=0.0,
            volume_tokens=0.0,
            n_trades=0,
            hl_range=0.0,
            rv=0.0,
        )

    def _start_ws_thread(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

    def _ws_loop(self) -> None:
        try:
            asyncio.run(self._ws_main())
        except Exception:  # pragma: no cover
            _log.exception("ws loop crashed")

    async def _ws_main(self) -> None:
        sub_payload = json.dumps(
            {"type": "market", "assets_ids": [self._info.yes_token_id]}
        )
        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(
                    CLOB_WS, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    await ws.send(sub_payload)
                    _log.info("ws connected; subscribed to %s", self._info.yes_token_id)
                    while not self._ws_stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            continue
                        self._handle_ws_message(raw)
            except Exception as e:
                if self._ws_stop.is_set():
                    return
                _log.warning("ws disconnected (%s) — reconnecting in 5s", e)
                await asyncio.sleep(5)

    def _handle_ws_message(self, raw) -> None:
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            parsed = json.loads(raw)
        except Exception:
            return
        events = parsed if isinstance(parsed, list) else [parsed]
        now = pd.Timestamp.now(tz="UTC")
        for ev in events:
            ev_type = ev.get("event_type") or ev.get("type") or ""
            if ev_type == "book":
                bb, ba = _best_from_book(ev)
                self._update_book(bb, ba, now)
            elif ev_type == "price_change":
                changes = ev.get("price_changes") or []
                for ch in changes:
                    if str(ch.get("asset_id")) != self._info.yes_token_id:
                        continue
                    try:
                        bb = float(ch["best_bid"]) if ch.get("best_bid") else None
                        ba = float(ch["best_ask"]) if ch.get("best_ask") else None
                    except (TypeError, ValueError):
                        continue
                    self._update_book(bb, ba, now)
            elif ev_type in ("last_trade_price", "trade"):
                asset = str(ev.get("asset_id") or ev.get("asset") or "")
                if asset != self._info.yes_token_id:
                    continue
                try:
                    price = float(ev["price"])
                    size = float(ev.get("size", 0.0) or 0.0)
                    ts_raw = ev.get("timestamp") or ev.get("created_at")
                    ts = _parse_ts(ts_raw, fallback=now)
                except (KeyError, TypeError, ValueError):
                    continue
                self._tick_queue.put(
                    _Tick(
                        ts=ts,
                        price=price,
                        volume_usd=size * price,
                        volume_tokens=size,
                        is_trade=True,
                    )
                )
                self._last_known_price = price

    def _update_book(
        self,
        best_bid: float | None,
        best_ask: float | None,
        now: pd.Timestamp,
    ) -> None:
        if best_bid is None and best_ask is None:
            return
        with self._bid_lock:
            if best_bid is not None:
                self._best_bid = best_bid
            if best_ask is not None:
                self._best_ask = best_ask
            bb, ba = self._best_bid, self._best_ask
        if bb is not None and ba is not None and ba >= bb:
            mid = 0.5 * (bb + ba)
            self._tick_queue.put(_Tick(ts=now, price=mid, is_trade=False))
            self._last_known_price = mid


def _best_from_book(ev: dict) -> tuple[float | None, float | None]:
    """Pull best bid/ask out of a CLOB ``book`` snapshot event."""
    bids = ev.get("bids") or []
    asks = ev.get("asks") or []
    best_bid = None
    best_ask = None
    try:
        if bids:
            best_bid = max(float(b["price"]) for b in bids if b.get("size"))
        if asks:
            best_ask = min(float(a["price"]) for a in asks if a.get("size"))
    except (TypeError, ValueError):
        return None, None
    return best_bid, best_ask


def _parse_ts(raw, *, fallback: pd.Timestamp) -> pd.Timestamp:
    if raw is None:
        return fallback
    if isinstance(raw, (int, float)):
        # ms or s?
        if raw > 1e12:  # ms
            return pd.to_datetime(int(raw), unit="ms", utc=True)
        return pd.to_datetime(int(raw), unit="s", utc=True)
    if isinstance(raw, str):
        if raw.isdigit():
            v = int(raw)
            unit = "ms" if v > 1e12 else "s"
            return pd.to_datetime(v, unit=unit, utc=True)
        try:
            return pd.to_datetime(raw, utc=True)
        except Exception:
            return fallback
    return fallback
