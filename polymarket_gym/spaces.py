from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import spaces

from polymarket_gym.config import EnvConfig
from polymarket_gym.feed import Bar
from polymarket_gym.features import apply_features, build as build_features

WINDOW_FEATURES = (
    "close",
    "log_return",
    "volume_usd_z",
    "volume_tokens_z",
    "n_trades_z",
    "hl_range",
    "rv",
)
N_WINDOW_FEATURES = len(WINDOW_FEATURES)

SCALAR_FEATURES = (
    "position_frac",
    "cash_frac",
    "portfolio_value_norm",
    "time_to_resolution_frac",
)
N_SCALAR_FEATURES = len(SCALAR_FEATURES)


def n_window_features(cfg: EnvConfig) -> int:
    return N_WINDOW_FEATURES + len(cfg.extra_features)


def build_observation_space(cfg: EnvConfig) -> spaces.Dict:
    return spaces.Dict(
        {
            "window": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.lookback, n_window_features(cfg)),
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_SCALAR_FEATURES,),
                dtype=np.float32,
            ),
        }
    )


def _causal_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score using stats computed only from the window itself (no episode-wide foresight)."""
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _window_from_history(history: list[Bar], cfg: EnvConfig) -> np.ndarray:
    n = cfg.lookback
    if len(history) < n:
        raise ValueError(
            f"history has {len(history)} bars, need at least lookback={n}"
        )
    window = history[-n:]
    closes = np.array([b.close for b in window], dtype=np.float64)
    prev_close = history[-n - 1].close if len(history) > n else window[0].close
    closes_for_ret = np.concatenate(([prev_close], closes))
    log_ret = np.log(closes_for_ret[1:]) - np.log(closes_for_ret[:-1])

    vol_usd = np.array([b.volume_usd for b in window], dtype=np.float64)
    vol_tok = np.array([b.volume_tokens for b in window], dtype=np.float64)
    n_trades = np.array([b.n_trades for b in window], dtype=np.float64)
    hl_range = np.array([b.hl_range for b in window], dtype=np.float64)
    rv = np.array([b.rv for b in window], dtype=np.float64)

    out = np.stack(
        [
            closes.astype(np.float32),
            log_ret.astype(np.float32),
            _causal_zscore(vol_usd),
            _causal_zscore(vol_tok),
            _causal_zscore(n_trades),
            hl_range.astype(np.float32),
            rv.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if cfg.extra_features:
        extra = _extra_features_block(history, cfg)
        out = np.concatenate([out, extra], axis=1)
    return out


def _history_to_df(history: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [b.open for b in history],
            "high": [b.high for b in history],
            "low": [b.low for b in history],
            "close": [b.close for b in history],
            "volume_usd": [b.volume_usd for b in history],
            "volume_tokens": [b.volume_tokens for b in history],
            "n_trades": [b.n_trades for b in history],
            "hl_range": [b.hl_range for b in history],
            "rv": [b.rv for b in history],
        },
        index=pd.Index([b.ts for b in history], name="ts"),
    )


def _extra_features_block(history: list[Bar], cfg: EnvConfig) -> np.ndarray:
    history_df = _history_to_df(history)
    window_df = history_df.tail(cfg.lookback)
    pipeline = build_features(list(cfg.extra_features))
    enriched = apply_features(pipeline, window_df, history_df)
    cols = list(cfg.extra_features)
    missing = [c for c in cols if c not in enriched.columns]
    if missing:
        raise ValueError(
            f"feature pipeline did not produce expected columns {missing}; "
            f"got {list(enriched.columns)}"
        )
    return enriched.loc[:, cols].to_numpy(dtype=np.float32, copy=True)


def pack_observation(
    history: list[Bar],
    *,
    position_tokens: float,
    cash: float,
    portfolio_value: float,
    bars_remaining: int,
    total_bars: int,
    cfg: EnvConfig,
) -> dict:
    window = _window_from_history(history, cfg)
    position_frac = 1.0 if position_tokens > 0.0 else 0.0
    cash_frac = float(cash / cfg.initial_cash) if cfg.initial_cash > 0 else 0.0
    pv_norm = float(portfolio_value / cfg.initial_cash) if cfg.initial_cash > 0 else 0.0
    if total_bars > 0:
        t_remaining = float(max(0, bars_remaining)) / float(total_bars)
    else:
        t_remaining = 0.0
    scalars = np.array(
        [position_frac, cash_frac, pv_norm, t_remaining], dtype=np.float32
    )
    return {"window": window, "scalars": scalars}
