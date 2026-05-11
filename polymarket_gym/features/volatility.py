from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from polymarket_gym.features.registry import register

@register
@dataclass
class BollingerBands:
    """Bollinger Bands normalized by current price."""
    window: int = 20
    num_std: float = 2.0
    name: str = "bollinger_bands"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        closes = history_df["close"].astype(float)
        sma = closes.rolling(window=self.window, min_periods=1).mean()
        std = closes.rolling(window=self.window, min_periods=1).std()
        
        upper = sma + (self.num_std * std)
        lower = sma - (self.num_std * std)
        
        # Normalize by current price to keep features scale-invariant
        window_df["bb_upper_norm"] = (upper.reindex(window_df.index) / window_df["close"]).fillna(1.0).astype(np.float32)
        window_df["bb_lower_norm"] = (lower.reindex(window_df.index) / window_df["close"]).fillna(1.0).astype(np.float32)
        return window_df

@register
@dataclass
class ATR:
    """Average True Range normalized by current price."""
    window: int = 14
    name: str = "atr"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        high = history_df["high"].astype(float)
        low = history_df["low"].astype(float)
        prev_close = history_df["close"].shift(1).astype(float)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.window, min_periods=1).mean()
        
        window_df["atr_norm"] = (atr.reindex(window_df.index) / window_df["close"]).fillna(0.0).astype(np.float32)
        return window_df
