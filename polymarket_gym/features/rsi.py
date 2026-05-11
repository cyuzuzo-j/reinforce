from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from polymarket_gym.features.registry import register

@register
@dataclass
class RSI:
    """Relative Strength Index (RSI) indicator."""
    window: int = 14
    name: str = "rsi"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        closes = history_df["close"].astype(float)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window, min_periods=1).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0) # Neutral RSI for warm-up
        
        window_df["rsi"] = (rsi.reindex(window_df.index).to_numpy() / 100.0).astype(np.float32)
        return window_df
