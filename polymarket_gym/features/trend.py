from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from polymarket_gym.features.registry import register

@register
@dataclass
class MACD:
    """Moving Average Convergence Divergence."""
    fast: int = 12
    slow: int = 26
    signal: int = 9
    name: str = "macd"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        closes = history_df["close"].astype(float)
        exp1 = closes.ewm(span=self.fast, adjust=False).mean()
        exp2 = closes.ewm(span=self.slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        
        # Normalize by price
        window_df["macd_line"] = (macd.reindex(window_df.index) / window_df["close"]).fillna(0.0).astype(np.float32)
        window_df["macd_signal"] = (signal.reindex(window_df.index) / window_df["close"]).fillna(0.0).astype(np.float32)
        return window_df
