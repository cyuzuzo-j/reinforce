from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from polymarket_gym.features import register


@register
@dataclass
class SMAClose:
    window: int = 10
    name: str = "sma_close"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        closes = history_df["close"].astype(float)
        sma = closes.rolling(self.window, min_periods=1).mean()
        window_df["sma_close"] = sma.reindex(window_df.index).to_numpy()
        return window_df
