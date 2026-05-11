from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_gym.features.registry import register


@register
@dataclass
class LogReturnMomentum:
    """Sum of log-returns over the last ``lookback`` bars of full history."""

    lookback: int = 5
    name: str = "log_return_momentum"

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        closes = history_df["close"].astype(float).clip(lower=1e-12)
        log_ret = np.log(closes).diff().fillna(0.0)
        momentum = log_ret.rolling(self.lookback, min_periods=1).sum()
        window_df["log_return_momentum"] = momentum.reindex(window_df.index).to_numpy()
        return window_df
