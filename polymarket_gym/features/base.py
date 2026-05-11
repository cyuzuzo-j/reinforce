from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureTransform(Protocol):
    """A single, self-contained feature engineering step.

    Each implementation lives in its own module under ``polymarket_gym.features``
    and is responsible for one column (or one transform) only. Implementations
    must be pure: no module-level mutable state, no I/O, no edits to inputs.

    ``apply`` receives:
      - ``window_df``: the lookback slice the model will see (one row per bar).
      - ``history_df``: every bar observed so far (``history_df.tail(len(window_df))``
        is ``window_df``). Use it for causal stats that need more than the
        lookback window.

    It must return a new DataFrame with the same index as ``window_df`` and
    the original columns preserved, plus any added/modified columns.
    """

    name: str

    def apply(self, window_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame: ...


def _validate_output(
    name: str,
    window_df: pd.DataFrame,
    out: pd.DataFrame,
) -> pd.DataFrame:
    """Cheap invariants that catch the common ways a transform breaks isolation."""
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"feature {name!r} returned {type(out).__name__}, expected DataFrame")
    if len(out) != len(window_df):
        raise ValueError(
            f"feature {name!r} changed row count: {len(window_df)} -> {len(out)}"
        )
    if not out.index.equals(window_df.index):
        raise ValueError(f"feature {name!r} altered the window index")
    missing = set(window_df.columns) - set(out.columns)
    if missing:
        raise ValueError(f"feature {name!r} dropped columns: {sorted(missing)}")
    return out


def apply_features(
    features: list[FeatureTransform],
    window_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply features in order on copies so each step starts from a clean slate.

    Each transform sees the cumulative dataframe from prior steps but operates
    on a copy, so a buggy in-place edit cannot leak back into ``window_df`` or
    cross-contaminate sibling transforms during A/B testing.
    """
    cur = window_df.copy()
    hist = history_df.copy()
    for f in features:
        out = f.apply(cur.copy(), hist)
        cur = _validate_output(f.name, cur, out)
    return cur
