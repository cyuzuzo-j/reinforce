from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader, build_bars

_log = logging.getLogger(__name__)

_MIN_STAGE_SIZE = 15  # floor: auto-lower conviction threshold if stage is smaller


@dataclass(frozen=True)
class CurriculumStages:
    stage1: list[str]  # easiest — high conviction
    stage2: list[str]  # medium conviction
    stage3: list[str]  # full training set
    stage1_conviction_used: float
    stage2_conviction_used: float


def curriculum_split(
    loader: MarketLoader,
    cfg: EnvConfig,
    train_ids: list[str],
    stage1_conviction: float = 0.35,
    stage2_conviction: float = 0.15,
) -> CurriculumStages:
    """Score each training market by directional conviction and return stage pools.

    Conviction = |yes_payoff - price at first observable bar (after lookback)|.
    Markets with higher conviction are 'easier' because a simple directional
    bet beats flat by a larger margin.

    If fewer than _MIN_STAGE_SIZE markets meet a threshold, the threshold is
    auto-lowered until the minimum is satisfied (or the full set is used).
    """
    t0 = time.monotonic()
    scores: dict[str, float] = {}

    for mid in train_ids:
        try:
            meta = loader.load_meta(mid)
            yes_payoff = meta.get("yes_payoff")
            if yes_payoff is None:
                continue
            trades = loader.load_trades(mid)
            bars_df = build_bars(
                trades,
                bar_size=cfg.bar_size,
                end_date=meta["end_date"],
                price_eps=cfg.price_eps,
            )
            if len(bars_df) <= cfg.lookback:
                continue
            initial_close = float(bars_df.iloc[cfg.lookback]["close"])
            scores[mid] = abs(yes_payoff - initial_close)
        except Exception as exc:
            _log.warning("curriculum_split: skipping market %s: %s", mid, exc)

    elapsed = time.monotonic() - t0
    _log.info(
        "curriculum_split: scored %d/%d markets in %.1fs",
        len(scores), len(train_ids), elapsed,
    )

    def _filter(threshold: float) -> list[str]:
        return [mid for mid in train_ids if scores.get(mid, 0.0) >= threshold]

    # Auto-lower stage1 threshold if too few markets qualify
    s1_thresh = stage1_conviction
    stage1_ids = _filter(s1_thresh)
    while len(stage1_ids) < _MIN_STAGE_SIZE and s1_thresh > 0.01:
        s1_thresh = round(s1_thresh - 0.05, 4)
        stage1_ids = _filter(s1_thresh)
    if not stage1_ids:
        stage1_ids = list(train_ids)
        s1_thresh = 0.0

    # Auto-lower stage2 threshold similarly; stage2 must be >= stage1
    s2_thresh = min(stage2_conviction, s1_thresh)
    stage2_ids = _filter(s2_thresh)
    while len(stage2_ids) < _MIN_STAGE_SIZE and s2_thresh > 0.01:
        s2_thresh = round(s2_thresh - 0.05, 4)
        stage2_ids = _filter(s2_thresh)
    if not stage2_ids:
        stage2_ids = list(train_ids)
        s2_thresh = 0.0

    if s1_thresh != stage1_conviction:
        _log.warning(
            "curriculum_split: stage1 threshold lowered %.2f → %.2f (%d markets)",
            stage1_conviction, s1_thresh, len(stage1_ids),
        )
    if s2_thresh != stage2_conviction:
        _log.warning(
            "curriculum_split: stage2 threshold lowered %.2f → %.2f (%d markets)",
            stage2_conviction, s2_thresh, len(stage2_ids),
        )

    _log.info(
        "curriculum stages: stage1=%d (conviction>=%.2f), stage2=%d (conviction>=%.2f), stage3=%d",
        len(stage1_ids), s1_thresh, len(stage2_ids), s2_thresh, len(train_ids),
    )
    return CurriculumStages(
        stage1=stage1_ids,
        stage2=stage2_ids,
        stage3=list(train_ids),
        stage1_conviction_used=s1_thresh,
        stage2_conviction_used=s2_thresh,
    )


def chronological_split(
    loader: MarketLoader,
    cfg: EnvConfig,
    eval_frac: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Split eligible markets into (train, eval) sorted ascending by end_date.

    The most recent `eval_frac` of markets go to eval. Both partitions are
    non-empty; raises RuntimeError otherwise.
    """
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac must be in (0,1), got {eval_frac}")

    ids = loader.eligible_market_ids(cfg)
    if not ids:
        raise RuntimeError("no eligible markets in loader")

    dated = []
    for mid in ids:
        meta = loader.load_meta(mid)
        dated.append((meta["end_date"], mid))
    dated.sort(key=lambda t: t[0])

    n = len(dated)
    n_eval = max(1, int(round(n * eval_frac)))
    n_train = n - n_eval
    if n_train < 1:
        raise RuntimeError(
            f"split leaves no train markets (n={n}, eval_frac={eval_frac})"
        )
    train_ids = [mid for _, mid in dated[:n_train]]
    eval_ids = [mid for _, mid in dated[n_train:]]
    return train_ids, eval_ids
