from __future__ import annotations

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader


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
