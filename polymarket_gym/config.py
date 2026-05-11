from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    bar_size: str = "1h"
    lookback: int = 32
    min_bars_per_episode: int = 64
    initial_cash: float = 1_000.0
    fee_bps: float = 10.0
    max_episode_steps: int | None = None
    terminal_settlement: bool = True
    invalid_action_penalty: float = 0.0
    price_eps: float = 1e-6
    seed: int | None = None
    extra_features: tuple[str, ...] = ()

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0
