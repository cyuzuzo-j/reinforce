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
    n_action_levels: int = 7        # Discrete(N): fracs evenly spaced in [0, 1]
    min_spread_bps: float = 50.0    # floor half-spread in bps (50 bps = 0.5%)
    spread_vol_factor: float = 2.0  # half_spread = factor / sqrt(volume_usd + 1)

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0
