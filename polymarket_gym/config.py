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
    invalid_action_penalty: float = 0.0  # retained for arg compat; unused
    price_eps: float = 1e-6
    seed: int | None = None
    extra_features: tuple[str, ...] = ()
    n_action_levels: int = 7        # per-side levels; total = 2N-1, centred at flat
    min_spread_bps: float = 50.0    # floor half-spread in bps (50 bps = 0.5%)
    spread_vol_factor: float = 2.0  # half_spread = factor / sqrt(volume_usd + 1)
    impact_factor: float = 0.5      # linear impact: cost += factor * notional / bar_volume
    impact_cap: float = 0.20        # cap impact contribution (same as spread cap)

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0

    @property
    def n_actions(self) -> int:
        return 2 * self.n_action_levels - 1

    @property
    def flat_action(self) -> int:
        return self.n_action_levels - 1

    @property
    def action_fracs(self) -> tuple[float, ...]:
        n = self.n_action_levels
        denom = max(n - 1, 1)
        return tuple((i - (n - 1)) / denom for i in range(2 * n - 1))
