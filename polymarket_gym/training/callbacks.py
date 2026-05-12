from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402


class StepRewardLoggerCallback(BaseCallback):
    """Logs per-step reward (and running episode return) to wandb every step.

    SB3's default logger only dumps once per rollout — for shorter trials or
    debugging this hides the moment-to-moment signal. This callback pushes a
    scalar to wandb on every env step (averaged across vec sub-envs).
    """

    def __init__(self, log_every: int = 1, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_every = max(1, int(log_every))
        self._step = 0
        self._ep_returns: np.ndarray | None = None

    def _on_step(self) -> bool:
        import wandb

        if wandb.run is None:
            return True

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None:
            return True

        rewards = np.asarray(rewards, dtype=np.float64)
        if self._ep_returns is None or self._ep_returns.shape != rewards.shape:
            self._ep_returns = np.zeros_like(rewards)
        self._ep_returns += rewards

        payload: dict[str, float] = {
            "step/reward_mean": float(rewards.mean()),
            "step/reward_sum": float(rewards.sum()),
        }
        if dones is not None:
            dones_arr = np.asarray(dones)
            if dones_arr.any():
                finished = self._ep_returns[dones_arr]
                payload["step/episode_return_last"] = float(finished[-1])
                payload["step/episode_return_mean"] = float(finished.mean())
                self._ep_returns[dones_arr] = 0.0

        self._step += 1
        if self._step % self._log_every == 0:
            wandb.log(payload, step=self.num_timesteps)
        return True


class EpisodeCounterCallback(BaseCallback):
    """Counts completed episodes across all sub-envs of a VecEnv."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.n_episodes: int = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode") is not None:
                self.n_episodes += 1
        return True


class VisualizationCallback(BaseCallback):
    """Every N completed training episodes, run a deterministic eval rollout
    and write a 2-panel PNG (rollout + training stats)."""

    def __init__(
        self,
        eval_env_fn: Callable[[], gym.Env],
        every_n_episodes: int,
        out_dir: str | Path,
        counter: EpisodeCounterCallback,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._eval_env_fn = eval_env_fn
        self._every = int(every_n_episodes)
        self._out_dir = Path(out_dir)
        self._counter = counter
        self._deterministic = deterministic
        self._last_trigger_bucket: int = 0

    def _init_callback(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        bucket = self._counter.n_episodes // self._every
        if self._every > 0 and bucket > self._last_trigger_bucket:
            self._last_trigger_bucket = bucket
            self._run_and_plot(episode_idx=self._counter.n_episodes)
        return True

    def _run_and_plot(self, episode_idx: int) -> None:
        env = self._eval_env_fn()
        try:
            obs, info = env.reset()
            market_id = info.get("market_id", "unknown")
            prices: list[float] = []
            actions: list[int] = []
            positions: list[float] = []
            pvs: list[float] = []
            rewards: list[float] = []
            fill_prices: list[float | None] = []

            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self._deterministic)
                action_int = int(np.asarray(action).item())
                obs, reward, terminated, truncated, info = env.step(action_int)
                done = bool(terminated or truncated)
                prices.append(float(info.get("bar_close", float("nan"))))
                actions.append(action_int)
                positions.append(float(info.get("position_tokens", 0.0)))
                pvs.append(float(info.get("pv", float("nan"))))
                rewards.append(float(reward))
                fp = info.get("last_fill_price")
                fill_prices.append(float(fp) if fp is not None else None)
        finally:
            env.close()

        ep_return = float(np.nansum(rewards))
        try:
            self.logger.record("eval/episode_return", ep_return)
            self.logger.record("eval/episode_length", len(rewards))
        except Exception:  # logger may not be ready in tests
            pass

        out_path = self._out_dir / f"ep_{episode_idx:06d}_{market_id}.png"
        self._render(out_path, market_id, prices, actions, positions, pvs, fill_prices, ep_return)

    def _render(
        self,
        out_path: Path,
        market_id: str,
        prices: list[float],
        actions: list[int],
        positions: list[float],
        pvs: list[float],
        fill_prices: list[float | None],
        ep_return: float,
    ) -> None:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]}
        )

        # --- top: eval rollout
        steps = np.arange(len(prices))
        ax_top.plot(steps, prices, color="steelblue", label="YES price", linewidth=1.2)

        in_position = np.array(positions) > 0.0
        if in_position.any():
            ax_top.fill_between(
                steps,
                np.nanmin(prices) if prices else 0.0,
                np.nanmax(prices) if prices else 1.0,
                where=in_position,
                color="green",
                alpha=0.08,
                label="long YES",
            )

        for i, (a, fp) in enumerate(zip(actions, fill_prices)):
            if fp is None:
                continue
            if a == 2:
                ax_top.scatter(i, fp, marker="^", color="green", s=60, zorder=5)
            elif a == 0:
                ax_top.scatter(i, fp, marker="v", color="red", s=60, zorder=5)

        ax_top.set_ylabel("price")
        ax_top.set_title(
            f"eval rollout — market {market_id} — return={ep_return:.2f}"
        )
        ax_top.grid(alpha=0.3)
        ax_top.legend(loc="upper left")

        ax_pv = ax_top.twinx()
        ax_pv.plot(steps, pvs, color="orange", linewidth=1.0, label="portfolio value")
        ax_pv.set_ylabel("PV", color="orange")
        ax_pv.tick_params(axis="y", labelcolor="orange")

        # --- bottom: training stats
        ep_buf = list(getattr(self.model, "ep_info_buffer", []) or [])
        if ep_buf:
            returns = [ep.get("r", 0.0) for ep in ep_buf]
            ax_bot.plot(returns, color="purple", label="train ep return")
            if len(returns) >= 10:
                w = min(50, len(returns))
                kernel = np.ones(w) / w
                rolling = np.convolve(returns, kernel, mode="valid")
                ax_bot.plot(
                    range(w - 1, w - 1 + len(rolling)),
                    rolling,
                    color="black",
                    linewidth=1.5,
                    label=f"rolling mean ({w})",
                )
        try:
            logged = dict(self.model.logger.name_to_value)
        except Exception:
            logged = {}
        loss_lines = []
        for key in ("train/policy_gradient_loss", "train/value_loss", "train/explained_variance"):
            if key in logged:
                loss_lines.append(f"{key.split('/')[-1]}={logged[key]:.4f}")
        if loss_lines:
            ax_bot.text(
                0.99,
                0.02,
                " | ".join(loss_lines),
                ha="right",
                va="bottom",
                transform=ax_bot.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
            )
        ax_bot.set_title("training stats (recent episodes)")
        ax_bot.set_xlabel("episode index in buffer")
        ax_bot.set_ylabel("return")
        ax_bot.grid(alpha=0.3)
        if ep_buf:
            ax_bot.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
