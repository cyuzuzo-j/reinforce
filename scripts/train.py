"""Train PPO on PolymarketDirectional-v0 with periodic visualization.

Example:
    python scripts/train.py --data-dir data/ --total-timesteps 500000 \
        --n-envs 4 --viz-every-n-episodes 50 --out-dir runs/ppo_demo/
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import numpy as np
import wandb
from functools import partial
from sbx import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from wandb.integration.sb3 import WandbCallback

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import polymarket_gym  # noqa: F401 — registers env
from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader
from polymarket_gym.policy import FlaxPolicyFeatures
from polymarket_gym.training.callbacks import (
    EpisodeCounterCallback,
    StepRewardLoggerCallback,
    VisualizationCallback,
)
from polymarket_gym.training.env_factory import make_env, make_vec_env
from polymarket_gym.training.splits import chronological_split

logger = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--markets-file", type=str, default="markets.parquet")
    p.add_argument("--quant-file", type=str, default="quant_sample.parquet")
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--viz-every-n-episodes", type=int, default=50)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"runs/ppo_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bar-size", type=str, default="1h")
    p.add_argument("--lookback", type=int, default=32)
    p.add_argument("--min-bars-per-episode", type=int, default=64)
    p.add_argument("--initial-cash", type=float, default=1_000.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--invalid-action-penalty", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=0.00017994879854930408)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.9648031979222836)
    p.add_argument("--gae-lambda", type=float, default=0.8655029100614116)
    p.add_argument("--ent-coef", type=float, default=0.09587668946325116)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--n-epochs", type=int, default=15)
    p.add_argument("--features-dim", type=int, default=64)
    p.add_argument("--net-arch-pi", type=int, nargs="+", default=[128, 64])
    p.add_argument("--net-arch-vf", type=int, nargs="+", default=[128, 128])
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv")
    p.add_argument("--extra-features", type=str, nargs="+", default=[])
    p.add_argument(
        "--wandb-project",
        type=str,
        default="polymarket-rl",
        help="Weights & Biases project name",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    markets_path = args.data_dir / args.markets_file
    quant_path = args.data_dir / args.quant_file
    if not markets_path.exists() or not quant_path.exists():
        logger.error(
            "missing data files: %s, %s — run scripts/build_sample.py first",
            markets_path,
            quant_path,
        )
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    (out_dir / "monitor").mkdir(exist_ok=True)

    # ── Weights & Biases ─────────────────────────────────────────────────
    run_config = {
        # PPO hypers
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "n_epochs": args.n_epochs,
        # Architecture
        "features_dim": args.features_dim,
        "net_arch_pi": args.net_arch_pi,
        "net_arch_vf": args.net_arch_vf,
        "cnn_channels": args.cnn_channels,
        # Environment
        "bar_size": args.bar_size,
        "lookback": args.lookback,
        "min_bars_per_episode": args.min_bars_per_episode,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
        "invalid_action_penalty": args.invalid_action_penalty,
        # Training
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=run_config,
        sync_tensorboard=False,
        save_code=True,
    )
    is_sweep = bool(getattr(run, "sweep_id", None))

    # Allow wandb sweep to override CLI args
    wc = wandb.config
    args.learning_rate = wc.get("learning_rate", args.learning_rate)
    args.n_steps = wc.get("n_steps", args.n_steps)
    args.batch_size = wc.get("batch_size", args.batch_size)
    args.gamma = wc.get("gamma", args.gamma)
    args.gae_lambda = wc.get("gae_lambda", args.gae_lambda)
    args.ent_coef = wc.get("ent_coef", args.ent_coef)
    args.clip_range = wc.get("clip_range", args.clip_range)
    args.n_epochs = wc.get("n_epochs", args.n_epochs)
    args.features_dim = wc.get("features_dim", args.features_dim)
    args.net_arch_pi = wc.get("net_arch_pi", args.net_arch_pi)
    args.net_arch_vf = wc.get("net_arch_vf", args.net_arch_vf)
    args.cnn_channels = wc.get("cnn_channels", args.cnn_channels)

    cfg = EnvConfig(
        bar_size=args.bar_size,
        lookback=args.lookback,
        min_bars_per_episode=args.min_bars_per_episode,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        invalid_action_penalty=args.invalid_action_penalty,
        seed=args.seed,
        extra_features=tuple(args.extra_features),
    )

    loader = MarketLoader(markets_path, quant_path)
    try:
        train_ids, eval_ids = chronological_split(loader, cfg, eval_frac=args.eval_frac)
    finally:
        loader.close()
    logger.info("split: %d train markets, %d eval markets", len(train_ids), len(eval_ids))

    train_env = make_vec_env(
        markets_path,
        quant_path,
        cfg,
        train_ids,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=out_dir / "monitor",
        subproc=args.subproc,
    )
    eval_env = make_vec_env(
        markets_path,
        quant_path,
        cfg,
        eval_ids,
        n_envs=1,
        seed=args.seed + 10_000,
        monitor_dir=out_dir / "monitor_eval",
    )

    # Since the env is now wrapped with FlattenObservation, observation space is a Box.
    # The first 4 + len(extra_features) elements are the scalars.
    n_scalars = 4 + len(args.extra_features)
    n_window_features = 7 + len(args.extra_features)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        policy_kwargs={
            "features_extractor_class": partial(
                FlaxPolicyFeatures,
                lookback=args.lookback,
                n_window_features=n_window_features,
                n_scalars=n_scalars,
                cnn_channels=args.cnn_channels,
            ),
            "features_extractor_kwargs": {
                "features_dim": args.features_dim,
            },
            "net_arch": {"pi": args.net_arch_pi, "vf": args.net_arch_vf},
        },
        seed=args.seed,
        verbose=1,
    )

    counter = EpisodeCounterCallback()
    viz_env_thunk = make_env(
        markets_path,
        quant_path,
        cfg,
        eval_ids,
        seed=args.seed + 99_999,
        monitor_dir=None,
    )
    viz_cb = VisualizationCallback(
        eval_env_fn=viz_env_thunk,
        every_n_episodes=args.viz_every_n_episodes,
        out_dir=out_dir / "viz",
        counter=counter,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(out_dir / "ckpt"),
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "ckpt"),
        log_path=str(out_dir / "eval_log"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        deterministic=True,
        n_eval_episodes=args.n_eval_episodes,
    )
    # In a sweep, skip model artifact uploads — they balloon storage for
    # little value when only the metric matters.
    wandb_cb = WandbCallback(
        model_save_path=None if is_sweep else str(out_dir / "ckpt"),
        model_save_freq=0 if is_sweep else max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        verbose=1,
    )
    step_log_cb = StepRewardLoggerCallback(log_every=1)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([counter, viz_cb, ckpt_cb, eval_cb, wandb_cb, step_log_cb]),
        progress_bar=False,
    )
    model.save(str(out_dir / "final_model.zip"))
    logger.info("training complete; model saved to %s", out_dir / "final_model.zip")

    # Log final reward summary so the sweep optimiser has a guaranteed value
    # even if the last rollout dump was suppressed by early termination.
    try:
        ep_buf = list(getattr(model, "ep_info_buffer", []) or [])
        if ep_buf:
            returns = [float(ep.get("r", 0.0)) for ep in ep_buf]
            lengths = [float(ep.get("l", 0.0)) for ep in ep_buf]
            final_metrics = {
                "final/ep_rew_mean": float(np.mean(returns)),
                "final/ep_rew_last": float(returns[-1]),
                "final/ep_len_mean": float(np.mean(lengths)),
                "final/n_episodes": len(returns),
            }
            wandb.log(final_metrics, step=model.num_timesteps)
            for k, v in final_metrics.items():
                run.summary[k] = v
    except Exception as e:
        logger.warning("failed to log final reward summary: %s", e)

    if not is_sweep:
        # Log final model as wandb artifact (skipped for sweeps).
        artifact = wandb.Artifact("ppo-model", type="model")
        artifact.add_file(str(out_dir / "final_model.zip"))
        run.log_artifact(artifact)

    train_env.close()
    eval_env.close()
    run.finish()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
