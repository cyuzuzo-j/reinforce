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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

import polymarket_gym  # noqa: F401 — registers env
from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader
from polymarket_gym.policy import PolicyFeatures
from polymarket_gym.training.callbacks import (
    EpisodeCounterCallback,
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
    p.add_argument("--n-envs", type=int, default=4)
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
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv")
    p.add_argument("--extra-features", type=str, nargs="+", default=[])
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
    (out_dir / "tb").mkdir(exist_ok=True)
    (out_dir / "monitor").mkdir(exist_ok=True)

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

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        policy_kwargs={
            "features_extractor_class": PolicyFeatures,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        tensorboard_log=str(out_dir / "tb"),
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
        n_eval_episodes=3,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([counter, viz_cb, ckpt_cb, eval_cb]),
        progress_bar=False,
    )
    model.save(str(out_dir / "final_model.zip"))
    logger.info("training complete; model saved to %s", out_dir / "final_model.zip")
    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
