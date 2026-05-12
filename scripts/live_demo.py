"""Run the best PPO checkpoint against live Polymarket data.

Picks an open binary market (or accepts ``--condition-id``), bootstraps
historical bars via REST, attaches a CLOB websocket subscription for live
ticks, and runs the trained policy at each bar boundary. Execution is
*simulated* via ``SimulatedVenue`` — no real orders are placed.

Example:
    python scripts/live_demo.py --duration-min 60
    python scripts/live_demo.py --condition-id 0xe9e3d24a… --bar-size 1m
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import polymarket_gym  # noqa: F401 — registers env
from polymarket_gym.config import EnvConfig
from polymarket_gym.env import PolymarketDirectionalEnv
from polymarket_gym.execution import SimulatedVenue
from polymarket_gym.live_feed import (
    WebsocketLiveFeed,
    fetch_market_by_condition_id,
    fetch_open_market,
)
from polymarket_gym.policy import FlaxPolicyFeatures

logger = logging.getLogger("live_demo")

ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY "}


def _default_ckpt() -> Path:
    """Latest ``runs/ppo_*/ckpt/best_model.zip``."""
    candidates = sorted(
        glob.glob(
            str(Path(__file__).resolve().parent / "runs" / "ppo_*" / "ckpt" / "best_model.zip")
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no best_model.zip under scripts/runs/ppo_*/ckpt/ — pass --ckpt"
        )
    return Path(candidates[-1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="path to a saved PPO model (default: latest runs/ppo_*/ckpt/best_model.zip)",
    )
    p.add_argument(
        "--condition-id",
        type=str,
        default=None,
        help="Polymarket conditionId (0x…) to trade. If omitted, auto-pick.",
    )
    p.add_argument("--min-volume-24h", type=float, default=200_000.0)
    p.add_argument("--price-min", type=float, default=0.15)
    p.add_argument("--price-max", type=float, default=0.85)
    p.add_argument("--bar-size", type=str, default="1h", help="must match training")
    p.add_argument("--lookback", type=int, default=32, help="must match training")
    p.add_argument("--bootstrap-hours-back", type=float, default=64.0)
    p.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="how many wall-clock minutes to run live before stopping",
    )
    p.add_argument(
        "--bar-close-timeout-sec",
        type=float,
        default=None,
        help="per-bar wall-clock timeout when waiting for the next boundary; "
        "defaults to duration-min * 60 so the outer duration cap is respected",
    )
    p.add_argument("--initial-cash", type=float, default=1_000.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ckpt = args.ckpt or _default_ckpt()
    if not ckpt.exists():
        logger.error("checkpoint not found: %s", ckpt)
        return 2
    logger.info("loading PPO checkpoint: %s", ckpt)

    info = (
        fetch_market_by_condition_id(args.condition_id)
        if args.condition_id
        else fetch_open_market(
            min_volume_24h=args.min_volume_24h,
            price_range=(args.price_min, args.price_max),
        )
    )
    logger.info("market: %s", info.market_id)
    logger.info("question: %s", info.question)
    logger.info("yes_token_id: %s", info.yes_token_id)
    logger.info("market end_date: %s", info.end_date)

    cfg = EnvConfig(
        bar_size=args.bar_size,
        lookback=args.lookback,
        min_bars_per_episode=args.lookback,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
    )

    bar_close_timeout_sec = (
        args.bar_close_timeout_sec
        if args.bar_close_timeout_sec is not None
        else args.duration_min * 60.0
    )
    feed = WebsocketLiveFeed(
        cfg,
        info,
        bootstrap_hours_back=args.bootstrap_hours_back,
        bar_close_timeout_sec=bar_close_timeout_sec,
    )
    base_env = PolymarketDirectionalEnv(config=cfg, feed=feed, venue=SimulatedVenue())
    env = gym.wrappers.FlattenObservation(base_env)

    # Lazy import to keep the script importable without sbx for non-running tests.
    from sbx import PPO

    # custom_objects fills in missing pieces if the saved policy's extractor
    # class wasn't pickled cleanly.
    n_scalars = 4
    n_window_features = 7
    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": partial(
                FlaxPolicyFeatures,
                lookback=cfg.lookback,
                n_window_features=n_window_features,
                n_scalars=n_scalars,
                cnn_channels=32,
            ),
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
    }
    model = PPO.load(str(ckpt), env=env, custom_objects=custom_objects)
    logger.info("model loaded; policy=%s", type(model.policy).__name__)

    obs, info_dict = env.reset()
    logger.info("env reset; bootstrap last close=%.4f", base_env._last_close)
    logger.info(
        "running for up to %.1f min wall-clock (bar_size=%s)",
        args.duration_min,
        args.bar_size,
    )

    # First inference on the bootstrap snapshot.
    action0, _ = model.predict(obs, deterministic=args.deterministic)
    action0 = int(action0)
    pv0 = base_env._cash + base_env._position_tokens * base_env._last_close
    print(
        f"\n[bootstrap]  bar_ts={feed._bars[-1].ts}  close={base_env._last_close:.4f}  "
        f"pv=${pv0:.2f}  action={ACTION_NAMES[action0]}({action0})"
    )

    # Live loop: each step waits for the next bar boundary.
    deadline = time.monotonic() + args.duration_min * 60.0
    step_count = 0
    try:
        # Apply the bootstrap action by stepping once — this triggers feed.advance().
        cur_action = action0
        while time.monotonic() < deadline:
            obs, reward, terminated, truncated, step_info = env.step(cur_action)
            step_count += 1
            pv = step_info.get("pv", 0.0)
            cash = step_info.get("cash", 0.0)
            pos_tokens = step_info.get("position_tokens", 0.0)
            bar_close = step_info.get("bar_close", 0.0)
            print(
                f"[step {step_count:3d}]  bar_close={bar_close:.4f}  reward={reward:+.4f}  "
                f"pv=${pv:.2f}  cash=${cash:.2f}  pos={pos_tokens:.3f} tok  "
                f"action(executed)={ACTION_NAMES[cur_action]}({cur_action})"
            )
            if terminated or truncated:
                logger.info(
                    "episode end (terminated=%s, truncated=%s)", terminated, truncated
                )
                break
            # Predict next action from the new observation.
            cur_action, _ = model.predict(obs, deterministic=args.deterministic)
            cur_action = int(cur_action)
            print(
                f"             next predicted action={ACTION_NAMES[cur_action]}({cur_action})"
            )
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        env.close()

    final_pv = base_env._cash + base_env._position_tokens * base_env._last_close
    pnl = final_pv - args.initial_cash
    print(
        f"\nfinal_pv=${final_pv:.2f}  pnl={pnl:+.2f} "
        f"({pnl / args.initial_cash * 100:+.2f}%)  steps={step_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
