#!/usr/bin/env python3
"""Weights & Biases hyperparameter sweep for PolymarketDirectional PPO.

Usage:
    # Create the sweep (once) and start an agent:
    python scripts/sweep.py

    # Or just create the sweep and print the ID:
    python scripts/sweep.py --create-only

    # Attach additional agents to an existing sweep:
    python scripts/sweep.py --sweep-id <SWEEP_ID>

    # Customise the number of runs per agent / per-trial budget:
    python scripts/sweep.py --count 30 --total-timesteps 500000

Each agent trial calls ``train.main`` in-process so that the wandb run created
by ``train.py`` IS the sweep trial's run — bayesian optimization and hyperband
early-termination both operate on the metrics sb3 logs via ``WandbCallback``.
"""
from __future__ import annotations

import argparse
import os
import sys

import wandb

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Sweep configuration ──────────────────────────────────────────────────────
#
# Design rationale:
#   - Metric: eval/mean_reward from EvalCallback (20 deterministic episodes),
#     far less noisy than rollout/ep_rew_mean.
#   - Search space: fully reopened. Prior sweep priors (narrowed n_steps,
#     fixed batch_size) were derived from a Discrete(3) + 10bps env that no
#     longer exists. Discrete(7) + spread model changes the optimal region.
#   - Hyperband min_iter=5: first cull after 5 eval checkpoints (~100K steps
#     at eval_freq=20K), giving runs time to warm up before pruning.
#
SWEEP_CONFIG = {
    "name": "polymarket-ppo-v3",
    "method": "bayes",
    "metric": {
        "name": "eval/mean_reward",
        "goal": "maximize",
    },
    "parameters": {
        # ── PPO core ──────────────────────────────────────────────────────
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "n_steps": {
            "values": [1024, 2048, 4096],
        },
        "batch_size": {
            "values": [64, 128, 256, 512],
        },
        "gamma": {
            "distribution": "uniform",
            "min": 0.9,
            "max": 0.999,
        },
        "gae_lambda": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 1.0,
        },
        "ent_coef": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 0.1,
        },
        "clip_range": {
            "values": [0.1, 0.2, 0.3],
        },
        "n_epochs": {
            "values": [5, 10, 15, 20],
        },
        # ── Architecture ──────────────────────────────────────────────────
        "features_dim": {
            "values": [64, 128, 256],
        },
        "cnn_channels": {
            "values": [16, 32, 64],
        },
        "net_arch_pi": {
            "values": [[64, 64], [128, 64], [128, 128], [256, 128]],
        },
        "net_arch_vf": {
            "values": [[64, 64], [128, 64], [128, 128], [256, 128]],
        },
        # ── Fee warmup ────────────────────────────────────────────────────
        "fee_warmup_episodes": {
            "values": [0, 5, 10, 20],
        },
        "fee_warmup_bps": {
            "values": [2.0, 5.0, 10.0],
        },
        # ── Curriculum ────────────────────────────────────────────────────
        "stage1_conviction": {
            "distribution": "uniform",
            "min": 0.20,
            "max": 0.50,
        },
        "stage2_conviction": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.25,
        },
        "stage1_threshold": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.5,
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,   # first cull after 5 eval checkpoints (~100K steps)
        "eta": 3,        # keep top 1/3 at each rung
    },
}

_TRIAL_ARGS = {
    "total_timesteps": 500_000,
    "n_eval_episodes": 50,
    "eval_freq": 20_000,
    "data_dir": os.path.join(_PROJECT_ROOT, "data"),
    "wandb_project": "polymarket-rl",
    "wandb_entity": None,
    # Curriculum is always enabled in sweep trials; thresholds/convictions
    # are swept via wandb.config and picked up by train.py.
    "stage2_threshold": 2.0,
    "max_steps_per_stage": 300_000,
}


def _trial_argv() -> list[str]:
    argv = [
        "--total-timesteps", str(_TRIAL_ARGS["total_timesteps"]),
        "--n-eval-episodes", str(_TRIAL_ARGS["n_eval_episodes"]),
        "--eval-freq", str(_TRIAL_ARGS["eval_freq"]),
        "--data-dir", _TRIAL_ARGS["data_dir"],
        "--wandb-project", _TRIAL_ARGS["wandb_project"],
        # env params fixed across all trials — not swept
        "--n-action-levels", "7",
        "--min-spread-bps", "50.0",
        "--spread-vol-factor", "2.0",
        "--n-envs", "4",
        # Curriculum always on; per-trial convictions/thresholds come from wandb.config
        "--curriculum",
        "--stage2-threshold", str(_TRIAL_ARGS["stage2_threshold"]),
        "--max-steps-per-stage", str(_TRIAL_ARGS["max_steps_per_stage"]),
    ]
    if _TRIAL_ARGS["wandb_entity"]:
        argv.extend(["--wandb-entity", _TRIAL_ARGS["wandb_entity"]])
    return argv


def train_fn() -> None:
    import train  # scripts/train.py — on sys.path via the prelude above

    argv = _trial_argv()
    print(f"[sweep] trial argv: {' '.join(argv)}", flush=True)
    try:
        rc = train.main(argv)
    except Exception as e:
        if wandb.run is not None:
            wandb.log({"sweep/error": 1, "sweep/error_msg": str(e)[:240]})
        raise
    if rc != 0 and wandb.run is not None:
        wandb.log({"sweep/error": 1, "sweep/exit_code": rc})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=str, default="polymarket-rl")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--sweep-id", type=str, default=None,
                   help="Attach to an existing sweep instead of creating a new one.")
    p.add_argument("--count", type=int, default=30,
                   help="Number of trials this agent will execute (default: 30).")
    p.add_argument("--create-only", action="store_true",
                   help="Create the sweep and print the ID, but don't start an agent.")
    p.add_argument("--total-timesteps", type=int, default=_TRIAL_ARGS["total_timesteps"],
                   help="Per-trial training budget (default: %(default)s).")
    p.add_argument("--n-eval-episodes", type=int, default=_TRIAL_ARGS["n_eval_episodes"],
                   help="Eval episodes per checkpoint (default: %(default)s).")
    p.add_argument("--data-dir", type=str, default=_TRIAL_ARGS["data_dir"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    _TRIAL_ARGS["total_timesteps"] = args.total_timesteps
    _TRIAL_ARGS["n_eval_episodes"] = args.n_eval_episodes
    _TRIAL_ARGS["data_dir"] = args.data_dir
    _TRIAL_ARGS["wandb_project"] = args.project
    _TRIAL_ARGS["wandb_entity"] = args.entity

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"[sweep] attaching to existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project=args.project,
            entity=args.entity,
        )
        print(f"[sweep] created sweep: {sweep_id}")

    if args.create_only:
        print(f"[sweep] run agents with: python scripts/sweep.py --sweep-id {sweep_id}")
        return 0

    wandb.agent(
        sweep_id,
        function=train_fn,
        project=args.project,
        entity=args.entity,
        count=args.count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
