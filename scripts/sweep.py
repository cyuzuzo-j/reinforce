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
    python scripts/sweep.py --count 30 --total-timesteps 100000

Each agent trial calls ``train.main`` in-process so that the wandb run created
by ``train.py`` IS the sweep trial's run — bayesian optimization and hyperband
early-termination both operate on the metrics sb3 logs via ``WandbCallback``.
"""
from __future__ import annotations

import argparse
import os
import sys

import wandb

# Make project root + scripts/ importable so `import train` works regardless
# of the caller's CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Sweep configuration ─────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "name": "polymarket-ppo-sweep",
    "method": "bayes",  # bayesian optimisation > random for sample-efficiency
    "metric": {
        "name": "rollout/ep_rew_mean",
        "goal": "maximize",
    },
    "parameters": {
        # ── PPO core hypers ──────────────────────────────────────────────
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "n_steps": {
            "values": [512, 1024, 2048, 4096],
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
            "values": [3, 5, 10, 15, 20],
        },
        # ── Network architecture ─────────────────────────────────────────
        "features_dim": {
            "values": [64, 128, 256],
        },
        "cnn_channels": {
            "values": [16, 32, 64],
        },
        "net_arch_pi": {
            "values": [[64, 64], [128, 128], [256, 128], [128, 64]],
        },
        "net_arch_vf": {
            "values": [[64, 64], [128, 128], [256, 128], [128, 64]],
        },
    },
    # Early-terminate runs that plateau (works in-process — the agent kills
    # the trial by signalling the run; sb3+WandbCallback then exit).
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 3,
    },
}


# Mutated by main() so train_fn closures pick up CLI overrides.
_TRIAL_ARGS = {
    "total_timesteps": 200_000,
    "data_dir": os.path.join(_PROJECT_ROOT, "data"),
    "wandb_project": "polymarket-rl",
    "wandb_entity": None,
}


def _trial_argv() -> list[str]:
    """Argv handed to ``train.main`` for a single sweep trial.

    Only the knobs that ``train.py`` does *not* read from ``wandb.config``
    need to be set here. The sampled hyperparameters in ``SWEEP_CONFIG``
    flow into the trial's wandb run via ``WANDB_SWEEP_ID`` and are picked
    up by ``train.py`` through its ``wandb.config.get(...)`` overrides.
    """
    argv = [
        "--total-timesteps", str(_TRIAL_ARGS["total_timesteps"]),
        "--data-dir", _TRIAL_ARGS["data_dir"],
        "--wandb-project", _TRIAL_ARGS["wandb_project"],
    ]
    if _TRIAL_ARGS["wandb_entity"]:
        argv.extend(["--wandb-entity", _TRIAL_ARGS["wandb_entity"]])
    return argv


def train_fn() -> None:
    """wandb.agent callback — runs one sweep trial in-process.

    Importantly we do *not* call ``wandb.init`` here: ``train.py`` does it
    itself, and when invoked from within a sweep agent's call frame, that
    init picks up ``WANDB_SWEEP_ID`` from the environment and creates a
    run associated with the sweep. The trial's sampled hyperparameters
    land in ``wandb.config``, which train.py reads to override its CLI
    defaults (see the ``wc.get(...)`` block in train.py).
    """
    import train  # scripts/train.py — on sys.path via the prelude above

    argv = _trial_argv()
    print(f"[sweep] trial argv: {' '.join(argv)}", flush=True)
    try:
        rc = train.main(argv)
    except Exception as e:
        # Surface the failure on the trial's wandb run, if still alive.
        if wandb.run is not None:
            wandb.log({"sweep/error": 1, "sweep/error_msg": str(e)[:240]})
        raise
    if rc != 0 and wandb.run is not None:
        wandb.log({"sweep/error": 1, "sweep/exit_code": rc})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project",
        type=str,
        default="polymarket-rl",
        help="wandb project name",
    )
    p.add_argument(
        "--entity",
        type=str,
        default=None,
        help="wandb entity (team/user)",
    )
    p.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Attach to an existing sweep instead of creating a new one.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of trials this agent will execute (default: 20).",
    )
    p.add_argument(
        "--create-only",
        action="store_true",
        help="Create the sweep and print the ID, but don't start an agent.",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=_TRIAL_ARGS["total_timesteps"],
        help="Per-trial training budget (default: %(default)s).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=_TRIAL_ARGS["data_dir"],
        help="Path passed as --data-dir to each trial's train.py.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    _TRIAL_ARGS["total_timesteps"] = args.total_timesteps
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
