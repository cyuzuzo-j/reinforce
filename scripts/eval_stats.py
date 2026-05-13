"""Evaluation statistics with baselines.

Compares a trained model against three constant policies — always-YES,
always-NO, uniform-random — over the held-out eval split. Outputs a
markdown table with mean return, std, and Sharpe (per-episode).

Usage:
    python scripts/eval_stats.py --model runs/.../final_model.zip \
        --data-dir data/ --out-md runs/.../baselines.md --n-episodes 20

Legacy mode (compute stats from an SB3 ``evaluations.npz``) is still
supported via ``--eval-npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from polymarket_gym.config import EnvConfig
from polymarket_gym.data import MarketLoader
from polymarket_gym.training import rollout
from polymarket_gym.training.env_factory import make_env
from polymarket_gym.training.splits import chronological_split


def _stats(returns: list[float]) -> dict[str, float]:
    arr = np.asarray(returns, dtype=np.float64)
    std = float(arr.std(ddof=0))
    sharpe = float(arr.mean() / std) if std > 1e-12 else 0.0
    return {"mean": float(arr.mean()), "std": std, "sharpe": sharpe, "n": int(arr.size)}


def _legacy_from_npz(path: Path, out_md: Path) -> None:
    data = np.load(path)
    md = (
        "## Training Evaluation Statistics\n\n"
        f"* **Timesteps trained:** {data['timesteps'][-1]}\n"
        f"* **Mean Reward:** {data['results'][-1].mean():.2f} ± {data['results'][-1].std():.2f}\n"
        f"* **Mean Episode Length:** {data['ep_lengths'][-1].mean():.2f}\n"
    )
    out_md.write_text(md)
    print(f"Stats written to {out_md}")


def _baseline_policies(cfg: EnvConfig, seed: int) -> dict[str, Callable[[object], int]]:
    long_yes = cfg.n_actions - 1
    long_no = 0
    rng = np.random.default_rng(seed)
    return {
        "always-YES": lambda _obs, _a=long_yes: _a,
        "always-NO": lambda _obs, _a=long_no: _a,
        "random": lambda _obs: int(rng.integers(0, cfg.n_actions)),
    }


def _model_policy(model_path: Path) -> Callable[[object], int]:
    from sbx import PPO
    model = PPO.load(str(model_path))

    def _act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    return _act


def _eval_policy(
    policy: Callable[[object], int],
    env_thunk: Callable,
    n_episodes: int,
    seed: int,
) -> list[float]:
    returns: list[float] = []
    env = env_thunk()
    try:
        for ep in range(n_episodes):
            returns.append(rollout(env, policy, seed=seed + ep)["return"])
    finally:
        env.close()
    return returns


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-npz", type=Path, help="legacy mode")
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--markets-file", type=str, default="markets.parquet")
    p.add_argument("--quant-file", type=str, default="quant_sample.parquet")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eval-frac", type=float, default=0.2)
    args = p.parse_args()

    if args.eval_npz is not None:
        if not args.eval_npz.exists():
            print(f"Eval file {args.eval_npz} not found.")
            return 1
        _legacy_from_npz(args.eval_npz, args.out_md)
        return 0

    if args.model is None:
        print("either --eval-npz or --model is required")
        return 2

    markets_path = args.data_dir / args.markets_file
    quant_path = args.data_dir / args.quant_file
    cfg = EnvConfig()
    loader = MarketLoader(markets_path, quant_path)
    try:
        _, eval_ids = chronological_split(loader, cfg, eval_frac=args.eval_frac)
    finally:
        loader.close()

    env_thunk = make_env(markets_path, quant_path, cfg, eval_ids, seed=args.seed)

    policies = _baseline_policies(cfg, args.seed)
    policies["trained"] = _model_policy(args.model)

    results = {
        name: _stats(_eval_policy(pol, env_thunk, args.n_episodes, args.seed))
        for name, pol in policies.items()
    }

    md = ["## Eval baselines\n", f"_n_episodes_ = {args.n_episodes}, eval markets = {len(eval_ids)}\n"]
    md.append("\n| Policy | Mean Return | Std | Sharpe |")
    md.append("|---|---:|---:|---:|")
    for name, s in results.items():
        md.append(f"| {name} | {s['mean']:+.4f} | {s['std']:.4f} | {s['sharpe']:+.3f} |")
    args.out_md.write_text("\n".join(md) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Stats written to {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
