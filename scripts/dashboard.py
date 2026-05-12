"""Streamlit dashboard for live-viewing a PPO training run.

Reads from a run directory created by ``scripts/train.py``:

  runs/<run_name>/
    monitor/monitor_*.monitor.csv     per-env episode rewards (sb3 Monitor)
    monitor_eval/monitor_*.monitor.csv eval-env episode rewards
    eval_log/evaluations.npz          EvalCallback metrics
    ckpt/                              checkpoint zips + best_model.zip
    viz/                               periodic episode visualization PNGs

Run with:
    streamlit run scripts/dashboard.py -- --run-dir scripts/runs/ppo_XXXX
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = SCRIPTS_DIR / "runs"


def list_runs() -> list[Path]:
    return sorted(
        [p for p in RUNS_DIR.glob("ppo_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--refresh-sec", type=int, default=5)
    # Streamlit passes its own argv beyond `--`; we read only known args.
    known, _ = p.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(ttl=4, show_spinner=False)
def load_monitor_csvs(run_dir: str, kind: str) -> pd.DataFrame:
    """Load all per-env monitor CSVs from ``run_dir/<kind>/``.

    Columns: ``r`` (episode reward), ``l`` (episode length, steps),
    ``t`` (wall-clock seconds since the env's t_start), plus injected
    ``env_idx`` and ``t_abs`` (UTC datetime of episode end).
    """
    dirpath = Path(run_dir) / kind
    if not dirpath.exists():
        return pd.DataFrame(columns=["r", "l", "t", "env_idx", "t_abs"])
    frames: list[pd.DataFrame] = []
    for csv in sorted(dirpath.glob("monitor_*.monitor.csv")):
        try:
            with open(csv) as f:
                header = f.readline()
            meta = json.loads(header.lstrip("#"))
            t_start = float(meta.get("t_start", 0.0))
            df = pd.read_csv(csv, skiprows=1)
            if df.empty:
                continue
            # The numeric suffix in the filename is the env (or eval) seed/idx.
            stem = csv.stem.replace("monitor_", "").replace(".monitor", "")
            df["env_idx"] = stem
            df["t_abs"] = pd.to_datetime(t_start + df["t"], unit="s", utc=True)
            frames.append(df)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not frames:
        return pd.DataFrame(columns=["r", "l", "t", "env_idx", "t_abs"])
    return pd.concat(frames, ignore_index=True).sort_values("t_abs")


@st.cache_data(ttl=4, show_spinner=False)
def load_eval_npz(run_dir: str) -> dict | None:
    path = Path(run_dir) / "eval_log" / "evaluations.npz"
    if not path.exists():
        return None
    try:
        z = np.load(path)
        return {
            "timesteps": z["timesteps"].astype(np.int64),
            "results": z["results"].astype(np.float64),  # (n_evals, n_episodes)
            "ep_lengths": z["ep_lengths"].astype(np.int64),
        }
    except (OSError, KeyError):
        return None


@st.cache_data(ttl=4, show_spinner=False)
def list_checkpoints(run_dir: str) -> list[tuple[int | None, Path, float]]:
    ckpt_dir = Path(run_dir) / "ckpt"
    if not ckpt_dir.exists():
        return []
    out = []
    for p in ckpt_dir.glob("*.zip"):
        steps: int | None = None
        # Filenames look like ppo_100000_steps.zip / best_model.zip / model.zip
        stem = p.stem
        if stem.startswith("ppo_") and stem.endswith("_steps"):
            try:
                steps = int(stem[len("ppo_") : -len("_steps")])
            except ValueError:
                steps = None
        out.append((steps, p, p.stat().st_mtime))
    out.sort(key=lambda r: (r[0] is None, r[0] or 0, r[2]))
    return out


@st.cache_data(ttl=4, show_spinner=False)
def latest_viz_png(run_dir: str) -> Path | None:
    viz = Path(run_dir) / "viz"
    if not viz.exists():
        return None
    pngs = sorted(viz.glob("*.png"), key=lambda p: p.stat().st_mtime)
    return pngs[-1] if pngs else None


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()


def render():
    args = parse_cli()
    st.set_page_config(page_title="PPO live dashboard", layout="wide")
    st_autorefresh(interval=args.refresh_sec * 1000, key="autorefresh")

    runs = list_runs()
    if not runs:
        st.error(f"No runs found under {RUNS_DIR}")
        return

    # Sidebar: run selector
    default_run = Path(args.run_dir) if args.run_dir else runs[0]
    run_names = [p.name for p in runs]
    try:
        default_idx = run_names.index(default_run.name)
    except ValueError:
        default_idx = 0
    with st.sidebar:
        st.markdown("### Run")
        selected_name = st.selectbox(
            "active run", run_names, index=default_idx, label_visibility="collapsed"
        )
        rolling_window = st.slider(
            "rolling-mean window (episodes)", 5, 200, 50
        )
        show_per_env = st.checkbox("show per-env scatter", value=True)
    run_dir = str(RUNS_DIR / selected_name)

    # ── header / status ────────────────────────────────────────────────────
    train_df = load_monitor_csvs(run_dir, "monitor")
    eval_df = load_monitor_csvs(run_dir, "monitor_eval")
    eval_npz = load_eval_npz(run_dir)
    ckpts = list_checkpoints(run_dir)
    viz_png = latest_viz_png(run_dir)

    latest_ckpt_steps = None
    latest_ckpt_mtime = None
    for steps, _, mtime in ckpts:
        if steps is not None:
            latest_ckpt_steps = steps if (
                latest_ckpt_steps is None or steps > latest_ckpt_steps
            ) else latest_ckpt_steps
        latest_ckpt_mtime = (
            mtime if latest_ckpt_mtime is None or mtime > latest_ckpt_mtime else latest_ckpt_mtime
        )

    age_str = "—"
    if latest_ckpt_mtime is not None:
        age_sec = time.time() - latest_ckpt_mtime
        if age_sec < 60:
            age_str = f"{age_sec:.0f}s ago"
        elif age_sec < 3600:
            age_str = f"{age_sec/60:.1f} min ago"
        else:
            age_str = f"{age_sec/3600:.1f} hr ago"

    st.markdown(f"## {selected_name}")
    cols = st.columns(5)
    cols[0].metric("episodes (train)", f"{len(train_df):,}")
    cols[1].metric(
        "mean reward (last 100)",
        f"{train_df.tail(100)['r'].mean():.1f}" if not train_df.empty else "—",
    )
    cols[2].metric(
        "latest eval mean",
        f"{eval_npz['results'][-1].mean():.1f}" if eval_npz is not None and eval_npz["results"].size else "—",
    )
    cols[3].metric("latest ckpt step", f"{latest_ckpt_steps:,}" if latest_ckpt_steps else "—")
    cols[4].metric("last update", age_str)

    # ── reward curves ──────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown("**Training episode reward**")
        if train_df.empty:
            st.info("no training episodes yet")
        else:
            t_df = train_df.sort_values("t_abs").reset_index(drop=True).copy()
            t_df["ep_idx"] = np.arange(len(t_df))
            t_df["rolling"] = _rolling_mean(t_df["r"], rolling_window)
            fig = go.Figure()
            if show_per_env:
                for env_idx, g in t_df.groupby("env_idx"):
                    fig.add_trace(
                        go.Scatter(
                            x=g["ep_idx"],
                            y=g["r"],
                            mode="markers",
                            name=f"env {env_idx}",
                            marker=dict(size=4, opacity=0.45),
                        )
                    )
            fig.add_trace(
                go.Scatter(
                    x=t_df["ep_idx"],
                    y=t_df["rolling"],
                    mode="lines",
                    name=f"rolling[{rolling_window}]",
                    line=dict(width=3, color="black"),
                )
            )
            fig.update_layout(
                xaxis_title="episode #",
                yaxis_title="episode reward",
                height=380,
                margin=dict(l=40, r=10, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Eval reward vs. timesteps**")
        if eval_npz is None or eval_npz["results"].size == 0:
            st.info("no eval data yet")
        else:
            ts = eval_npz["timesteps"]
            results = eval_npz["results"]
            mean = results.mean(axis=1)
            best = results.max(axis=1)
            worst = results.min(axis=1)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([ts, ts[::-1]]),
                    y=np.concatenate([best, worst[::-1]]),
                    fill="toself",
                    fillcolor="rgba(99,110,250,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="min/max",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ts, y=mean, mode="lines+markers", name="mean",
                    line=dict(width=2),
                )
            )
            fig.update_layout(
                xaxis_title="timesteps",
                yaxis_title="eval reward",
                height=380,
                margin=dict(l=40, r=10, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── episode visualization + checkpoint timeline ────────────────────────
    left2, right2 = st.columns([3, 2])
    with left2:
        st.markdown("**Latest episode visualization**")
        if viz_png is not None:
            st.image(str(viz_png), caption=viz_png.name, use_container_width=True)
        else:
            st.info("no viz PNGs yet — VisualizationCallback emits these every "
                    "`--viz-every-n-episodes` episodes")
    with right2:
        st.markdown("**Checkpoints**")
        if not ckpts:
            st.info("no checkpoints yet")
        else:
            ckpt_rows = []
            for steps, p, mtime in ckpts:
                ckpt_rows.append(
                    {
                        "steps": steps if steps is not None else "—",
                        "file": p.name,
                        "size_MB": round(p.stat().st_size / 1e6, 2),
                        "age": _fmt_age(time.time() - mtime),
                    }
                )
            st.dataframe(
                pd.DataFrame(ckpt_rows),
                use_container_width=True,
                hide_index=True,
                height=380,
            )

    # ── per-env reward distribution ────────────────────────────────────────
    if not train_df.empty:
        st.markdown("**Per-env reward distribution (all episodes)**")
        fig = go.Figure()
        for env_idx, g in train_df.groupby("env_idx"):
            fig.add_trace(go.Box(y=g["r"], name=f"env {env_idx}", boxmean="sd"))
        fig.update_layout(
            yaxis_title="episode reward",
            height=300,
            margin=dict(l=40, r=10, t=10, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def _fmt_age(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    if sec < 86400:
        return f"{sec/3600:.1f}h"
    return f"{sec/86400:.1f}d"


if __name__ == "__main__":
    render()
