"""Build a small local sample of the SII-WANGZJ Polymarket dataset.

Steps:
  1. Download markets.parquet (68 MB) via huggingface_hub.
  2. Filter to cleanly-resolved binary markets above a volume floor; take top-N.
  3. Pull only those markets' trades from quant.parquet:
     - First attempt: remote DuckDB over httpfs with predicate pushdown.
     - Fallback: hf_hub_download the full quant.parquet, filter locally, delete cache.

Usage:
    python scripts/build_sample.py --top-n 200 --min-volume 10000 --out data/

The resulting files live at <out>/markets.parquet and <out>/quant_sample.parquet
and are consumed by polymarket_gym.data.MarketLoader.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

from polymarket_gym.data import parse_outcome_prices

logger = logging.getLogger("build_sample")

REPO_ID = "SII-WANGZJ/Polymarket_data"
MARKETS_FILE = "markets.parquet"
QUANT_FILE = "quant.parquet"
REMOTE_QUANT_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{QUANT_FILE}"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-n", type=int, default=200, help="number of markets to keep")
    p.add_argument(
        "--min-volume",
        type=float,
        default=10_000.0,
        help="minimum lifetime USD volume for a market to be eligible",
    )
    p.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    p.add_argument(
        "--remote-timeout-sec",
        type=int,
        default=600,
        help="seconds to wait on remote DuckDB before falling back to full download",
    )
    p.add_argument("--no-remote", action="store_true", help="skip remote DuckDB attempt")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _download_markets(out_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading markets.parquet from HF")
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=MARKETS_FILE,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    return Path(local)


def _filter_markets(markets_path: Path, top_n: int, min_volume: float) -> pd.DataFrame:
    df = pd.read_parquet(markets_path)
    if "market_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "market_id"})
    df["market_id"] = df["market_id"].astype(str)
    outcome_col = (
        "outcome_prices" if "outcome_prices" in df.columns else "outcomePrices"
    )
    closed_col = "closed" if "closed" in df.columns else None
    volume_col = (
        "volume" if "volume" in df.columns else ("volumeNum" if "volumeNum" in df.columns else None)
    )
    if volume_col is None:
        logger.warning("no volume column found; skipping volume floor")
        df["_volume"] = 0.0
    else:
        df["_volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)

    keep_mask = pd.Series(False, index=df.index)
    for i, raw in enumerate(df[outcome_col]):
        try:
            payoffs = parse_outcome_prices(raw)
        except ValueError:
            continue
        if payoffs in ((1.0, 0.0), (0.0, 1.0)):
            keep_mask.iloc[i] = True
    if closed_col is not None:
        keep_mask &= df[closed_col].astype(bool)
    keep_mask &= df["_volume"] >= min_volume

    eligible = df.loc[keep_mask].sort_values("_volume", ascending=False).head(top_n)
    logger.info("eligible markets: %d (top-N=%d, min_volume=%.0f)", len(eligible), top_n, min_volume)
    return eligible.drop(columns=["_volume"])


def _try_remote_quant(market_ids: list[str], out_path: Path, timeout_sec: int) -> bool:
    logger.info("attempting remote DuckDB over httpfs (predicate pushdown)")
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except duckdb.Error as e:
        logger.warning("httpfs unavailable: %s", e)
        con.close()
        return False
    placeholders = ",".join(["?"] * len(market_ids))
    sql = (
        f"COPY (SELECT * FROM read_parquet('{REMOTE_QUANT_URL}') "
        f"WHERE market_id IN ({placeholders})) "
        f"TO '{out_path}' (FORMAT 'parquet')"
    )
    params = list(market_ids)
    start = time.monotonic()
    try:
        con.execute(sql, params)
    except duckdb.Error as e:
        logger.warning("remote DuckDB failed: %s", e)
        return False
    finally:
        con.close()
    elapsed = time.monotonic() - start
    if elapsed > timeout_sec:
        logger.warning("remote DuckDB took %.0fs (> %ds budget)", elapsed, timeout_sec)
    return out_path.exists()


def _local_quant_filter(market_ids: list[str], out_path: Path, cache_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    logger.info("falling back to local download of full quant.parquet (~21 GB)")
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=QUANT_FILE,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    try:
        con = duckdb.connect()
        placeholders = ",".join(["?"] * len(market_ids))
        sql = (
            f"COPY (SELECT * FROM read_parquet('{local}') "
            f"WHERE market_id IN ({placeholders})) "
            f"TO '{out_path}' (FORMAT 'parquet')"
        )
        con.execute(sql, list(market_ids))
        con.close()
    finally:
        try:
            Path(local).unlink(missing_ok=True)
            logger.info("removed cached quant.parquet")
        except OSError as e:
            logger.warning("could not remove cached quant.parquet: %s", e)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    markets_local = _download_markets(out_dir)
    if markets_local != out_dir / MARKETS_FILE:
        target = out_dir / MARKETS_FILE
        if markets_local != target:
            target.write_bytes(markets_local.read_bytes())
    filtered = _filter_markets(out_dir / MARKETS_FILE, args.top_n, args.min_volume)
    filtered.to_parquet(out_dir / MARKETS_FILE, index=False)
    if filtered.empty:
        logger.error("no eligible markets after filtering")
        return 2
    market_ids = filtered["market_id"].astype(str).tolist()
    quant_out = out_dir / "quant_sample.parquet"
    ok = False
    if not args.no_remote:
        ok = _try_remote_quant(market_ids, quant_out, args.remote_timeout_sec)
    if not ok:
        cache_dir = Path(os.environ.get("HF_HUB_CACHE", out_dir / ".hf_cache"))
        _local_quant_filter(market_ids, quant_out, cache_dir)
    size_mb = quant_out.stat().st_size / 1e6 if quant_out.exists() else 0.0
    logger.info("wrote %s (%.1f MB)", quant_out, size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
