#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enrich v4 merged daily table with leakage-safe lags, rolling windows,
deltas, and targets.

- Input:  <merged_root>/day=YYYY-MM-DD/*.parquet
- Output: <out_root>/day=YYYY-MM-DD/part-*.parquet

Choices baked in:
- Rolling windows centralized here (3/7/14/30 for Reddit signals).
- Market rolls: ret1 std (5/14/30), vol mean (5/20); relative volume using 5 and 20.
- Targets included by default: ret1_lead1, ret1_lead2, ret1_lead5.
- No zero-filling for missing Reddit days.
- Append-only; skip existing partitions unless --force.
"""

import argparse
import os
import glob
import uuid
from typing import List, Sequence

import numpy as np
import pandas as pd


def _list_days_between(start: str, end: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    return list(pd.date_range(s, e, freq="D"))


def _existing_day_written(out_root: str, day: pd.Timestamp) -> bool:
    dpath = os.path.join(out_root, f"day={day.date().isoformat()}")
    return os.path.isdir(dpath) and bool(glob.glob(os.path.join(dpath, "*.parquet")))


def _read_day_partition(root: str, day: pd.Timestamp) -> pd.DataFrame:
    d_dir = os.path.join(root, f"day={day.date().isoformat()}")
    parts = glob.glob(os.path.join(d_dir, "*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _add_lags(g: pd.DataFrame, cols: Sequence[str], lags=(1, 2, 3)) -> pd.DataFrame:
    g = g.sort_values("day").copy()
    for L in lags:
        for c in cols:
            if c in g.columns:
                g[f"{c}_lag{L}"] = g[c].shift(L)
    return g


def _add_rolling(
    g: pd.DataFrame,
    sum_cols: Sequence[str],
    mean_cols: Sequence[str],
    windows=(3, 7, 14, 30),
) -> pd.DataFrame:
    g = g.sort_values("day").copy()
    for W in windows:
        for c in sum_cols:
            if c in g.columns:
                g[f"{c}_roll{W}_sum"] = g[c].rolling(W, min_periods=1).sum()
        for c in mean_cols:
            if c in g.columns:
                g[f"{c}_roll{W}_mean"] = g[c].rolling(W, min_periods=1).mean()
    return g


def _add_market_rolls(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("day").copy()
    # ret1 rolling std: 5,14,30
    for W in (5, 14, 30):
        if "ret1" in g.columns:
            g[f"ret1_roll{W}_std"] = g["ret1"].rolling(W, min_periods=3).std()
    # volume rolling mean: 5,20 + relative volume
    for W in (5, 20):
        if "Volume" in g.columns:
            g[f"vol_roll{W}_mean"] = g["Volume"].rolling(W, min_periods=3).mean()
            g[f"rel_volume_{W}"] = g["Volume"] / g[f"vol_roll{W}_mean"]
    return g


def _add_deltas_and_extras(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("day").copy()
    # Deltas for key Reddit signals
    if "chatter_count" in g.columns and "chatter_count_lag1" in g.columns:
        g["delta_chatter"] = g["chatter_count"] - g["chatter_count_lag1"]
    if "sent_mean" in g.columns and "sent_mean_lag1" in g.columns:
        g["delta_sent_mean"] = g["sent_mean"] - g["sent_mean_lag1"]
    if "engagement_sum" in g.columns and "engagement_sum_lag1" in g.columns:
        g["delta_engagement_sum"] = g["engagement_sum"] - g["engagement_sum_lag1"]

    # Market extras
    if set(["High", "Low", "Close"]).issubset(g.columns):
        g["intraday_range"] = (g["High"] - g["Low"]) / g["Close"]
    if "Close" in g.columns:
        g["Close_lag5"] = g["Close"].shift(5)
        g["momentum_5"] = g["Close"] / g["Close_lag5"] - 1
    return g


def _add_targets(g: pd.DataFrame) -> pd.DataFrame:
    """
    Add future returns: ret1_lead1/2/5 computed from Adj Close if available else Close.
    """
    g = g.sort_values("day").copy()
    price_col = "Adj Close" if "Adj Close" in g.columns else "Close"
    for H in (1, 2, 5):
        g[f"ret1_lead{H}"] = g[price_col].pct_change(-H)
    return g


def enrich_features_v4(
    merged_root: str,
    out_root: str,
    start: str,
    end: str,
    lags=(1, 2, 3),
    rolls=(3, 7, 14, 30),
    include_targets: bool = True,
    force: bool = False,
) -> None:
    days = _list_days_between(start, end)
    os.makedirs(out_root, exist_ok=True)

    # Columns for families
    reddit_sum_cols = ("chatter_count", "engagement_sum", "forward_true_count")
    reddit_mean_cols = (
        "sent_mean",
        "sent_weighted",
        "avg_conf",
        "median_conf",
        "forward_share",
        "value_mean",
    )

    # Lags apply to:
    lag_cols = (
        "chatter_count",
        "unique_authors",
        "user_concentration",
        "sent_mean",
        "sent_weighted",
        "avg_conf",
        "median_conf",
        "engagement_sum",
        "comments_sum",
        "score_sum",
        "subreddit_entropy",
        "forward_true_count",
        "forward_share",
        "value_mean",
        "value_median",
    )

    written = 0
    for d in days:
        if not force and _existing_day_written(out_root, d):
            continue

        df = _read_day_partition(merged_root, d)
        if df.empty:
            continue

        df["day"] = pd.to_datetime(df["day"]).dt.tz_localize(None)
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df.sort_values(["symbol", "day"]).reset_index(drop=True)

        # Apply per-symbol transforms
        def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("day").copy()
            g = _add_lags(g, lag_cols, lags=lags)
            g = _add_rolling(g, reddit_sum_cols, reddit_mean_cols, windows=rolls)
            g = _add_market_rolls(g)
            g = _add_deltas_and_extras(g)
            if include_targets:
                g = _add_targets(g)
            return g

        enriched = df.groupby("symbol", group_keys=False).apply(_per_symbol)
        if enriched.empty:
            continue

        d_dir = os.path.join(out_root, f"day={d.date().isoformat()}")
        os.makedirs(d_dir, exist_ok=True)
        out_path = os.path.join(d_dir, f"part-{uuid.uuid4().hex}.parquet")
        enriched.to_parquet(out_path, index=False)
        written += 1
        print(f"[WRITE] {out_path}  rows={len(enriched)}")

    print(f"[DONE] merged_enriched_v4: wrote {written} day partitions (skipped existing: {len(days) - written}).")


def main():
    ap = argparse.ArgumentParser(
        description="Enrich v4 merged daily table with lags, rolls, deltas, and targets."
    )
    ap.add_argument("--merged-root", required=True, help="Path to featuresets/merged_v4")
    ap.add_argument("--out-root", required=True, help="Output root for featuresets/merged_enriched_v4")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lags", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--rolls", type=int, nargs="*", default=[3, 7, 14, 30])
    ap.add_argument("--no-targets", action="store_true", help="Disable adding future return targets")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    enrich_features_v4(
        merged_root=args.merged_root,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        lags=tuple(args.lags),
        rolls=tuple(args.rolls),
        include_targets=not args.no_targets,
        force=args.force,
    )


if __name__ == "__main__":
    main()
