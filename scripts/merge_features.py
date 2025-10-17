#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge reddit_daily and prices_yahoo on (symbol, day) to produce a single tidy table.

- Input:
    data/derived/featuresets/reddit_daily/day=YYYY-MM-DD/*.parquet
    data/derived/featuresets/prices_yahoo/day=YYYY-MM-DD/*.parquet
- Output:
    data/derived/featuresets/merged/day=YYYY-MM-DD/part-*.parquet

Behavior:
- Inner-join on (symbol, day) (default).
- Append-only; skips existing day partitions unless --force.
"""

import argparse
import os
import glob
import uuid
from typing import List

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


def merge_features(reddit_root: str, prices_root: str, out_root: str, start: str, end: str, force: bool = False) -> None:
    days = _list_days_between(start, end)
    os.makedirs(out_root, exist_ok=True)

    written = 0
    for d in days:
        if not force and _existing_day_written(out_root, d):
            continue

        r_df = _read_day_partition(reddit_root, d)
        p_df = _read_day_partition(prices_root, d)
        if r_df.empty or p_df.empty:
            continue

        # Normalize keys/types
        for df in (r_df, p_df):
            df["day"] = pd.to_datetime(df["day"]).dt.tz_localize(None)
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

        merged = pd.merge(
            r_df, p_df,
            on=["symbol", "day"],
            how="inner",
            validate="m:1"
        ).sort_values(["symbol", "day"]).reset_index(drop=True)

        if merged.empty:
            continue

        d_dir = os.path.join(out_root, f"day={d.date().isoformat()}")
        os.makedirs(d_dir, exist_ok=True)
        out_path = os.path.join(d_dir, f"part-{uuid.uuid4().hex}.parquet")
        merged.to_parquet(out_path, index=False)
        written += 1
        print(f"[WRITE] {out_path}  rows={len(merged)}")

    print(f"[DONE] merged: wrote {written} day partitions (skipped existing: {len(days) - written}).")


def main():
    ap = argparse.ArgumentParser(description="Merge reddit_daily and prices_yahoo per day on (symbol, day).")
    ap.add_argument("--reddit-root", required=True, help="Path to featuresets/reddit_daily")
    ap.add_argument("--prices-root", required=True, help="Path to featuresets/prices_yahoo")
    ap.add_argument("--out-root", required=True, help="Output root for featuresets/merged")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    merge_features(
        reddit_root=args.reddit_root,
        prices_root=args.prices_root,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
