#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build daily Reddit features per (symbol, day) from annotated/exploded parquet.
- Input:  data/derived/submission_tickers/YYYY-MM-DD/{annotated, exploded}.parquet
- Output: data/derived/featuresets/reddit_daily/day=YYYY-MM-DD/part-*.parquet

Notes:
- No subreddit_top_share (per user request).
- No rolling windows here (centralized later in enrich_features.py).
- Append-only: skips existing day partitions unless --force.
"""

import argparse
import os
import glob
import uuid
from typing import List

import numpy as np
import pandas as pd


# ------------------------- helpers -------------------------

def _existing_day_written(out_root: str, day: pd.Timestamp) -> bool:
    dpath = os.path.join(out_root, f"day={day.date().isoformat()}")
    return os.path.isdir(dpath) and bool(glob.glob(os.path.join(dpath, "*.parquet")))


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _read_concat(files: List[str]) -> pd.DataFrame:
    dfs = []
    for fp in sorted(files):
        try:
            dfs.append(pd.read_parquet(fp))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ------------------------- main logic -------------------------

def build_reddit_daily(in_root: str, out_root: str, start: str, end: str, force: bool = False) -> None:
    # Discover inputs (we’ll filter rows by window later)
    annot_glob = os.path.join(in_root, "*", "annotated.parquet")
    explo_glob = os.path.join(in_root, "*", "exploded.parquet")

    annot_files = glob.glob(annot_glob)
    expl_files = glob.glob(explo_glob)
    if not annot_files or not expl_files:
        raise FileNotFoundError("Missing annotated/exploded parquet files under submission_tickers/*/")

    annot = _read_concat(annot_files)
    expl = _read_concat(expl_files)

    if annot.empty or expl.empty:
        print("[INFO] No rows found in annotated or exploded; nothing to do.")
        return

    # Minimal columns needed from annotated
    keep_ann = ["id", "created_utc", "score", "num_comments", "subreddit", "author"]
    keep_ann = [c for c in keep_ann if c in annot.columns]
    ann_join = annot[keep_ann].copy()

    # Merge exploded + (subset of) annotated on id
    df = expl.merge(ann_join, on="id", how="left", validate="m:1")

    # ---- Normalize subreddit column (handle suffixes & missing) ----
    # After merge, pandas may create subreddit_x / subreddit_y; coalesce to a single 'subreddit'
    if "subreddit" not in df.columns:
        if "subreddit_x" in df.columns or "subreddit_y" in df.columns:
            df["subreddit"] = df.get("subreddit_x")
            if "subreddit_y" in df.columns:
                df["subreddit"] = df["subreddit"].fillna(df["subreddit_y"])
        else:
            # If neither side had subreddit, create a null column so entropy step can handle it.
            df["subreddit"] = pd.NA
    for c in ("subreddit_x", "subreddit_y"):
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")

    # ---- Types / cleaning ----
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["day"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce").dt.floor("D")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df.get("confidence", np.nan), errors="coerce").clip(lower=0, upper=1)

    # Filter essentials
    df = df.dropna(subset=["symbol", "day", "sentiment_score"])
    df = df[df["symbol"].str.len() > 0]

    # Limit to date window
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    df = df[(df["day"] >= start_ts) & (df["day"] <= end_ts)]
    if df.empty:
        print("[INFO] No Reddit rows in the requested date window.")
        return

    # ---- Base aggregations per (symbol, day) ----
    grp = df.groupby(["day", "symbol"], sort=True)

    base = grp.agg(
        chatter_count=("id", "count"),
        sent_mean=("sentiment_score", "mean"),
        sent_w_num=("sentiment_score", lambda s: float(np.sum(s * df.loc[s.index, "confidence"]))),
        sent_w_den=("confidence", "sum"),
        avg_conf=("confidence", "mean"),
        median_conf=("confidence", "median"),
        score_sum=("score", "sum"),
        score_mean=("score", "mean"),
        comments_sum=("num_comments", "sum"),
        comments_mean=("num_comments", "mean"),
        unique_authors=("author", lambda s: s.nunique()),
    ).reset_index()

    base["sent_weighted"] = np.where(
        base["sent_w_den"] > 0, base["sent_w_num"] / base["sent_w_den"], np.nan
    )
    base.drop(columns=["sent_w_num", "sent_w_den"], inplace=True)

    # Engagement composites
    base["engagement_sum"] = base["score_sum"] + base["comments_sum"]
    base["engagement_mean"] = base["score_mean"] + base["comments_mean"]
    base["user_concentration"] = np.where(
        base["chatter_count"] > 0, base["unique_authors"] / base["chatter_count"], np.nan
    )

    # ---- Subreddit entropy per (symbol, day) ----
    if df["subreddit"].notna().any():
        sub_counts = (
            df.dropna(subset=["subreddit"])
              .groupby(["day", "symbol", "subreddit"])
              .size()
              .reset_index(name="n")
        )
        entropy = (
            sub_counts.groupby(["day", "symbol"])["n"]
                      .apply(lambda c: _shannon_entropy(c.values))
                      .reset_index(name="subreddit_entropy")
        )
    else:
        # If subreddit is entirely missing, emit NaN entropy per (day, symbol)
        entropy = (
            base[["day", "symbol"]]
            .drop_duplicates()
            .assign(subreddit_entropy=np.nan)
        )

    out = (
        base.merge(entropy, on=["day", "symbol"], how="left")
            .sort_values(["day", "symbol"])
            .reset_index(drop=True)
    )

    # ---- Write per day (append-only) ----
    os.makedirs(out_root, exist_ok=True)
    days = out["day"].dropna().drop_duplicates().sort_values().tolist()
    written = 0

    for d in days:
        d = pd.Timestamp(d).normalize()
        d_iso = d.date().isoformat()
        d_dir = os.path.join(out_root, f"day={d_iso}")
        if not force and _existing_day_written(out_root, d):
            # skip existing partition
            continue
        os.makedirs(d_dir, exist_ok=True)
        part = out[out["day"] == d].copy()
        # Ensure clean dtypes
        part["day"] = pd.to_datetime(part["day"]).dt.tz_localize(None)
        # Unique part name
        part_path = os.path.join(d_dir, f"part-{uuid.uuid4().hex}.parquet")
        part.to_parquet(part_path, index=False)
        written += 1
        print(f"[WRITE] {part_path}  rows={len(part)}")

    print(f"[DONE] reddit_daily: wrote {written} day partitions (skipped existing: {len(days) - written}).")


def main():
    ap = argparse.ArgumentParser(description="Build daily Reddit features per (symbol, day).")
    ap.add_argument("--in-root", required=True, help="Path to data/derived/submission_tickers")
    ap.add_argument("--out-root", required=True, help="Output root for reddit_daily/")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--force", action="store_true", help="Overwrite existing day partitions")
    args = ap.parse_args()

    build_reddit_daily(
        in_root=args.in_root,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
