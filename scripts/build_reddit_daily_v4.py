#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build daily Reddit features per (symbol, day) from v4 exploded parquet
(submissions + comments).

Inputs (example):
  data_v4/derived/submission_tickers/day=YYYY-MM-DD/exploded.parquet
  data_v4/derived/comment_tickers/day=YYYY-MM-DD/exploded.parquet

Output:
  <out_root>/day=YYYY-MM-DD/part-*.parquet

Notes:
- Aggregates across BOTH submissions and comments.
- Uses v4 columns: symbol, sentiment_score, conf, is_forward, value_score,
  asset_type, kind, created_utc, subreddit, score, num_comments, author.
- No rolling windows here; those are added later (enrich_features_v4.py).
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


def _read_exploded_for_day(root: str, day: pd.Timestamp) -> pd.DataFrame:
    """
    Read exploded parquet for a given day from a root like:
      data_v4/derived/submission_tickers  (expects day=YYYY-MM-DD/exploded.parquet)
    """
    day_iso = day.date().isoformat()
    p = os.path.join(root, f"day={day_iso}", "exploded.parquet")
    if not os.path.isfile(p):
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as e:
        print(f"[WARN] Failed to read exploded for {day_iso} under {root}: {e}")
        return pd.DataFrame()


def _list_days_between(start: str, end: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    return list(pd.date_range(s, e, freq="D"))


# ------------------------- main logic -------------------------


def build_reddit_daily_v4(
    sub_root: str,
    cmt_root: str,
    out_root: str,
    start: str,
    end: str,
    force: bool = False,
) -> None:
    days = _list_days_between(start, end)
    os.makedirs(out_root, exist_ok=True)

    written = 0

    for day in days:
        day_iso = day.date().isoformat()
        if not force and _existing_day_written(out_root, day):
            print(f"[SKIP] reddit_daily_v4 for {day_iso} already exists.")
            continue

        sub_df = _read_exploded_for_day(sub_root, day)
        cmt_df = _read_exploded_for_day(cmt_root, day)

        if sub_df.empty and cmt_df.empty:
            print(f"[INFO] No exploded rows for {day_iso} (submissions or comments)")
            continue

        df = pd.concat([sub_df, cmt_df], ignore_index=True)

        # ---- Clean / normalize basic columns ----
        df["symbol"] = (
            df.get("symbol", "")
            .astype(str)
            .str.upper()
            .str.strip()
        )
        df = df[df["symbol"].str.len() > 0]

        df["created_utc"] = pd.to_numeric(df.get("created_utc", np.nan), errors="coerce")
        df["day"] = (
            pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
            .dt.floor("D")
        )

        # Some rows might have weird timestamps; force to current day when missing
        df["day"] = df["day"].fillna(day).dt.floor("D")

        # Score / num_comments mostly exist for submissions; treat NaN as 0
        df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
        df["num_comments"] = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)

        df["sentiment_score"] = pd.to_numeric(
            df.get("sentiment_score", np.nan), errors="coerce"
        )
        df["conf"] = pd.to_numeric(df.get("conf", np.nan), errors="coerce").clip(lower=0, upper=1)

        # Boolean-ish is_forward
        is_fwd = df.get("is_forward")
        df["is_forward"] = is_fwd.where(is_fwd.isin([True, False]), np.nan)

        df["value_score"] = pd.to_numeric(df.get("value_score", np.nan), errors="coerce")

        # Subreddit + author
        df["subreddit"] = df.get("subreddit")
        df["author"] = df.get("author")

        # Asset type (may be missing, but keep if present)
        df["asset_type"] = df.get("asset_type")

        # Filter essentials
        df = df.dropna(subset=["symbol", "sentiment_score"])
        if df.empty:
            print(f"[INFO] No valid Reddit rows with sentiment for {day_iso}")
            continue

        # Limit to *this* day (in case there is any spill-over)
        df = df[df["day"] == day]
        if df.empty:
            print(f"[INFO] After clamping to day={day_iso}, no rows remain.")
            continue

        # ---- Aggregations per (day, symbol) ----
        grp = df.groupby(["day", "symbol"], sort=True)

        base = grp.agg(
            chatter_count=("id", "count"),
            sent_mean=("sentiment_score", "mean"),
            sent_w_num=("sentiment_score",
                        lambda s: float(np.sum(s * df.loc[s.index, "conf"]))),
            sent_w_den=("conf", "sum"),
            avg_conf=("conf", "mean"),
            median_conf=("conf", "median"),
            score_sum=("score", "sum"),
            score_mean=("score", "mean"),
            comments_sum=("num_comments", "sum"),
            comments_mean=("num_comments", "mean"),
            unique_authors=("author", lambda s: s.nunique()),
            # Forward / value stats
            forward_true_count=("is_forward",
                                lambda x: float((x == True).sum())),
            forward_total=("is_forward",
                           lambda x: float(x.notna().sum())),
            value_mean=("value_score", "mean"),
            value_median=("value_score", "median"),
        ).reset_index()

        # Weighted sentiment
        base["sent_weighted"] = np.where(
            base["sent_w_den"] > 0, base["sent_w_num"] / base["sent_w_den"], np.nan
        )
        base.drop(columns=["sent_w_num", "sent_w_den"], inplace=True)

        # Forward share
        base["forward_share"] = np.where(
            base["forward_total"] > 0,
            base["forward_true_count"] / base["forward_total"],
            np.nan,
        )

        # Engagement composites
        base["engagement_sum"] = base["score_sum"] + base["comments_sum"]
        base["engagement_mean"] = base["score_mean"] + base["comments_mean"]
        base["user_concentration"] = np.where(
            base["chatter_count"] > 0,
            base["unique_authors"] / base["chatter_count"],
            np.nan,
        )

        # Dominant asset_type per (day, symbol) if available
        if df["asset_type"].notna().any():
            asset_mode = (
                df.dropna(subset=["asset_type"])
                .groupby(["day", "symbol"])["asset_type"]
                .agg(lambda s: s.value_counts().idxmax())
                .reset_index()
            )
            base = base.merge(asset_mode, on=["day", "symbol"], how="left")
        else:
            base["asset_type"] = np.nan

        # Subreddit entropy per (day, symbol)
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

        # ---- Write per day ----
        d_dir = os.path.join(out_root, f"day={day_iso}")
        os.makedirs(d_dir, exist_ok=True)
        out_path = os.path.join(d_dir, f"part-{uuid.uuid4().hex}.parquet")
        # Ensure 'day' is naive datetime
        out["day"] = pd.to_datetime(out["day"]).dt.tz_localize(None)
        out.to_parquet(out_path, index=False)
        written += 1
        print(f"[WRITE] reddit_daily_v4 {out_path}  rows={len(out)}")

    print(f"[DONE] reddit_daily_v4: wrote {written} day partitions (skipped existing where applicable).")


def main():
    ap = argparse.ArgumentParser(
        description="Build v4 daily Reddit features per (symbol, day) from submission+comment exploded parquets."
    )
    ap.add_argument("--sub-root", required=True, help="Path to data_v4/derived/submission_tickers")
    ap.add_argument("--cmt-root", required=True, help="Path to data_v4/derived/comment_tickers")
    ap.add_argument("--out-root", required=True, help="Output root for featuresets/reddit_daily_v4")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--force", action="store_true", help="Overwrite existing day partitions")
    args = ap.parse_args()

    build_reddit_daily_v4(
        sub_root=args.sub_root,
        cmt_root=args.cmt_root,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
