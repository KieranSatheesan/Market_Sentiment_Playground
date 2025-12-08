#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetch OHLCV from Yahoo Finance for symbols present in v4 reddit_daily,
compute basic market labels, and write per-day parquet partitions.

- Input:  <reddit_daily_root>/day=YYYY-MM-DD/*.parquet
- Output: <out_root>/day=YYYY-MM-DD/part-*.parquet

Behavior:
- Discovers symbols from reddit_daily_v4 for the requested window.
- Batch-downloads via yfinance; skips unfetchable tickers.
- Uses Adjusted Close for returns if present, else Close.
- Computes ret1, ret5, ret1_roll_std (ret_roll=5), vol_roll_mean (vol_roll=5).
- Append-only; skips existing days unless --force.

Note:
- yfinance treats `end` as exclusive. We add +1 day to include the final day,
  then clamp rows back to [start, end] before writing.
"""

import argparse
import os
import glob
import uuid
import time
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def _list_days_between(start: str, end: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    return list(pd.date_range(s, e, freq="D"))


def _existing_day_written(out_root: str, day: pd.Timestamp) -> bool:
    dpath = os.path.join(out_root, f"day={day.date().isoformat()}")
    return os.path.isdir(dpath) and bool(glob.glob(os.path.join(dpath, "*.parquet")))


def _discover_symbols(
    reddit_daily_root: str, start: str, end: str
) -> Tuple[Set[str], List[pd.Timestamp]]:
    days = _list_days_between(start, end)
    symbols: Set[str] = set()
    found_days: List[pd.Timestamp] = []
    for d in days:
        d_dir = os.path.join(reddit_daily_root, f"day={d.date().isoformat()}")
        parts = glob.glob(os.path.join(d_dir, "*.parquet"))
        if not parts:
            continue
        try:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            syms = (
                df["symbol"]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
                .unique()
                .tolist()
            )
            symbols.update(syms)
            found_days.append(d)
        except Exception as e:
            print(f"[WARN] Failed to read reddit_daily_v4 {d.date()}: {e}")
            continue
    return symbols, found_days


def _download_batch(
    tickers: List[str],
    start_inclusive: str,
    end_exclusive: str,
    max_retries: int = 3,
    sleep: float = 2.0,
) -> pd.DataFrame:
    """
    Return tidy long OHLCV for successfully fetched tickers in the batch.
    Skips unfetchables (no non-empty frames returned).

    Parameters:
      start_inclusive: YYYY-MM-DD
      end_exclusive:   YYYY-MM-DD (yfinance will not include this date)
    """
    if not tickers:
        return pd.DataFrame()

    last_err = None
    for i in range(max_retries):
        try:
            # IMPORTANT: pass the list directly, not a joined string
            data = yf.download(
                tickers=tickers,
                start=start_inclusive,
                end=end_exclusive,  # exclusive per yfinance
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] Batch download error (attempt {i+1}/{max_retries}): {e}")
            time.sleep(sleep * (1.5 ** i))
    else:
        print(f"[WARN] Batch download failed after {max_retries} attempts: {last_err}")
        return pd.DataFrame()

    if data is None or (isinstance(data, pd.DataFrame) and data.dropna(how="all").empty):
        return pd.DataFrame()

    long_rows = []

    # Multi-ticker case → MultiIndex columns, one top level per ticker
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        present = list(data.columns.levels[0])
        missing_in_batch = sorted(set(tickers) - set(present))
        if missing_in_batch:
            print(f"[INFO] In this batch, no data returned for: {missing_in_batch}")

        for t in tickers:
            if t not in present:
                continue
            df_t = data[t]
            if not isinstance(df_t, pd.DataFrame) or df_t.dropna(how="all").empty:
                continue
            tmp = df_t.reset_index().rename(columns={"Date": "day"})
            tmp["symbol"] = t
            keep = ["day", "symbol"] + [
                c
                for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                if c in tmp.columns
            ]
            long_rows.append(tmp[keep])

    # Single-ticker case → flat columns
    elif isinstance(data, pd.DataFrame):
        if len(tickers) == 1:
            t = tickers[0]
            df_t = data
            if not df_t.dropna(how="all").empty:
                tmp = df_t.reset_index().rename(columns={"Date": "day"})
                tmp["symbol"] = t
                keep = ["day", "symbol"] + [
                    c
                    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                    if c in tmp.columns
                ]
                long_rows.append(tmp[keep])
        else:
            # Unexpected: multiple tickers requested but got flat columns.
            # Log and skip rather than mis-assign.
            print(
                "[WARN] Unexpected flat DataFrame for multi-ticker batch; "
                f"tickers={tickers}. Skipping this batch."
            )

    if not long_rows:
        return pd.DataFrame()

    return pd.concat(long_rows, ignore_index=True)


def fetch_prices_v4(
    reddit_daily_root: str,
    out_root: str,
    start: str,
    end: str,
    batch: int = 150,
    ret_roll: int = 5,
    vol_roll: int = 5,
    force: bool = False,
) -> None:
    # Discover symbols from reddit_daily (inclusive window)
    symbols, days_with_reddit = _discover_symbols(reddit_daily_root, start, end)
    if not symbols:
        print("[INFO] No symbols discovered in reddit_daily_v4 for this window.")
        return

    symbols = sorted({s.strip().upper() for s in symbols if s})
    print(
        f"[INFO] Symbols discovered in reddit_daily_v4: {len(symbols)}; "
        f"days with reddit activity: {len(days_with_reddit)}"
    )

    # yfinance end is exclusive → add +1 day to include requested final day
    start_dt = pd.to_datetime(start).normalize()
    end_dt = pd.to_datetime(end).normalize()
    dl_start = start_dt.strftime("%Y-%m-%d")
    dl_end = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Download all data for full range in batches
    long_frames = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        print(f"[INFO] Downloading batch {i//batch + 1} with {len(chunk)} symbols...")
        df_long = _download_batch(chunk, dl_start, dl_end)
        if not df_long.empty:
            long_frames.append(df_long)

    if not long_frames:
        print("[INFO] No price data fetched at all.")
        return

    prices = pd.concat(long_frames, ignore_index=True)
    prices["day"] = pd.to_datetime(prices["day"]).dt.tz_localize(None)
    prices["symbol"] = prices["symbol"].astype(str).str.upper().str.strip()

    # Clamp to the original inclusive window [start_dt, end_dt]
    prices = prices[(prices["day"] >= start_dt) & (prices["day"] <= end_dt)]
    if prices.empty:
        print(
            "[INFO] Price rows exist but none within the requested window after clamping."
        )
        return

    prices = prices.sort_values(["symbol", "day"]).reset_index(drop=True)

    # Compute returns using Adjusted Close if present
    price_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
    prices["ret1"] = prices.groupby("symbol")[price_col].pct_change(1)
    prices["ret5"] = prices.groupby("symbol")[price_col].pct_change(5)

    # Rolling labels
    prices["ret1_roll_std"] = (
        prices.groupby("symbol")["ret1"]
        .rolling(ret_roll, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    if "Volume" in prices.columns:
        prices["vol_roll_mean"] = (
            prices.groupby("symbol")["Volume"]
            .rolling(vol_roll, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        prices["vol_roll_mean"] = np.nan

    # Helpful summary + missing-symbols log
    fetched_syms = sorted(prices["symbol"].unique().tolist())
    print(
        f"[INFO] Fetched rows for {len(fetched_syms)} / {len(symbols)} symbols "
        f"within {start} → {end}"
    )
    missing_syms = sorted(set(symbols) - set(fetched_syms))
    if missing_syms:
        os.makedirs(out_root, exist_ok=True)
        missing_path = os.path.join(
            out_root, f"missing_symbols_{start}_to_{end}.txt"
        )
        with open(missing_path, "w", encoding="utf-8") as f:
            for s in missing_syms:
                f.write(s + "\n")
        print(
            f"[INFO] Symbols with no price data in this window: {len(missing_syms)} "
            f"(written to {missing_path})"
        )

    # Write per-day partitions
    os.makedirs(out_root, exist_ok=True)
    days = sorted(prices["day"].dropna().dt.normalize().unique())

    written = 0
    for d in days:
        d = pd.Timestamp(d).normalize()
        d_iso = d.date().isoformat()
        d_dir = os.path.join(out_root, f"day={d_iso}")
        if not force and _existing_day_written(out_root, d):
            continue
        os.makedirs(d_dir, exist_ok=True)
        part = prices[prices["day"] == d].copy()
        if part.empty:
            continue
        out_path = os.path.join(d_dir, f"part-{uuid.uuid4().hex}.parquet")
        part.to_parquet(out_path, index=False)
        written += 1
        print(f"[WRITE] {out_path}  rows={len(part)}")

    print(
        f"[DONE] prices_yahoo_v4: wrote {written} day partitions "
        f"(skipped existing: {len(days) - written})."
    )


def main():
    ap = argparse.ArgumentParser(
        description="Fetch Yahoo OHLCV and compute basic labels per (symbol, day) for v4 reddit_daily."
    )
    ap.add_argument(
        "--reddit-daily-root", required=True, help="Path to featuresets/reddit_daily_v4"
    )
    ap.add_argument(
        "--out-root", required=True, help="Output root for featuresets/prices_yahoo_v4"
    )
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--ret-roll", type=int, default=5)
    ap.add_argument("--vol-roll", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fetch_prices_v4(
        reddit_daily_root=args.reddit_daily_root,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        batch=args.batch,
        ret_roll=args.ret_roll,
        vol_roll=args.vol_roll,
        force=args.force,
    )


if __name__ == "__main__":
    main()
