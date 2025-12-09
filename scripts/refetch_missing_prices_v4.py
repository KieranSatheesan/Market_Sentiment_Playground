#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refetch missing Yahoo Finance OHLCV for a given window and append them
to existing prices_yahoo_v4 partitions, avoiding duplicates on (symbol, day).

Intended use:
  - First run fetch_prices_v4.py (main pipeline).
  - It writes missing_symbols_YYYY-MM-DD_to_YYYY-MM-DD.txt.
  - Then run this script to retry JUST those symbols (with throttling).

Behavior:
  - Reads a text file with one symbol per line (ignores blanks and '#'-comments).
  - Fetches OHLCV for [start, end] (inclusive) via yfinance.
  - Computes ret1, ret5, ret1_roll_std, vol_roll_mean (same logic as fetch_prices_v4).
  - Loads existing prices_yahoo_v4 for [start, end], and drops any
    (symbol, day) pairs that are already present.
  - Writes new parquet parts in out_root/day=YYYY-MM-DD/ with only new rows.
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


# -------- helpers --------

def _list_days_between(start: str, end: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    return list(pd.date_range(s, e, freq="D"))


def _read_existing_pairs(out_root: str, start: str, end: str) -> Set[Tuple[str, pd.Timestamp]]:
    """
    Read existing prices_yahoo_v4 partitions in [start, end] and return
    the set of (symbol, day) pairs already present.
    """
    days = _list_days_between(start, end)
    pairs: Set[Tuple[str, pd.Timestamp]] = set()

    for d in days:
        d_dir = os.path.join(out_root, f"day={d.date().isoformat()}")
        parts = glob.glob(os.path.join(d_dir, "*.parquet"))
        if not parts:
            continue
        try:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Failed to read existing prices for {d.date()}: {e}")
            continue

        if df.empty or "symbol" not in df.columns or "day" not in df.columns:
            continue

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["day"] = pd.to_datetime(df["day"]).dt.normalize()

        for sym, day_val in zip(df["symbol"], df["day"]):
            pairs.add((sym, day_val))

    print(f"[INFO] Existing (symbol, day) pairs in prices_v4: {len(pairs)}")
    return pairs


def _load_missing_symbols(path: str) -> List[str]:
    """
    Load missing symbols from a text file.
    Expected format: one symbol per line. Lines starting with '#' are ignored.
    If the file has extra junk, we try to strip brackets and quotes.
    """
    syms: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try to be robust against "['TSLA']" or "TSLA: reason..."
            # Take first token, strip brackets/quotes/commas.
            token = line.split()[0]
            token = token.strip("[],'\"")
            if not token:
                continue
            syms.add(token.upper().strip())
    out = sorted(syms)
    print(f"[INFO] Loaded {len(out)} missing symbols from {path}")
    return out


def _download_batch(
    tickers: List[str],
    start_inclusive: str,
    end_exclusive: str,
    max_retries: int = 3,
    sleep: float = 2.0,
) -> pd.DataFrame:
    """
    Return tidy long OHLCV for successfully fetched tickers in the batch.
    Skips unfetchables (no empty frames returned).
    """
    if not tickers:
        return pd.DataFrame()

    last_err = None
    for attempt in range(max_retries):
        try:
            data = yf.download(
                " ".join(tickers),
                start=start_inclusive,
                end=end_exclusive,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] Batch download error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(sleep * (1.5 ** attempt))
    else:
        print(f"[WARN] Batch download failed for tickers {tickers[:5]}... : {last_err}")
        return pd.DataFrame()

    long_rows = []
    # Multi-ticker: MultiIndex columns [ticker, field]
    if isinstance(data.columns, pd.MultiIndex):
        present = list(data.columns.levels[0])
        for t in tickers:
            if t not in present:
                continue
            df_t = data[t]
            if not isinstance(df_t, pd.DataFrame) or df_t.dropna(how="all").empty:
                continue
            tmp = df_t.reset_index().rename(columns={"Date": "day"})
            tmp["symbol"] = t
            keep = ["day", "symbol"] + [
                c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                if c in tmp.columns
            ]
            long_rows.append(tmp[keep])
    else:
        # Single-ticker flat DF case – ignore to avoid mislabeling
        if isinstance(data, pd.DataFrame) and not data.dropna(how="all").empty:
            pass

    if not long_rows:
        return pd.DataFrame()
    return pd.concat(long_rows, ignore_index=True)


# -------- main logic --------

def refetch_missing_prices_v4(
    missing_file: str,
    out_root: str,
    start: str,
    end: str,
    batch: int = 25,
    batch_sleep: float = 5.0,
    ret_roll: int = 5,
    vol_roll: int = 5,
) -> None:
    missing_syms = _load_missing_symbols(missing_file)
    if not missing_syms:
        print("[INFO] No missing symbols to refetch – file was empty.")
        return

    # Discover existing (symbol, day) to avoid duplicates
    existing_pairs = _read_existing_pairs(out_root, start, end)

    start_dt = pd.to_datetime(start).normalize()
    end_dt = pd.to_datetime(end).normalize()
    dl_start = start_dt.strftime("%Y-%m-%d")
    dl_end = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    long_frames = []
    total = len(missing_syms)
    if batch <= 0:
        batch = 25

    batch_idx = 0
    for i in range(0, total, batch):
        chunk = missing_syms[i : i + batch]
        batch_idx += 1
        print(f"[INFO] Refetch batch {batch_idx} ({len(chunk)} symbols)...")
        df_long = _download_batch(chunk, dl_start, dl_end)
        if not df_long.empty:
            long_frames.append(df_long)
        if batch_sleep and batch_sleep > 0:
            time.sleep(batch_sleep)

    if not long_frames:
        print("[INFO] No new price data fetched for missing symbols.")
        return

    prices_new = pd.concat(long_frames, ignore_index=True)
    prices_new["day"] = pd.to_datetime(prices_new["day"]).dt.tz_localize(None)
    prices_new["symbol"] = prices_new["symbol"].astype(str).str.upper().str.strip()

    # Clamp to [start_dt, end_dt]
    prices_new = prices_new[(prices_new["day"] >= start_dt) & (prices_new["day"] <= end_dt)]
    if prices_new.empty:
        print("[INFO] New price rows exist but none in requested window after clamping.")
        return

    # Drop (symbol, day) that already exist
    prices_new["day_norm"] = prices_new["day"].dt.normalize()
    idx_new = list(zip(prices_new["symbol"], prices_new["day_norm"]))
    mask_new = [pair not in existing_pairs for pair in idx_new]
    prices_new = prices_new.loc[mask_new].copy()
    if prices_new.empty:
        print("[INFO] All (symbol, day) rows for missing symbols already present; nothing to append.")
        return

    prices_new = prices_new.sort_values(["symbol", "day"]).reset_index(drop=True)

    # Compute returns using Adjusted Close if present
    price_col = "Adj Close" if "Adj Close" in prices_new.columns else "Close"
    prices_new["ret1"] = prices_new.groupby("symbol")[price_col].pct_change(1)
    prices_new["ret5"] = prices_new.groupby("symbol")[price_col].pct_change(5)

    # Rolling labels
    prices_new["ret1_roll_std"] = (
        prices_new.groupby("symbol")["ret1"]
        .rolling(ret_roll, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    prices_new["vol_roll_mean"] = (
        prices_new.groupby("symbol")["Volume"]
        .rolling(vol_roll, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Write as *additional* parquet parts per day
    os.makedirs(out_root, exist_ok=True)
    written = 0
    for d in sorted(prices_new["day_norm"].dropna().unique()):
        d = pd.Timestamp(d).normalize()
        d_iso = d.date().isoformat()
        d_dir = os.path.join(out_root, f"day={d_iso}")
        os.makedirs(d_dir, exist_ok=True)

        part = prices_new[prices_new["day_norm"] == d].copy()
        if part.empty:
            continue
        part.drop(columns=["day_norm"], inplace=True)
        out_path = os.path.join(d_dir, f"part-refetch-{uuid.uuid4().hex}.parquet")
        part.to_parquet(out_path, index=False)
        written += 1
        print(f"[WRITE] {out_path}  rows={len(part)}")

    print(f"[DONE] refetch_missing_prices_v4: wrote {written} day partitions.")


def main():
    ap = argparse.ArgumentParser(
        description="Refetch missing Yahoo OHLCV for v4 prices_yahoo using a missing_symbols.txt file."
    )
    ap.add_argument("--missing-file", required=True, help="Path to missing_symbols_YYYY-MM-DD_to_YYYY-MM-DD.txt")
    ap.add_argument("--out-root", required=True, help="Output root for featuresets/prices_yahoo_v4")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--batch", type=int, default=25, help="Tickers per yfinance batch")
    ap.add_argument("--batch-sleep", type=float, default=5.0, help="Seconds to sleep between batches")
    ap.add_argument("--ret-roll", type=int, default=5)
    ap.add_argument("--vol-roll", type=int, default=5)
    args = ap.parse_args()

    refetch_missing_prices_v4(
        missing_file=args.missing_file,
        out_root=args.out_root,
        start=args.start,
        end=args.end,
        batch=args.batch,
        batch_sleep=args.batch_sleep,
        ret_roll=args.ret_roll,
        vol_roll=args.vol_roll,
    )


if __name__ == "__main__":
    main()
