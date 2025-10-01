# scripts/clean_reddit_data.py
import argparse
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------- helpers ----------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def list_day_files(in_root: Path) -> list[Path]:
    # expects .../processed/YYYY-MM-DD/submissions_*.parquet
    days = []
    for day_dir in sorted(in_root.glob("*")):
        if day_dir.is_dir():
            parts = list(day_dir.glob("submissions_*.parquet"))
            if parts:
                days.append((day_dir.name, parts))
    return days

def read_parquets(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df

def basic_normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split())

# spam heuristics (conservative)
RE_REPEAT = re.compile(r"(.)\1{4,}")               # aaaaa or !!!!!!
RE_URL    = re.compile(r"https?://\S+")
RE_WORD   = re.compile(r"[A-Za-z0-9]+")

def spam_flags(title: str, body: str) -> dict:
    t = title or ""
    b = body or ""
    t_norm = basic_normalize_text(t)
    b_norm = basic_normalize_text(b)

    joined = f"{t_norm} {b_norm}".strip()

    # empty after strip
    is_empty = (len(t_norm) == 0 and len(b_norm) == 0)

    # excessive repetition
    has_repeat = bool(RE_REPEAT.search(joined))

    # symbol ratio (non-word chars overwhelming)
    total_chars = len(joined)
    word_chars = sum(len(m.group(0)) for m in RE_WORD.finditer(joined))
    sym_ratio = 1.0 if total_chars == 0 else max(0.0, 1 - (word_chars / total_chars))
    too_many_symbols = sym_ratio > 0.55 and total_chars >= 40  # avoid penalizing short strings

    # all-caps title (but ignore short titles)
    title_is_all_caps = (len(t_norm) >= 8 and t_norm.upper() == t_norm and any(c.isalpha() for c in t_norm))

    # url-only body (very common spam)
    body_is_mostly_urls = False
    if len(b_norm) >= 8:
        url_chars = sum(len(m.group(0)) for m in RE_URL.finditer(b_norm))
        body_is_mostly_urls = (url_chars / max(len(b_norm), 1)) > 0.7

    return {
        "is_empty_text": is_empty,
        "has_repeat": has_repeat,
        "too_many_symbols": too_many_symbols,
        "title_all_caps": title_is_all_caps,
        "body_mostly_urls": body_is_mostly_urls,
    }

def mark_time_dupes(df: pd.DataFrame, window_sec: int = 3600) -> pd.Series:
    """
    Mark near-duplicate posts where (author, title) repeats within window_sec.
    Assumes df has 'author', 'title', 'created_utc'.
    We keep the earliest occurrence, mark later repeats as duplicates.
    """
    if df.empty:
        return pd.Series([], dtype=bool)

    # normalize keys
    a = (df["author"].astype(str).str.lower().fillna(""))
    t = (df["title"].astype(str).str.strip().str.lower().fillna(""))
    c = pd.to_numeric(df["created_utc"], errors="coerce").fillna(0).astype(np.int64)

    tmp = pd.DataFrame({"author": a, "title": t, "created_utc": c, "__row_id": np.arange(len(df))})
    tmp.sort_values(["author", "title", "created_utc", "__row_id"], inplace=True)
    dup_mask = np.zeros(len(df), dtype=bool)

    # walk each (author, title) group and mark events within window_sec
    for _, g in tmp.groupby(["author", "title"], sort=False):
        times = g["created_utc"].to_numpy()
        idxs  = g["__row_id"].to_numpy()
        if len(times) <= 1:
            continue
        # keep the first in each "burst"
        last_keep_time = times[0]
        for i in range(1, len(times)):
            if times[i] - last_keep_time <= window_sec:
                dup_mask[idxs[i]] = True
            else:
                last_keep_time = times[i]

    return pd.Series(dup_mask, index=df.index, dtype=bool)

# ---------- main cleaning ----------

def clean_day(df: pd.DataFrame, window_sec: int, drop_nsfw: bool) -> pd.DataFrame:
    if df.empty:
        return df

    # Expected columns (graceful if some are missing)
    for c in ["id","created_utc","subreddit","title","selftext","score","num_comments","author","over_18"]:
        if c not in df.columns:
            df[c] = np.nan

    # Type coercions
    df["id"] = df["id"].astype(str)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce").astype("Int64")
    df["subreddit"] = df["subreddit"].astype(str)
    df["title"] = df["title"].astype(str)
    df["selftext"] = df["selftext"].astype(str)
    df["author"] = df["author"].astype(str)
    if "over_18" in df.columns:
        df["over_18"] = df["over_18"].astype("boolean")

    # 1) exact dedup by id
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    after = len(df)
    print(f"  - exact id dedup: {before} -> {after} (-{before-after})")

    # 2) near-dup by (author, title) within window
    dup_mask = mark_time_dupes(df, window_sec=window_sec)
    n_near = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    print(f"  - near-dup (author+title within {window_sec}s): removed {n_near}")

    # 3) spam flags (conservative)
    flags = df.apply(lambda r: spam_flags(r.get("title",""), r.get("selftext","")), axis=1, result_type="expand")
    for col in flags.columns:
        df[col] = flags[col].astype(bool)
    spam_mask = (
        df["is_empty_text"]
        | df["has_repeat"]
        | df["too_many_symbols"]
        | df["title_all_caps"]
        | df["body_mostly_urls"]
    )
    n_spam = int(spam_mask.sum())
    df = df.loc[~spam_mask].copy()
    print(f"  - spam-like filtered: removed {n_spam}")

    # 4) optional: drop NSFW
    if drop_nsfw and "over_18" in df.columns:
        n_nsfw = int(df["over_18"].fillna(False).sum())
        df = df.loc[~df["over_18"].fillna(False)].copy()
        print(f"  - NSFW removed: {n_nsfw}")

    # final normalize whitespace on text columns (cheap)
    df["title"] = df["title"].map(basic_normalize_text)
    df["selftext"] = df["selftext"].map(basic_normalize_text)

    return df

def main():
    ap = argparse.ArgumentParser(description="Clean Reddit bronze submissions into a cleaned mirror.")
    ap.add_argument("--in_root", required=True, help="Path to bronze root, e.g. data/RedditDumps/processed")
    ap.add_argument("--out_root", required=True, help="Path to cleaned root, e.g. data/RedditDumps/cleaned")
    ap.add_argument("--window_sec", type=int, default=3600, help="Time window (seconds) for near-dup detection")
    ap.add_argument("--drop_nsfw", action="store_true", help="Drop over_18 posts")
    ap.add_argument("--max_days", type=int, default=None, help="Process at most N day folders (for smoke tests)")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    by_day = list_day_files(in_root)
    if args.max_days is not None:
        by_day = by_day[: args.max_days]

    if not by_day:
        print("[INFO] No day folders with submissions_* found.")
        return

    for day, files in by_day:
        print(f"\n[DAY] {day} | files={len(files)}")
        df = read_parquets(files)
        if df.empty:
            print("  - no rows")
            continue

        cleaned = clean_day(df, window_sec=args.window_sec, drop_nsfw=args.drop_nsfw)
        day_out = out_root / day
        ensure_dir(day_out)
        out_path = day_out / "submissions_clean.parquet"
        cleaned.to_parquet(out_path, index=False)
        print(f"  - wrote {len(cleaned):,} rows → {out_path}")

if __name__ == "__main__":
    main()
