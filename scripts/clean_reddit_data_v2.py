# scripts/clean_reddit_data_v2.py
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------- helpers ----------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def list_day_files(in_root: Path, kind: str) -> list[Tuple[str, list[Path]]]:
    """
    Returns list of (day, [files]) under in_root.
    - RS expects files named submissions_*.parquet
    - RC expects files named comments_*.parquet
    And in both cases, day directories are 'day=YYYY-MM-DD'.
    """
    assert kind in {"rs", "rc"}
    pattern = "submissions_*.parquet" if kind == "rs" else "comments_*.parquet"
    days = []
    for day_dir in sorted(in_root.glob("day=*")):
        if day_dir.is_dir():
            parts = list(day_dir.glob(pattern))
            if parts:
                days.append((day_dir.name.split("=", 1)[-1], parts))  # "day=YYYY-MM-DD" -> "YYYY-MM-DD"
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
    return pd.concat(dfs, ignore_index=True)

def basic_normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split())

# spam heuristics (conservative)
RE_REPEAT = re.compile(r"(.)\1{4,}")               # aaaaa or !!!!!! etc.
RE_URL    = re.compile(r"https?://\S+")
RE_WORD   = re.compile(r"[A-Za-z0-9]+")

def spam_flags(title: str, body: str) -> dict:
    # Both strings already normalized ideally
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

def mark_time_dupes(df: pd.DataFrame, keys: list[str], window_sec: int) -> pd.Series:
    """
    Mark near-duplicate rows where 'keys' repeat within window_sec (on increasing created_utc).
    Keeps the earliest occurrence, marks later repeats as duplicates.
    Expects 'created_utc' column.
    """
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)

    # build normalized columns for grouping
    data = {
        "__row_id": np.arange(len(df)),
        "created_utc": pd.to_numeric(df["created_utc"], errors="coerce").fillna(0).astype(np.int64),
    }
    for k in keys:
        data[k] = df[k].astype(str).str.lower().fillna("")

    tmp = pd.DataFrame(data)
    sort_cols = keys + ["created_utc", "__row_id"]
    tmp.sort_values(sort_cols, inplace=True)

    dup_mask = np.zeros(len(df), dtype=bool)

    group = tmp.groupby(keys, sort=False)
    for _, g in group:
        times = g["created_utc"].to_numpy()
        idxs  = g["__row_id"].to_numpy()
        if len(times) <= 1:
            continue
        last_keep_time = times[0]
        for i in range(1, len(times)):
            if times[i] - last_keep_time <= window_sec:
                dup_mask[idxs[i]] = True
            else:
                last_keep_time = times[i]

    return pd.Series(dup_mask, index=df.index, dtype=bool)

# ---------- main cleaning ----------

def clean_day_rs(df: pd.DataFrame, window_sec: int, drop_nsfw: bool) -> pd.DataFrame:
    if df.empty:
        return df

    # ensure expected columns exist
    for c in ["id","created_utc","subreddit","title","selftext","score","num_comments","author","over_18","permalink","url"]:
        if c not in df.columns:
            df[c] = np.nan

    # type coercions
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
    print(f"  - exact id dedup: {before} -> {len(df)} (-{before-len(df)})")

    # 2) near-dup by (author, title)
    dup_mask = mark_time_dupes(df, keys=["author","title"], window_sec=window_sec)
    n_near = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    print(f"  - near-dup (author+title within {window_sec}s): removed {n_near}")

    # 3) spam flags
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

    # 4) optional NSFW
    if drop_nsfw and "over_18" in df.columns:
        n_nsfw = int(df["over_18"].fillna(False).sum())
        df = df.loc[~df["over_18"].fillna(False)].copy()
        print(f"  - NSFW removed: {n_nsfw}")

    # normalize whitespace + build unified text
    df["title"] = df["title"].map(basic_normalize_text)
    df["selftext"] = df["selftext"].map(basic_normalize_text)
    df["text"] = (df["title"].fillna("") + "\n\n" + df["selftext"].fillna("")).str.strip()

    # drop rows with empty final text (double safety)
    n_empty = int((df["text"].str.len() == 0).sum())
    if n_empty:
        df = df.loc[df["text"].str.len() > 0].copy()
        print(f"  - removed {n_empty} rows with empty final text")

    return df

def clean_day_rc(df: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    if df.empty:
        return df

    # ensure expected columns exist
    for c in ["id","created_utc","subreddit","body","score","author","parent_id","link_id","permalink"]:
        if c not in df.columns:
            df[c] = np.nan

    # remove deleted/removed bodies
    body_norm = df["body"].astype(str).str.strip().str.lower()
    removed_mask = body_norm.isin({"[deleted]","[removed]",""})
    n_removed = int(removed_mask.sum())
    df = df.loc[~removed_mask].copy()
    if n_removed:
        print(f"  - removed deleted/removed/empty comments: {n_removed}")

    # type coercions
    df["id"] = df["id"].astype(str)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce").astype("Int64")
    df["subreddit"] = df["subreddit"].astype(str)
    df["body"] = df["body"].astype(str)
    df["author"] = df["author"].astype(str)
    df["parent_id"] = df["parent_id"].astype(str)
    df["link_id"] = df["link_id"].astype(str)

    # 1) exact dedup by id
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"  - exact id dedup: {before} -> {len(df)} (-{before-len(df)})")

    # 2) near-dup by (author, body_seed) within window
    body_seed = df["body"].astype(str).str.lower().str.strip().str.slice(0, 160)
    df["__body_seed"] = body_seed
    dup_mask = mark_time_dupes(df.assign(body_seed=body_seed), keys=["author","body_seed"], window_sec=window_sec)
    n_near = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    df.drop(columns=["__body_seed"], errors="ignore", inplace=True)
    print(f"  - near-dup (author+body_seed within {window_sec}s): removed {n_near}")

    # 3) spam flags (treat comment body as 'body'; no title)
    flags = df.apply(lambda r: spam_flags("", r.get("body","")), axis=1, result_type="expand")
    for col in flags.columns:
        df[col] = flags[col].astype(bool)
    spam_mask = (
        df["is_empty_text"]
        | df["has_repeat"]
        | df["too_many_symbols"]
        | df["body_mostly_urls"]
    )
    n_spam = int(spam_mask.sum())
    df = df.loc[~spam_mask].copy()
    print(f"  - spam-like filtered: removed {n_spam}")

    # normalize + build unified text
    df["body"] = df["body"].map(basic_normalize_text)
    df["text"] = df["body"].fillna("").str.strip()

    # drop empty final text
    n_empty = int((df["text"].str.len() == 0).sum())
    if n_empty:
        df = df.loc[df["text"].str.len() > 0].copy()
        print(f"  - removed {n_empty} rows with empty final text")

    return df

def main():
    ap = argparse.ArgumentParser(description="Clean Reddit processed parquet (RS or RC) into cleaned day partitions with unified 'text'.")
    ap.add_argument("--kind", choices=["rs","rc"], required=True, help="rs=submissions, rc=comments")
    ap.add_argument("--in_root", required=True, help="Processed root, e.g. data_v2/RedditDumps/processed/RS_2025-05")
    ap.add_argument("--out_root", required=True, help="Cleaned root, e.g. data_v2/RedditDumps/cleaned/RS_2025-05")
    ap.add_argument("--window_sec", type=int, default=3600, help="Near-dup time window (seconds)")
    ap.add_argument("--drop_nsfw", action="store_true", help="(RS only) Drop over_18 posts")
    ap.add_argument("--max_days", type=int, default=None, help="Process at most N day folders (for smoke tests)")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    by_day = list_day_files(in_root, kind=args.kind)
    if args.max_days is not None:
        by_day = by_day[: args.max_days]

    if not by_day:
        print(f"[INFO] No day folders with {'submissions' if args.kind=='rs' else 'comments'}_* found.")
        return

    for day, files in by_day:
        print(f"\n[DAY] {day} | files={len(files)} | kind={args.kind}")
        df = read_parquets(files)
        if df.empty:
            print("  - no rows")
            continue

        if args.kind == "rs":
            cleaned = clean_day_rs(df, window_sec=args.window_sec, drop_nsfw=args.drop_nsfw)
            out_name = "submissions_clean.parquet"
        else:
            cleaned = clean_day_rc(df, window_sec=args.window_sec)
            out_name = "comments_clean.parquet"

        day_out = out_root / f"day={day}"
        ensure_dir(day_out)
        out_path = day_out / out_name
        cleaned.to_parquet(out_path, index=False)
        print(f"  - wrote {len(cleaned):,} rows → {out_path}")

if __name__ == "__main__":
    main()
