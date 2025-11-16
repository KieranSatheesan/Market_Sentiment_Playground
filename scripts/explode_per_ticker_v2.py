# scripts/explode_per_ticker_v2.py
"""
Explode annotated parquet (per doc) into per-(doc × ticker) rows.

Inputs
- --annotated_parquet  e.g. data_v2/derived/comment_tickers/day=2025-05-01/annotated.parquet
- --out_parquet        e.g. data_v2/derived/comment_tickers/day=2025-05-01/exploded.parquet

Output columns (one row per ticker mention):
- id, kind ("submission"/"comment")
- symbol
- sentiment_label, sentiment_score  (per-ticker sentiment, -1..1)
- conf
- is_forward, value_score
- passthrough from the doc level (if present): created_utc, subreddit, score (Reddit karma),
  num_comments (for RS), parent_id/link_id/permalink/author (for RC)
"""

import argparse
from pathlib import Path
import pandas as pd


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        # pyarrow list-like
        return list(x)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()

    ann_path = Path(args.annotated_parquet)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ann_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print(f"[INFO] empty input → {out_path}")
        return

    rows = []
    # Pass-through doc-level columns (Reddit score stays as `score`)
    passthrough_cols = [
        c
        for c in [
            "created_utc",
            "subreddit",
            "score",          # Reddit upvotes
            "num_comments",
            "parent_id",
            "link_id",
            "permalink",
            "author",
        ]
        if c in df.columns
    ]

    for _, r in df.iterrows():
        items = ensure_list(r.get("ticker_items"))
        if not isinstance(items, list) or not items:
            continue

        for t in items:
            if not isinstance(t, dict):
                continue

            sym = t.get("symbol")
            if not sym:
                continue

            # Handle both v2-style (label/score) and v3-style (sentiment_label/sentiment_score)
            sentiment_label = t.get("label", t.get("sentiment_label"))
            sentiment_score = t.get("sentiment_score", t.get("score"))
            conf = t.get("conf")

            rows.append(
                {
                    "id": r.get("id"),
                    "kind": r.get("kind"),
                    "symbol": sym,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "conf": conf,
                    "is_forward": r.get("is_forward"),
                    "value_score": r.get("value_score"),
                    **{c: r.get(c) for c in passthrough_cols},
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"Exploded {len(out):,} rows → {out_path}")


if __name__ == "__main__":
    main()
