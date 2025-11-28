# scripts/explode_per_ticker_v4.py

import argparse
from pathlib import Path
import pandas as pd


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
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
        print(f"[v4] [INFO] empty input → {out_path}")
        return

    passthrough = [
        c
        for c in [
            "created_utc",
            "subreddit",
            "score",
            "num_comments",
            "parent_id",
            "link_id",
            "permalink",
            "author",
            "__day__",
        ]
        if c in df.columns
    ]

    rows = []
    for _, r in df.iterrows():
        items = ensure_list(r.get("ticker_items"))
        for t in items:
            if not isinstance(t, dict):
                continue
            sym = t.get("symbol")
            if not sym:
                continue
            rows.append(
                {
                    "id": r.get("id"),
                    "kind": r.get("kind"),
                    "symbol": sym,
                    "sentiment_label": t.get("sentiment_label"),
                    "sentiment_score": t.get("sentiment_score"),
                    "conf": t.get("conf"),
                    "is_forward": r.get("is_forward"),
                    "value_score": r.get("value_score"),
                    **{c: r.get(c) for c in passthrough},
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"[v4] Exploded {len(out):,} rows → {out_path}")


if __name__ == "__main__":
    main()
