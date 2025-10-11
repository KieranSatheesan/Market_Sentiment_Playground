import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.annotated_parquet)
    rows = []

    for _, r in df.iterrows():
        items = r["ticker_items"]

        # Convert to plain list safely
        if items is None:
            items = []
        elif hasattr(items, "tolist"):
            items = items.tolist()
        elif not isinstance(items, list):
            items = list(items)

        for t in items:
            if not isinstance(t, dict):
                continue
            rows.append({
                "id": r["id"],
                "subreddit": r.get("subreddit"),
                "symbol": t.get("symbol"),
                "sentiment_label": t.get("sentiment_label"),
                "sentiment_score": t.get("sentiment_score"),
                "confidence": t.get("confidence"),
                "evidence": t.get("evidence"),
            })

    df_out = pd.DataFrame(rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.out_parquet, index=False)
    print(f"Exploded {len(df_out)} ticker rows → {args.out_parquet}")

if __name__ == "__main__":
    main()
