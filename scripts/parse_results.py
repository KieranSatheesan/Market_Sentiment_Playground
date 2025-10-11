# scripts/parse_results.py  (replace your current file with this)

import json, argparse
from pathlib import Path
import pandas as pd
from pydantic import BaseModel

class Ticker(BaseModel):
    symbol: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    evidence: str | None = None

class PostResult(BaseModel):
    submission_id: str
    tickers: list[Ticker]

def load_clean_day(day_dir: Path) -> pd.DataFrame:
    p = day_dir / "submissions_clean.parquet"
    if p.exists(): return pd.read_parquet(p)
    parts = list(day_dir.glob("submissions_*.parquet"))
    return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True) if parts else pd.DataFrame()

def extract_output_text(obj: dict) -> str | None:
    """
    Handles different Responses API shapes.

    Preferred:
      obj["response"]["body"]["output"][0]["content"][0]["text"]
    Fallbacks (older shapes):
      obj["response"]["output_text"]
      obj["response"]["output"][0]["content"][0]["text"]
    """
    resp = obj.get("response") or {}
    # 1) Newest: response.body.output[..].content[..].text
    body = resp.get("body")
    if isinstance(body, dict):
        out = body.get("output")
        if isinstance(out, list) and out:
            msg0 = out[0]
            if isinstance(msg0, dict):
                content = msg0.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                        return c0["text"]

    # 2) Older convenience field
    txt = resp.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt

    # 3) Older: response.output[..].content[..].text
    out = resp.get("output")
    if isinstance(out, list) and out:
        msg0 = out[0]
        if isinstance(msg0, dict):
            content = msg0.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                    return c0["text"]

    return None

def parse_output_text(txt: str) -> list[PostResult]:
    try:
        payload = json.loads(txt)
        arr = payload.get("results", [])
        return [PostResult(**x) for x in arr]
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_day_dir", required=True)
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    day_dir = Path(args.clean_day_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean = load_clean_day(day_dir)
    if df_clean.empty:
        raise SystemExit("No cleaned data found.")

    id_map: dict[str, list[dict]] = {}

    with open(args.results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = extract_output_text(obj)
            if not txt:
                continue
            for res in parse_output_text(txt):
                id_map[res.submission_id] = [t.model_dump() for t in res.tickers]

    meta = pd.DataFrame({"id": df_clean["id"].astype(str)})
    meta["ticker_items"] = meta["id"].map(id_map).apply(lambda x: x if isinstance(x, list) else [])
    meta["tickers"] = meta["ticker_items"].apply(lambda L: [d["symbol"] for d in L])

    merged = df_clean.merge(meta, on="id", how="left")
    merged.to_parquet(out_dir / "annotated.parquet", index=False)
    (out_dir / "_SUCCESS").write_text("")

    print(
        "Annotated:", len(merged),
        "rows; non-empty ticker lists:",
        (merged["tickers"].str.len().fillna(0) > 0).sum()
    )

if __name__ == "__main__":
    main()
