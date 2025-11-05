# scripts/parse_results_v2.py
"""
Parse OpenAI Batch results (Responses API) into annotated parquet for a single day & kind.

Inputs
- --kind {rs,rc}
- --clean_day_dir  e.g. data_v2/RedditDumps/cleaned/RS_2025-05/day=2025-05-01
                    or data_v2/RedditDumps/cleaned/RC_2025-05/day=2025-05-01
- --results_jsonl  e.g. batch/Results/submissions/day=2025-05-01/part-00000.jsonl
- (optional) --alias_map  tiny CSV with columns [raw,normalized] for symbol fixes (e.g., BRKB -> BRK.B)
- --out_dir        e.g. data_v2/derived/submission_tickers/day=2025-05-01  (for rs)
                    or   data_v2/derived/comment_tickers/day=2025-05-01     (for rc)

Output
- annotated.parquet with original clean columns +:
    ticker_items: list[dict] of normalized tickers per doc
    tickers:      list[str]  symbols per doc
    is_forward:   bool|None
    value_score:  float|None
    kind:         "submission"|"comment"
- _failed.jsonl capturing any items we could not parse (by submission_id)

Tolerances
- Accept both legacy keys (sentiment_label/sentiment_score/confidence) and new (label/score/conf).
- Accept payload either as {"results":[...]} or a bare array [...], or (rarely) a single object.
- Accept multiple Responses API shapes: response.body.output[..].content[..].text, response.output_text, etc.
"""

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def read_clean(kind: str, day_dir: Path) -> pd.DataFrame:
    if kind == "rs":
        fn = "submissions_clean.parquet"
    else:
        fn = "comments_clean.parquet"
    p = day_dir / fn
    if not p.exists():
        raise SystemExit(f"Missing cleaned parquet: {p}")
    df = pd.read_parquet(p)
    # Ensure required columns are present
    if "id" not in df.columns:
        raise SystemExit("Cleaned parquet missing 'id' column.")
    if "text" not in df.columns:
        # fallback construction if someone ran an older cleaner
        if kind == "rs":
            title = df.get("title", pd.Series([""] * len(df))).astype(str)
            body = df.get("selftext", pd.Series([""] * len(df))).astype(str)
            df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
        else:
            df["text"] = df.get("body", pd.Series([""] * len(df))).astype(str)
    df["id"] = df["id"].astype(str)
    return df


def extract_output_text(obj: Dict[str, Any]) -> Optional[str]:
    """Handle multiple known Response API shapes; return the textual JSON payload if found."""
    resp = obj.get("response") or {}
    # Newer: response.body.output[..].content[..].text
    body = resp.get("body")
    if isinstance(body, dict):
        out = body.get("output")
        if isinstance(out, list) and out:
            m0 = out[0]
            if isinstance(m0, dict):
                content = m0.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                        return c0["text"]
    # Convenience field
    txt = resp.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt
    # Older: response.output[..].content[..].text
    out = resp.get("output")
    if isinstance(out, list) and out:
        m0 = out[0]
        if isinstance(m0, dict):
            content = m0.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                    return c0["text"]
    return None


def load_alias_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        print(f"[WARN] alias_map not found: {path}", file=sys.stderr)
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        raw = str(r.get("raw", "")).strip().upper()
        norm = str(r.get("normalized", "")).strip().upper()
        if raw and norm:
            out[raw] = norm
    return out


def normalize_one_ticker(d: Dict[str, Any], alias: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Normalize a single ticker dict to unified keys:
      symbol, label, score, conf
    Also pass through original keys if needed later.
    """
    if not isinstance(d, dict):
        return None

    # Symbol normalization: uppercase; preserve suffix
    sym = (d.get("symbol") or d.get("ticker") or "").strip().upper()
    if not sym:
        return None
    sym = alias.get(sym, sym)

    # Field normalization: accept both legacy and new keys
    label = d.get("label", d.get("sentiment_label"))
    score = d.get("score", d.get("sentiment_score"))
    conf  = d.get("conf", d.get("confidence"))

    # Coerce types defensively
    try:
        if score is not None:
            score = float(score)
    except Exception:
        score = None
    try:
        if conf is not None:
            conf = float(conf)
    except Exception:
        conf = None
    if isinstance(label, str):
        label = label.strip().lower()
        if label not in ("positive", "negative", "neutral"):
            # Soft-fix common variants
            if label in ("bullish",): label = "positive"
            elif label in ("bearish",): label = "negative"
            else: label = "neutral"  # fallback

    out = {
        "symbol": sym,
        "label": label,
        "score": score,
        "conf": conf,
    }
    return out


def parse_payload_text(txt: str) -> List[Dict[str, Any]]:
    """
    Return a list of per-doc objects. Each should contain:
      submission_id, tickers (list), and optionally is_forward, value_score
    Accepts:
      {"results":[...]}
      [...]
      {...} (single object)
    """
    try:
        obj = json.loads(txt)
    except Exception:
        return []
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "submission_id" in obj:
        return [obj]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["rs", "rc"])
    ap.add_argument("--clean_day_dir", required=True)
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alias_map", default=None, help="CSV with columns raw,normalized")
    args = ap.parse_args()

    kind = args.kind
    clean_day_dir = Path(args.clean_day_dir)
    results_jsonl = Path(args.results_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean = read_clean(kind, clean_day_dir)
    alias = load_alias_map(Path(args.alias_map)) if args.alias_map else {}

    # Build a map from submission_id -> parsed object
    parsed_by_id: Dict[str, Dict[str, Any]] = {}
    failed_ids: List[Dict[str, Any]] = []

    if not results_jsonl.exists():
        raise SystemExit(f"Missing results file: {results_jsonl}")

    with results_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                envelope = json.loads(line)
            except Exception:
                continue
            txt = extract_output_text(envelope)
            if not txt:
                continue
            items = parse_payload_text(txt)
            for it in items:
                sid = str(it.get("submission_id", "")).strip()
                if not sid:
                    continue
                # Normalize fields
                raw_tickers = it.get("tickers") or []
                norm_tickers = []
                for t in raw_tickers:
                    nt = normalize_one_ticker(t, alias)
                    if nt:
                        norm_tickers.append(nt)
                # Forward/value
                is_forward = it.get("is_forward")
                if isinstance(is_forward, str):
                    is_forward = is_forward.strip().lower() in {"true","1","yes","y"}
                elif not isinstance(is_forward, bool):
                    is_forward = None
                value_score = it.get("value_score")
                try:
                    if value_score is not None:
                        value_score = float(value_score)
                except Exception:
                    value_score = None

                parsed_by_id[sid] = {
                    "tickers": norm_tickers,
                    "is_forward": is_forward,
                    "value_score": value_score,
                }

    # Merge onto cleaned data
    meta = pd.DataFrame({"id": df_clean["id"].astype(str)})
    meta["tmp"] = meta["id"].map(parsed_by_id)

    def _extract(lst_or_none, key):
        if isinstance(lst_or_none, dict):
            return lst_or_none.get(key)
        return None

    meta["ticker_items"] = meta["tmp"].apply(lambda d: _extract(d, "tickers") or [])
    meta["is_forward"]   = meta["tmp"].apply(lambda d: _extract(d, "is_forward"))
    meta["value_score"]  = meta["tmp"].apply(lambda d: _extract(d, "value_score"))
    meta.drop(columns=["tmp"], inplace=True)
    meta["tickers"] = meta["ticker_items"].apply(lambda L: [t.get("symbol") for t in (L or []) if isinstance(t, dict)])

    annotated = df_clean.merge(meta, on="id", how="left")
    annotated["kind"] = "submission" if kind == "rs" else "comment"

    out_parquet = out_dir / "annotated.parquet"
    annotated.to_parquet(out_parquet, index=False)
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    # Emit a small failure log (ids present in clean but missing from parsed_by_id)
    missing_mask = ~annotated["id"].isin(parsed_by_id.keys())
    missing_ids = annotated.loc[missing_mask, "id"].astype(str).tolist()
    if missing_ids:
        with (out_dir / "_failed.jsonl").open("w", encoding="utf-8") as ff:
            for sid in missing_ids:
                ff.write(json.dumps({"id": sid}) + "\n")

    n_docs = len(annotated)
    n_with_tickers = int(annotated["tickers"].apply(lambda x: len(x) if isinstance(x,list) else 0).gt(0).sum())
    print(f"Annotated: {n_docs} docs; with >=1 ticker: {n_with_tickers} → {out_parquet}")

if __name__ == "__main__":
    main()
