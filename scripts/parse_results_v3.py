# scripts/parse_results_v3.py

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

def read_clean(kind: str, day_dir: Path) -> pd.DataFrame:
    fn = "submissions_clean.parquet" if kind == "rs" else "comments_clean.parquet"
    p = day_dir / fn
    if not p.exists():
        raise SystemExit(f"Missing cleaned parquet: {p}")
    df = pd.read_parquet(p)

    if "id" not in df.columns:
        raise SystemExit("Cleaned parquet missing 'id'.")

    if "text" not in df.columns:
        if kind == "rs":
            title = df.get("title", pd.Series([""] * len(df))).astype(str)
            body  = df.get("selftext", pd.Series([""] * len(df))).astype(str)
            df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
        else:
            df["text"] = df.get("body", pd.Series([""] * len(df))).astype(str)

    df["id"] = df["id"].astype(str)
    return df

def extract_output_text(env: Dict[str, Any]) -> Optional[str]:
    resp = env.get("response") or {}
    body = resp.get("body")

    if isinstance(body, dict):
        if body.get("status") and body["status"] != "completed":
            return None
        out = body.get("output")
        if isinstance(out, list) and out:
            m0 = out[0]
            if isinstance(m0, dict):
                content = m0.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    txt = c0.get("text")
                    if isinstance(txt, str):
                        return txt

    txt = resp.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt

    out = resp.get("output")
    if isinstance(out, list) and out:
        m0 = out[0]
        if isinstance(m0, dict):
            content = m0.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                txt = c0.get("text")
                if isinstance(txt, str):
                    return txt

    return None

def parse_payload_text(txt: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(txt)
    except Exception:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and ("submission_id" in obj or "comment_id" in obj):
        return [obj]
    return []

def iter_result_lines(results_path: Path):
    if results_path.is_dir():
        parts = sorted(results_path.glob("part-*.jsonl"))
        if not parts:
            raise SystemExit(f"No part-*.jsonl in {results_path}")
        for p in parts:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    yield line
    else:
        if not results_path.exists():
            raise SystemExit(f"Missing results file: {results_path}")
        with results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line

def normalize_one_ticker(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict):
        return None
    sym = (d.get("symbol") or d.get("ticker") or "").strip().upper()
    if not sym:
        return None

    label = d.get("sentiment_label", d.get("label", d.get("sentiment")))
    score = d.get("sentiment_score", d.get("score"))
    conf  = d.get("conf", d.get("confidence"))

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
        if label not in ("positive","negative","neutral"):
            if label in ("bullish",): label = "positive"
            elif label in ("bearish",): label = "negative"
            else: label = "neutral"

    return {
        "symbol": sym,
        "sentiment_label": label,
        "sentiment_score": score,
        "conf": conf,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["rs","rc"])
    ap.add_argument("--clean_day_dir", required=True)
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    kind = args.kind
    clean_day_dir = Path(args.clean_day_dir)
    results_path  = Path(args.results_jsonl)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean = read_clean(kind, clean_day_dir)

    parsed_by_id: Dict[str, Dict[str, Any]] = {}

    for line in iter_result_lines(results_path):
        line = line.strip()
        if not line:
            continue
        try:
            env = json.loads(line)
        except Exception:
            continue

        txt = extract_output_text(env)
        if not txt:
            continue

        for it in parse_payload_text(txt):
            if kind == "rs":
                did = str(it.get("submission_id") or it.get("id") or "").strip()
            else:
                did = str(
                    it.get("comment_id")
                    or it.get("submission_id")
                    or it.get("id")
                    or ""
                ).strip()
            if not did:
                continue

            raw_tickers = it.get("tickers") or []
            norm = []
            for t in raw_tickers:
                nt = normalize_one_ticker(t)
                if nt:
                    norm.append(nt)

            is_forward = it.get("is_forward")
            if isinstance(is_forward, str):
                is_forward = is_forward.strip().lower() in {"true","1","yes","y"}
            elif not isinstance(is_forward, bool) and is_forward is not None:
                is_forward = None

            value_score = it.get("value_score")
            try:
                if value_score is not None:
                    value_score = float(value_score)
            except Exception:
                value_score = None

            parsed_by_id[did] = {
                "tickers": norm,
                "is_forward": is_forward,
                "value_score": value_score,
            }

    meta = pd.DataFrame({"id": df_clean["id"].astype(str)})
    meta["tmp"] = meta["id"].map(parsed_by_id)

    def _get(d, k):
        return d.get(k) if isinstance(d, dict) else None

    meta["ticker_items"] = meta["tmp"].apply(lambda d: _get(d,"tickers") or [])
    meta["is_forward"]   = meta["tmp"].apply(lambda d: _get(d,"is_forward"))
    meta["value_score"]  = meta["tmp"].apply(lambda d: _get(d,"value_score"))
    meta.drop(columns=["tmp"], inplace=True)

    meta["tickers"] = meta["ticker_items"].apply(
        lambda L: [t.get("symbol") for t in (L or []) if isinstance(t, dict)]
    )

    annotated = df_clean.merge(meta, on="id", how="left")
    annotated["kind"] = "submission" if kind == "rs" else "comment"

    out_parquet = out_dir / "annotated.parquet"
    annotated.to_parquet(out_parquet, index=False)
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    missing = annotated.loc[~annotated["id"].isin(parsed_by_id.keys()), "id"].astype(str)
    if not missing.empty:
        with (out_dir / "_failed.jsonl").open("w", encoding="utf-8") as ff:
            for did in missing:
                ff.write(json.dumps({"id": did}) + "\n")

    n_docs = len(annotated)
    n_with = int(
        annotated["tickers"].apply(lambda x: len(x) if isinstance(x,list) else 0).gt(0).sum()
    )
    print(f"[v3] Annotated: {n_docs} docs; with >=1 ticker: {n_with} → {out_parquet}")

if __name__ == "__main__":
    main()
