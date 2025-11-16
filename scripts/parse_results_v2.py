# scripts/parse_results_v2.py
"""
Parse OpenAI Batch results (Responses API) into annotated parquet for a single day & kind.

Supports:
- Single results JSONL file; OR
- A directory containing multiple part-*.jsonl files for that day.

(...docstring unchanged...)
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

    if "id" not in df.columns:
        raise SystemExit("Cleaned parquet missing 'id' column.")

    if "text" not in df.columns:
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

    # Require completed status if present
    status = resp.get("status")
    if status and status != "completed":
        return None

    # Newer: response.body.output[..].content[..].text
    body = resp.get("body")
    if isinstance(body, dict):
        b_status = body.get("status")
        if b_status and b_status != "completed":
            return None

        out = body.get("output")
        if isinstance(out, list) and out:
            m0 = out[0]
            if isinstance(m0, dict):
                content = m0.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                        return c0["text"]

    # Convenience field (some clients expose this)
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
    if not isinstance(d, dict):
        return None

    sym = (d.get("symbol") or d.get("ticker") or "").strip().upper()
    if not sym:
        return None
    sym = alias.get(sym, sym)

    label = d.get("label", d.get("sentiment_label"))
    score = d.get("score", d.get("sentiment_score"))
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
        if label not in ("positive", "negative", "neutral"):
            if label in ("bullish",):
                label = "positive"
            elif label in ("bearish",):
                label = "negative"
            else:
                label = "neutral"

    return {
        "symbol": sym,
        "label": label,
        "score": score,
        "conf": conf,
    }


def parse_payload_text(txt: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(txt)
    except Exception:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "submission_id" in obj:
        return [obj]
    return []


def iter_result_lines(results_path: Path):
    """Yield JSONL lines from a file or all part-*.jsonl under a directory."""
    if results_path.is_dir():
        parts = sorted(results_path.glob("part-*.jsonl"))
        if not parts:
            raise SystemExit(f"No part-*.jsonl files in {results_path}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["rs", "rc"])
    ap.add_argument("--clean_day_dir", required=True)
    ap.add_argument("--results_jsonl", required=True,
                    help="Either a single JSONL file or a directory containing part-*.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alias_map", default=None, help="CSV with columns raw,normalized")
    args = ap.parse_args()

    kind = args.kind
    clean_day_dir = Path(args.clean_day_dir)
    results_path = Path(args.results_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean = read_clean(kind, clean_day_dir)
    alias = load_alias_map(Path(args.alias_map)) if args.alias_map else {}

    parsed_by_id: Dict[str, Dict[str, Any]] = {}

    for line in iter_result_lines(results_path):
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

            raw_tickers = it.get("tickers") or []
            norm_tickers = []
            for t in raw_tickers:
                nt = normalize_one_ticker(t, alias)
                if nt:
                    norm_tickers.append(nt)

            is_forward = it.get("is_forward")
            if isinstance(is_forward, str):
                is_forward = is_forward.strip().lower() in {"true", "1", "yes", "y"}
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

    def _extract(d, key):
        if isinstance(d, dict):
            return d.get(key)
        return None

    meta["ticker_items"] = meta["tmp"].apply(lambda d: _extract(d, "tickers") or [])
    meta["is_forward"]   = meta["tmp"].apply(lambda d: _extract(d, "is_forward"))
    meta["value_score"]  = meta["tmp"].apply(lambda d: _extract(d, "value_score"))
    meta.drop(columns=["tmp"], inplace=True)
    meta["tickers"] = meta["ticker_items"].apply(
        lambda L: [t.get("symbol") for t in (L or []) if isinstance(t, dict)]
    )

    annotated = df_clean.merge(meta, on="id", how="left")
    annotated["kind"] = "submission" if kind == "rs" else "comment"

    out_parquet = out_dir / "annotated.parquet"
    annotated.to_parquet(out_parquet, index=False)
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    # Log ids in clean with no parsed entry (for QA)
    missing_mask = ~annotated["id"].isin(parsed_by_id.keys())
    missing_ids = annotated.loc[missing_mask, "id"].astype(str).tolist()
    if missing_ids:
        with (out_dir / "_failed.jsonl").open("w", encoding="utf-8") as ff:
            for sid in missing_ids:
                ff.write(json.dumps({"id": sid}) + "\n")

    n_docs = len(annotated)
    n_with_tickers = int(
        annotated["tickers"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).gt(0).sum()
    )
    print(f"Annotated: {n_docs} docs; with >=1 ticker: {n_with_tickers} → {out_parquet}")


if __name__ == "__main__":
    main()
