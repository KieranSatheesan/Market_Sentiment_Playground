# scripts/parse_results_v4.py
"""
Parse v4 submission-centric Batch results into annotated parquet(s).

- kind=rc : comments
- kind=rs : submissions

Differences vs v3:
- Works over a date RANGE instead of a single day dir.
- Reads all relevant cleaned RS/RC parquet files under clean_root.
- Writes a SINGLE annotated parquet for the range:
    data_v4/derived/comment_tickers_v4/comments_annotated_<start>_to_<end>.parquet
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import datetime as dt


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def iter_clean_parquets(clean_root: Path, kind: str) -> List[Path]:
    """
    kind="rs" -> RS_*/day=*/submissions_clean.parquet
    kind="rc" -> RC_*/day=*/comments_clean.parquet
    """
    out: List[Path] = []
    prefix = "RS_" if kind == "rs" else "RC_"
    fname = "submissions_clean.parquet" if kind == "rs" else "comments_clean.parquet"
    for top in sorted(clean_root.glob(prefix + "*")):
        for day_dir in sorted(top.glob("day=*")):
            p = day_dir / fname
            if p.exists():
                out.append(p)
    return out


def load_clean_range(clean_root: Path, kind: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    parts = []
    for p in iter_clean_parquets(clean_root, kind):
        # parse day from directory name
        day_dir = p.parent
        name = day_dir.name  # day=YYYY-MM-DD
        if not name.startswith("day="):
            continue
        day = dt.date.fromisoformat(name.split("=", 1)[1])
        if day < start or day > end:
            continue
        df = pd.read_parquet(p)
        df["id"] = df["id"].astype(str)
        df["__day__"] = day
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df_all = pd.concat(parts, ignore_index=True)

    # Build text column if missing
    if "text" not in df_all.columns:
        if kind == "rs":
            title = df_all.get("title", "").astype(str)
            body = df_all.get("selftext", "").astype(str)
            df_all["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
        else:
            df_all["text"] = df_all.get("body", "").astype(str)
    return df_all


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


def iter_result_lines(results_root: Path) -> Any:
    """
    Iterate over all part-*.jsonl in results_root/submissions
    """
    sub_dir = results_root / "submissions"
    parts = sorted(sub_dir.glob("part-*.jsonl"))
    if not parts:
        raise SystemExit(f"No part-*.jsonl under {sub_dir}")
    for p in parts:
        with p.open("r", encoding="utf-8") as fh:
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
    conf = d.get("conf", d.get("confidence"))

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
        "sentiment_label": label,
        "sentiment_score": score,
        "conf": conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["rs", "rc"])
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    kind = args.kind
    clean_root = Path(args.clean_root)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = parse_date(args.start)
    end = parse_date(args.end)

    print(f"[v4] Loading clean {kind} from {start} to {end}...")
    df_clean = load_clean_range(clean_root, kind, start, end)
    if df_clean.empty:
        raise SystemExit("No cleaned data found in given range.")

    df_clean["id"] = df_clean["id"].astype(str)

    parsed_by_id: Dict[str, Dict[str, Any]] = {}

    print("[v4] Parsing Batch results...")
    for line in iter_result_lines(results_root):
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
            norm_tickers = []
            for t in raw_tickers:
                nt = normalize_one_ticker(t)
                if nt:
                    norm_tickers.append(nt)

            is_forward = it.get("is_forward")
            if isinstance(is_forward, str):
                is_forward = is_forward.strip().lower() in {"true", "1", "yes", "y"}
            elif not isinstance(is_forward, bool) and is_forward is not None:
                is_forward = None

            value_score = it.get("value_score")
            try:
                if value_score is not None:
                    value_score = float(value_score)
            except Exception:
                value_score = None

            parsed_by_id[did] = {
                "tickers": norm_tickers,
                "is_forward": is_forward,
                "value_score": value_score,
            }

    meta = pd.DataFrame({"id": df_clean["id"]})
    meta["tmp"] = meta["id"].map(parsed_by_id)

    def _get(d, k):
        return d.get(k) if isinstance(d, dict) else None

    meta["ticker_items"] = meta["tmp"].apply(lambda d: _get(d, "tickers") or [])
    meta["is_forward"] = meta["tmp"].apply(lambda d: _get(d, "is_forward"))
    meta["value_score"] = meta["tmp"].apply(lambda d: _get(d, "value_score"))
    meta.drop(columns=["tmp"], inplace=True)

    meta["tickers"] = meta["ticker_items"].apply(
        lambda L: [t.get("symbol") for t in (L or []) if isinstance(t, dict)]
    )

    annotated = df_clean.merge(meta, on="id", how="left")
    annotated["kind"] = "submission" if kind == "rs" else "comment"

    out_name = (
        f"{'submissions' if kind=='rs' else 'comments'}"
        f"_annotated_{start.isoformat()}_to_{end.isoformat()}.parquet"
    )
    out_parquet = out_dir / out_name
    annotated.to_parquet(out_parquet, index=False)

    missing = annotated.loc[
        ~annotated["id"].isin(parsed_by_id.keys()), "id"
    ].astype(str)
    if not missing.empty:
        failed_path = out_dir / f"{'submissions' if kind=='rs' else 'comments'}_failed.jsonl"
        with failed_path.open("w", encoding="utf-8") as ff:
            for did in missing:
                ff.write(json.dumps({"id": did}) + "\n")

    n_docs = len(annotated)
    n_with = int(
        annotated["tickers"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).gt(0).sum()
    )
    print(
        f"[v4] Annotated {n_docs} {kind} docs; with >=1 ticker: {n_with} → {out_parquet}"
    )


if __name__ == "__main__":
    main()
