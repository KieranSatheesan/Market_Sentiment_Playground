# scripts/parse_results_v4.py
"""
Parse v4 submission-centric Batch results into annotated parquet(s).

- kind=rs : submissions
- kind=rc : comments

Works over a date RANGE instead of a single day dir.

For v4, Batch results are written under Results_v4/submissions and
contain BOTH submission and comment outputs. For kind=rc we re-use
the same batch files, but only keep rows with a comment_id.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import datetime as dt


# -------------------- helpers for cleaned data --------------------


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
    parts: List[pd.DataFrame] = []

    for p in iter_clean_parquets(clean_root, kind):
        day_dir = p.parent
        name = day_dir.name  # day=YYYY-MM-DD
        if not name.startswith("day="):
            continue
        day = dt.date.fromisoformat(name.split("=", 1)[1])
        if day < start or day > end:
            continue

        df = pd.read_parquet(p)
        if "id" not in df.columns:
            raise SystemExit(f"Cleaned parquet missing 'id': {p}")

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


# -------------------- helpers for Batch output --------------------


def extract_output_text(env: Dict[str, Any]) -> Optional[str]:
    """
    Drill into the Batch line to extract the model's text output.
    Compatible with /v1/responses style payloads.
    """
    resp = env.get("response") or {}
    body = resp.get("body")

    # Newer responses: body["output"][0]["content"][0]["text"]
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

    # Fallbacks
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
    """
    Model is supposed to return:
      {"results":[ {...}, {...}, ... ]}
    but we defensively accept a few variants.
    """
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


def iter_result_lines(results_root: Path, kind: str):
    """
    In v4, Batch results live under:
        results_root/submissions/part-*.jsonl

    For kind="rs": we read from 'submissions/'.
    For kind="rc": we ALSO read from 'submissions/' and then filter
                   down to comment rows (comment_id present).
    """
    sub_dir = results_root / "submissions"

    if not sub_dir.exists():
        raise SystemExit(f"[v4] Missing results dir: {sub_dir}")

    parts = sorted(sub_dir.glob("part-*.jsonl"))
    if not parts:
        raise SystemExit(f"[v4] No part-*.jsonl under {sub_dir}")

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


# -------------------- main --------------------


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
        raise SystemExit("[v4] No cleaned data found in given range.")

    df_clean["id"] = df_clean["id"].astype(str)

    parsed_by_id: Dict[str, Dict[str, Any]] = {}
    lines_seen = 0
    objects_seen = 0

    print("[v4] Parsing Batch results...")
    for line in iter_result_lines(results_root, kind):
        line = line.strip()
        if not line:
            continue
        lines_seen += 1

        try:
            env = json.loads(line)
        except Exception:
            continue

        txt = extract_output_text(env)
        if not txt:
            continue

        for it in parse_payload_text(txt):
            objects_seen += 1

            # --- Decide which ID we key on ---
            if kind == "rs":
                # For submissions: only keep actual submission rows
                did = str(it.get("submission_id") or "").strip()
                if not did:
                    # Ignore any stray comment-style objects if present
                    continue
            else:
                # For comments: only keep actual comment rows
                cid = it.get("comment_id")
                if not cid:
                    # Skip the submission-level result in each group
                    continue
                did = str(cid).strip()

            if not did:
                continue

            # --- Normalise tickers ---
            raw_tickers = it.get("tickers") or []
            norm_tickers: List[Dict[str, Any]] = []
            for t in raw_tickers:
                nt = normalize_one_ticker(t)
                if nt:
                    norm_tickers.append(nt)

            # --- Normalise is_forward (allow None) ---
            is_forward = it.get("is_forward")
            if isinstance(is_forward, str):
                low = is_forward.strip().lower()
                if low in {"true", "1", "yes", "y"}:
                    is_forward = True
                elif low in {"false", "0", "no", "n"}:
                    is_forward = False
                else:
                    is_forward = None
            elif not isinstance(is_forward, bool) and is_forward is not None:
                is_forward = None

            # --- Normalise value_score ---
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

    print(f"[v4] Parsed {lines_seen} batch lines, {objects_seen} result objects.")
    print(f"[v4] Unique IDs with annotations: {len(parsed_by_id)}")

    id_keys = set(parsed_by_id.keys())

    # For comments, we really only care about rows we actually annotated in this run.
    # This also avoids hauling 4.4M comments through the merge during a small smoke test.
    if kind == "rc":
        before = len(df_clean)
        df_clean = df_clean[df_clean["id"].isin(id_keys)].copy()
        print(f"[v4] Filtered rc clean data from {before} → {len(df_clean)} rows using parsed IDs.")

    try:
        # Build meta-frame keyed by clean 'id'
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
            f"{'submissions' if kind == 'rs' else 'comments'}"
            f"_annotated_{start.isoformat()}_to_{end.isoformat()}.parquet"
        )
        out_parquet = out_dir / out_name
        annotated.to_parquet(out_parquet, index=False)

        missing = annotated.loc[
            ~annotated["id"].isin(id_keys), "id"
        ].astype(str)
        if not missing.empty:
            failed_path = out_dir / (
                f"{'submissions' if kind == 'rs' else 'comments'}_failed.jsonl"
            )
            with failed_path.open("w", encoding="utf-8") as ff:
                for did in missing:
                    ff.write(json.dumps({"id": did}) + "\n")

        n_docs = len(annotated)
        n_with = int(
            annotated["tickers"]
            .apply(lambda x: len(x) if isinstance(x, list) else 0)
            .gt(0)
            .sum()
        )
        print(
            f"[v4] Annotated {n_docs} {kind} docs; with >=1 ticker: {n_with} → {out_parquet}"
        )
    except Exception as e:
        # If anything goes wrong in the heavy part, surface it clearly
        print(f"[v4] ERROR while building / writing annotated {kind} data: {repr(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
