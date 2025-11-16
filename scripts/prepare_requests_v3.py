# scripts/prepare_requests_v3.py

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any
import pandas as pd
import datetime as dt

# ---------- tiny system prompt (shared for all batches) ----------

SYSTEM_PROMPT = """
You read INPUT ITEMS and return exactly:
{"results":[ ... ]}

Same order, one output per input.

If kind == "submission":
- Input:
  {"kind":"submission","submission_id":"<SID>","text":"..."}
- Output:
  {
    "submission_id": "<SID>",
    "is_forward": true|false,
    "value_score": 0.0-1.0,
    "tickers": [
      {
        "symbol": "AAPL",
        "sentiment_label": "positive"|"neutral"|"negative",
        "sentiment_score": -1.0..1.0,
        "conf": 0.0-1.0
      }
    ]
  }

If kind == "comment":
- Input:
  {
    "kind":"comment",
    "comment_id":"<CID>",
    "submission_id":"<SID or null>",
    "comment_text":"...",
    "submission_text":"..."   # context only
  }
- Use submission_text ONLY to disambiguate tickers.
- Judge is_forward, value_score, and sentiment based on comment_text itself.
- Output:
  {
    "comment_id": "<CID>",
    "submission_id": "<SID or null>",
    "is_forward": true|false|null,
    "value_score": 0.0-1.0,
    "tickers": [ same ticker schema as above ]
  }

Rules (minimal):
- Equities only; obvious stock tickers (A–Z, 1–6 chars) plus common suffixes (BRK.B, RY.TO, etc). Max 5 unique.
- Do not invent tickers. Only when clearly about the stock.
- sentiment_score:
    > +0.15 → positive
    < -0.15 → negative
    else    → neutral
- is_forward: future-looking price/targets/plays → true; purely backward/none → false; unclear for comments → null allowed.
- value_score:
    0.00–0.20 memes/low info
    0.21–0.50 vague opinions
    0.51–0.80 some reasoning or partial thesis
    0.81–1.00 detailed, specific, falsifiable thesis
- If no tickers: "tickers": [] but still set is_forward/value_score.
No extra keys, no explanations.
""".strip()


# ---------- helpers ----------

def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_submissions_for_day(clean_root: Path, day: dt.date) -> pd.DataFrame:
    ym = day.strftime("%Y-%m")
    day_dir = clean_root / f"RS_{ym}" / f"day={day:%Y-%m-%d}"
    p = day_dir / "submissions_clean.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    if "text" not in df.columns:
        title = df.get("title", pd.Series([""] * len(df))).astype(str)
        body = df.get("selftext", pd.Series([""] * len(df))).astype(str)
        df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()

    df["id"] = df["id"].astype(str)
    return df[["id", "text"]]


def load_comments_for_day(clean_root: Path, day: dt.date) -> pd.DataFrame:
    ym = day.strftime("%Y-%m")
    day_dir = clean_root / f"RC_{ym}" / f"day={day:%Y-%m-%d}"
    p = day_dir / "comments_clean.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    if "text" not in df.columns:
        df["text"] = df.get("body", pd.Series([""] * len(df))).astype(str)

    df["id"] = df["id"].astype(str)
    return df


def chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_submission_items(df_sub: pd.DataFrame, max_chars: int) -> List[Dict[str, Any]]:
    items = []
    for _, r in df_sub.iterrows():
        text = str(r["text"])[:max_chars].strip()
        if not text:
            continue
        items.append({
            "kind": "submission",
            "submission_id": str(r["id"]),
            "text": text,
        })
    return items


def build_comment_items(df_cmt: pd.DataFrame,
                        sub_map: Dict[str, str],
                        max_chars: int,
                        min_comment_len: int) -> List[Dict[str, Any]]:
    items = []
    for _, r in df_cmt.iterrows():
        cid = str(r["id"])
        ctext = str(r["text"])[:max_chars].strip()
        if not ctext:
            continue
        if min_comment_len and len(ctext) < min_comment_len:
            continue

        sub_id = None
        if "link_id" in r and pd.notna(r["link_id"]):
            lid = str(r["link_id"])
            sub_id = lid[3:] if lid.startswith("t3_") else lid

        sub_text = sub_map.get(sub_id, "")

        items.append({
            "kind": "comment",
            "comment_id": cid,
            "submission_id": sub_id,
            "comment_text": ctext,
            "submission_text": sub_text,
        })
    return items


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--kind", choices=["rs", "rc", "both"], default="both")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--min_comment_len", type=int, default=0)
    ap.add_argument("--max_requests_per_file", type=int, default=800)
    ap.add_argument("--text_format", choices=["none","json_object","text","json_schema"],
                    default="json_object")
    ap.add_argument("--max_output_tokens", type=int, default=512)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    start_date = dt.date.fromisoformat(args.start)
    end_date   = dt.date.fromisoformat(args.end)
    kinds = ["rs", "rc"] if args.kind == "both" else [args.kind]

    manifest_rows = []
    total_requests = 0

    for day in daterange(start_date, end_date):
        df_sub = load_submissions_for_day(clean_root, day)
        sub_map = {row["id"]: row["text"] for _, row in df_sub.iterrows()}

        for kind in kinds:
            if kind == "rs":
                if df_sub.empty:
                    print(f"[INFO] RS {day} -> no submissions")
                    continue
                items = build_submission_items(df_sub, args.max_chars)
                kind_dir = "submissions"
            else:
                df_cmt = load_comments_for_day(clean_root, day)
                if df_cmt.empty:
                    print(f"[INFO] RC {day} -> no comments")
                    continue
                items = build_comment_items(df_cmt, sub_map,
                                            max_chars=args.max_chars,
                                            min_comment_len=args.min_comment_len)
                kind_dir = "comments"

            if not items:
                print(f"[INFO] {kind.upper()} {day} -> no items after filtering")
                continue

            out_day_dir = out_root / kind_dir / f"day={day:%Y-%m-%d}"
            ensure_dir(out_day_dir)

            part_idx = 0
            written_in_part = 0
            total_written_for_day = 0

            def open_part(idx: int):
                p = out_day_dir / f"part-{idx:05d}.jsonl"
                return p.open("w", encoding="utf-8"), p

            f, part_path = open_part(part_idx)

            for group_idx, group in enumerate(chunk(items, args.group_size)):
                if written_in_part >= args.max_requests_per_file:
                    f.close()
                    manifest_rows.append({
                        "kind": kind_dir,
                        "day": f"{day:%Y-%m-%d}",
                        "requests": written_in_part,
                        "group_size": args.group_size,
                        "path": str(part_path),
                    })
                    part_idx += 1
                    written_in_part = 0
                    f, part_path = open_part(part_idx)

                user_content = (
                    'Return JSON only: {"results":[...]} with one output per INPUT ITEM, same order.\n'
                    "BEGIN_INPUTS:\n" +
                    "\n".join(json.dumps(obj, ensure_ascii=False) for obj in group)
                )

                body: Dict[str, Any] = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.text_format != "none":
                    body["text"] = {"format": {"type": args.text_format}}

                first_id = group[0].get("comment_id") or group[0].get("submission_id") or "na"
                custom_id = f"{kind}:{day:%Y-%m-%d}:{first_id}:{group_idx:04d}"

                line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written_in_part += 1
                total_written_for_day += 1

            f.close()
            manifest_rows.append({
                "kind": kind_dir,
                "day": f"{day:%Y-%m-%d}",
                "docs": len(items),
                "requests": written_in_part,
                "group_size": args.group_size,
                "path": str(part_path),
            })
            total_requests += total_written_for_day

            print(f"[WRITE] {out_day_dir} items={len(items)} requests={total_written_for_day} (parts={part_idx+1})")

    if manifest_rows:
        mpath = out_root / "MANIFEST_v3.json"
        with mpath.open("w", encoding="utf-8") as mf:
            json.dump(manifest_rows, mf, ensure_ascii=False, indent=2)
        total_docs = sum((m.get("docs") or 0) for m in manifest_rows)
        print(f"\nDone v3. Day-parts={len(manifest_rows)}, Docs={total_docs}, Requests={total_requests}")
        print(f"Manifest: {mpath}")
    else:
        print("No v3 JSONL written.")

if __name__ == "__main__":
    main()
