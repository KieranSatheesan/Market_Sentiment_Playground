# scripts/prepare_requests_v4.py

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any
import pandas as pd
import datetime as dt

# ---------- system prompt (v4, per-submission, thread-aware) ----------

SYSTEM_PROMPT_V4 = """
You read INPUT ITEMS and return exactly:
{"results":[ ... ]}

Same order, one output per INPUT ITEM.

There are two kinds of INPUT ITEM:

1) Submissions
   {
     "kind": "submission",
     "submission_id": "<SID>",
     "text": "..."
   }

   Output:
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

2) Comments
   {
     "kind": "comment",
     "comment_id": "<CID>",
     "submission_id": "<SID or null>",
     "comment_text": "...",
     "submission_text": "..."
   }

- All comment INPUT ITEMS in a single call refer to the SAME submission_id and submission_text.
- Use submission_text and other comments in this call as CONTEXT ONLY.
- For each comment, judge is_forward, value_score and sentiment based on comment_text itself, helped by context.

Output for each comment item:
{
  "comment_id": "<CID>",
  "submission_id": "<SID or null>",
  "is_forward": true|false|null,
  "value_score": 0.0-1.0,
  "tickers": [ same ticker schema as above ]
}

Ticker rules (strict):
- Equities/ETFs only: obvious stock tickers (A–Z, 1–6 chars) plus common suffixes (e.g. BRK.B, RY.TO).
- The ticker must clearly refer to a REAL traded equity/ETF. Do NOT invent symbols.
- Ignore random all-caps words that are not clearly a stock.
- Max 5 unique symbols per item.
- If the text is clearly about multiple tickers, include all of them with separate sentiment.

Sentiment:
- sentiment_score in [-1.0,1.0].
- > +0.15 → positive
- < -0.15 → negative
- otherwise → neutral
- sentiment_label must be "positive", "neutral" or "negative" consistent with sentiment_score.

is_forward:
- true  → future-looking positions, targets, expected moves, planned trades.
- false → purely backward-looking or general talk with no specific forward play.
- null  → allowed for comments that are ambiguous.

value_score (rough guideline):
- 0.00–0.20: memes/low information, jokes, one-liners.
- 0.21–0.50: vague opinions without detailed reasoning.
- 0.51–0.80: some reasoning, partial thesis, concrete arguments.
- 0.81–1.00: detailed, specific, falsifiable thesis with clear reasoning.

If no tickers for an item:
- "tickers": [] but still set is_forward and value_score sensibly.

Return JSON only, no explanations, no extra keys.
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
    return df


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


def extract_submission_id(link_id) -> str | None:
    if link_id is None or pd.isna(link_id):
        return None
    s = str(link_id)
    if s.startswith("t3_"):
        return s[3:]
    return s or None


def build_comment_groups_by_submission(
    df_cmt: pd.DataFrame,
    sub_map: Dict[str, str],
    max_chars: int,
    min_comment_len: int,
    max_comments_per_request: int,
) -> List[List[Dict[str, Any]]]:
    """
    Returns a list of groups, where each group is a list of INPUT ITEMS for one request.
    Each group:
      - all items share the same submission_id (or None)
      - contains up to max_comments_per_request comments
    """
    df = df_cmt.copy()

    # attach submission_id from link_id
    df["submission_id"] = df.get("link_id").apply(extract_submission_id)

    # sort by (submission_id, created_utc, id)
    if "created_utc" in df.columns:
        df = df.sort_values(["submission_id", "created_utc", "id"])
    else:
        df = df.sort_values(["submission_id", "id"])

    groups: List[List[Dict[str, Any]]] = []

    for sid, grp in df.groupby("submission_id", dropna=False):
        sid_str = None if sid is None or (isinstance(sid, float) and pd.isna(sid)) else str(sid)
        sub_text = sub_map.get(sid_str, "")

        comment_items: List[Dict[str, Any]] = []
        for _, r in grp.iterrows():
            cid = str(r["id"])
            ctext = str(r["text"])[:max_chars].strip()
            if not ctext:
                continue
            if min_comment_len and len(ctext) < min_comment_len:
                continue

            comment_items.append({
                "kind": "comment",
                "comment_id": cid,
                "submission_id": sid_str,
                "comment_text": ctext,
                "submission_text": sub_text,
            })

        if not comment_items:
            continue

        # Now chunk this submission's comments into groups of up to max_comments_per_request
        for chunk_items in chunk(comment_items, max_comments_per_request):
            groups.append(chunk_items)

    return groups


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--kind", choices=["rs", "rc", "both"], default="both")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--group_size", type=int, default=15, help="Max comments per request for RC; batch size for RS")
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
                # same behaviour as v3: simple batching over submissions
                if df_sub.empty:
                    print(f"[v4] [INFO] RS {day} -> no submissions")
                    continue
                items = build_submission_items(df_sub, args.max_chars)
                if not items:
                    print(f"[v4] [INFO] RS {day} -> no items after filtering")
                    continue
                kind_dir = "submissions"
                # groups are plain chunks of submissions
                groups: List[List[Dict[str, Any]]] = list(chunk(items, args.group_size))
            else:
                # RC: group comments by submission_id
                df_cmt = load_comments_for_day(clean_root, day)
                if df_cmt.empty:
                    print(f"[v4] [INFO] RC {day} -> no comments")
                    continue

                groups = build_comment_groups_by_submission(
                    df_cmt,
                    sub_map=sub_map,
                    max_chars=args.max_chars,
                    min_comment_len=args.min_comment_len,
                    max_comments_per_request=args.group_size,
                )
                if not groups:
                    print(f"[v4] [INFO] RC {day} -> no comment groups after filtering")
                    continue
                kind_dir = "comments"

            out_day_dir = out_root / kind_dir / f"day={day:%Y-%m-%d}"
            ensure_dir(out_day_dir)

            part_idx = 0
            written_in_part = 0
            total_written_for_day = 0

            def open_part(idx: int):
                p = out_day_dir / f"part-{idx:05d}.jsonl"
                return p.open("w", encoding="utf-8"), p

            f, part_path = open_part(part_idx)

            for group_idx, group in enumerate(groups):
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
                    "All INPUT ITEMS in this call refer to the same submission thread.\n"
                    "BEGIN_INPUTS:\n" +
                    "\n".join(json.dumps(obj, ensure_ascii=False) for obj in group)
                )

                body: Dict[str, Any] = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT_V4},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.text_format != "none":
                    body["text"] = {"format": {"type": args.text_format}}

                first = group[0]
                first_id = first.get("comment_id") or first.get("submission_id") or "na"
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
                "docs": sum(len(g) for g in groups),
                "requests": total_written_for_day,
                "group_size": args.group_size,
                "path": str(part_path),
            })
            total_requests += total_written_for_day

            print(f"[v4] [WRITE] {out_day_dir} groups={len(groups)} requests={total_written_for_day} (parts={part_idx+1})")

    if manifest_rows:
        mpath = out_root / "MANIFEST_v4.json"
        with mpath.open("w", encoding="utf-8") as mf:
            json.dump(manifest_rows, mf, ensure_ascii=False, indent=2)
        total_docs = sum((m.get("docs") or 0) for m in manifest_rows)
        print(f"\n[v4] Done. Day-parts={len(manifest_rows)}, Docs={total_docs}, Requests={total_requests}")
        print(f"[v4] Manifest: {mpath}")
    else:
        print("[v4] No JSONL written.")

if __name__ == "__main__":
    main()
