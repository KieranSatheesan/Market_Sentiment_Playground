# scripts/prepare_requests_v4.py
"""
Submission-centric request builder (v4).

- Groups by submission_id across a full time range.
- Each request only contains:
    - 1 submission item
    - up to N comment items for that submission (max_comments_per_request).
- Comments are sorted chronologically (created_utc) before chunking.
- Supports:
    (a) date_range mode: all submissions whose created_utc is between [--start, --end]
    (b) submission_ids mode: only submissions listed in --submission_ids_file

Outputs:
    batch/Requests_v4/submissions/part-00000.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
import datetime as dt

SYSTEM_PROMPT_V4 = """
You read INPUT ITEMS and return exactly:
{"results":[ ... ]}

Same order, one output per INPUT ITEM.

Each INPUT ITEM looks like one of:

1) Submission item:
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

2) Comment item:
   {
     "kind": "comment",
     "comment_id": "<CID>",
     "submission_id": "<SID or null>",
     "comment_text": "...",
     "submission_text": "..."   # context only
   }

   Use submission_text ONLY to disambiguate tickers or context.
   Judge is_forward, value_score, and sentiment based on comment_text itself.

   Output:
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
- If there are no tickers for an item: "tickers": [] but still set is_forward/value_score.
Return JSON only with {"results":[...]} and nothing else.
""".strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def iter_rs_parquets(clean_root: Path) -> Iterable[Path]:
    for rs_dir in sorted(clean_root.glob("RS_*")):
        for day_dir in sorted(rs_dir.glob("day=*")):
            p = day_dir / "submissions_clean.parquet"
            if p.exists():
                yield p


def iter_rc_parquets(clean_root: Path) -> Iterable[Path]:
    for rc_dir in sorted(clean_root.glob("RC_*")):
        for day_dir in sorted(rc_dir.glob("day=*")):
            p = day_dir / "comments_clean.parquet"
            if p.exists():
                yield p


def load_all_submissions(clean_root: Path) -> pd.DataFrame:
    parts = []
    for p in iter_rs_parquets(clean_root):
        df = pd.read_parquet(p)
        df["id"] = df["id"].astype(str)
        # build text if missing
        if "text" not in df.columns:
            title = df.get("title", "").astype(str)
            body = df.get("selftext", "").astype(str)
            df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df_all = pd.concat(parts, ignore_index=True)
    return df_all


def load_all_comments(clean_root: Path) -> pd.DataFrame:
    parts = []
    for p in iter_rc_parquets(clean_root):
        df = pd.read_parquet(p)
        df["id"] = df["id"].astype(str)
        if "text" not in df.columns:
            df["text"] = df.get("body", "").astype(str)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df_all = pd.concat(parts, ignore_index=True)
    return df_all


def add_created_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df["date"] = df["created_dt"].dt.date
    else:
        df["created_dt"] = pd.NaT
        df["date"] = pd.NaT
    return df


def derive_submission_id_from_link_id(link_id: Any) -> Optional[str]:
    if link_id is None or (isinstance(link_id, float) and pd.isna(link_id)):
        return None
    s = str(link_id)
    return s[3:] if s.startswith("t3_") else s


def chunk_list(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_request_body(
    model: str,
    max_output_tokens: int,
    items: List[Dict[str, Any]],
    text_format: str,
) -> Dict[str, Any]:
    user_content = (
        'Return JSON only: {"results":[...]} with one output per INPUT ITEM, same order.\n'
        "BEGIN_INPUTS:\n"
        + "\n".join(json.dumps(obj, ensure_ascii=False) for obj in items)
    )

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT_V4},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
    }
    if text_format != "none":
        body["text"] = {"format": {"type": text_format}}
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument(
        "--mode",
        choices=["date_range", "submission_ids"],
        default="date_range",
        help="How to choose which submissions to process.",
    )
    ap.add_argument(
        "--submission_ids_file",
        help="If mode=submission_ids, path to a text file with one submission_id per line.",
    )
    ap.add_argument("--start", help="Submission date start (YYYY-MM-DD) for date_range mode")
    ap.add_argument("--end", help="Submission date end (YYYY-MM-DD) for date_range mode")
    ap.add_argument(
        "--comment_start",
        help="Comment date start (YYYY-MM-DD). Defaults to --start if omitted.",
    )
    ap.add_argument(
        "--comment_end",
        help="Comment date end (YYYY-MM-DD). Defaults to --end if omitted.",
    )
    ap.add_argument("--max_comments_per_request", type=int, default=15)
    ap.add_argument("--max_requests_per_file", type=int, default=800)
    ap.add_argument("--text_format", choices=["none", "json_object", "text", "json_schema"],
                    default="json_object")
    ap.add_argument("--max_output_tokens", type=int, default=1200)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    # -------- Load full RS/RC universe once --------
    print("[v4] Loading all submissions...")
    df_sub_all = load_all_submissions(clean_root)
    df_sub_all = add_created_date_cols(df_sub_all)

    print("[v4] Loading all comments...")
    df_cmt_all = load_all_comments(clean_root)
    df_cmt_all = add_created_date_cols(df_cmt_all)

    # Derive submission_id for comments
    df_cmt_all["submission_id"] = df_cmt_all["link_id"].apply(
        derive_submission_id_from_link_id
    )

    # -------- Filter submissions we will process --------
    if args.mode == "date_range":
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required in date_range mode")
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
        mask = (df_sub_all["date"] >= start_date) & (df_sub_all["date"] <= end_date)
        df_sub = df_sub_all[mask].copy()
        print(f"[v4] Submissions in [{start_date}..{end_date}]: {len(df_sub)}")

        comment_start = parse_date(args.comment_start) if args.comment_start else start_date
        comment_end = parse_date(args.comment_end) if args.comment_end else end_date
    else:
        # submission_ids mode
        if not args.submission_ids_file:
            raise SystemExit("--submission_ids_file is required in submission_ids mode")
        sids = []
        with open(args.submission_ids_file, "r", encoding="utf-8") as fh:
            for line in fh:
                sid = line.strip()
                if sid:
                    sids.append(sid)
        sids = sorted(set(sids))
        df_sub = df_sub_all[df_sub_all["id"].isin(sids)].copy()
        print(f"[v4] Found {len(df_sub)} submissions matching {len(sids)} IDs")

        # For smoke tests, you can set comment_start/comment_end wide
        if args.comment_start and args.comment_end:
            comment_start = parse_date(args.comment_start)
            comment_end = parse_date(args.comment_end)
        else:
            # fallback: min/max over all comments
            comment_start = df_cmt_all["date"].min()
            comment_end = df_cmt_all["date"].max()

    # Filter comments to a window, but they can still refer to earlier submissions
    mask_c = (df_cmt_all["date"] >= comment_start) & (df_cmt_all["date"] <= comment_end)
    df_cmt = df_cmt_all[mask_c].copy()
    print(f"[v4] Comments in [{comment_start}..{comment_end}]: {len(df_cmt)}")

    # Build submission text map (id -> text)
    sub_text_map: Dict[str, str] = {}
    for _, r in df_sub_all.iterrows():
        sid = str(r["id"])
        sub_text_map[sid] = str(r["text"])

    # Group comments by submission_id within the filtered comment set
    grouped_comments = df_cmt.groupby("submission_id")

    # Output: single dir "submissions"
    out_sub_dir = out_root / "submissions"
    ensure_dir(out_sub_dir)

    part_idx = 0
    written_in_part = 0
    total_requests = 0

    def open_part(idx: int):
        p = out_sub_dir / f"part-{idx:05d}.jsonl"
        f = p.open("w", encoding="utf-8")
        return f, p

    f, part_path = open_part(part_idx)

    # -------- Build requests per submission --------
    for _, sub_row in df_sub.sort_values("created_dt").iterrows():
        sid = str(sub_row["id"])
        sub_text = sub_text_map.get(sid, "")
        if not sub_text:
            continue

        # All comments (within the comment date window) that link to this submission
        try:
            sub_comments = grouped_comments.get_group(sid).copy()
        except KeyError:
            sub_comments = pd.DataFrame()

        # Sort by created_dt if available
        if "created_dt" in sub_comments.columns:
            sub_comments = sub_comments.sort_values("created_dt")

        # Build comment items list
        comment_items: List[Dict[str, Any]] = []
        for _, c in sub_comments.iterrows():
            ctext = str(c["text"]).strip()
            if not ctext:
                continue
            comment_items.append(
                {
                    "kind": "comment",
                    "comment_id": str(c["id"]),
                    "submission_id": sid,
                    "comment_text": ctext,
                    "submission_text": sub_text,
                }
            )

        # Always include the submission itself as an item
        submission_item = {
            "kind": "submission",
            "submission_id": sid,
            "text": str(sub_text).strip(),
        }

        if not comment_items:
            # We can still send the submission by itself
            items_batch = [submission_item]
            body = build_request_body(
                args.model,
                args.max_output_tokens,
                items_batch,
                args.text_format,
            )
            custom_id = f"sub:{sid}:0000:subonly"
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }

            if written_in_part >= args.max_requests_per_file:
                f.close()
                part_idx += 1
                written_in_part = 0
                f, part_path = open_part(part_idx)

            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            written_in_part += 1
            total_requests += 1
            continue

        # Chunk comments for this submission
        for chunk_idx, chunk_comments in enumerate(
            chunk_list(comment_items, args.max_comments_per_request)
        ):
            items_batch = [submission_item] + chunk_comments
            body = build_request_body(
                args.model,
                args.max_output_tokens,
                items_batch,
                args.text_format,
            )
            custom_id = f"sub:{sid}:{chunk_idx:04d}"

            if written_in_part >= args.max_requests_per_file:
                f.close()
                part_idx += 1
                written_in_part = 0
                f, part_path = open_part(part_idx)

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            written_in_part += 1
            total_requests += 1

    f.close()
    print(f"[v4] Wrote {total_requests} requests into {part_idx+1} part file(s) under {out_sub_dir}")


if __name__ == "__main__":
    main()
