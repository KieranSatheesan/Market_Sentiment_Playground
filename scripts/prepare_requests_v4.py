# scripts/prepare_requests_v4.py (v4 submission+comments context, with file splitting)

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable
import pandas as pd
import datetime as dt

SYSTEM_PROMPT = """
You read INPUT ITEMS and must reply with only:
{"results":[ ... ]}
One result per INPUT ITEM, same order.

Each INPUT ITEM is JSON:
- Submission: {"kind":"submission","submission_id":"<SID>","text":"..."}
- Comment:    {"kind":"comment","comment_id":"<CID>","submission_id":"<SID>","comment_text":"..."}

INPUT ITEMS always start with the submission; the rest are its comments.

For each submission:
{"submission_id":"<SID>","is_forward":true|false|null,"value_score":0.0-1.0,"tickers":[...]}

For each comment:
{"comment_id":"<CID>","submission_id":"<SID>","is_forward":true|false|null,"value_score":0.0-1.0,"tickers":[...]}

Tickers:
- Only real traded stocks/ETFs/REITs; symbols 1–6 A–Z, optional suffix like ".TO" or ".L".
- Max 5 per item; NEVER treat generic words (gold, oil, market, crypto, etc.) as symbols.
- You may infer symbols from clear context, but never invent them.
- Each ticker:
  {"symbol":"AAPL","sentiment_label":"positive|neutral|negative","sentiment_score":-1.0..1.0,"conf":0.0-1.0}

sentiment_score (for this ticker in this text):
- -1.0 to -0.8 : very strong negative
- -0.7 to -0.4 : clearly negative
- -0.3 to -0.1 : mildly negative
-  0.0         : neutral / cannot infer
-  0.1 to 0.3  : mildly positive
-  0.4 to 0.7  : clearly positive
-  0.8 to 1.0  : very strong positive
sentiment_label must match the sign of sentiment_score.

is_forward (future-looking trading relevance of this text):
- true  = clear forward view or trading action (prediction, target, “will buy/sell”, “will pump/dump”, etc.)
- false = past events, questions, commentary, or no clear forward view
- null  = only when intent is genuinely impossible to infer

value_score (trading info value of this text alone):
- 0.0      : no trading value / pure meme
- 0.1–0.3  : very low value (vague, emotional, noisy)
- 0.4–0.6  : relevant but shallow or vague
- 0.7–0.8  : moderately useful reasoning
- 0.9–1.0  : highly actionable, specific, falsifiable thesis

Use "text" for submissions and "comment_text" for comments.
If no tickers: "tickers":[] but still set is_forward and value_score.
No extra fields. No explanations.
""".strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_all_submissions(clean_root: Path) -> pd.DataFrame:
    """
    Load *all* submissions across RS_* dirs.
    Returns columns: id (str), title, selftext, text, created_utc, subreddit.
    """
    rows = []
    for rs_dir in sorted(clean_root.glob("RS_*")):
        for day_dir in sorted(rs_dir.glob("day=*")):
            p = day_dir / "submissions_clean.parquet"
            if not p.exists():
                continue

            # cheap schema read
            schema_df = pd.read_parquet(p).head(0)
            wanted_cols = ["id", "title", "selftext", "created_utc", "subreddit"]
            cols = [c for c in wanted_cols if c in schema_df.columns]

            df = pd.read_parquet(p, columns=cols or None)
            if "id" not in df.columns:
                continue
            df["id"] = df["id"].astype(str)

            title = df.get("title", "").astype(str)
            body  = df.get("selftext", "").astype(str)
            df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
            rows.append(df)

    if not rows:
        print("[v4] WARNING: no submissions found under", clean_root)
        return pd.DataFrame(columns=["id","title","selftext","text","created_utc","subreddit"])

    out = pd.concat(rows, ignore_index=True)
    if "created_utc" in out.columns:
        out["created_dt"] = pd.to_datetime(out["created_utc"], unit="s", utc=True)
        out["date"] = out["created_dt"].dt.date
    else:
        out["created_dt"] = pd.NaT
        out["date"] = pd.NaT

    print(f"[v4] Loaded {len(out):,} submissions total")
    return out


def load_all_comments_for_submissions(
    clean_root: Path,
    submission_ids: List[str],
) -> pd.DataFrame:
    """
    Load ALL comments whose link_id refers to any of the given submission_ids.
    Looks across all RC_* dirs and all days.
    """
    target_link_ids = {f"t3_{sid}" for sid in submission_ids}
    rows = []

    for rc_dir in sorted(clean_root.glob("RC_*")):
        for day_dir in sorted(rc_dir.glob("day=*")):
            p = day_dir / "comments_clean.parquet"
            if not p.exists():
                continue

            # cheap schema read
            schema_df = pd.read_parquet(p).head(0)
            wanted_cols = [
                "id", "body", "link_id", "created_utc",
                "subreddit", "score", "parent_id", "permalink", "author"
            ]
            cols = [c for c in wanted_cols if c in schema_df.columns]

            df = pd.read_parquet(p, columns=cols or None)
            if df.empty:
                continue
            df["id"] = df["id"].astype(str)
            if "link_id" in df.columns:
                df["link_id"] = df["link_id"].astype(str)
            else:
                continue

            mask = df["link_id"].isin(target_link_ids)
            if not mask.any():
                continue

            df = df[mask].copy()
            df["comment_text"] = df.get("body", "").astype(str)
            rows.append(df)

    if not rows:
        print("[v4] WARNING: no comments found for selected submissions.")
        return pd.DataFrame(columns=[
            "id","link_id","comment_text","created_utc",
            "subreddit","score","parent_id","permalink","author"
        ])

    out = pd.concat(rows, ignore_index=True)
    if "created_utc" in out.columns:
        out["created_dt"] = pd.to_datetime(out["created_utc"], unit="s", utc=True)
    print(f"[v4] Loaded {len(out):,} comments for {len(submission_ids)} submissions")
    return out


def chunk(lst, n) -> Iterable[list]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--seed_ids_file", help="Text file; one submission id (e.g. 1lgdft5) per line")
    ap.add_argument("--start", help="optional YYYY-MM-DD lower bound on submission date")
    ap.add_argument("--end", help="optional YYYY-MM-DD upper bound on submission date")
    ap.add_argument("--max_comments_per_req", type=int, default=15)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--max_output_tokens", type=int, default=800)
    ap.add_argument("--text_format", choices=["none","json_object","text","json_schema"],
                    default="json_object")
    ap.add_argument("--max_requests_per_file", type=int, default=800,
                    help="Maximum number of requests per .jsonl file (part-XXXXX.jsonl)")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_root   = Path(args.out_dir)
    ensure_dir(out_root)
    req_dir = out_root / "submissions"
    ensure_dir(req_dir)

    # 1) Load all submissions
    df_subs = load_all_submissions(clean_root)
    if df_subs.empty:
        print("[v4] No submissions loaded; exiting.")
        return

    # 2) Optional date filter
    if args.start:
        start_date = dt.date.fromisoformat(args.start)
        df_subs = df_subs[df_subs["date"] >= start_date]
    if args.end:
        end_date = dt.date.fromisoformat(args.end)
        df_subs = df_subs[df_subs["date"] <= end_date]

    print(f"[v4] After date filter: {len(df_subs):,} submissions remain")

    # 3) Seed filter (handle BOM / weird whitespace robustly)
    if args.seed_ids_file:
        seed_path = Path(args.seed_ids_file)

        # Read with utf-8-sig to transparently strip BOM if present
        raw_lines = seed_path.read_text(encoding="utf-8-sig").splitlines()

        seeds: List[str] = []
        for ln in raw_lines:
            s = ln.strip()
            if s.startswith("\ufeff"):
                s = s.lstrip("\ufeff")
            if s:
                seeds.append(s)

        # dedupe while preserving order
        seeds = list(dict.fromkeys(seeds))

        print(f"[v4] Seed IDs ({len(seeds)}): {seeds}")

        df_subs = df_subs[df_subs["id"].astype(str).isin(seeds)].copy()
        print(f"[v4] Found {len(df_subs):,} submissions matching seed IDs")

        if df_subs.empty:
            print("[v4] Nothing matches the provided seed IDs – aborting.")
            return

    # 4) Load all comments for these submissions (across *all* days)
    sub_ids = df_subs["id"].tolist()
    df_comments = load_all_comments_for_submissions(clean_root, sub_ids)

    # 5) Build and write requests grouped per submission, splitting across part-*.jsonl
    part_idx = -1
    current_in_file = 0
    total_written = 0
    f = None  # current file handle

    def write_line(line: str):
        nonlocal part_idx, current_in_file, total_written, f
        # Open new file if needed
        if f is None or current_in_file >= args.max_requests_per_file:
            if f is not None:
                f.close()
            part_idx += 1
            part_path = req_dir / f"part-{part_idx:05d}.jsonl"
            print(f"[v4] Opening {part_path} for writing...")
            f = part_path.open("w", encoding="utf-8")
            current_in_file = 0
        f.write(line + "\n")
        current_in_file += 1
        total_written += 1

    for _, sub in df_subs.iterrows():
        sid = sub["id"]
        stext = str(sub["text"])[:args.max_chars].strip()
        if not stext:
            continue

        cmt_for_sub = (
            df_comments[df_comments["link_id"] == f"t3_{sid}"]
            if not df_comments.empty else pd.DataFrame()
        )

        if cmt_for_sub.empty:
            # Still send one request with just the submission (so it gets annotated)
            groups = [[]]
        else:
            comments = [
                {
                    "kind": "comment",
                    "comment_id": r["id"],
                    "submission_id": sid,
                    "comment_text": str(r["comment_text"])[:args.max_chars].strip(),
                    "created_utc": int(r.get("created_utc", 0)),
                }
                for _, r in cmt_for_sub.sort_values("created_utc").iterrows()
            ]
            groups = list(chunk(comments, args.max_comments_per_req))

        for gi, comment_chunk in enumerate(groups):
            items: List[Dict[str, Any]] = []
            # First item: submission
            items.append({
                "kind": "submission",
                "submission_id": sid,
                "text": stext,
            })
            # Remaining items: comments
            items.extend(comment_chunk)

            user_content = (
                'Return JSON only: {"results":[...]} with one output per INPUT ITEM, same order.\n'
                "BEGIN_INPUTS:\n" +
                "\n".join(json.dumps(obj, ensure_ascii=False) for obj in items)
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

            custom_id = f"sub:{sid}:{gi:04d}"
            line = json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                ensure_ascii=False,
            )
            write_line(line)

    if f is not None:
        f.close()

    if total_written == 0:
        print("[v4] WARNING: no requests actually written – check filters / seeds.")
    else:
        num_files = part_idx + 1
        print(f"[v4] Wrote {total_written} requests into {num_files} file(s) under {req_dir}")


if __name__ == "__main__":
    main()
