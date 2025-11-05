# scripts/prepare_requests_v2.py
"""
prepare_requests_v2.py
Build OpenAI Batch API request JSONL files from CLEANED submissions (RS) and/or comments (RC).

Key features:
- --kind {rs, rc, both}
- --start YYYY-MM-DD --end YYYY-MM-DD (inclusive)
- Outputs under: batch/Requests/<submissions|comments>/day=YYYY-MM-DD/part-00000.jsonl
- Optional splitting per day: --max_requests_per_file (default 800)
- Optional structured outputs flag: --text_format {none,json_object,text,json_schema} (default json_object)
- Optional comment gating: --min_comment_len (default 0)
- Truncation: --max_chars per doc
- Writes MANIFEST.json with counts
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple
import pandas as pd
import datetime as dt

# ------------------------------ helpers ------------------------------
def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_day_cleaned(kind: str, clean_root: Path, day: dt.date) -> pd.DataFrame:
    """
    kind: 'rs' (submissions) or 'rc' (comments)
    Layout:
      RS: <clean_root>/RS_YYYY-MM/day=YYYY-MM-DD/submissions_clean.parquet
      RC: <clean_root>/RC_YYYY-MM/day=YYYY-MM-DD/comments_clean.parquet
    """
    ym = day.strftime("%Y-%m")
    base = "RS_" if kind == "rs" else "RC_"
    day_dir = clean_root / f"{base}{ym}" / f"day={day:%Y-%m-%d}"
    fname = "submissions_clean.parquet" if kind == "rs" else "comments_clean.parquet"
    fpath = day_dir / fname
    if not fpath.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(fpath)
    except Exception as e:
        print(f"[WARN] failed to read {fpath}: {e}")
        return pd.DataFrame()

    # Standardize available text -> 'text'
    if "text" not in df.columns:
        if kind == "rs":
            title = df.get("title", pd.Series([""] * len(df), index=df.index)).astype(str)
            body  = df.get("selftext", pd.Series([""] * len(df), index=df.index)).astype(str)
            df["text"] = (title.str.strip() + "\n\n" + body.str.strip()).str.strip()
        else:
            df["text"] = df.get("body", pd.Series([""] * len(df), index=df.index)).astype(str)

    df["id"] = df["id"].astype(str)
    return df

def chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def compact_user_message(doc_id: str, allow_etf: bool, text: str) -> str:
    # extremely compact per-doc payload
    return f"id={doc_id}\nallow_etf={'true' if allow_etf else 'false'}\ntext:\n{text}"

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True, help="data_v2/RedditDumps/cleaned")
    ap.add_argument("--out_dir", required=True, help="batch/Requests")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--kind", choices=["rs", "rc", "both"], default="both")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=1500, help="Trim text per doc")
    ap.add_argument("--allow_etf", type=str, default="false", help="true/false passed to prompt")
    ap.add_argument("--min_comment_len", type=int, default=0, help="Only for rc; drop comments with len(text)<N")
    ap.add_argument("--system_prompt_path", default="prompts/ticker_system_v2.txt")
    ap.add_argument("--max_requests_per_file", type=int, default=800,
                    help="Split each day into multiple parts when request groups exceed this number.")
    # NOTE: 'json' is NOT a valid type for Responses API text.format.type.
    ap.add_argument("--text_format", choices=["none","json_object","text","json_schema"], default="json_object",
                    help="Structured outputs switch for Responses API.")
    ap.add_argument("--max_output_tokens", type=int, default=256)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_root   = Path(args.out_dir)
    ensure_dir(out_root)

    # Load system prompt
    spath = Path(args.system_prompt_path)
    if not spath.exists():
        raise SystemExit(f"Missing prompt file: {spath}")
    system_prompt = spath.read_text(encoding="utf-8")

    allow_etf_bool = str(args.allow_etf).strip().lower() in {"1", "true", "yes", "y"}

    start_date = dt.date.fromisoformat(args.start)
    end_date   = dt.date.fromisoformat(args.end)

    kinds = ["rs", "rc"] if args.kind == "both" else [args.kind]
    total_requests = 0
    manifest_rows: List[Dict[str, Any]] = []

    for day in daterange(start_date, end_date):
        for kind in kinds:
            df = read_day_cleaned(kind, clean_root, day)
            if df.empty:
                print(f"[INFO] {kind.upper()} {day} -> no data")
                continue

            # Optional RC gating by length
            if kind == "rc" and args.min_comment_len > 0:
                df = df[df["text"].astype(str).str.len() >= args.min_comment_len].copy()
                if df.empty:
                    print(f"[INFO] RC {day} after len>={args.min_comment_len} -> empty")
                    continue

            # Prepare docs
            docs: List[Tuple[str, str]] = []
            clipped = df["text"].astype(str).str.slice(0, args.max_chars)
            for rid, body in zip(df["id"].astype(str).values, clipped.values):
                if not body:
                    continue
                docs.append((rid, body))

            if not docs:
                print(f"[INFO] {kind.upper()} {day} -> no docs after trimming")
                continue

            kind_name = "submissions" if kind == "rs" else "comments"
            out_day_dir = out_root / kind_name / f"day={day:%Y-%m-%d}"
            ensure_dir(out_day_dir)

            # rolling writer with part counter
            part_idx = 0
            written_in_part = 0
            total_written_for_day = 0

            def open_new_part(idx: int):
                path = out_day_dir / f"part-{idx:05d}.jsonl"
                fh = path.open("w", encoding="utf-8")
                return fh, path

            f, out_path = open_new_part(part_idx)

            for group_idx, group in enumerate(chunk(docs, args.group_size)):
                # rollover if this part reached its cap
                if written_in_part >= args.max_requests_per_file:
                    f.close()
                    manifest_rows.append({
                        "kind": kind_name,
                        "day": f"{day:%Y-%m-%d}",
                        "docs": None,
                        "requests": written_in_part,
                        "group_size": args.group_size,
                        "path": str(out_path),
                    })
                    part_idx += 1
                    written_in_part = 0
                    f, out_path = open_new_part(part_idx)

                user_prompt = (
                    "Return ONE JSON object with a 'results' array, one entry PER input item, IN THE SAME ORDER.\nBEGIN:\n"
                    + "\n".join(
                        json.dumps(
                            {
                                "submission_id": rid,
                                "user": compact_user_message(rid, allow_etf_bool, text),
                            },
                            ensure_ascii=False,
                        )
                        for rid, text in group
                    )
                )

                # Build body (Responses API)
                body: Dict[str, Any] = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_output_tokens": args.max_output_tokens,
                }

                # text.format must be an object with a valid type
                if args.text_format != "none":
                    body["text"] = {"format": {"type": args.text_format}}

                custom_id = f"{kind}:{day:%Y-%m-%d}:{group[0][0]}:{group_idx:04d}"
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
            # record last open part
            manifest_rows.append({
                "kind": kind_name,
                "day": f"{day:%Y-%m-%d}",
                "docs": len(docs),
                "requests": written_in_part,
                "group_size": args.group_size,
                "path": str(out_path),
            })
            total_requests += total_written_for_day
            print(f"[WRITE] {out_day_dir}  docs={len(docs)}  requests={total_written_for_day} (parts={part_idx+1})")

    # Write a global manifest for the run
    if manifest_rows:
        manifest_path = Path(args.out_dir) / "MANIFEST.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_rows, mf, ensure_ascii=False, indent=2)
        total_docs = sum(m.get("docs", 0) or 0 for m in manifest_rows)
        distinct_day_parts = len(manifest_rows)
        print(f"\nDone. Day-parts={distinct_day_parts}, Docs={total_docs}, Requests={total_requests}")
        print(f"Manifest: {manifest_path}")
    else:
        print("No JSONL written (no matching days/kinds).")

if __name__ == "__main__":
    main()
