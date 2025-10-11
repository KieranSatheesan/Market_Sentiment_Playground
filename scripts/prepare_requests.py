"""
prepare_requests.py
Builds OpenAI Batch API request JSONL files from cleaned Reddit submissions.

- Groups multiple posts per API call (default 16).
- Two schema modes remain ("lite" vs "full") but we DO NOT enforce server-side schema.
  We ask for a JSON object and validate locally in parse_results.py.
- Uses the Responses API with text.format = json_object.

Usage:
  py scripts\prepare_requests.py ^
    --clean_root "data\RedditDumps\cleaned" ^
    --out_dir "batch\Requests" ^
    --model "gpt-4.1-mini" ^
    --schema full ^
    --group-size 16 ^
    --max-chars 1500
"""

import json
import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any
import pandas as pd

# ------------------------------ JSON Schemas (kept for reference only) ------------------------------
LITE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "submission_id": {"type": "string"},
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[A-Z]{1,6}$"},
                        "minItems": 0,
                        "maxItems": 5,
                    },
                },
                "required": ["submission_id", "tickers"],
            }
        }
    },
    "required": ["results"],
}

FULL_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "submission_id": {"type": "string"},
                    "tickers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "symbol": {"type": "string", "pattern": "^[A-Z]{1,6}$"},
                                "sentiment_label": {
                                    "type": "string",
                                    "enum": ["positive", "neutral", "negative"],
                                },
                                "sentiment_score": {
                                    "type": "number",
                                    "minimum": -1,
                                    "maximum": 1,
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "evidence": {"type": "string"},
                            },
                            "required": [
                                "symbol",
                                "sentiment_label",
                                "sentiment_score",
                                "confidence",
                            ],
                        },
                        "minItems": 0,
                        "maxItems": 5,
                    },
                },
                "required": ["submission_id", "tickers"],
            }
        }
    },
    "required": ["results"],
}

# ------------------------------ Helpers ------------------------------
def load_day(day_dir: Path) -> pd.DataFrame:
    p = day_dir / "submissions_clean.parquet"
    if p.exists():
        return pd.read_parquet(p)
    parts = sorted(day_dir.glob("submissions_*.parquet"))
    if parts:
        return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    return pd.DataFrame()

def iter_day_dirs(clean_root: Path) -> Iterable[Path]:
    for d in sorted(clean_root.iterdir()):
        if d.is_dir():
            yield d

def chunk(lst: List[Dict[str, str]], n: int) -> Iterable[List[Dict[str, str]]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare OpenAI Batch request JSONL files from cleaned Reddit submissions.")
    ap.add_argument("--clean_root", required=True, help="Path to data/RedditDumps/cleaned")
    ap.add_argument("--out_dir", required=True, help="Where to write request JSONL files (e.g., batch/Requests)")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model (e.g., gpt-4.1-mini, gpt-5-nano)")
    ap.add_argument("--schema", choices=["lite", "full"], default="full", help="Output schema type (affects prompt expectation only)")
    ap.add_argument("--group-size", type=int, default=16, help="Posts per API call")
    ap.add_argument("--max-chars", type=int, default=1500, help="Trim title+selftext to this many chars")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt_path = Path("prompts") / "ticker_system.txt"
    if not system_prompt_path.exists():
        raise SystemExit(f"Missing prompt file: {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # choose schema purely for doc; NOT sent to server anymore
    json_schema = FULL_SCHEMA if args.schema == "full" else LITE_SCHEMA

    total_days = 0
    total_groups = 0

    for day_dir in iter_day_dirs(clean_root):
        day = day_dir.name
        df = load_day(day_dir)
        if df.empty:
            continue

        posts: List[Dict[str, str]] = []
        for _, r in df.iterrows():
            rid = str(r["id"])
            title = (r.get("title") or "").strip()
            body = (r.get("selftext") or "").strip()
            text = (title + "\n\n" + body)[: args.max_chars]
            posts.append({"submission_id": rid, "text": text})

        if not posts:
            continue

        out_path = out_dir / f"req_{day}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for group in chunk(posts, args.group_size):
                user_prompt = (
                    "You will receive JSON lines for multiple posts. "
                    "Return a SINGLE JSON OBJECT with a 'results' array containing one object PER input post, "
                    "in the SAME ORDER as provided. Do not include any extra text.\nBEGIN:\n"
                    + "\n".join(json.dumps(x, ensure_ascii=False) for x in group)
                )

                body = {
                    "model": args.model,  # e.g., "gpt-4.1-mini"
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                }

                line = {
                    "custom_id": f"{day}_{group[0]['submission_id']}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                total_groups += 1

        total_days += 1
        print(f"Wrote {out_path}  (posts={len(posts)}, requests={((len(posts)-1)//args.group_size)+1})")

    if total_days == 0:
        print("No day folders with data found under:", clean_root)
    else:
        print(f"Done. Days: {total_days}, total grouped requests: {total_groups}")

if __name__ == "__main__":
    main()
