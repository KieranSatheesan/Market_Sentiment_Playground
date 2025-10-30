# scripts/filter_month.py
import os, io, json, time, argparse, datetime as dt
from typing import Dict, Any, List
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import yaml
except ImportError:
    yaml = None

# --- Default target subs; can be overridden by --subs or YAML
TARGET_SUBS = {s.lower() for s in [
    "stocks","wallstreetbets","investing","trading","stockmarket","pennystocks"
]}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def date_str_from_utc(ts: float) -> str:
    return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

# --- Schemas (fixed to avoid drift)
def make_schema_rs() -> pa.schema:
    return pa.schema([
        ("id", pa.string()),
        ("created_utc", pa.int64()),
        ("subreddit", pa.string()),
        ("title", pa.string()),
        ("selftext", pa.string()),
        ("score", pa.int32()),
        ("num_comments", pa.int32()),
        ("author", pa.string()),
        ("permalink", pa.string()),
        ("url", pa.string()),
        ("over_18", pa.bool_()),
    ])

def make_schema_rc() -> pa.schema:
    # Comments don't have title/selftext; main text is "body".
    # Keep link_id (submission), parent_id (comment or submission), permalink.
    return pa.schema([
        ("id", pa.string()),
        ("created_utc", pa.int64()),
        ("subreddit", pa.string()),
        ("body", pa.string()),
        ("score", pa.int32()),
        ("author", pa.string()),
        ("parent_id", pa.string()),
        ("link_id", pa.string()),
        ("permalink", pa.string()),
    ])

SCHEMA_RS = make_schema_rs()
SCHEMA_RC = make_schema_rc()

def _s(x):  return "" if x is None else str(x)
def _i32(x):
    try: return int(x)
    except: return 0
def _i64(x):
    try: return int(x)
    except: return 0
def _b(x):
    return bool(x) if isinstance(x, bool) else (str(x).lower() == "true")

def coerce_record_rs(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":           _s(rec.get("id")),
        "created_utc":  _i64(rec.get("created_utc")),
        "subreddit":    _s(rec.get("subreddit")),
        "title":        _s(rec.get("title")),
        "selftext":     _s(rec.get("selftext")),
        "score":        _i32(rec.get("score")),
        "num_comments": _i32(rec.get("num_comments")),
        "author":       _s(rec.get("author")),
        "permalink":    _s(rec.get("permalink")),
        "url":          _s(rec.get("url")),
        "over_18":      _b(rec.get("over_18")),
    }

def coerce_record_rc(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":          _s(rec.get("id")),
        "created_utc": _i64(rec.get("created_utc")),
        "subreddit":   _s(rec.get("subreddit")),
        "body":        _s(rec.get("body")),
        "score":       _i32(rec.get("score")),
        "author":      _s(rec.get("author")),
        "parent_id":   _s(rec.get("parent_id")),
        "link_id":     _s(rec.get("link_id")),
        "permalink":   _s(rec.get("permalink")),
    }

def write_day_part(out_root: str, day: str, rows: List[Dict[str, Any]], part_idx: int, kind: str) -> int:
    """Write a list of dict rows to a parquet part for this day, return next part index."""
    if not rows:
        return part_idx
    schema = SCHEMA_RS if kind == "rs" else SCHEMA_RC
    table = pa.Table.from_pylist(rows, schema=schema)
    day_dir = os.path.join(out_root, f"day={day}")
    ensure_dir(day_dir)
    basename = "submissions" if kind == "rs" else "comments"
    out_file = os.path.join(day_dir, f"{basename}_{part_idx:04d}.parquet")
    pq.write_table(table, out_file, compression="zstd")
    return part_idx + 1

def process_month(
    zst_path: str,
    out_root: str,
    log_path: str,
    kind: str,                          # "rs" or "rc"
    flush_every: int = 50_000,
    drop_nsfw: bool = False,            # applies to RS only
    max_lines: int | None = None,
    target_subs: set[str] | None = None,
) -> None:
    subs = {s.lower() for s in (target_subs or TARGET_SUBS)}
    ensure_dir(out_root); ensure_dir(os.path.dirname(log_path))

    buffers: Dict[str, List[Dict[str, Any]]] = {}
    part_idx: Dict[str, int] = {}

    count_in = 0
    kept = 0
    t0 = time.time()

    with open(zst_path, "rb") as fh, open(log_path, "a", encoding="utf-8") as lg:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                if not line:
                    break
                count_in += 1
                if max_lines and count_in > max_lines:
                    break

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sub = (rec.get("subreddit") or "").lower()
                if sub not in subs:
                    continue

                utc = rec.get("created_utc")
                if utc is None:
                    continue

                if kind == "rs":
                    typed = coerce_record_rs(rec)
                    if drop_nsfw and typed.get("over_18"):
                        continue
                else:
                    # kind == "rc"
                    # skip completely empty or deleted comments
                    body = rec.get("body")
                    if body is None or str(body).strip() == "" or str(body).strip().lower() in {"[deleted]", "[removed]"}:
                        continue
                    typed = coerce_record_rc(rec)

                day = date_str_from_utc(float(typed["created_utc"]))
                if day not in buffers:
                    buffers[day] = []
                    part_idx.setdefault(day, 0)

                buffers[day].append(typed)
                kept += 1

                # Opportunistic flush
                if kept % flush_every == 0:
                    for d, rows in list(buffers.items()):
                        if rows:
                            part_idx[d] = write_day_part(out_root, d, rows, part_idx[d], kind)
                            buffers[d].clear()
                    lg.write(f"{time.strftime('%F %T')} [{kind}] processed={count_in} kept={kept}\n")
                    lg.flush()

        # Final flush
        for d, rows in buffers.items():
            if rows:
                part_idx[d] = write_day_part(out_root, d, rows, part_idx[d], kind)

        dt_sec = time.time() - t0
        lg.write(f"DONE ({kind}) {os.path.basename(zst_path)} in {dt_sec:.1f}s lines={count_in} kept={kept}\n")
        lg.flush()

def load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml is not installed. pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to YAML config file", default=None)
    ap.add_argument("--kind", choices=["rs","rc"], required=True, help="rs=submissions, rc=comments")
    ap.add_argument("--zst", help="Path to RS_YYYY-MM.zst or RC_YYYY-MM.zst")
    ap.add_argument("--out", help="Output root folder for daily parquet")
    ap.add_argument("--log", help="Path to .log file")
    ap.add_argument("--flush_every", type=int, default=None)
    ap.add_argument("--drop_nsfw", action="store_true", help="(RS only) drop over_18")
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--subs", nargs="*", help="Override target subreddits list")
    args = ap.parse_args()

    # Load config then allow CLI overrides
    cfg = {}
    if args.config:
        cfg = load_yaml(args.config) or {}

    zst = args.zst or cfg.get("zst")
    out = args.out or cfg.get("out")
    log = args.log or cfg.get("log")
    kind = args.kind or cfg.get("kind")
    flush_every = args.flush_every or cfg.get("flush_every", 50_000)
    drop_nsfw = args.drop_nsfw or bool(cfg.get("drop_nsfw", False))
    max_lines = args.max_lines or cfg.get("max_lines")
    subs = set(args.subs) if args.subs else set(cfg.get("subs", [])) or TARGET_SUBS

    if not (zst and out and log and kind):
        raise SystemExit("Missing required args: --kind, --zst, --out, --log (or provide in --config)")

    process_month(
        zst_path=zst,
        out_root=out,
        log_path=log,
        kind=kind,
        flush_every=flush_every,
        drop_nsfw=drop_nsfw,
        max_lines=max_lines,
        target_subs=subs,
    )
