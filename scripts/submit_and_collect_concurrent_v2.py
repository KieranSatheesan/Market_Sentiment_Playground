# scripts/submit_and_collect_concurrent_v2.py
"""
Submit & collect OpenAI Batch API jobs for RS/RC annotation requests.

Layout (requests produced by prepare_requests_v2.py):
  batch/Requests/submissions/day=YYYY-MM-DD/part-00000.jsonl
  batch/Requests/comments/day=YYYY-MM-DD/part-00000.jsonl

Results:
  batch/Results/submissions/day=YYYY-MM-DD/part-00000.jsonl
  batch/Results/comments/day=YYYY-MM-DD/part-00000.jsonl

Adds a debug dump when a batch fails without an error_file_id.
"""

import os
import argparse
import time
import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


KIND_DIR = {"rs": "submissions", "rc": "comments"}


def parse_day_from_dir(day_dir: Path) -> dt.date | None:
    # expects .../day=YYYY-MM-DD
    if day_dir.name.startswith("day="):
        try:
            return dt.date.fromisoformat(day_dir.name.split("=", 1)[1])
        except Exception:
            return None
    return None


def iter_request_files(
    requests_root: Path,
    kinds: List[str],
    start: dt.date | None,
    end: dt.date | None,
) -> List[Tuple[str, dt.date, Path]]:
    """
    Returns a list of (kind, day, req_file_path), sorted by (kind, day, path).
    """
    items: List[Tuple[str, dt.date, Path]] = []
    for k in kinds:
        kdir = requests_root / KIND_DIR[k]
        if not kdir.exists():
            continue
        for day_dir in sorted(kdir.glob("day=*")):
            day = parse_day_from_dir(day_dir)
            if day is None:
                continue
            if start and day < start:
                continue
            if end and day > end:
                continue
            for part in sorted(day_dir.glob("part-*.jsonl")):
                items.append((k, day, part))
    items.sort(key=lambda t: (t[0], t[1], t[2].name))
    return items


def result_paths(results_root: Path, kind: str, day: dt.date, req_part_path: Path) -> Tuple[Path, Path]:
    """
    Mirror the requests path under results.
    Returns (ok_path, err_path)
    """
    out_dir = results_root / KIND_DIR[kind] / f"day={day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = req_part_path.name  # part-00000.jsonl
    ok = out_dir / fname
    err = out_dir / (fname.replace(".jsonl", ".errors.jsonl"))
    return ok, err


def submit_batch(client: OpenAI, req_path: Path, desc: str):
    with req_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"desc": desc, "req_path": str(req_path)},
    )
    return batch


def download_file(client: OpenAI, file_id: str, out_path: Path):
    content = client.files.content(file_id)
    out_path.write_bytes(content.read())


def dump_debug(json_obj: Any, debug_dir: Path, batch_id: str):
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / f"{batch_id}.json"
    try:
        out.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DEBUG] Saved batch debug → {out}")
    except Exception as e:
        print(f"[DEBUG] Failed to write debug JSON: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_root", required=True, help="batch/Requests")
    ap.add_argument("--results_root", required=True, help="batch/Results")
    ap.add_argument("--kind", choices=["rs", "rc", "both"], default="both")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive", default=None)
    ap.add_argument("--end", help="YYYY-MM-DD inclusive", default=None)
    ap.add_argument("--desc", default="reddit_llm_v2")
    ap.add_argument("--max_in_flight", type=int, default=10)
    ap.add_argument("--poll_sec", type=int, default=15)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")

    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    kinds = ["rs", "rc"] if args.kind == "both" else [args.kind]

    requests_root = Path(args.requests_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    debug_root = results_root / "debug"

    # Build queue (skip any request file that already has a result OK/ERR)
    queue: List[Tuple[str, dt.date, Path, Path, Path]] = []
    for kind, day, req_path in iter_request_files(requests_root, kinds, start, end):
        ok, err = result_paths(results_root, kind, day, req_path)
        if ok.exists() or err.exists():
            continue
        queue.append((kind, day, req_path, ok, err))

    print(f"Total request files to submit: {len(queue)}")
    if not queue:
        return

    client = OpenAI()

    in_flight: Dict[str, Tuple[str, dt.date, Path, Path, Path]] = {}  # batch_id -> (kind, day, req, ok, err)
    last_status: Dict[str, str] = {}

    # Prime pipeline
    while queue and len(in_flight) < args.max_in_flight:
        kind, day, req, ok, err = queue.pop(0)
        b = submit_batch(client, req, args.desc)
        in_flight[b.id] = (kind, day, req, ok, err)
        print(f"Submitted {req} -> batch {b.id}")

    # Poll loop
    while in_flight or queue:
        for batch_id in list(in_flight.keys()):
            kind, day, req, ok, err = in_flight[batch_id]
            b = client.batches.retrieve(batch_id)
            if last_status.get(batch_id) != b.status:
                print(f"[{batch_id}] {KIND_DIR[kind]} day={day} status={b.status}")
                last_status[batch_id] = b.status

            if b.status in ("completed", "failed", "expired", "cancelled"):
                # Save outputs if any
                out_ids = []
                if getattr(b, "output_file_id", None):
                    out_ids = [b.output_file_id]
                elif getattr(b, "output_file_ids", None):
                    out_ids = [x["id"] if isinstance(x, dict) else x for x in b.output_file_ids]

                if out_ids:
                    with ok.open("wb") as fh:
                        for fid in out_ids:
                            content = client.files.content(fid)
                            fh.write(content.read())
                    print(f"Saved results  → {ok}")
                else:
                    err_id = getattr(b, "error_file_id", None)
                    if err_id:
                        download_file(client, err_id, err)
                        print(f"Saved errors   → {err}")
                    else:
                        # No artifacts; dump the batch object for root-cause, and also write a slim marker
                        marker = {"batch_id": batch_id, "status": b.status, "metadata": getattr(b, "metadata", {})}
                        err.write_text(json.dumps(marker), encoding="utf-8")
                        print(f"No output. Wrote marker → {err}")
                        try:
                            # b is a pydantic-like object; convert via .model_dump() if available, else dict()
                            dumpable = getattr(b, "model_dump", None)
                            if callable(dumpable):
                                dump_debug(dumpable(), debug_root, batch_id)
                            else:
                                # Best effort
                                dump_debug(json.loads(b.model_dump_json()), debug_root, batch_id)
                        except Exception:
                            # Ultimate fallback: write minimal info
                            dump_debug(marker, debug_root, batch_id)

                # remove from active
                del in_flight[batch_id]

                # backfill
                if queue and len(in_flight) < args.max_in_flight:
                    k2, d2, r2, ok2, er2 = queue.pop(0)
                    b2 = submit_batch(client, r2, args.desc)
                    in_flight[b2.id] = (k2, d2, r2, ok2, er2)
                    print(f"Submitted {r2} -> batch {b2.id}")

        time.sleep(args.poll_sec)

    print("All batches processed.")


if __name__ == "__main__":
    main()
