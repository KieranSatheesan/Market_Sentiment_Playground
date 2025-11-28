# scripts/submit_and_collect_concurrent_v4.py
"""
Submit & collect OpenAI Batch API jobs for v4 submission-centric requests.

Requests:
    batch/Requests_v4/submissions/part-00000.jsonl

Results:
    batch/Results_v4/submissions/part-00000.jsonl
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def iter_request_files(requests_root: Path) -> List[Path]:
    out: List[Path] = []
    sub_dir = requests_root / "submissions"
    if sub_dir.exists():
        for part in sorted(sub_dir.glob("part-*.jsonl")):
            out.append(part)
    return out


def result_paths(results_root: Path, req_part_path: Path) -> Path:
    out_dir = results_root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = req_part_path.name
    return out_dir / fname


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_root", required=True, help="e.g. batch/Requests_v4")
    ap.add_argument("--results_root", required=True, help="e.g. batch/Results_v4")
    ap.add_argument("--desc", default="reddit_llm_v4")
    ap.add_argument("--max_in_flight", type=int, default=10)
    ap.add_argument("--poll_sec", type=int, default=15)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")

    requests_root = Path(args.requests_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    debug_root = results_root / "debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    req_files = iter_request_files(requests_root)
    print(f"[v4] Total request files to submit: {len(req_files)}")
    if not req_files:
        return

    client = OpenAI()

    queue: List[Tuple[Path, Path]] = []
    for req in req_files:
        out_path = result_paths(results_root, req)
        if out_path.exists():
            continue
        queue.append((req, out_path))

    in_flight: Dict[str, Tuple[Path, Path]] = {}
    last_status: Dict[str, str] = {}

    # Prime
    while queue and len(in_flight) < args.max_in_flight:
        req, out_path = queue.pop(0)
        b = submit_batch(client, req, args.desc)
        in_flight[b.id] = (req, out_path)
        print(f"Submitted {req} -> batch {b.id}")

    # Poll loop
    while in_flight or queue:
        for batch_id in list(in_flight.keys()):
            req, out_path = in_flight[batch_id]
            b = client.batches.retrieve(batch_id)
            if last_status.get(batch_id) != b.status:
                print(f"[{batch_id}] status={b.status}")
                last_status[batch_id] = b.status

            if b.status in ("completed", "failed", "expired", "cancelled"):
                # Collect outputs or errors
                output_ids: List[str] = []
                if getattr(b, "output_file_id", None):
                    output_ids = [b.output_file_id]
                elif getattr(b, "output_file_ids", None):
                    output_ids = [
                        x["id"] if isinstance(x, dict) else x
                        for x in b.output_file_ids
                    ]
                if output_ids:
                    with out_path.open("wb") as fh:
                        for fid in output_ids:
                            content = client.files.content(fid)
                            fh.write(content.read())
                    print(f"Saved results → {out_path}")
                else:
                    # Write a minimal marker
                    marker = {
                        "batch_id": batch_id,
                        "status": b.status,
                        "metadata": getattr(b, "metadata", {}),
                    }
                    err_path = out_path.with_suffix(".errors.jsonl")
                    err_path.write_text(json.dumps(marker), encoding="utf-8")
                    print(f"No output. Wrote marker → {err_path}")

                # Remove from active
                del in_flight[batch_id]

                # Backfill
                if queue and len(in_flight) < args.max_in_flight:
                    r2, out2 = queue.pop(0)
                    b2 = submit_batch(client, r2, args.desc)
                    in_flight[b2.id] = (r2, out2)
                    print(f"Submitted {r2} -> batch {b2.id}")

        time.sleep(args.poll_sec)

    print("[v4] All batches processed.")


if __name__ == "__main__":
    main()
