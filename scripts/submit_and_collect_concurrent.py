# scripts/submit_and_collect_concurrent.py
import os, argparse, time, json
from pathlib import Path
from openai import OpenAI

def submit_batch(client: OpenAI, req_path: Path, desc: str):
    with req_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"desc": desc}
    )
    return batch

def download_file(client: OpenAI, file_id: str, out_path: Path):
    content = client.files.content(file_id)
    out_path.write_bytes(content.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--desc", default="reddit_ticker_sentiment_full")
    ap.add_argument("--max_in_flight", type=int, default=10)  # set 10–15 if you want
    ap.add_argument("--poll_sec", type=int, default=15)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")

    client = OpenAI()
    req_dir = Path(args.requests_dir)
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Build our todo list: only days without results (.jsonl or .errors.jsonl)
    queue = []
    for req in sorted(req_dir.glob("req_*.jsonl")):
        day = req.stem.replace("req_", "")
        out_ok = res_dir / f"res_req_{day}.jsonl"
        out_err = res_dir / f"res_req_{day}.errors.jsonl"
        if out_ok.exists() or out_err.exists():
            continue
        queue.append((day, req, out_ok, out_err))

    print(f"Total days remaining: {len(queue)}")
    if not queue:
        return

    in_flight = {}  # batch_id -> (day, out_ok, out_err)
    # Prime up to max_in_flight
    while queue and len(in_flight) < args.max_in_flight:
        day, req, out_ok, out_err = queue.pop(0)
        batch = submit_batch(client, req, args.desc)
        in_flight[batch.id] = (day, out_ok, out_err)
        print(f"Submitted {req.name} -> {batch.id}")

    # Poll loop
    last_status = {}
    while in_flight or queue:
        for batch_id in list(in_flight.keys()):
            day, out_ok, out_err = in_flight[batch_id]
            b = client.batches.retrieve(batch_id)
            if last_status.get(batch_id) != b.status:
                print(f"[{batch_id}] {day} status={b.status}")
                last_status[batch_id] = b.status

            if b.status in ("completed", "failed", "expired", "cancelled"):
                # Prefer output file(s)
                out_ids = []
                if getattr(b, "output_file_id", None):
                    out_ids = [b.output_file_id]
                elif getattr(b, "output_file_ids", None):
                    out_ids = [x["id"] if isinstance(x, dict) else x for x in b.output_file_ids]

                if out_ids:
                    with out_ok.open("wb") as out:
                        for fid in out_ids:
                            content = client.files.content(fid)
                            out.write(content.read())
                    print(f"Saved results → {out_ok}")
                else:
                    err_id = getattr(b, "error_file_id", None)
                    if err_id:
                        download_file(client, err_id, out_err)
                        print(f"Saved errors  → {out_err}")
                    else:
                        print(f"No output for {day} (status={b.status})")

                del in_flight[batch_id]

                # Backfill new submission to keep concurrency level
                if queue:
                    d2, r2, ok2, er2 = queue.pop(0)
                    b2 = submit_batch(client, r2, args.desc)
                    in_flight[b2.id] = (d2, ok2, er2)
                    print(f"Submitted {r2.name} -> {b2.id}")

        time.sleep(args.poll_sec)

    print("All batches done.")

if __name__ == "__main__":
    main()
