# scripts/submit_and_collect.py
import os, time, argparse
from pathlib import Path
from openai import OpenAI

def submit_batch(client, req_path: Path, desc: str):
    with req_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"desc": desc}
    )
    return batch

def wait_for_batch(client, batch_id: str, poll=15):
    last = None
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status != last:
            print(f"[{batch_id}] status={b.status}")
            last = b.status
        if b.status in ("failed","expired","cancelled","completed"):
            return b
        time.sleep(poll)

def download_file(client, file_id: str, out_path: Path):
    content = client.files.content(file_id)
    out_path.write_bytes(content.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--desc", default="reddit_ticker_sentiment")
    args = ap.parse_args()

    client = OpenAI()
    req_dir = Path(args.requests_dir)
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    for req_file in sorted(req_dir.glob("req_*.jsonl")):
        out_file = res_dir / f"res_{req_file.name}"
        if out_file.exists():
            print("Skipping", req_file.name, "(already done)")
            continue

        print("Submitting", req_file.name)
        batch = submit_batch(client, req_file, args.desc)
        print("Submitted:", batch.id)
        done = wait_for_batch(client, batch.id)

        # Print counts and IDs for transparency
        print("request_counts:", getattr(done, "request_counts", None))
        print("output_file_id:", getattr(done, "output_file_id", None))
        print("output_file_ids:", getattr(done, "output_file_ids", None))
        print("error_file_id:", getattr(done, "error_file_id", None))

        if done.status != "completed":
            # Save error file if present, then continue
            err_id = getattr(done, "error_file_id", None)
            if err_id:
                err_path = res_dir / (out_file.stem + ".errors.jsonl")
                download_file(client, err_id, err_path)
                print("Saved errors →", err_path)
            print("❌ Batch did not complete:", done.status)
            continue

        # Collect output IDs
        out_ids = []
        single = getattr(done, "output_file_id", None)
        multi  = getattr(done, "output_file_ids", None)
        if single:
            out_ids = [single]
        elif isinstance(multi, (list, tuple)):
            for x in multi:
                if isinstance(x, str):
                    out_ids.append(x)
                elif isinstance(x, dict) and "id" in x:
                    out_ids.append(x["id"])

        if not out_ids:
            # Completed but no outputs → fetch error file if present
            err_id = getattr(done, "error_file_id", None)
            if err_id:
                err_path = res_dir / (out_file.stem + ".errors.jsonl")
                download_file(client, err_id, err_path)
                print("Saved errors →", err_path)
            print("❌ No output files returned.")
            continue

        # Concatenate outputs if multiple
        with out_file.open("wb") as out:
            for fid in out_ids:
                content = client.files.content(fid)
                out.write(content.read())
        print("Saved results →", out_file)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env")
    main()
