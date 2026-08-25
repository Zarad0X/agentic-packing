#!/usr/bin/env python3
"""Fetch only the official Objaverse metadata shards needed by a candidate pool."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata"


def required_shards(candidate_path: Path) -> list[str]:
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    shards = {
        candidate["object_path"].split("/")[1]
        for pool in data["categories"].values()
        for candidate in pool["candidates"]
    }
    return sorted(shards)


def validate_shard(path: Path) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid Objaverse metadata shard: {path}")


def fetch_shard(shard: str, output_dir: Path, retries: int = 3) -> tuple[str, str]:
    target = output_dir / f"{shard}.json.gz"
    if target.is_file():
        validate_shard(target)
        return shard, "cached"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"{BASE_URL}/{shard}.json.gz",
        headers={"User-Agent": "agentic-packing-asset-curator/0.1"},
    )
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with opener.open(request, timeout=90) as response, part.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            validate_shard(part)
            os.replace(part, target)
            return shard, "downloaded"
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = required_shards(args.candidates)
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_shard, shard, args.output_dir): shard for shard in shards}
        for future in as_completed(futures):
            shard, status = future.result()
            counts[status] += 1
            print(f"{status:10s} {shard}")
    print(json.dumps({"required": len(shards), **counts}, indent=2))


if __name__ == "__main__":
    main()
