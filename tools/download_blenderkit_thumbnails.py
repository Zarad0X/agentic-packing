#!/usr/bin/env python3
"""Download public thumbnails referenced by a BlenderKit shortlist."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _download(candidate: dict[str, Any], output_dir: Path) -> tuple[str, str]:
    category_dir = output_dir / candidate["category"]
    category_dir.mkdir(parents=True, exist_ok=True)
    target = category_dir / f"{candidate['asset_base_id']}.jpg"
    if target.is_file() and target.stat().st_size > 0:
        return str(target), "cached"
    request = urllib.request.Request(
        candidate["thumbnail_url"],
        headers={"User-Agent": "agentic-packing-asset-curator/0.1"},
    )
    part = target.with_suffix(".jpg.part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as stream:
                while chunk := response.read(256 * 1024):
                    stream.write(chunk)
            part.replace(target)
            return str(target), "downloaded"
        except (OSError, urllib.error.URLError):
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    data = json.loads(args.shortlist.read_text(encoding="utf-8"))
    candidates = [
        candidate
        for pool in data["categories"].values()
        for candidate in pool["candidates"]
    ]
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download, candidate, args.output_dir) for candidate in candidates]
        for future in as_completed(futures):
            _, status = future.result()
            counts[status] += 1
    print(json.dumps({"candidates": len(candidates), **counts}, indent=2))


if __name__ == "__main__":
    main()
