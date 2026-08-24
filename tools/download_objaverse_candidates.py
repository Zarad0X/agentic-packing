#!/usr/bin/env python3
"""Download an explicit reviewed Objaverse shortlist subset with SHA-256 provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_glb(path: Path) -> None:
    if path.stat().st_size < 20:
        raise ValueError(f"GLB is unexpectedly small: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"glTF":
            raise ValueError(f"Downloaded file is not a GLB: {path}")


def _download(candidate: dict[str, Any], output_dir: Path, retries: int = 3) -> dict[str, Any]:
    target = output_dir / f"{candidate['uid']}.glb"
    expected_size = int(candidate["glb_size"])
    if target.is_file():
        _validate_glb(target)
        status = "cached" if target.stat().st_size == expected_size else "cached_size_drift"
    else:
        target.unlink(missing_ok=True)
        part = target.with_suffix(".glb.part")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            candidate["download_url"],
            headers={"User-Agent": "physcensis-reproduction-asset-curator/0.1"},
        )
        for attempt in range(1, retries + 1):
            try:
                with opener.open(request, timeout=120) as response, part.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                _validate_glb(part)
                os.replace(part, target)
                status = (
                    "downloaded"
                    if target.stat().st_size == expected_size
                    else "downloaded_size_drift"
                )
                break
            except Exception:
                part.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                time.sleep(float(attempt))
        else:
            raise AssertionError("unreachable")
    return {
        **candidate,
        "filename": target.name,
        "downloaded_size": target.stat().st_size,
        "sha256": _sha256(target),
        "local_path": str(target),
        "download_status": status,
    }


def _parse_selections(values: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for value in values:
        try:
            category, uid = value.split("=", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --select {value!r}; expected category=uid") from exc
        result.setdefault(category.strip(), set()).add(uid.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--select", action="append", default=[], metavar="CATEGORY=UID")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    try:
        selections = _parse_selections(args.select)
    except ValueError as exc:
        parser.error(str(exc))
    if not selections:
        parser.error("At least one --select is required")

    shortlist = json.loads(args.shortlist.read_text(encoding="utf-8"))
    candidates = []
    missing = []
    for category, uids in selections.items():
        pool = shortlist["categories"].get(category, {}).get("candidates", [])
        by_uid = {candidate["uid"]: candidate for candidate in pool}
        for uid in sorted(uids):
            if uid not in by_uid:
                missing.append(f"{category}={uid}")
                continue
            candidates.append({**by_uid[uid], "category": category})
    if missing:
        parser.error("Selections not found in shortlist: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download, candidate, args.output_dir): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            downloaded.append(result)
            print(f"{result['download_status']:10s} {result['category']:10s} {result['uid']}")
    downloaded.sort(key=lambda item: (item["category"], item["uid"]))
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_shortlist": str(args.shortlist),
                "assets": downloaded,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
